"""
SPLINT: Spreading-Pressure Liquid INTerface Theory.

This module implements the **Real Adsorbed Solution Theory** (RAST)
framework where activity coefficients γ_i depend on the adsorbed-
phase composition **and** the spreading pressure π.  This extends
classical activity-coefficient models (Margules, Wilson, NRTL) from
bulk liquid mixtures to the confined adsorbed phase inside MOF pores.

Why spreading-pressure dependence?
──────────────────────────────────
In bulk liquids, activity coefficients γ_i(x, T) depend only on
composition and temperature.  In adsorption, the pore environment
changes with loading, so γ_i should also depend on the spreading
pressure π (which is a proxy for total surface coverage).  The
SPLINT approach parameterises:

    ln γ_i = f(x, π, T, MOF descriptors)

where *f* can be a classical functional form (Margules, Wilson)
with π-dependent parameters, or a flexible spline/polynomial.

Components
──────────
1.  **MargulesSPLINT** — symmetric 1-parameter Margules model with
    π-dependent interaction parameter ``A(π)``.
2.  **WilsonSPLINT** — Wilson equation with π-dependent binary
    parameters ``Λ_ij(π)``.
3.  **SplineSPLINT** — non-parametric: B-spline interpolation of
    ``ln γ`` as a function of (x, π), fitted to GCMC data.
4.  **SPLINTModel** — unified interface that wraps any of the above
    and integrates with the IAST solver for RAST predictions.

Integration
───────────
*   ``iast.py`` provides the base IAST solution + spreading pressure.
*   ``SPLINTModel.predict()`` takes IAST results, computes γ, and
    iterates to self-consistency (RAST fixed-point loop).
*   Compared against the neural mixture model in benchmarking.

References
──────────
[1] Myers (1983). Activity Coefficients of Mixtures Adsorbed on
    Heterogeneous Surfaces. AIChE Journal.
[2] Siperstein & Myers (2001). Mixed-Gas Adsorption. AIChE Journal.
[3] Cessford et al. (2012). Evaluation of Ideal Adsorbed Solution
    Theory (IAST). J. Phys. Chem. C.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import interpolate, optimize

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SPLINTConfig:
    """
    SPLINT model hyperparameters.

    Attributes
    ──────────
    model_type     : ``'margules'``, ``'wilson'``, or ``'spline'``.
    n_components   : Number of adsorbate species.
    max_rast_iter  : Maximum RAST fixed-point iterations.
    rast_tol       : RAST convergence tolerance.
    pi_ref         : Reference spreading pressure for normalisation.
    spline_degree  : B-spline degree (for ``'spline'`` model).
    spline_knots   : Number of interior knots per dimension.
    """

    model_type: str = "margules"
    n_components: int = 3
    max_rast_iter: int = 50
    rast_tol: float = 1e-8
    pi_ref: float = 10.0
    spline_degree: int = 3
    spline_knots: int = 8


# ═══════════════════════════════════════════════════════════════════════
# 2.  MARGULES SPLINT
# ═══════════════════════════════════════════════════════════════════════

class MargulesSPLINT:
    """
    Margules activity-coefficient model with π-dependent parameters.

    For a binary (i, j)::

        ln γ_i = A_ij(π) · x_j²

    where ``A_ij(π) = a₀ + a₁·(π/π_ref) + a₂·(π/π_ref)²``.

    For multicomponent systems, the one-constant Margules form is
    extended via pairwise summation.

    Parameters
    ----------
    n_components : Number of species.
    pi_ref       : Reference spreading pressure for normalisation.
    """

    def __init__(self, n_components: int = 3, pi_ref: float = 10.0):
        self.n_components = n_components
        self.pi_ref = pi_ref

        # Polynomial coefficients for A_ij(π): [n_comp, n_comp, 3]
        # A_ij = a0 + a1*(π/π_ref) + a2*(π/π_ref)²
        self._A_coeffs = np.zeros((n_components, n_components, 3))

    @property
    def n_params(self) -> int:
        """Number of free parameters (upper-triangle pairs × 3 coeffs)."""
        n = self.n_components
        return n * (n - 1) // 2 * 3

    def set_params(self, i: int, j: int, a0: float, a1: float = 0.0, a2: float = 0.0) -> None:
        """Set the interaction parameters for pair (i, j)."""
        self._A_coeffs[i, j] = [a0, a1, a2]
        self._A_coeffs[j, i] = [a0, a1, a2]  # symmetric

    def _A(self, i: int, j: int, pi: float) -> float:
        """Evaluate A_ij at spreading pressure π."""
        t = pi / self.pi_ref
        c = self._A_coeffs[i, j]
        return c[0] + c[1] * t + c[2] * t ** 2

    def activity_coefficients(
        self,
        x: np.ndarray,
        pi: float,
    ) -> np.ndarray:
        """
        Compute activity coefficients γ_i.

        Parameters
        ----------
        x  : ``[C]`` adsorbed-phase mole fractions.
        pi : Reduced spreading pressure πA/(RT).

        Returns
        -------
        ``[C]`` activity coefficients.
        """
        C = self.n_components
        ln_gamma = np.zeros(C)

        for i in range(C):
            for j in range(C):
                if i == j:
                    continue
                A_ij = self._A(i, j, pi)
                ln_gamma[i] += A_ij * x[j] ** 2

        return np.exp(ln_gamma)

    def fit(
        self,
        x_data: np.ndarray,
        pi_data: np.ndarray,
        gamma_data: np.ndarray,
    ) -> Dict[str, float]:
        """
        Fit Margules parameters from GCMC-derived activity coefficients.

        Parameters
        ----------
        x_data     : ``[N, C]`` compositions.
        pi_data    : ``[N]`` spreading pressures.
        gamma_data : ``[N, C]`` activity coefficients from GCMC.

        Returns
        -------
        Dict with RMSE per component.
        """
        C = self.n_components

        def objective(params_flat):
            # Unpack
            idx = 0
            for i in range(C):
                for j in range(i + 1, C):
                    self._A_coeffs[i, j] = params_flat[idx:idx + 3]
                    self._A_coeffs[j, i] = params_flat[idx:idx + 3]
                    idx += 3

            residuals = []
            for n in range(len(x_data)):
                gamma_pred = self.activity_coefficients(x_data[n], pi_data[n])
                residuals.extend((np.log(gamma_pred) - np.log(gamma_data[n].clip(min=1e-6))).tolist())
            return residuals

        x0 = np.zeros(self.n_params)
        result = optimize.least_squares(objective, x0, method="trf", max_nfev=5000)

        # Compute RMSE
        rmse = {}
        for i in range(C):
            gamma_pred_all = np.array([
                self.activity_coefficients(x_data[n], pi_data[n])[i]
                for n in range(len(x_data))
            ])
            rmse[f"comp_{i}"] = float(np.sqrt(np.mean((gamma_pred_all - gamma_data[:, i]) ** 2)))

        return rmse


# ═══════════════════════════════════════════════════════════════════════
# 3.  WILSON SPLINT
# ═══════════════════════════════════════════════════════════════════════

class WilsonSPLINT:
    """
    Wilson equation with π-dependent binary parameters.

    For multicomponent::

        ln γ_i = 1 − ln(Σ_j x_j Λ_ij) − Σ_k (x_k Λ_ki / Σ_j x_j Λ_kj)

    where ``Λ_ij(π) = exp(−(λ₀ + λ₁·π/π_ref) / RT_ref)``
    (for adsorption, RT_ref is absorbed into the parameters).

    Parameters
    ----------
    n_components : Number of species.
    pi_ref       : Reference spreading pressure.
    """

    def __init__(self, n_components: int = 3, pi_ref: float = 10.0):
        self.n_components = n_components
        self.pi_ref = pi_ref

        # Lambda parameters: Λ_ij = exp(−(l0 + l1*(π/π_ref)))
        # Diagonal: Λ_ii = 1
        self._lambda_coeffs = np.zeros((n_components, n_components, 2))

    def set_params(self, i: int, j: int, l0: float, l1: float = 0.0) -> None:
        """Set Wilson binary parameters for pair (i, j)."""
        self._lambda_coeffs[i, j] = [l0, l1]

    def _Lambda(self, pi: float) -> np.ndarray:
        """Return ``[C, C]`` Wilson Λ matrix at spreading pressure π."""
        t = pi / self.pi_ref
        C = self.n_components
        L = np.eye(C)
        for i in range(C):
            for j in range(C):
                if i == j:
                    continue
                c = self._lambda_coeffs[i, j]
                L[i, j] = np.exp(-(c[0] + c[1] * t))
        return L

    def activity_coefficients(
        self,
        x: np.ndarray,
        pi: float,
    ) -> np.ndarray:
        """
        Compute Wilson activity coefficients.

        Parameters
        ----------
        x  : ``[C]`` mole fractions.
        pi : Spreading pressure.

        Returns
        -------
        ``[C]`` activity coefficients.
        """
        C = self.n_components
        L = self._Lambda(pi)
        x = np.asarray(x, dtype=np.float64).clip(min=1e-15)

        ln_gamma = np.zeros(C)
        for i in range(C):
            # Term 1: -ln(Σ_j x_j Λ_ij)
            sum_j = np.dot(x, L[i, :])
            ln_gamma[i] = -np.log(max(sum_j, 1e-15))

            # Term 2: -Σ_k (x_k Λ_ki / Σ_j x_j Λ_kj)
            for k in range(C):
                denom = np.dot(x, L[k, :])
                ln_gamma[i] -= x[k] * L[k, i] / max(denom, 1e-15)

        # +1 for the Wilson normalization
        ln_gamma += 1.0

        return np.exp(ln_gamma.clip(-10, 10))


# ═══════════════════════════════════════════════════════════════════════
# 4.  SPLINE SPLINT
# ═══════════════════════════════════════════════════════════════════════

class SplineSPLINT:
    """
    Non-parametric B-spline interpolation of ln(γ_i) as a function
    of adsorbed-phase composition x and spreading pressure π.

    This is the most flexible SPLINT variant — no assumed functional
    form — but requires more data to fit.

    For a ternary system, the input space is (x₁, x₂, π) and one
    spline is fitted per component.

    Parameters
    ----------
    n_components : Number of species.
    degree       : B-spline degree.
    n_knots      : Interior knots per dimension.
    """

    def __init__(
        self,
        n_components: int = 3,
        degree: int = 3,
        n_knots: int = 8,
    ):
        self.n_components = n_components
        self.degree = degree
        self.n_knots = n_knots

        # One spline per component (fitted lazily)
        self._splines: List[Optional[Any]] = [None] * n_components
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        x_data: np.ndarray,
        pi_data: np.ndarray,
        gamma_data: np.ndarray,
        smoothing: float = 0.1,
    ) -> Dict[str, float]:
        """
        Fit splines from observed activity coefficients.

        Parameters
        ----------
        x_data     : ``[N, C]`` compositions.
        pi_data    : ``[N]`` spreading pressures.
        gamma_data : ``[N, C]`` activity coefficients.
        smoothing  : Smoothing factor for ``RBFInterpolator``.

        Returns
        -------
        Dict with per-component RMSE.
        """
        C = self.n_components
        N = len(x_data)

        # Input features: [x_1, ..., x_{C-1}, π]
        # (last mole fraction is determined by closure)
        features = np.column_stack([x_data[:, :-1], pi_data])

        rmse = {}
        for i in range(C):
            ln_gamma_i = np.log(gamma_data[:, i].clip(min=1e-10))

            try:
                self._splines[i] = interpolate.RBFInterpolator(
                    features, ln_gamma_i,
                    kernel="thin_plate_spline",
                    smoothing=smoothing,
                )
            except Exception:
                # Fallback to linear interpolation
                self._splines[i] = interpolate.LinearNDInterpolator(
                    features, ln_gamma_i, fill_value=0.0,
                )

            # RMSE
            ln_pred = self._splines[i](features)
            rmse[f"comp_{i}"] = float(np.sqrt(np.mean((ln_pred - ln_gamma_i) ** 2)))

        self._fitted = True
        return rmse

    def activity_coefficients(
        self,
        x: np.ndarray,
        pi: float,
    ) -> np.ndarray:
        """
        Evaluate spline-interpolated activity coefficients.

        Parameters
        ----------
        x  : ``[C]`` mole fractions.
        pi : Spreading pressure.

        Returns
        -------
        ``[C]`` activity coefficients.
        """
        if not self._fitted:
            raise RuntimeError("SplineSPLINT not fitted. Call fit() first.")

        C = self.n_components
        features = np.concatenate([x[:-1], [pi]]).reshape(1, -1)

        gamma = np.ones(C)
        for i in range(C):
            if self._splines[i] is not None:
                ln_g = float(self._splines[i](features).ravel()[0])
                gamma[i] = np.exp(np.clip(ln_g, -10, 10))

        return gamma


# ═══════════════════════════════════════════════════════════════════════
# 5.  UNIFIED SPLINT MODEL
# ═══════════════════════════════════════════════════════════════════════

class SPLINTModel:
    """
    Unified SPLINT interface for RAST predictions.

    Wraps any activity-coefficient backend (Margules, Wilson, Spline)
    and provides a self-consistent RAST solver that iterates between
    IAST with γ ≠ 1 until convergence.

    Parameters
    ----------
    iast_calc : An ``IASTCalculator`` (from ``iast.py``).
    config    : ``SPLINTConfig`` or dict.

    Example
    ───────
    >>> from src.models.mixture.iast import IASTCalculator, Langmuir
    >>> iast = IASTCalculator([Langmuir(5,0.8), Langmuir(4,0.05)])
    >>> splint = SPLINTModel(iast, SPLINTConfig(model_type='margules'))
    >>> splint.backend.set_params(0, 1, a0=0.5)
    >>> result = splint.predict(y=[0.15, 0.85], P_total=1.0)
    """

    def __init__(
        self,
        iast_calc: Any,
        config: Optional[Union[SPLINTConfig, Dict]] = None,
    ):
        self.iast = iast_calc

        if config is None:
            config = SPLINTConfig()
        elif isinstance(config, dict):
            config = SPLINTConfig(**{
                k: v for k, v in config.items()
                if k in SPLINTConfig.__dataclass_fields__
            })
        self.config = config

        # Build backend
        C = config.n_components
        if config.model_type == "margules":
            self.backend = MargulesSPLINT(C, config.pi_ref)
        elif config.model_type == "wilson":
            self.backend = WilsonSPLINT(C, config.pi_ref)
        elif config.model_type == "spline":
            self.backend = SplineSPLINT(C, config.spline_degree, config.spline_knots)
        else:
            raise ValueError(
                f"Unknown model_type '{config.model_type}'. "
                "Choose 'margules', 'wilson', or 'spline'."
            )

    def predict(
        self,
        y: Sequence[float],
        P_total: float,
    ) -> Dict[str, Any]:
        """
        Predict mixture adsorption via self-consistent RAST.

        1.  Start from IAST solution (γ = 1).
        2.  Compute γ_i from the activity-coefficient backend.
        3.  Re-solve with modified Raoult's law: ``P_i = x_i γ_i P_i⁰``.
        4.  Iterate until convergence.

        Parameters
        ----------
        y       : Gas-phase mole fractions.
        P_total : Total pressure [bar].

        Returns
        -------
        Dict with ``"loadings"``, ``"gamma"``, ``"iast_loadings"``,
        ``"x"``, ``"selectivity"``, ``"converged"``, ``"iterations"``.
        """
        y = np.asarray(y, dtype=np.float64)
        C = self.config.n_components

        # Step 1: IAST baseline
        iast_result = self.iast.predict(y, P_total)
        x = iast_result["x"].copy()
        pi = iast_result["spreading_pressure"]
        q_iast = iast_result["loadings"].copy()

        if np.any(np.isnan(q_iast)):
            return self._nan_result(C, q_iast)

        # Step 2–4: RAST fixed-point iteration
        gamma = np.ones(C)
        converged = False

        for iteration in range(self.config.max_rast_iter):
            gamma_new = self.backend.activity_coefficients(x, pi)

            # Modified partial pressures: P_i = y_i · P / γ_i
            # (In RAST, the hypothetical pressure becomes P_i⁰/γ_i)
            P0_modified = np.zeros(C)
            for i in range(C):
                if x[i] > 1e-15 and gamma_new[i] > 1e-15:
                    P0_modified[i] = y[i] * P_total / (x[i] * gamma_new[i])
                else:
                    P0_modified[i] = P_total

            # Re-compute x from modified pressures
            x_new = np.zeros(C)
            for i in range(C):
                if P0_modified[i] > 1e-15:
                    x_new[i] = y[i] * P_total / (gamma_new[i] * P0_modified[i])

            x_sum = x_new.sum()
            if x_sum > 1e-15:
                x_new /= x_sum

            # Update spreading pressure (average)
            pi_new = 0.0
            for i in range(C):
                if x_new[i] > 1e-15:
                    pi_new += x_new[i] * self.iast.isotherms[i].spreading_pressure(P0_modified[i])

            # Check convergence
            dx = np.abs(x_new - x).max()
            dgamma = np.abs(gamma_new - gamma).max()

            x = x_new
            gamma = gamma_new
            pi = pi_new

            if dx < self.config.rast_tol and dgamma < self.config.rast_tol:
                converged = True
                break

        # Compute final loadings via reciprocal mixing rule
        recip = 0.0
        for i in range(C):
            if x[i] > 1e-15:
                P0_i = y[i] * P_total / (x[i] * gamma[i]) if gamma[i] > 1e-15 else P_total
                q_i0 = float(self.iast.isotherms[i].loading(np.array([P0_i]))[0])
                if q_i0 > 1e-15:
                    recip += x[i] / q_i0

        n_total = 1.0 / max(recip, 1e-15)
        loadings = x * n_total

        # Selectivity
        selectivity = None
        if C >= 2 and loadings[1] > 1e-15 and y[1] > 1e-15:
            selectivity = float((loadings[0] / loadings[1]) / (y[0] / y[1]))

        if not converged:
            logger.warning(
                f"RAST did not converge in {self.config.max_rast_iter} iterations "
                f"(dx={dx:.2e}, dγ={dgamma:.2e})."
            )

        return {
            "loadings": loadings,
            "gamma": gamma,
            "iast_loadings": q_iast,
            "x": x,
            "spreading_pressure": pi,
            "selectivity": selectivity,
            "converged": converged,
            "iterations": iteration + 1,
        }

    def predict_batch(
        self,
        y_batch: np.ndarray,
        P_batch: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Batch RAST predictions."""
        B = y_batch.shape[0]
        C = self.config.n_components

        loadings = np.zeros((B, C))
        gammas = np.zeros((B, C))
        sel = np.full(B, np.nan)

        for i in range(B):
            result = self.predict(y_batch[i], float(P_batch[i]))
            loadings[i] = result["loadings"]
            gammas[i] = result["gamma"]
            if result["selectivity"] is not None:
                sel[i] = result["selectivity"]

        return {"loadings": loadings, "gamma": gammas, "selectivity": sel}

    def _nan_result(self, C: int, q_iast: np.ndarray) -> Dict[str, Any]:
        return {
            "loadings": np.full(C, np.nan),
            "gamma": np.ones(C),
            "iast_loadings": q_iast,
            "x": np.full(C, np.nan),
            "spreading_pressure": np.nan,
            "selectivity": np.nan,
            "converged": False,
            "iterations": 0,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "model_type": self.config.model_type,
            "n_components": self.config.n_components,
            "backend": type(self.backend).__name__,
        }


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "SPLINTConfig",
    "SPLINTModel",
    "MargulesSPLINT",
    "WilsonSPLINT",
    "SplineSPLINT",
]