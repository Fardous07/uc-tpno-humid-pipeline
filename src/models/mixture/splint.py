"""
SPLINT: Spreading-Pressure Liquid INTerface Theory.
[docstring unchanged]
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
    A_ij(π) = a₀ + a₁·(π/π_ref) + a₂·(π/π_ref)²
    """

    def __init__(self, n_components: int = 3, pi_ref: float = 10.0):
        self.n_components = n_components
        self.pi_ref = pi_ref
        self._A_coeffs = np.zeros((n_components, n_components, 3))

    @property
    def n_params(self) -> int:
        n = self.n_components
        return n * (n - 1) // 2 * 3

    def set_params(self, i: int, j: int, a0: float, a1: float = 0.0, a2: float = 0.0) -> None:
        self._A_coeffs[i, j] = [a0, a1, a2]
        self._A_coeffs[j, i] = [a0, a1, a2]

    def _A(self, i: int, j: int, pi: float) -> float:
        t = pi / self.pi_ref
        c = self._A_coeffs[i, j]
        return c[0] + c[1] * t + c[2] * t ** 2

    def activity_coefficients(self, x: np.ndarray, pi: float) -> np.ndarray:
        C = self.n_components
        ln_gamma = np.zeros(C)
        for i in range(C):
            for j in range(C):
                if i == j:
                    continue
                ln_gamma[i] += self._A(i, j, pi) * x[j] ** 2
        return np.exp(ln_gamma)

    def fit(
        self,
        x_data: np.ndarray,
        pi_data: np.ndarray,
        gamma_data: np.ndarray,
    ) -> Dict[str, float]:
        C = self.n_components

        def objective(params_flat):
            idx = 0
            for i in range(C):
                for j in range(i + 1, C):
                    self._A_coeffs[i, j] = params_flat[idx:idx + 3]
                    self._A_coeffs[j, i] = params_flat[idx:idx + 3]
                    idx += 3
            residuals = []
            for n in range(len(x_data)):
                gamma_pred = self.activity_coefficients(x_data[n], pi_data[n])
                residuals.extend(
                    (np.log(gamma_pred) - np.log(gamma_data[n].clip(min=1e-6))).tolist()
                )
            return residuals

        x0 = np.zeros(self.n_params)
        optimize.least_squares(objective, x0, method="trf", max_nfev=5000)

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
    ln γ_i = 1 − ln(Σ_j x_j Λ_ij) − Σ_k (x_k Λ_ki / Σ_j x_j Λ_kj)
    """

    def __init__(self, n_components: int = 3, pi_ref: float = 10.0):
        self.n_components = n_components
        self.pi_ref = pi_ref
        self._lambda_coeffs = np.zeros((n_components, n_components, 2))

    def set_params(self, i: int, j: int, l0: float, l1: float = 0.0) -> None:
        self._lambda_coeffs[i, j] = [l0, l1]

    def _Lambda(self, pi: float) -> np.ndarray:
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

    def activity_coefficients(self, x: np.ndarray, pi: float) -> np.ndarray:
        C = self.n_components
        L = self._Lambda(pi)
        x = np.asarray(x, dtype=np.float64).clip(min=1e-15)

        ln_gamma = np.zeros(C)
        for i in range(C):
            sum_j = np.dot(x, L[i, :])
            ln_gamma[i] = -np.log(max(sum_j, 1e-15))
            for k in range(C):
                denom = np.dot(x, L[k, :])
                ln_gamma[i] -= x[k] * L[k, i] / max(denom, 1e-15)
        ln_gamma += 1.0
        return np.exp(ln_gamma.clip(-10, 10))


# ═══════════════════════════════════════════════════════════════════════
# 4.  SPLINE SPLINT
# ═══════════════════════════════════════════════════════════════════════

class SplineSPLINT:
    """Non-parametric B-spline interpolation of ln(γ_i)(x, π)."""

    def __init__(self, n_components: int = 3, degree: int = 3, n_knots: int = 8):
        self.n_components = n_components
        self.degree = degree
        self.n_knots = n_knots
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
        C = self.n_components
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
                self._splines[i] = interpolate.LinearNDInterpolator(
                    features, ln_gamma_i, fill_value=0.0,
                )
            ln_pred = self._splines[i](features)
            rmse[f"comp_{i}"] = float(np.sqrt(np.mean((ln_pred - ln_gamma_i) ** 2)))

        self._fitted = True
        return rmse

    def activity_coefficients(self, x: np.ndarray, pi: float) -> np.ndarray:
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
    computing γ_i and re-solving the IAST system with γ until convergence.

    FIX in predict():
    The original RAST loop computed
        P0_modified[i] = y_i * P / (x_i * γ_i)
        x_new[i]       = y_i * P / (γ_i * P0_modified[i])
    which substitutes back to x_new[i] = x[i] identically — a circular
    derivation.  dx = 0 on the first pass, so converged = True
    immediately and x never changed.

    Fix: after computing γ, re-solve the full IAST system with the
    modified sum constraint Σ (y_i * P / (γ_i * P_i⁰)) = 1 while
    keeping the equal-spreading-pressure condition unchanged.  This
    gives genuinely new P_i⁰ values, from which x_new = y * P / (γ * P⁰)
    is derived.  The loop then converges to the RAST fixed point.
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

    # ------------------------------------------------------------------
    # FIX: helper to re-solve IAST with γ incorporated
    # ------------------------------------------------------------------

    def _solve_rast(
        self,
        y: np.ndarray,
        P_total: float,
        gamma: np.ndarray,
        P0_prev: np.ndarray,
        active: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Solve the modified IAST system with activity coefficients γ_i.

        Equal spreading pressures still hold:
            π_i(P_i⁰) = π_j(P_j⁰)   for all i, j

        The sum constraint is modified to include γ:
            Σ_i  y_i * P / (γ_i * P_i⁰) = 1

        Returns updated P0 array, or None on failure.
        """
        n_active = len(active)

        def equations(log_P0_active: np.ndarray) -> np.ndarray:
            P0_a = np.exp(log_P0_active)
            resid = np.zeros(n_active)

            sp_ref = self.iast.isotherms[active[0]].spreading_pressure(P0_a[0])
            for k in range(1, n_active):
                sp_k = self.iast.isotherms[active[k]].spreading_pressure(P0_a[k])
                resid[k] = sp_k - sp_ref

            # Modified sum: Σ y_i * P / (γ_i * P_i⁰) = 1
            s = 0.0
            for k in range(n_active):
                i = active[k]
                g = max(gamma[i], 1e-15)
                s += y[i] * P_total / (g * P0_a[k])
            resid[0] = s - 1.0
            return resid

        log_P0_init = np.log(P0_prev[active].clip(min=1e-15))
        result = optimize.root(
            equations, log_P0_init,
            method="hybr",
            options={"maxfev": 500, "xtol": self.config.rast_tol},
        )
        if not result.success:
            result = optimize.root(
                equations, log_P0_init,
                method="lm",
                options={"maxiter": 500, "xtol": self.config.rast_tol},
            )
        if not result.success:
            return None

        P0_new = P0_prev.copy()
        P0_new[active] = np.exp(result.x)
        return P0_new

    # ------------------------------------------------------------------
    # RAST fixed-point predict
    # ------------------------------------------------------------------

    def predict(
        self,
        y: Sequence[float],
        P_total: float,
    ) -> Dict[str, Any]:
        """
        Predict mixture adsorption via self-consistent RAST.

        1.  Start from IAST solution (γ = 1).
        2.  Compute γ_i from the activity-coefficient backend.
        3.  Re-solve IAST with modified Raoult's law: γ incorporated in
            the sum constraint (via _solve_rast).
        4.  Derive x_new = y_i * P / (γ_i * P_i⁰_new).
        5.  Iterate until convergence in both x and γ.
        """
        y = np.asarray(y, dtype=np.float64)
        C = self.config.n_components

        # Step 1: IAST baseline (γ = 1)
        iast_result = self.iast.predict(y, P_total)
        x = iast_result["x"].copy()
        pi = iast_result["spreading_pressure"]
        q_iast = iast_result["loadings"].copy()
        P0 = iast_result["P0"].copy()

        if np.any(np.isnan(q_iast)):
            return self._nan_result(C, q_iast)

        active = np.where(y > 1e-15)[0]
        if len(active) == 0:
            return self._nan_result(C, q_iast)

        gamma = np.ones(C)
        converged = False
        dx = float("inf")
        dgamma = float("inf")

        for iteration in range(self.config.max_rast_iter):
            # Step 2: compute γ from current x and π
            gamma_new = self.backend.activity_coefficients(x, pi)

            # Step 3: re-solve IAST with γ incorporated
            #
            # FIX: previously P0 was recomputed as y_i*P/(x_i*γ_i) and then
            # x_new = y_i*P/(γ_i*P0_modified) which trivially equals x_i.
            # Now we actually solve the modified IAST system so P0 genuinely
            # changes and x_new ≠ x.
            P0_new = self._solve_rast(y, P_total, gamma_new, P0, active)
            if P0_new is None:
                logger.warning(
                    f"RAST sub-solver failed at iteration {iteration}; "
                    "keeping previous P0."
                )
                P0_new = P0.copy()

            # Step 4: derive x from new P0 and γ
            x_new = np.zeros(C)
            for i in active:
                g = max(gamma_new[i], 1e-15)
                x_new[i] = y[i] * P_total / (g * max(P0_new[i], 1e-15))
            x_sum = x_new.sum()
            if x_sum > 1e-15:
                x_new /= x_sum

            # Update spreading pressure (x-weighted average of π_i(P_i⁰))
            pi_new = float(np.sum(
                x_new[i] * self.iast.isotherms[i].spreading_pressure(P0_new[i])
                for i in active
                if x_new[i] > 1e-15
            ))

            # Check convergence
            dx = np.abs(x_new - x).max()
            dgamma = np.abs(gamma_new - gamma).max()

            x = x_new
            gamma = gamma_new
            P0 = P0_new
            pi = pi_new

            if dx < self.config.rast_tol and dgamma < self.config.rast_tol:
                converged = True
                break

        # Compute final loadings via reciprocal mixing rule
        recip = 0.0
        for i in active:
            if x[i] > 1e-15:
                g = max(gamma[i], 1e-15)
                P0_i = max(P0[i], 1e-15)
                q_i0 = float(self.iast.isotherms[i].loading(np.array([P0_i]))[0])
                if q_i0 > 1e-15:
                    recip += x[i] / q_i0

        n_total = 1.0 / max(recip, 1e-15)
        loadings = x * n_total

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