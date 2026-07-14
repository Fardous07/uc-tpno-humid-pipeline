"""
Ideal Adsorbed Solution Theory (IAST) for multicomponent adsorption.

IAST (Myers & Prausnitz, 1965) predicts mixture adsorption equilibria
from **single-component isotherms** alone, assuming the adsorbed
phase behaves as an ideal solution.  It is the standard baseline for
multicomponent adsorption prediction in MOFs and the benchmark
against which the neural mixture model is compared.

How it works
────────────
1.  Fit single-component isotherms q_i(P) to pure-component data
    (Langmuir, dual-site Langmuir, Freundlich, Sips, BET, or
    interpolated).
2.  Compute the **spreading pressure** π for each component via
    numerical integration of q_i(P)/P from 0 to P_i⁰.
3.  At equilibrium, all spreading pressures must be equal:
    ``π₁(P₁⁰) = π₂(P₂⁰) = … = π_N(P_N⁰)``.
4.  Combined with the Raoult's-law analogue ``P_i = x_i · P_i⁰``
    and ``Σ x_i = 1``, solve for the mole fractions x_i and total
    loading n_total.

Components
──────────
*  **IsothermModel** (Protocol) — interface for pure-component
   isotherms with ``loading(P)`` and ``spreading_pressure(P)``
   methods.
*  **Langmuir / DualSiteLangmuir / Freundlich / Sips / BET /
   InterpolatedIsotherm** — concrete isotherm classes.
*  **fit_isotherm** — convenience function that fits an isotherm to
   (P, q) data with least-squares optimisation.
*  **IASTCalculator** — the main solver class.  Given gas-phase
   composition y_i and total pressure P, returns per-component
   loadings q_i.
*  **reverse_iast** — given a target loading, find the pressure at
   which it is achieved (useful for process design).

Integration
───────────
The ``IASTCalculator`` is used:
*   As a physics baseline in ``src/evaluation/benchmarking.py``.
*   As a constraint / regulariser in the neural mixture model.
*   For PVSA process simulation where fast mixture prediction is
    needed (``src/models/process/pvsa.py``).

References
──────────
[1] Myers & Prausnitz (1965). Thermodynamics of Mixed‐Gas
    Adsorption. AIChE Journal.
[2] Simon et al. (2016). pyIAST: Ideal Adsorbed Solution Theory
    (IAST) Python Package. Computer Physics Communications.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Protocol, Sequence,
    Tuple, Type, Union, runtime_checkable,
)

import numpy as np
from scipy import integrate, optimize

logger = logging.getLogger(__name__)

# Gas constant [bar·cm³·mol⁻¹·K⁻¹] — convenient for adsorption units
_R_cm3 = 83.14462618


# ═══════════════════════════════════════════════════════════════════════
# 1.  ISOTHERM PROTOCOL & MODELS
# ═══════════════════════════════════════════════════════════════════════

@runtime_checkable
class IsothermModel(Protocol):
    """Interface that every isotherm must implement."""

    def loading(self, pressure: np.ndarray) -> np.ndarray:
        """Return loading q [mol/kg] at given pressure(s) [bar]."""
        ...

    def spreading_pressure(self, pressure: float) -> float:
        r"""
        Reduced spreading pressure πA/(RT) via numerical integration:

        .. math::
            \frac{\pi A}{RT} = \int_0^P \frac{q(P')}{P'} \, dP'
        """
        ...


# ── Langmuir ─────────────────────────────────────────────────────

@dataclass
class Langmuir:
    """
    Langmuir isotherm: ``q = q_sat · K·P / (1 + K·P)``.

    Parameters
    ----------
    q_sat : Saturation capacity [mol/kg].
    K     : Langmuir constant [1/bar].
    """

    q_sat: float
    K: float

    def loading(self, pressure: np.ndarray) -> np.ndarray:
        P = np.asarray(pressure, dtype=np.float64)
        return self.q_sat * self.K * P / (1.0 + self.K * P)

    def spreading_pressure(self, pressure: float) -> float:
        # Analytical: q_sat * ln(1 + K*P)
        return self.q_sat * np.log(1.0 + self.K * max(pressure, 0.0))

    @property
    def henry(self) -> float:
        """Henry's law constant K_H = q_sat · K [mol/(kg·bar)]."""
        return self.q_sat * self.K

    def params(self) -> Dict[str, float]:
        return {"q_sat": self.q_sat, "K": self.K}


# ── Dual-Site Langmuir ───────────────────────────────────────────

@dataclass
class DualSiteLangmuir:
    """
    Dual-site Langmuir (DSL): ``q = q1·K1·P/(1+K1·P) + q2·K2·P/(1+K2·P)``.

    Parameters
    ----------
    q_sat1, K1 : Site 1 capacity and affinity.
    q_sat2, K2 : Site 2 capacity and affinity.
    """

    q_sat1: float
    K1: float
    q_sat2: float
    K2: float

    def loading(self, pressure: np.ndarray) -> np.ndarray:
        P = np.asarray(pressure, dtype=np.float64)
        return (self.q_sat1 * self.K1 * P / (1.0 + self.K1 * P)
                + self.q_sat2 * self.K2 * P / (1.0 + self.K2 * P))

    def spreading_pressure(self, pressure: float) -> float:
        P = max(pressure, 0.0)
        return (self.q_sat1 * np.log(1.0 + self.K1 * P)
                + self.q_sat2 * np.log(1.0 + self.K2 * P))

    @property
    def henry(self) -> float:
        return self.q_sat1 * self.K1 + self.q_sat2 * self.K2

    def params(self) -> Dict[str, float]:
        return {"q_sat1": self.q_sat1, "K1": self.K1,
                "q_sat2": self.q_sat2, "K2": self.K2}


# ── Freundlich ───────────────────────────────────────────────────

@dataclass
class Freundlich:
    """
    Freundlich isotherm: ``q = K_F · P^(1/n)``.

    Note: no saturation capacity — diverges at high pressure.  The
    spreading-pressure integral diverges at P → ∞, so IAST with
    Freundlich requires a pressure ceiling.

    Parameters
    ----------
    K_F : Freundlich constant [mol/(kg·bar^(1/n))].
    n   : Freundlich exponent (≥ 1).
    """

    K_F: float
    n: float

    def loading(self, pressure: np.ndarray) -> np.ndarray:
        P = np.asarray(pressure, dtype=np.float64).clip(min=0.0)
        return self.K_F * P ** (1.0 / self.n)

    def spreading_pressure(self, pressure: float) -> float:
        # Analytical: K_F * n * P^(1/n)
        P = max(pressure, 0.0)
        return self.K_F * self.n * P ** (1.0 / self.n)

    @property
    def henry(self) -> float:
        # Only defined for n = 1; otherwise infinite at P → 0⁺
        if abs(self.n - 1.0) < 1e-6:
            return self.K_F
        return float("inf")

    def params(self) -> Dict[str, float]:
        return {"K_F": self.K_F, "n": self.n}


# ── Sips (Langmuir–Freundlich) ───────────────────────────────────

@dataclass
class Sips:
    """
    Sips isotherm: ``q = q_sat · (K·P)^(1/n) / (1 + (K·P)^(1/n))``.

    Combines Langmuir saturation with Freundlich heterogeneity.

    Parameters
    ----------
    q_sat : Saturation capacity [mol/kg].
    K     : Affinity constant [1/bar].
    n     : Heterogeneity exponent (≥ 1; n = 1 → Langmuir).
    """

    q_sat: float
    K: float
    n: float

    def loading(self, pressure: np.ndarray) -> np.ndarray:
        P = np.asarray(pressure, dtype=np.float64).clip(min=0.0)
        KP = (self.K * P) ** (1.0 / self.n)
        return self.q_sat * KP / (1.0 + KP)

    def spreading_pressure(self, pressure: float) -> float:
        # No closed-form; numerical integration
        P = max(pressure, 0.0)
        if P < 1e-15:
            return 0.0

        inv_n = 1.0 / self.n

        def integrand(p: float) -> float:
            if p < 1e-12:
                # Taylor expansion near zero: q/p ≈ q_sat * K^(1/n) * p^(1/n - 1)
                return self.q_sat * (self.K ** inv_n) * (max(p, 1e-30) ** (inv_n - 1.0))
            kp = (self.K * p) ** inv_n
            return self.q_sat * kp / (p * (1.0 + kp))

        result, _ = integrate.quad(integrand, 1e-10, P, limit=200)
        return max(result, 0.0)

    def params(self) -> Dict[str, float]:
        return {"q_sat": self.q_sat, "K": self.K, "n": self.n}


# ── BET ──────────────────────────────────────────────────────────

@dataclass
class BET:
    """
    BET isotherm: ``q = q_m · C·x / ((1−x)(1−x+C·x))``
    where ``x = P / P_sat``.

    Parameters
    ----------
    q_m   : Monolayer capacity [mol/kg].
    C     : BET constant (related to adsorption energy).
    P_sat : Saturation vapour pressure [bar].
    """

    q_m: float
    C: float
    P_sat: float

    def loading(self, pressure: np.ndarray) -> np.ndarray:
        P = np.asarray(pressure, dtype=np.float64).clip(min=0.0)
        x = P / self.P_sat
        x = x.clip(max=0.9999)  # avoid divergence at P_sat
        return self.q_m * self.C * x / ((1.0 - x) * (1.0 - x + self.C * x))

    def spreading_pressure(self, pressure: float) -> float:
        P = max(pressure, 0.0)
        if P < 1e-15:
            return 0.0

        def integrand(p: float) -> float:
            if p < 1e-15:
                return self.q_m * self.C / self.P_sat
            x = min(p / self.P_sat, 0.9999)
            q = self.q_m * self.C * x / ((1.0 - x) * (1.0 - x + self.C * x))
            return q / p

        result, _ = integrate.quad(integrand, 1e-10, P, limit=200)
        return result

    def params(self) -> Dict[str, float]:
        return {"q_m": self.q_m, "C": self.C, "P_sat": self.P_sat}


# ── Interpolated (from tabular data) ────────────────────────────

@dataclass
class InterpolatedIsotherm:
    """
    Isotherm from tabular (P, q) data with piecewise-linear interpolation.

    Useful when no analytical model fits well, or for simulation-
    generated isotherms.

    Parameters
    ----------
    pressures : 1-D array of pressures [bar] (ascending).
    loadings  : 1-D array of loadings [mol/kg].
    fill_value: Loading value to return beyond data range.
    """

    pressures: np.ndarray
    loadings: np.ndarray
    fill_value: float = 0.0

    def __post_init__(self):
        self.pressures = np.asarray(self.pressures, dtype=np.float64)
        self.loadings = np.asarray(self.loadings, dtype=np.float64)
        assert len(self.pressures) == len(self.loadings)
        # Sort by pressure
        order = np.argsort(self.pressures)
        self.pressures = self.pressures[order]
        self.loadings = self.loadings[order]

    def loading(self, pressure: np.ndarray) -> np.ndarray:
        P = np.asarray(pressure, dtype=np.float64)
        return np.interp(P, self.pressures, self.loadings,
                         left=0.0, right=self.fill_value)

    def spreading_pressure(self, pressure: float) -> float:
        P = max(pressure, 0.0)
        if P < 1e-15:
            return 0.0

        def integrand(p: float) -> float:
            q = float(np.interp(p, self.pressures, self.loadings, left=0.0))
            return q / max(p, 1e-10)

        result, _ = integrate.quad(integrand, 1e-10, P, limit=200)
        return result


# ═══════════════════════════════════════════════════════════════════════
# 2.  ISOTHERM FITTING
# ═══════════════════════════════════════════════════════════════════════

_MODEL_REGISTRY: Dict[str, Type] = {
    "langmuir": Langmuir,
    "dsl": DualSiteLangmuir,
    "freundlich": Freundlich,
    "sips": Sips,
    "bet": BET,
}


def fit_isotherm(
    pressures: np.ndarray,
    loadings: np.ndarray,
    model: str = "langmuir",
    p0: Optional[Dict[str, float]] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> IsothermModel:
    """
    Fit a pure-component isotherm to (P, q) data.

    Parameters
    ----------
    pressures : 1-D array of pressures [bar].
    loadings  : 1-D array of loadings [mol/kg].
    model     : ``'langmuir'``, ``'dsl'``, ``'freundlich'``,
                ``'sips'``, ``'bet'``, or ``'interpolated'``.
    p0        : Initial parameter guesses (optional).
    bounds    : Parameter bounds (optional).

    Returns
    -------
    A fitted isotherm instance.
    """
    P = np.asarray(pressures, dtype=np.float64)
    q = np.asarray(loadings, dtype=np.float64)

    if model == "interpolated":
        return InterpolatedIsotherm(P, q)

    if model not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model}'. Choose from "
            f"{list(_MODEL_REGISTRY) + ['interpolated']}"
        )

    # Default initial guesses and bounds
    q_max = q.max() if len(q) > 0 else 1.0
    K_guess = float(q_max / max(P.mean(), 1e-8))

    defaults: Dict[str, Dict] = {
        "langmuir": {
            "p0": [q_max * 1.2, K_guess],
            "bounds": ([0, 0], [q_max * 10, K_guess * 100]),
            "names": ["q_sat", "K"],
        },
        "dsl": {
            "p0": [q_max * 0.6, K_guess, q_max * 0.6, K_guess * 0.1],
            "bounds": ([0, 0, 0, 0], [q_max * 10, K_guess * 100] * 2),
            "names": ["q_sat1", "K1", "q_sat2", "K2"],
        },
        "freundlich": {
            "p0": [q_max * 0.5, 1.5],
            "bounds": ([0, 1.0], [q_max * 100, 20.0]),
            "names": ["K_F", "n"],
        },
        "sips": {
            "p0": [q_max * 1.2, K_guess, 1.2],
            "bounds": ([0, 0, 0.5], [q_max * 10, K_guess * 100, 10.0]),
            "names": ["q_sat", "K", "n"],
        },
        "bet": {
            "p0": [q_max, 50.0, max(P.max(), 1.0) * 5],
            "bounds": ([0, 1, P.max() * 1.01], [q_max * 10, 1e6, 1e4]),
            "names": ["q_m", "C", "P_sat"],
        },
    }

    info = defaults[model]
    param_names = info["names"]
    x0 = info["p0"]
    lb, ub = info["bounds"]

    # Override with user-supplied values
    if p0 is not None:
        for i, name in enumerate(param_names):
            if name in p0:
                x0[i] = p0[name]
    if bounds is not None:
        for i, name in enumerate(param_names):
            if name in bounds:
                lb[i], ub[i] = bounds[name]

    cls = _MODEL_REGISTRY[model]

    def residuals(params):
        try:
            iso = cls(*params)
            q_pred = iso.loading(P)
            return q_pred - q
        except Exception:
            return np.full_like(q, 1e10)

    result = optimize.least_squares(
        residuals, x0, bounds=(lb, ub), method="trf",
        max_nfev=5000, ftol=1e-12, xtol=1e-12,
    )

    if not result.success:
        logger.warning(f"Isotherm fit ({model}) did not converge: {result.message}")

    fitted = cls(*result.x)
    logger.debug(
        f"Fitted {model}: params={fitted.params()}, "
        f"RMSE={np.sqrt(np.mean(result.fun**2)):.4e}"
    )
    return fitted


# ═══════════════════════════════════════════════════════════════════════
# 3.  IAST CALCULATOR
# ═══════════════════════════════════════════════════════════════════════

class IASTCalculator:
    """
    Ideal Adsorbed Solution Theory solver.

    Given pure-component isotherms and gas-phase conditions, compute
    mixture adsorption equilibria.

    Parameters
    ----------
    isotherms     : List of fitted ``IsothermModel`` instances, one
                    per component.
    component_names: Human-readable labels (e.g. ``['CO2','N2','H2O']``).
    temperature   : Temperature [K] (for thermodynamic consistency
                    checks; IAST itself is temperature-implicit via
                    the isotherms).
    tolerance     : Convergence tolerance for the IAST root-finder.
    max_iter      : Maximum Newton iterations.

    Example
    ───────
    >>> iso_co2 = Langmuir(q_sat=5.0, K=0.8)
    >>> iso_n2  = Langmuir(q_sat=4.0, K=0.05)
    >>> iast = IASTCalculator([iso_co2, iso_n2], ['CO2', 'N2'])
    >>> result = iast.predict(y=[0.15, 0.85], P_total=1.0)
    >>> result['loadings']   # per-component loadings [mol/kg]
    >>> result['selectivity']  # CO2/N2 selectivity
    """

    def __init__(
        self,
        isotherms: Sequence[IsothermModel],
        component_names: Optional[Sequence[str]] = None,
        temperature: float = 298.15,
        tolerance: float = 1e-10,
        max_iter: int = 200,
    ):
        self.isotherms = list(isotherms)
        self.n_components = len(isotherms)
        self.component_names = (
            list(component_names) if component_names
            else [f"comp_{i}" for i in range(self.n_components)]
        )
        self.temperature = temperature
        self.tolerance = tolerance
        self.max_iter = max_iter

        assert len(self.component_names) == self.n_components

    # ── Core solver ──────────────────────────────────────────────

    def predict(
        self,
        y: Sequence[float],
        P_total: float,
    ) -> Dict[str, Any]:
        """
        Predict mixture adsorption at given gas-phase mole fractions
        and total pressure.

        Parameters
        ----------
        y       : Gas-phase mole fractions (must sum to 1).
        P_total : Total pressure [bar].

        Returns
        -------
        Dict with keys:

        *  ``loadings``    — ``[n_components]`` per-component loadings
           [mol/kg].
        *  ``total_loading``— total loading [mol/kg].
        *  ``x``           — adsorbed-phase mole fractions.
        *  ``P0``          — hypothetical pure-component pressures.
        *  ``spreading_pressure`` — equilibrium πA/(RT).
        *  ``selectivity`` — CO₂/N₂ selectivity (if 2+ components).
        """
        y = np.asarray(y, dtype=np.float64)
        assert len(y) == self.n_components
        assert abs(y.sum() - 1.0) < 1e-6, f"Mole fractions must sum to 1, got {y.sum()}"

        partial_P = y * P_total  # [n_components]

        # Handle single-component case
        if self.n_components == 1:
            q = float(self.isotherms[0].loading(np.array([P_total]))[0])
            return {
                "loadings": np.array([q]),
                "total_loading": q,
                "x": np.array([1.0]),
                "P0": np.array([P_total]),
                "spreading_pressure": self.isotherms[0].spreading_pressure(P_total),
                "selectivity": None,
            }

        # Handle components with zero mole fraction
        active = np.where(y > 1e-15)[0]
        if len(active) == 0:
            return self._zero_result()

        # Initial guess for P0: P_i^0 = P_i / x_i ≈ P_total (Raoult-like)
        P0_init = np.array([
            partial_P[i] / max(y[i], 1e-15) for i in range(self.n_components)
        ])
        P0_init = P0_init.clip(min=1e-15, max=1e6)

        # Solve: find P0 such that all spreading pressures are equal
        # and Σ(y_i * P_total / P0_i) = 1
        try:
            P0 = self._solve(y, P_total, P0_init, active)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"IAST solver failed: {e}. Returning NaN.")
            return self._nan_result()

        # Compute results
        x = np.zeros(self.n_components)
        for i in active:
            x[i] = y[i] * P_total / P0[i]

        # Normalise (numerical safety)
        x_sum = x.sum()
        if x_sum > 0:
            x /= x_sum

        # Total loading via reciprocal mixing rule
        recip_sum = 0.0
        for i in active:
            q_i0 = float(self.isotherms[i].loading(np.array([P0[i]]))[0])
            if q_i0 > 1e-15:
                recip_sum += x[i] / q_i0

        n_total = 1.0 / max(recip_sum, 1e-15)
        loadings = x * n_total

        # Spreading pressure (should be same for all active)
        sp = self.isotherms[active[0]].spreading_pressure(P0[active[0]])

        # Selectivity
        selectivity = None
        if self.n_components >= 2 and y[0] > 1e-15 and y[1] > 1e-15:
            if loadings[1] > 1e-15 and y[1] > 1e-15:
                selectivity = (loadings[0] / loadings[1]) / (y[0] / y[1])

        return {
            "loadings": loadings,
            "total_loading": n_total,
            "x": x,
            "P0": P0,
            "spreading_pressure": sp,
            "selectivity": selectivity,
        }

    def _solve(
        self,
        y: np.ndarray,
        P_total: float,
        P0_init: np.ndarray,
        active: np.ndarray,
    ) -> np.ndarray:
        """
        Solve the IAST system using ``scipy.optimize.root``.

        Unknowns: ``P0[active]`` (hypothetical pure-component pressures).
        Equations:
            π_i(P0_i) = π_j(P0_j) for all active pairs
            Σ y_i · P_total / P0_i = 1
        """
        n_active = len(active)

        def equations(log_P0_active):
            P0_a = np.exp(log_P0_active)  # work in log-space for positivity
            resid = np.zeros(n_active)

            # Spreading pressure of the first active component
            sp_ref = self.isotherms[active[0]].spreading_pressure(P0_a[0])

            # Equations: π_i - π_ref = 0 for i > 0
            for k in range(1, n_active):
                sp_k = self.isotherms[active[k]].spreading_pressure(P0_a[k])
                resid[k] = sp_k - sp_ref

            # Sum equation: Σ y_i · P / P0_i = 1
            s = 0.0
            for k in range(n_active):
                i = active[k]
                s += y[i] * P_total / P0_a[k]
            resid[0] = s - 1.0

            return resid

        log_P0_init = np.log(P0_init[active].clip(min=1e-15))
        result = optimize.root(
            equations, log_P0_init,
            method="hybr",
            options={"maxfev": self.max_iter * 50, "xtol": self.tolerance},
        )

        if not result.success:
            # Try with different method
            result = optimize.root(
                equations, log_P0_init,
                method="lm",
                options={"maxiter": self.max_iter * 50, "xtol": self.tolerance},
            )

        if not result.success:
            raise RuntimeError(f"IAST root-finding failed: {result.message}")

        P0 = P0_init.copy()
        P0[active] = np.exp(result.x)
        return P0

    # ── Batch prediction ─────────────────────────────────────────

    def predict_batch(
        self,
        y_batch: np.ndarray,
        P_total_batch: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Vectorised IAST over a batch of conditions.

        Parameters
        ----------
        y_batch       : ``[B, n_components]`` mole fractions.
        P_total_batch : ``[B]`` total pressures.

        Returns
        -------
        Dict with ``"loadings"`` ``[B, n_components]``,
        ``"selectivity"`` ``[B]``, etc.
        """
        B = y_batch.shape[0]
        loadings = np.zeros((B, self.n_components))
        selectivities = np.full(B, np.nan)
        x_ads = np.zeros((B, self.n_components))

        for i in range(B):
            result = self.predict(y_batch[i], float(P_total_batch[i]))
            loadings[i] = result["loadings"]
            x_ads[i] = result["x"]
            if result["selectivity"] is not None:
                selectivities[i] = result["selectivity"]

        return {
            "loadings": loadings,
            "x": x_ads,
            "selectivity": selectivities,
        }

    # ── Isotherm sweep ───────────────────────────────────────────

    def sweep_pressure(
        self,
        y: Sequence[float],
        P_range: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute mixture isotherms over a range of total pressures
        at fixed composition.

        Returns ``"loadings"`` ``[n_P, n_components]`` and
        ``"selectivity"`` ``[n_P]``.
        """
        y = np.asarray(y)
        P_range = np.asarray(P_range)
        n_P = len(P_range)

        loadings = np.zeros((n_P, self.n_components))
        sel = np.full(n_P, np.nan)

        for i, P in enumerate(P_range):
            result = self.predict(y, float(P))
            loadings[i] = result["loadings"]
            if result["selectivity"] is not None:
                sel[i] = result["selectivity"]

        return {"pressures": P_range, "loadings": loadings, "selectivity": sel}

    def sweep_composition(
        self,
        P_total: float,
        n_points: int = 50,
    ) -> Dict[str, np.ndarray]:
        """
        Binary mixture: sweep y₁ from 0 to 1 at fixed total pressure.

        Only valid for 2-component systems.
        """
        assert self.n_components == 2, "Composition sweep only for binary mixtures."

        y1_range = np.linspace(0.01, 0.99, n_points)
        loadings = np.zeros((n_points, 2))
        sel = np.full(n_points, np.nan)

        for i, y1 in enumerate(y1_range):
            result = self.predict([y1, 1.0 - y1], P_total)
            loadings[i] = result["loadings"]
            if result["selectivity"] is not None:
                sel[i] = result["selectivity"]

        return {"y1": y1_range, "loadings": loadings, "selectivity": sel}

    # ── Helpers ──────────────────────────────────────────────────

    def _zero_result(self) -> Dict[str, Any]:
        return {
            "loadings": np.zeros(self.n_components),
            "total_loading": 0.0,
            "x": np.zeros(self.n_components),
            "P0": np.zeros(self.n_components),
            "spreading_pressure": 0.0,
            "selectivity": None,
        }

    def _nan_result(self) -> Dict[str, Any]:
        return {
            "loadings": np.full(self.n_components, np.nan),
            "total_loading": np.nan,
            "x": np.full(self.n_components, np.nan),
            "P0": np.full(self.n_components, np.nan),
            "spreading_pressure": np.nan,
            "selectivity": np.nan,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "n_components": self.n_components,
            "components": self.component_names,
            "isotherm_types": [type(iso).__name__ for iso in self.isotherms],
            "temperature": self.temperature,
        }


# ═══════════════════════════════════════════════════════════════════════
# 4.  REVERSE IAST
# ═══════════════════════════════════════════════════════════════════════

def reverse_iast(
    iast: IASTCalculator,
    y: Sequence[float],
    target_loading: float,
    component: int = 0,
    P_bounds: Tuple[float, float] = (1e-4, 100.0),
    tolerance: float = 1e-6,
) -> Optional[float]:
    """
    Find the total pressure at which a target per-component loading
    is achieved.

    Uses bisection on ``iast.predict(y, P).loadings[component]``.

    Parameters
    ----------
    iast           : Fitted IASTCalculator.
    y              : Gas-phase mole fractions.
    target_loading : Desired loading [mol/kg] for ``component``.
    component      : Component index.
    P_bounds       : Pressure search range [bar].

    Returns
    -------
    Total pressure [bar], or *None* if the target is not achievable.
    """
    y = list(y)

    def objective(log_P: float) -> float:
        P = np.exp(log_P)
        result = iast.predict(y, float(P))
        q = result["loadings"][component]
        if np.isnan(q):
            return 1e10  # push solver away from NaN regions
        return q - target_loading

    try:
        sol = optimize.brentq(
            objective,
            np.log(P_bounds[0]),
            np.log(P_bounds[1]),
            xtol=tolerance,
            maxiter=200,
        )
        return float(np.exp(sol))
    except ValueError:
        logger.warning(
            f"reverse_iast: target {target_loading:.3f} mol/kg not "
            f"achievable in [{P_bounds[0]}, {P_bounds[1]}] bar."
        )
        return None


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    # Isotherms
    "IsothermModel",
    "Langmuir",
    "DualSiteLangmuir",
    "Freundlich",
    "Sips",
    "BET",
    "InterpolatedIsotherm",
    # Fitting
    "fit_isotherm",
    # IAST
    "IASTCalculator",
    "reverse_iast",
]