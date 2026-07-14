"""
Pressure-Vacuum Swing Adsorption (PVSA) cycle simulator.

[docstring unchanged — see original file]
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

R_GAS = 8.314462618  # J mol⁻¹ K⁻¹


# ═══════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ColumnGeometry:
    """Packed-bed column dimensions."""

    length: float = 1.0
    diameter: float = 0.3
    void_frac: float = 0.37
    particle_void: float = 0.35
    particle_dia: float = 2e-3

    @property
    def cross_area(self) -> float:
        return math.pi * (self.diameter / 2.0) ** 2

    @property
    def volume(self) -> float:
        return self.cross_area * self.length

    @property
    def adsorbent_volume(self) -> float:
        return self.volume * (1.0 - self.void_frac)

    @property
    def total_void(self) -> float:
        return self.void_frac + (1.0 - self.void_frac) * self.particle_void


@dataclass
class PVSAParameters:
    """PVSA operating conditions."""

    P_feed: float = 1.0
    P_blow: float = 0.1
    P_vac: float = 0.03
    T_feed: float = 313.15
    y_CO2: float = 0.15
    y_N2: float = 0.75
    y_H2O: float = 0.10
    feed_vel: float = 0.5
    t_feed: float = 120.0
    t_blow: float = 30.0
    t_evac: float = 120.0
    t_press: float = 30.0
    rho_ads: float = 1000.0
    cp_ads: float = 900.0
    eta_vac: float = 0.72
    eta_comp: float = 0.80
    gamma_gas: float = 1.4

    @property
    def y_feed(self) -> np.ndarray:
        return np.array([self.y_CO2, self.y_N2, self.y_H2O])

    @property
    def cycle_time(self) -> float:
        return self.t_feed + self.t_blow + self.t_evac + self.t_press

    def validate(self) -> None:
        """
        Check parameter sanity.

        FIX: replaced assert with ValueError so checks survive
        ``python -O`` (which strips assert statements).
        """
        if not (self.P_feed > self.P_blow >= self.P_vac > 0):
            raise ValueError(
                f"Pressure ordering violated: P_feed={self.P_feed} > "
                f"P_blow={self.P_blow} >= P_vac={self.P_vac} > 0"
            )
        if not (0 < self.y_CO2 < 1 and 0 < self.y_N2 < 1):
            raise ValueError(
                f"Mole fractions out of range: y_CO2={self.y_CO2}, y_N2={self.y_N2}"
            )
        if not (abs(self.y_CO2 + self.y_N2 + self.y_H2O - 1.0) < 0.01):
            raise ValueError("Feed mole fractions do not sum to 1.")


# ═══════════════════════════════════════════════════════════════════════
# 2.  CYCLE RESULT
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PVSACycleResult:
    """Output of one PVSA cycle simulation."""

    q_ads: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q_des: np.ndarray = field(default_factory=lambda: np.zeros(3))
    delta_q: np.ndarray = field(default_factory=lambda: np.zeros(3))

    n_feed: float = 0.0
    n_CO2_feed: float = 0.0
    n_CO2_product: float = 0.0
    n_CO2_raffinate: float = 0.0

    purity: float = 0.0
    recovery: float = 0.0
    productivity: float = 0.0
    energy_kJ_mol: float = 0.0
    energy_MJ_ton: float = 0.0

    P_ads: float = 0.0
    P_des: float = 0.0

    valid: bool = True
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "q_ads": self.q_ads.tolist(),
            "q_des": self.q_des.tolist(),
            "delta_q": self.delta_q.tolist(),
            "purity": self.purity,
            "recovery": self.recovery,
            "productivity": self.productivity,
            "energy_kJ_mol": self.energy_kJ_mol,
            "energy_MJ_ton": self.energy_MJ_ton,
            "valid": self.valid,
        }


# ═══════════════════════════════════════════════════════════════════════
# 3.  PVSA SIMULATOR
# ═══════════════════════════════════════════════════════════════════════

class PVSASimulator:
    """
    Equilibrium-theory PVSA cycle simulator.
    """

    def __init__(
        self,
        isotherm_fn: Callable,
        params: Optional[PVSAParameters] = None,
        column: Optional[ColumnGeometry] = None,
        n_components: int = 3,
    ):
        self.isotherm_fn = isotherm_fn
        self.params = params or PVSAParameters()
        self.column = column or ColumnGeometry()
        self.n_components = n_components

    def _get_loadings(self, y: np.ndarray, P: float) -> np.ndarray:
        try:
            q = self.isotherm_fn(y, P, self.params.T_feed)
            if isinstance(q, dict):
                q = q.get("loadings", np.zeros(self.n_components))
            return np.asarray(q, dtype=np.float64)
        except Exception as e:
            logger.warning(f"Isotherm evaluation failed: {e}")
            return np.zeros(self.n_components)

    def run_cycle(self) -> PVSACycleResult:
        """
        Simulate one complete PVSA cycle and return performance.
        """
        p = self.params
        col = self.column
        result = PVSACycleResult()

        # Step 1: Adsorption equilibrium at P_feed
        y_feed = p.y_feed
        q_ads = self._get_loadings(y_feed, p.P_feed)
        result.q_ads = q_ads
        result.P_ads = p.P_feed

        # Steps 2–3: Desorption equilibrium at P_vac
        delta_q_approx = q_ads * (1.0 - p.P_vac / p.P_feed)
        q_total = delta_q_approx.sum()
        if q_total > 1e-10:
            y_des = delta_q_approx / q_total
        else:
            y_des = y_feed

        q_des = self._get_loadings(y_des, p.P_vac)
        result.q_des = q_des
        result.P_des = p.P_vac

        # Step 4: Working capacity
        delta_q = (q_ads - q_des).clip(min=0.0)
        result.delta_q = delta_q

        # Step 5: Mass balance
        m_ads = col.adsorbent_volume * p.rho_ads                        # kg
        n_CO2_captured = delta_q[0] * m_ads                             # mol
        n_total_desorbed = delta_q.sum() * m_ads                        # mol

        n_void = (col.volume * col.total_void * p.P_feed * 1e5
                  / (R_GAS * p.T_feed))                                 # mol in void at P_feed
        n_CO2_void = n_void * p.y_CO2

        n_CO2_product = n_CO2_captured + n_CO2_void * (1.0 - p.P_vac / p.P_feed)
        n_product_total = n_total_desorbed + n_void * (1.0 - p.P_vac / p.P_feed)

        result.n_CO2_product = max(n_CO2_product, 0.0)

        n_feed = (col.cross_area * p.feed_vel * p.t_feed
                  * p.P_feed * 1e5 / (R_GAS * p.T_feed))
        result.n_feed = n_feed
        result.n_CO2_feed = n_feed * p.y_CO2

        # Purity
        result.purity = (n_CO2_product / n_product_total
                         if n_product_total > 1e-10 else 0.0)

        # Recovery
        result.recovery = (min(n_CO2_product / result.n_CO2_feed, 1.0)
                           if result.n_CO2_feed > 1e-10 else 0.0)

        result.n_CO2_raffinate = result.n_CO2_feed - n_CO2_product

        # Productivity [mol CO₂ / (kg · s)]
        result.productivity = n_CO2_captured / max(m_ads * p.cycle_time, 1e-10)

        # Energy
        energy = self._compute_energy(n_CO2_captured)
        result.energy_kJ_mol = energy
        result.energy_MJ_ton = energy * 1e-3 * 1e6 / 44.01

        # Validity check
        if delta_q[0] < 1e-6:
            result.valid = False
            result.message = "Negligible CO₂ working capacity."
        elif result.purity < 0.01:
            result.valid = False
            result.message = "CO₂ purity too low."

        return result

    def _compute_energy(self, n_CO2: float) -> float:
        """
        Compute energy consumption [kJ/mol CO₂].

        FIX: n_evac now uses p.P_vac (suction pressure) instead of
        p.P_blow.  The vacuum pump suction is at P_vac — that is the
        pressure of the gas being pumped — so n_evac must be computed
        at P_vac.  Using P_blow overestimated the moles by a factor of
        P_blow/P_vac (e.g. 0.1/0.03 ≈ 3.3×), hence overestimated
        vacuum-pump work by the same factor.
        """
        p = self.params
        col = self.column

        if n_CO2 < 1e-10:
            return 0.0

        gamma = p.gamma_gas
        ratio = gamma / (gamma - 1.0)

        # Moles in void at END of evacuation (suction-side pressure = P_vac)
        # FIX: was p.P_blow; must be p.P_vac
        n_evac = (col.volume * col.total_void * p.P_vac * 1e5
                  / (R_GAS * p.T_feed))

        W_vac = 0.0
        if p.P_vac < p.P_blow and n_evac > 0:
            compression_ratio = p.P_blow / max(p.P_vac, 1e-6)
            W_vac = (n_evac * R_GAS * p.T_feed * ratio
                     * ((compression_ratio ** (1.0 / ratio)) - 1.0)
                     / p.eta_vac)

        energy_kJ = W_vac / 1000.0 / max(n_CO2, 1e-10)
        return max(energy_kJ, 0.0)

    # ── Parameter sweeps ─────────────────────────────────────────

    def sweep_vacuum(self, P_vac_range: np.ndarray) -> List[PVSACycleResult]:
        """
        Sweep evacuation pressure and return results.

        FIX: use try/finally so P_vac is always restored even if
        run_cycle() raises an exception.
        """
        results = []
        original = self.params.P_vac
        try:
            for P_vac in P_vac_range:
                self.params.P_vac = float(P_vac)
                results.append(self.run_cycle())
        finally:
            self.params.P_vac = original
        return results

    def sweep_feed_composition(
        self, y_CO2_range: np.ndarray,
    ) -> List[PVSACycleResult]:
        """
        Sweep CO₂ feed fraction (adjusting N₂ accordingly).

        FIX: use try/finally so composition is always restored.
        """
        results = []
        orig_co2, orig_n2 = self.params.y_CO2, self.params.y_N2
        try:
            for y_co2 in y_CO2_range:
                self.params.y_CO2 = float(y_co2)
                self.params.y_N2 = 1.0 - float(y_co2) - self.params.y_H2O
                results.append(self.run_cycle())
        finally:
            self.params.y_CO2, self.params.y_N2 = orig_co2, orig_n2
        return results

    def pareto_frontier(
        self, P_vac_range: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute purity–recovery Pareto frontier over vacuum pressures."""
        results = self.sweep_vacuum(P_vac_range)
        return {
            "P_vac": P_vac_range,
            "purity": np.array([r.purity for r in results]),
            "recovery": np.array([r.recovery for r in results]),
            "energy_kJ_mol": np.array([r.energy_kJ_mol for r in results]),
            "productivity": np.array([r.productivity for r in results]),
        }


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "PVSAParameters",
    "ColumnGeometry",
    "PVSASimulator",
    "PVSACycleResult",
]