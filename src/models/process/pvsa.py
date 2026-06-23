"""
Pressure-Vacuum Swing Adsorption (PVSA) cycle simulator.

This module implements a simplified but physically grounded PVSA
process model for evaluating MOF adsorbents in post-combustion
CO₂ capture from humid flue gas.  It converts adsorption isotherm
data (from IAST or neural predictions) into engineering performance
metrics (purity, recovery, productivity, energy).

Process cycle (4-step Skarstrom-based)
──────────────────────────────────────
1.  **Pressurisation** — feed the column to adsorption pressure
    P_high with flue gas.
2.  **Adsorption (feed)** — flue gas flows through the packed bed;
    CO₂ is selectively adsorbed; N₂-rich product exits.
3.  **Blowdown** — reduce column pressure to P_low; desorbed gas
    (CO₂-enriched) is collected.
4.  **Evacuation / purge** — apply vacuum to P_vac (or light purge)
    to regenerate the adsorbent for the next cycle.

The model uses **equilibrium theory** (sharp-front / constant-pattern
approximation) rather than full PDE column dynamics, making it fast
enough for high-throughput screening of thousands of MOFs while
retaining the key physics.

Components
──────────
*  ``PVSAParameters`` — all process operating conditions.
*  ``ColumnGeometry``  — packed-bed dimensions and void fractions.
*  ``PVSASimulator``   — the main simulator class.
*  ``PVSACycleResult`` — per-cycle output container.

Integration
───────────
*  ``iast.py`` or ``neural_mixture.py`` supplies q_i(P, y, T).
*  ``kpi.py`` computes standardised KPIs from the cycle result.
*  ``surrogate.py`` can emulate this simulator for fast optimisation.

References
──────────
[1] Ruthven (1984). Principles of Adsorption and Adsorption Processes.
[2] Sholl & Lively (2016). Seven Chemical Separations to Change the
    World. Nature.
[3] Burns et al. (2020). Prediction of MOF Performance in Vacuum
    Swing Adsorption Systems. JACS.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Physical constants
R_GAS = 8.314462618  # J mol⁻¹ K⁻¹


# ═══════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ColumnGeometry:
    """
    Packed-bed column dimensions.

    Attributes
    ──────────
    length       : Column length [m].
    diameter     : Column inner diameter [m].
    void_frac    : Interparticle void fraction (ε_b).
    particle_void: Intraparticle void fraction (ε_p).
    particle_dia : Adsorbent particle diameter [m].
    """

    length: float = 1.0
    diameter: float = 0.3
    void_frac: float = 0.37
    particle_void: float = 0.35
    particle_dia: float = 2e-3

    @property
    def cross_area(self) -> float:
        """Column cross-sectional area [m²]."""
        return math.pi * (self.diameter / 2.0) ** 2

    @property
    def volume(self) -> float:
        """Column volume [m³]."""
        return self.cross_area * self.length

    @property
    def adsorbent_volume(self) -> float:
        """Volume occupied by adsorbent [m³]."""
        return self.volume * (1.0 - self.void_frac)

    @property
    def total_void(self) -> float:
        """Total void fraction (inter + intra)."""
        return self.void_frac + (1.0 - self.void_frac) * self.particle_void


@dataclass
class PVSAParameters:
    """
    PVSA operating conditions.

    Attributes
    ──────────
    P_feed    : Feed (adsorption) pressure [bar].
    P_blow    : Blowdown pressure [bar].
    P_vac     : Evacuation (vacuum) pressure [bar].
    T_feed    : Feed temperature [K].
    y_CO2     : CO₂ mole fraction in feed gas.
    y_N2      : N₂ mole fraction in feed gas.
    y_H2O     : H₂O mole fraction in feed gas.
    feed_vel  : Superficial feed velocity [m/s].
    t_feed    : Feed step duration [s].
    t_blow    : Blowdown step duration [s].
    t_evac    : Evacuation step duration [s].
    t_press   : Pressurisation step duration [s].
    rho_ads   : Adsorbent bulk density [kg/m³].
    cp_ads    : Adsorbent heat capacity [J/(kg·K)].
    eta_vac   : Vacuum pump isentropic efficiency.
    eta_comp  : Compressor isentropic efficiency.
    gamma_gas : Heat capacity ratio of gas (Cp/Cv).
    """

    P_feed: float = 1.0
    P_blow: float = 0.1
    P_vac: float = 0.03
    T_feed: float = 313.15      # 40 °C (typical flue gas)
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
        """Feed composition array [CO₂, N₂, H₂O]."""
        return np.array([self.y_CO2, self.y_N2, self.y_H2O])

    @property
    def cycle_time(self) -> float:
        """Total cycle time [s]."""
        return self.t_feed + self.t_blow + self.t_evac + self.t_press

    def validate(self) -> None:
        """Check parameter sanity."""
        assert self.P_feed > self.P_blow >= self.P_vac > 0
        assert 0 < self.y_CO2 < 1 and 0 < self.y_N2 < 1
        assert abs(self.y_CO2 + self.y_N2 + self.y_H2O - 1.0) < 0.01


# ═══════════════════════════════════════════════════════════════════════
# 2.  CYCLE RESULT
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PVSACycleResult:
    """
    Output of one PVSA cycle simulation.

    All quantities are per cycle unless noted.
    """

    # Loadings [mol/kg]
    q_ads: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q_des: np.ndarray = field(default_factory=lambda: np.zeros(3))
    delta_q: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Amounts [mol]
    n_feed: float = 0.0          # total moles fed
    n_CO2_feed: float = 0.0      # CO₂ in feed
    n_CO2_product: float = 0.0   # CO₂ in product (desorbed)
    n_CO2_raffinate: float = 0.0 # CO₂ lost in raffinate

    # Performance
    purity: float = 0.0          # CO₂ purity in product [−]
    recovery: float = 0.0        # CO₂ recovery fraction [−]
    productivity: float = 0.0    # mol CO₂ / (kg ads · s)
    energy_kJ_mol: float = 0.0   # kJ per mol CO₂ captured
    energy_MJ_ton: float = 0.0   # MJ per tonne CO₂ captured

    # Pressure profile
    P_ads: float = 0.0
    P_des: float = 0.0

    # Flags
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

    Uses the equilibrium (sharp-front) approximation: at each
    pressure, the adsorbent is in full equilibrium with the local
    gas phase.  This ignores mass-transfer limitations and axial
    dispersion, but is a good first-order screening tool.

    Parameters
    ----------
    isotherm_fn : Callable ``(y, P_total, T) → loadings [C]``
                  that returns per-component loadings [mol/kg].
                  This can be an ``IASTCalculator.predict`` wrapper
                  or a neural model.
    params      : ``PVSAParameters``.
    column      : ``ColumnGeometry``.
    n_components: Number of species (default 3: CO₂, N₂, H₂O).

    Example
    ───────
    >>> from src.models.mixture.iast import IASTCalculator, Langmuir
    >>> iast = IASTCalculator([Langmuir(5,0.8), Langmuir(4,0.05), Langmuir(8,2.0)])
    >>> def iso_fn(y, P, T):
    ...     return iast.predict(y, P)['loadings']
    >>> sim = PVSASimulator(iso_fn)
    >>> result = sim.run_cycle()
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
        """Evaluate isotherm at given composition and pressure."""
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

        Steps
        ─────
        1.  Adsorption at P_feed → q_ads (equilibrium loadings).
        2.  Blowdown to P_blow → intermediate desorption.
        3.  Evacuation to P_vac → q_des (residual loadings).
        4.  Working capacity Δq = q_ads − q_des.
        5.  Compute CO₂ balance, purity, recovery, energy.
        """
        p = self.params
        col = self.column
        result = PVSACycleResult()

        # ── Step 1: Adsorption equilibrium at P_feed ─────────────
        y_feed = p.y_feed
        q_ads = self._get_loadings(y_feed, p.P_feed)
        result.q_ads = q_ads
        result.P_ads = p.P_feed

        # ── Step 2-3: Desorption equilibrium at P_vac ────────────
        # During blowdown/evacuation, the gas composition shifts
        # toward the more strongly adsorbed species (CO₂, H₂O).
        # Approximate desorption composition:
        delta_q_approx = q_ads * (1.0 - p.P_vac / p.P_feed)
        q_total = delta_q_approx.sum()
        if q_total > 1e-10:
            y_des = delta_q_approx / q_total
        else:
            y_des = y_feed

        q_des = self._get_loadings(y_des, p.P_vac)
        result.q_des = q_des
        result.P_des = p.P_vac

        # ── Step 4: Working capacity ─────────────────────────────
        delta_q = (q_ads - q_des).clip(min=0.0)
        result.delta_q = delta_q

        # ── Step 5: Mass balance ─────────────────────────────────
        # Moles of adsorbent per column
        m_ads = col.adsorbent_volume * p.rho_ads  # [kg]

        # CO₂ captured per cycle = Δq_CO₂ · m_ads
        n_CO2_captured = delta_q[0] * m_ads  # [mol]

        # Total desorbed gas (all components)
        n_total_desorbed = delta_q.sum() * m_ads  # [mol]

        # CO₂ in the void space desorbed during blowdown
        n_void = (col.volume * col.total_void * p.P_feed * 1e5
                  / (R_GAS * p.T_feed))  # mol in void at P_feed
        n_CO2_void = n_void * p.y_CO2

        # Product = desorbed + void gas expelled during blowdown
        n_CO2_product = n_CO2_captured + n_CO2_void * (1.0 - p.P_vac / p.P_feed)
        n_product_total = n_total_desorbed + n_void * (1.0 - p.P_vac / p.P_feed)

        result.n_CO2_product = max(n_CO2_product, 0.0)

        # Feed gas processed per cycle
        n_feed = (col.cross_area * p.feed_vel * p.t_feed
                  * p.P_feed * 1e5 / (R_GAS * p.T_feed))
        result.n_feed = n_feed
        result.n_CO2_feed = n_feed * p.y_CO2

        # ── Purity ───────────────────────────────────────────────
        if n_product_total > 1e-10:
            result.purity = n_CO2_product / n_product_total
        else:
            result.purity = 0.0

        # ── Recovery ─────────────────────────────────────────────
        if result.n_CO2_feed > 1e-10:
            result.recovery = min(n_CO2_product / result.n_CO2_feed, 1.0)
        else:
            result.recovery = 0.0

        result.n_CO2_raffinate = result.n_CO2_feed - n_CO2_product

        # ── Productivity [mol CO₂ / (kg ads · cycle)] ───────────
        result.productivity = n_CO2_captured / max(m_ads * p.cycle_time, 1e-10)

        # ── Energy ───────────────────────────────────────────────
        energy = self._compute_energy(n_CO2_captured)
        result.energy_kJ_mol = energy
        result.energy_MJ_ton = energy * 1e-3 * 1e6 / 44.01  # kJ/mol → MJ/tonne CO₂

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

        Includes:
        *  Vacuum pump work (isentropic compression from P_vac to 1 atm).
        *  Compression work (if product needs pressurisation).
        """
        p = self.params
        col = self.column

        if n_CO2 < 1e-10:
            return 0.0

        gamma = p.gamma_gas
        ratio = gamma / (gamma - 1.0)

        # Vacuum pump: compress gas from P_vac to ~P_blow
        # n_gas evacuated ≈ void volume at P_vac
        n_evac = (col.volume * col.total_void * p.P_blow * 1e5
                  / (R_GAS * p.T_feed))

        W_vac = 0.0
        if p.P_vac < p.P_blow and n_evac > 0:
            compression_ratio = p.P_blow / max(p.P_vac, 1e-6)
            W_vac = (n_evac * R_GAS * p.T_feed * ratio
                     * ((compression_ratio ** (1.0 / ratio)) - 1.0)
                     / p.eta_vac)

        # Convert J → kJ, per mol CO₂
        energy_kJ = W_vac / 1000.0 / max(n_CO2, 1e-10)
        return max(energy_kJ, 0.0)

    # ── Parameter sweep ──────────────────────────────────────────

    def sweep_vacuum(
        self,
        P_vac_range: np.ndarray,
    ) -> List[PVSACycleResult]:
        """Sweep evacuation pressure and return results."""
        results = []
        original = self.params.P_vac
        for P_vac in P_vac_range:
            self.params.P_vac = float(P_vac)
            results.append(self.run_cycle())
        self.params.P_vac = original
        return results

    def sweep_feed_composition(
        self,
        y_CO2_range: np.ndarray,
    ) -> List[PVSACycleResult]:
        """Sweep CO₂ feed fraction (adjusting N₂ accordingly)."""
        results = []
        orig_co2, orig_n2 = self.params.y_CO2, self.params.y_N2
        for y_co2 in y_CO2_range:
            self.params.y_CO2 = float(y_co2)
            self.params.y_N2 = 1.0 - float(y_co2) - self.params.y_H2O
            results.append(self.run_cycle())
        self.params.y_CO2, self.params.y_N2 = orig_co2, orig_n2
        return results

    def pareto_frontier(
        self,
        P_vac_range: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute purity–recovery Pareto frontier over vacuum pressures.

        Returns arrays suitable for plotting.
        """
        results = self.sweep_vacuum(P_vac_range)
        purities = np.array([r.purity for r in results])
        recoveries = np.array([r.recovery for r in results])
        energies = np.array([r.energy_kJ_mol for r in results])
        productivities = np.array([r.productivity for r in results])

        return {
            "P_vac": P_vac_range,
            "purity": purities,
            "recovery": recoveries,
            "energy_kJ_mol": energies,
            "productivity": productivities,
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