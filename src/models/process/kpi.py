"""
Key Performance Indicators (KPIs) for MOF CO₂ capture evaluation.

This module computes standardised metrics that translate raw
adsorption data and PVSA cycle results into engineering-relevant
performance numbers.  KPIs are used for:

*   **Screening** — rank thousands of MOFs by a composite score.
*   **Pareto analysis** — identify trade-offs (purity vs recovery
    vs energy).
*   **Techno-economic assessment** — estimate capture cost [$/tonne].

KPI hierarchy
─────────────
1.  **Material-level** (from isotherms only):
        *  Working capacity Δq_CO₂ [mol/kg]
        *  CO₂/N₂ selectivity (IAST)
        *  Regenerability (Δq / q_ads)
        *  Heat of adsorption Q_st [kJ/mol]
        *  Adsorbent Performance Indicator (API)

2.  **Process-level** (from PVSA simulation):
        *  CO₂ purity [%]
        *  CO₂ recovery [%]
        *  Productivity [mol CO₂ / (kg·h)]
        *  Specific energy [MJ/tonne CO₂]
        *  Adsorbent cost metric [$/tonne CO₂]

3.  **Composite scores**:
        *  Weighted multi-objective score
        *  Pareto rank / dominance count
        *  DOE benchmark compliance (≥ 95% purity, ≥ 90% recovery)

References
──────────
[1] Bae & Snurr (2011). Development and Evaluation of Porous
    Materials for Carbon Dioxide Separation and Capture. Angew. Chem.
[2] NETL (2019). Quality Guidelines for Energy System Studies:
    Carbon Capture Approaches for Natural Gas Combined Cycle Systems.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1.  DOE / INDUSTRY TARGETS
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CaptureTargets:
    """
    Standard capture targets from DOE / NETL guidelines.

    These define the minimum acceptable performance for a viable
    post-combustion CO₂ capture process.
    """

    purity_min: float = 0.95           # ≥ 95% CO₂ purity
    recovery_min: float = 0.90         # ≥ 90% CO₂ recovery
    energy_max_MJ_ton: float = 1500.0  # ≤ 1.5 GJ/tonne CO₂
    productivity_min: float = 1e-3     # mol/(kg·s) — indicative
    cost_max_USD_ton: float = 40.0     # ≤ $40/tonne — aspirational


DOE_TARGETS = CaptureTargets()


# ═══════════════════════════════════════════════════════════════════════
# 2.  MATERIAL-LEVEL KPIs
# ═══════════════════════════════════════════════════════════════════════

def working_capacity(
    q_ads: Union[float, np.ndarray],
    q_des: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Δq = q_ads − q_des [mol/kg].  Clip at zero."""
    return np.maximum(np.asarray(q_ads) - np.asarray(q_des), 0.0)


def selectivity(
    q_CO2: float,
    q_N2: float,
    y_CO2: float = 0.15,
    y_N2: float = 0.75,
) -> float:
    """
    IAST selectivity: ``S = (q_CO2/q_N2) / (y_CO2/y_N2)``.
    """
    if q_N2 < 1e-15 or y_N2 < 1e-15:
        return float("inf")
    return (q_CO2 / q_N2) / (y_CO2 / y_N2)


def regenerability(
    q_ads: float,
    q_des: float,
) -> float:
    """Fraction of adsorbed CO₂ that can be recovered: Δq / q_ads."""
    if q_ads < 1e-15:
        return 0.0
    return max((q_ads - q_des) / q_ads, 0.0)


def adsorbent_performance_indicator(
    delta_q_CO2: float,
    selectivity_CO2_N2: float,
    alpha: float = 0.5,
) -> float:
    """
    Adsorbent Performance Indicator (API).

    ``API = Δq_CO₂^α · S_{CO₂/N₂}^{1−α}``

    A geometric-mean composite that balances capacity and selectivity.
    Bae & Snurr (2011) used α = 0.5.
    """
    if delta_q_CO2 <= 0 or selectivity_CO2_N2 <= 0:
        return 0.0
    return delta_q_CO2 ** alpha * selectivity_CO2_N2 ** (1.0 - alpha)


def sorbent_selection_parameter(
    delta_q_CO2: float,
    delta_q_N2: float,
    selectivity_CO2_N2: float,
) -> float:
    """
    Sorbent Selection Parameter (SSP) — Rege & Yang (2001):

    ``SSP = Δq_CO₂ / Δq_N₂ · S²``

    Higher is better; heavily penalises low selectivity.
    """
    if delta_q_N2 < 1e-15:
        return float("inf") if delta_q_CO2 > 0 else 0.0
    return (delta_q_CO2 / delta_q_N2) * selectivity_CO2_N2 ** 2


# ═══════════════════════════════════════════════════════════════════════
# 3.  PROCESS-LEVEL KPIs
# ═══════════════════════════════════════════════════════════════════════

def specific_energy(
    energy_kJ_mol: float,
) -> float:
    """Convert kJ/mol CO₂ → MJ/tonne CO₂."""
    return energy_kJ_mol * 1e-3 * 1e6 / 44.01


def capture_cost_estimate(
    energy_MJ_ton: float,
    electricity_price: float = 60.0,    # $/MWh
    adsorbent_cost_kg: float = 15.0,    # $/kg MOF
    adsorbent_lifetime_yr: float = 5.0,
    productivity_mol_kg_s: float = 1e-3,
    annual_hours: float = 8000.0,
) -> float:
    """
    Simplified capture cost estimate [$/tonne CO₂].

    Two major components:
    *  Energy cost = energy × electricity price.
    *  Adsorbent amortisation = cost / (lifetime × annual production).
    """
    # Energy cost
    energy_MWh = energy_MJ_ton / 3600.0
    cost_energy = energy_MWh * electricity_price

    # Adsorbent amortisation
    annual_prod_ton = (productivity_mol_kg_s * 44.01e-6  # mol/s → ton/s
                       * annual_hours * 3600.0)          # per year
    if annual_prod_ton > 1e-10:
        cost_ads = adsorbent_cost_kg / (adsorbent_lifetime_yr * annual_prod_ton)
    else:
        cost_ads = float("inf")

    return cost_energy + cost_ads


# ═══════════════════════════════════════════════════════════════════════
# 4.  KPI CALCULATOR (UNIFIED)
# ═══════════════════════════════════════════════════════════════════════

class KPICalculator:
    """
    Compute all KPIs from a ``PVSACycleResult`` or raw data.

    Parameters
    ----------
    targets : ``CaptureTargets`` for compliance checking.
    weights : Dict of KPI name → weight for composite scoring.
              Defaults balance purity, recovery, energy, productivity.
    """

    DEFAULT_WEIGHTS = {
        "purity": 0.25,
        "recovery": 0.25,
        "energy": 0.25,
        "productivity": 0.25,
    }

    def __init__(
        self,
        targets: Optional[CaptureTargets] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.targets = targets or DOE_TARGETS
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)

    def from_cycle_result(
        self,
        result: Any,
        y_CO2: float = 0.15,
        y_N2: float = 0.75,
    ) -> Dict[str, Any]:
        """
        Compute all KPIs from a ``PVSACycleResult``.

        Parameters
        ----------
        result : ``PVSACycleResult`` (or any object with matching attrs).
        y_CO2, y_N2 : Feed mole fractions (for selectivity).

        Returns
        -------
        Dict with all material + process KPIs, compliance flags,
        and composite score.
        """
        dq = np.asarray(result.delta_q)
        q_ads = np.asarray(result.q_ads)
        q_des = np.asarray(result.q_des)

        # Material KPIs
        sel = selectivity(q_ads[0], q_ads[1], y_CO2, y_N2) if q_ads[1] > 1e-15 else 0.0
        regen = regenerability(q_ads[0], q_des[0])
        api = adsorbent_performance_indicator(dq[0], sel)
        ssp = sorbent_selection_parameter(dq[0], max(dq[1], 1e-15), sel)

        # Process KPIs
        kpis: Dict[str, Any] = {
            # Material
            "delta_q_CO2": float(dq[0]),
            "delta_q_N2": float(dq[1]) if len(dq) > 1 else 0.0,
            "delta_q_H2O": float(dq[2]) if len(dq) > 2 else 0.0,
            "selectivity_CO2_N2": sel,
            "regenerability": regen,
            "API": api,
            "SSP": ssp,

            # Process
            "purity": result.purity,
            "recovery": result.recovery,
            "productivity_mol_kg_s": result.productivity,
            "energy_kJ_mol": result.energy_kJ_mol,
            "energy_MJ_ton": result.energy_MJ_ton,
        }

        # Cost estimate
        kpis["capture_cost_USD_ton"] = capture_cost_estimate(
            result.energy_MJ_ton,
            productivity_mol_kg_s=result.productivity,
        )

        # Compliance
        kpis["meets_purity"] = result.purity >= self.targets.purity_min
        kpis["meets_recovery"] = result.recovery >= self.targets.recovery_min
        kpis["meets_energy"] = result.energy_MJ_ton <= self.targets.energy_max_MJ_ton
        kpis["meets_all"] = (kpis["meets_purity"]
                             and kpis["meets_recovery"]
                             and kpis["meets_energy"])

        # Composite score
        kpis["composite_score"] = self._composite_score(kpis)

        return kpis

    def from_isotherms(
        self,
        q_ads: np.ndarray,
        q_des: np.ndarray,
        y_CO2: float = 0.15,
        y_N2: float = 0.75,
    ) -> Dict[str, float]:
        """
        Material-level KPIs from isotherm data only (no PVSA needed).

        Parameters
        ----------
        q_ads : ``[C]`` loadings at adsorption conditions.
        q_des : ``[C]`` loadings at desorption conditions.

        Returns
        -------
        Dict with material KPIs.
        """
        q_ads = np.asarray(q_ads)
        q_des = np.asarray(q_des)
        dq = working_capacity(q_ads, q_des)

        sel = selectivity(q_ads[0], q_ads[1], y_CO2, y_N2) if len(q_ads) > 1 and q_ads[1] > 1e-15 else 0.0
        regen = regenerability(q_ads[0], q_des[0])
        api = adsorbent_performance_indicator(dq[0], sel)

        return {
            "delta_q_CO2": float(dq[0]),
            "delta_q_N2": float(dq[1]) if len(dq) > 1 else 0.0,
            "selectivity_CO2_N2": sel,
            "regenerability": regen,
            "API": api,
        }

    def _composite_score(self, kpis: Dict[str, Any]) -> float:
        """
        Weighted composite score in [0, 1].

        Each KPI is normalised to [0, 1] by the DOE targets, then
        combined via weighted sum.
        """
        t = self.targets
        w = self.weights

        # Normalise: 1.0 = meets target, >1 = exceeds, <1 = below
        scores = {
            "purity": min(kpis["purity"] / t.purity_min, 1.0),
            "recovery": min(kpis["recovery"] / t.recovery_min, 1.0),
            "energy": min(t.energy_max_MJ_ton / max(kpis["energy_MJ_ton"], 1.0), 1.0),
            "productivity": min(kpis["productivity_mol_kg_s"] / t.productivity_min, 1.0),
        }

        total_w = sum(w.get(k, 0.0) for k in scores)
        if total_w < 1e-10:
            return 0.0

        return sum(w.get(k, 0.0) * v for k, v in scores.items()) / total_w

    # ── Batch + ranking ──────────────────────────────────────────

    def rank_mofs(
        self,
        results: Sequence[Any],
        names: Optional[Sequence[str]] = None,
        sort_by: str = "composite_score",
    ) -> List[Dict[str, Any]]:
        """
        Compute KPIs for multiple MOFs and return ranked list.

        Parameters
        ----------
        results : Sequence of ``PVSACycleResult`` objects.
        names   : Optional MOF identifiers.
        sort_by : KPI key to sort by (descending).

        Returns
        -------
        List of dicts, sorted by ``sort_by``.
        """
        if names is None:
            names = [f"MOF_{i}" for i in range(len(results))]

        rows = []
        for name, res in zip(names, results):
            kpis = self.from_cycle_result(res)
            kpis["name"] = name
            rows.append(kpis)

        rows.sort(key=lambda r: r.get(sort_by, 0.0), reverse=True)

        # Add rank
        for i, row in enumerate(rows):
            row["rank"] = i + 1

        return rows

    def pareto_dominant(
        self,
        kpi_list: Sequence[Dict[str, Any]],
        objectives: Sequence[str] = ("purity", "recovery"),
        minimize: Sequence[str] = (),
    ) -> List[int]:
        """
        Return indices of Pareto-dominant solutions.

        Parameters
        ----------
        kpi_list   : List of KPI dicts (from ``rank_mofs`` or similar).
        objectives : Keys to consider for dominance.
        minimize   : Which objectives to minimise (rest are maximised).

        Returns
        -------
        List of indices into ``kpi_list`` that are on the Pareto front.
        """
        n = len(kpi_list)
        values = np.zeros((n, len(objectives)))
        for i, kpi in enumerate(kpi_list):
            for j, obj in enumerate(objectives):
                v = kpi.get(obj, 0.0)
                if obj in minimize:
                    v = -v  # flip for minimisation
                values[i, j] = v

        is_pareto = np.ones(n, dtype=bool)
        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i == j or not is_pareto[j]:
                    continue
                # j dominates i if j ≥ i in all objectives and j > i in at least one
                if np.all(values[j] >= values[i]) and np.any(values[j] > values[i]):
                    is_pareto[i] = False
                    break

        return [i for i in range(n) if is_pareto[i]]


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "CaptureTargets",
    "DOE_TARGETS",
    "KPICalculator",
    "working_capacity",
    "selectivity",
    "regenerability",
    "adsorbent_performance_indicator",
    "sorbent_selection_parameter",
    "specific_energy",
    "capture_cost_estimate",
]