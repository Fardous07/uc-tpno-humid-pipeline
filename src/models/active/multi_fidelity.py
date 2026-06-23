"""
Multi-fidelity surrogate modeling and fidelity selection.

This module provides the **modeling** layer for multi-fidelity
active learning — complementing the optimisation loop in
``acquisition.py``.  It answers the question: *given a limited
compute budget, which fidelity should we use for the next
evaluation, and how do we combine data from different fidelities?*

Components
──────────
1.  **CostBudget** — tracks cumulative cost and decides whether the
    budget allows another evaluation at a given fidelity.
2.  **FidelityCorrelation** — estimates the correlation ρ between
    adjacent fidelity levels from observed data, guiding when cheap
    simulations are informative enough.
3.  **AutoRegressiveModel** (AR1) — the classic Kennedy & O'Hagan
    (2000) model: ``f_high(x) = ρ · f_low(x) + δ(x)`` where δ is
    a GP correction.  Enables prediction at the highest fidelity
    using data from all levels.
4.  **FidelitySelector** — policy that decides which fidelity to
    assign to a candidate based on the current budget, correlation
    estimates, and candidate uncertainty.
5.  **MultiFidelityDataset** — container that manages observations
    across fidelity levels with alignment, deduplication, and
    convenience accessors.

Integration
───────────
``acquisition.py`` owns the BO loop and proposes **what** to evaluate;
this module decides **at which fidelity** and maintains the surrogate
that fuses information across levels.

References
──────────
[1] Kennedy & O'Hagan (2000). Predicting the Output of a Complex
    Computer Code When Fast Approximations Are Available. Biometrika.
[2] Perdikaris et al. (2017). Nonlinear Information Fusion Algorithms
    for Data-Efficient Multi-fidelity Modelling. Proc. R. Soc. A.
[3] Takeno et al. (2020). Multi-fidelity Bayesian Optimization with
    Max-value Entropy Search and its Parallelization. ICML.
[4] Kandasamy et al. (2017). Multi-fidelity Bayesian Optimisation
    with Continuous Approximations. ICML.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import copy
import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple, Union,
)

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1.  COST-BUDGET TRACKER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CostBudget:
    """
    Tracks cumulative simulation cost across fidelity levels.

    Parameters
    ----------
    total_budget : Maximum total CPU-hours (or arbitrary cost units).
    per_fidelity : Optional per-fidelity sub-budgets (dict mapping
                   fidelity index → max cost).
    """

    total_budget: float = 1000.0
    per_fidelity: Optional[Dict[int, float]] = None

    def __post_init__(self):
        self._spent_total: float = 0.0
        self._spent_per_fid: Dict[int, float] = {}
        self._history: List[Dict[str, Any]] = []

    # ── Queries ──────────────────────────────────────────────────

    @property
    def spent(self) -> float:
        """Total cost spent so far."""
        return self._spent_total

    @property
    def remaining(self) -> float:
        """Remaining budget."""
        return max(self.total_budget - self._spent_total, 0.0)

    @property
    def fraction_spent(self) -> float:
        return self._spent_total / max(self.total_budget, 1e-12)

    def can_afford(self, cost: float) -> bool:
        """Check whether a single evaluation at the given cost fits."""
        return (self._spent_total + cost) <= self.total_budget

    def can_afford_fidelity(self, fidelity_index: int, cost: float) -> bool:
        """Check both global and per-fidelity budgets."""
        if not self.can_afford(cost):
            return False
        if self.per_fidelity is not None and fidelity_index in self.per_fidelity:
            fid_spent = self._spent_per_fid.get(fidelity_index, 0.0)
            if (fid_spent + cost) > self.per_fidelity[fidelity_index]:
                return False
        return True

    # ── Mutations ────────────────────────────────────────────────

    def record(self, fidelity_index: int, cost: float, metadata: Optional[Dict] = None) -> None:
        """Record a completed evaluation."""
        self._spent_total += cost
        self._spent_per_fid[fidelity_index] = self._spent_per_fid.get(fidelity_index, 0.0) + cost
        entry = {"fidelity": fidelity_index, "cost": cost, "cumulative": self._spent_total}
        if metadata:
            entry.update(metadata)
        self._history.append(entry)

    def reset(self) -> None:
        self._spent_total = 0.0
        self._spent_per_fid.clear()
        self._history.clear()

    # ── Summary ──────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        return {
            "total_budget": self.total_budget,
            "spent": self._spent_total,
            "remaining": self.remaining,
            "fraction_spent": self.fraction_spent,
            "per_fidelity": dict(self._spent_per_fid),
            "n_evaluations": len(self._history),
        }

    def __repr__(self) -> str:
        return f"CostBudget(spent={self.spent:.1f}/{self.total_budget:.1f})"


# ═══════════════════════════════════════════════════════════════════════
# 2.  FIDELITY CORRELATION ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

class FidelityCorrelation:
    """
    Estimate the Pearson correlation ρ between adjacent fidelity
    levels from paired observations.

    If ρ is high (say > 0.9), the cheap fidelity is a reliable proxy
    and the ``FidelitySelector`` can spend more budget there.  If ρ
    is low, the cheap fidelity is misleading and we should invest in
    the expensive one.

    Parameters
    ----------
    n_fidelities : Number of distinct fidelity levels.
    min_pairs    : Minimum paired observations before returning a
                   correlation estimate (below this, return ``None``).
    """

    def __init__(self, n_fidelities: int = 3, min_pairs: int = 5):
        self.n_fidelities = n_fidelities
        self.min_pairs = min_pairs

        # Paired observations: (fid_low, fid_high) → [(y_low, y_high), ...]
        self._pairs: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}

    def add_pair(
        self,
        fid_low: int,
        fid_high: int,
        y_low: float,
        y_high: float,
    ) -> None:
        """Record a paired observation at two fidelities (same x)."""
        key = (min(fid_low, fid_high), max(fid_low, fid_high))
        self._pairs.setdefault(key, []).append((y_low, y_high))

    def add_pairs_batch(
        self,
        fid_low: int,
        fid_high: int,
        y_low: np.ndarray,
        y_high: np.ndarray,
    ) -> None:
        """Add a batch of paired observations."""
        key = (min(fid_low, fid_high), max(fid_low, fid_high))
        self._pairs.setdefault(key, [])
        for yl, yh in zip(y_low.ravel(), y_high.ravel()):
            self._pairs[key].append((float(yl), float(yh)))

    def correlation(self, fid_low: int, fid_high: int) -> Optional[float]:
        """
        Pearson correlation between fid_low and fid_high.

        Returns ``None`` if fewer than ``min_pairs`` observations.
        """
        key = (min(fid_low, fid_high), max(fid_low, fid_high))
        pairs = self._pairs.get(key, [])
        if len(pairs) < self.min_pairs:
            return None

        y_l = np.array([p[0] for p in pairs])
        y_h = np.array([p[1] for p in pairs])

        std_l = y_l.std()
        std_h = y_h.std()
        if std_l < 1e-12 or std_h < 1e-12:
            return 1.0  # constant → perfectly correlated

        return float(np.corrcoef(y_l, y_h)[0, 1])

    def all_correlations(self) -> Dict[Tuple[int, int], Optional[float]]:
        """Return correlations for all observed fidelity pairs."""
        return {key: self.correlation(*key) for key in self._pairs}

    def scaling_factor(self, fid_low: int, fid_high: int) -> Optional[float]:
        """
        Linear scaling factor ρ such that ``y_high ≈ ρ · y_low``.

        This is the slope of the least-squares fit (used by AR1).
        """
        key = (min(fid_low, fid_high), max(fid_low, fid_high))
        pairs = self._pairs.get(key, [])
        if len(pairs) < self.min_pairs:
            return None

        y_l = np.array([p[0] for p in pairs])
        y_h = np.array([p[1] for p in pairs])

        denom = np.dot(y_l, y_l)
        if denom < 1e-12:
            return 1.0
        return float(np.dot(y_l, y_h) / denom)


# ═══════════════════════════════════════════════════════════════════════
# 3.  AUTO-REGRESSIVE MULTI-FIDELITY MODEL (AR1)
# ═══════════════════════════════════════════════════════════════════════

class AutoRegressiveModel(nn.Module):
    """
    Kennedy & O'Hagan (2000) AR1 multi-fidelity model:

        f₁(x) = δ₁(x)
        fₜ(x) = ρₜ · fₜ₋₁(x) + δₜ(x)    for t = 2, …, T

    Each δₜ is modelled as a small MLP correction.  ρₜ is a learnable
    scalar (or can be fixed from ``FidelityCorrelation``).

    This provides a principled way to **transfer** cheap low-fidelity
    predictions to the high-fidelity target, learning only the
    residual correction δ.

    Parameters
    ----------
    input_dim    : Dimension of the input features x.
    hidden_dim   : Hidden dimension of the correction MLPs.
    n_fidelities : Number of fidelity levels.
    learn_rho    : Whether ρ is learnable or fixed at initialisation.
    init_rho     : Initial ρ values (list of length n_fidelities − 1).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_fidelities: int = 3,
        learn_rho: bool = True,
        init_rho: Optional[List[float]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_fidelities = n_fidelities

        # Scaling factors ρ (one per fidelity transition)
        if init_rho is None:
            init_rho = [1.0] * (n_fidelities - 1)
        assert len(init_rho) == n_fidelities - 1

        if learn_rho:
            self.rho = nn.ParameterList([
                nn.Parameter(torch.tensor(float(r)))
                for r in init_rho
            ])
        else:
            self.rho = nn.ParameterList()
            for r in init_rho:
                p = nn.Parameter(torch.tensor(float(r)))
                p.requires_grad = False
                self.rho.append(p)

        # Correction MLPs δ_t(x) for each fidelity level
        self.deltas = nn.ModuleList()
        for t in range(n_fidelities):
            self.deltas.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            ))

    def forward_at_fidelity(
        self,
        x: torch.Tensor,
        fidelity: int,
    ) -> torch.Tensor:
        """
        Predict at a specific fidelity level.

        Parameters
        ----------
        x        : ``[B, input_dim]`` input features.
        fidelity : Target fidelity index (0-based).

        Returns
        -------
        ``[B, 1]`` prediction at the requested fidelity.
        """
        # f_0 = δ_0(x)
        f = self.deltas[0](x)

        # f_t = ρ_t · f_{t-1} + δ_t(x)
        for t in range(1, fidelity + 1):
            rho_t = self.rho[t - 1]
            delta_t = self.deltas[t](x)
            f = rho_t * f + delta_t

        return f

    def forward(
        self,
        x: torch.Tensor,
        fidelities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mixed-fidelity forward pass: each sample in the batch may
        request a different fidelity.

        Parameters
        ----------
        x          : ``[B, input_dim]``
        fidelities : ``[B]`` integer fidelity indices.

        Returns
        -------
        ``[B, 1]`` predictions.
        """
        unique_fids = fidelities.unique().tolist()
        out = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)

        for fid in unique_fids:
            mask = fidelities == int(fid)
            out[mask] = self.forward_at_fidelity(x[mask], int(fid))

        return out

    def predict_highest(self, x: torch.Tensor) -> torch.Tensor:
        """Predict at the highest fidelity."""
        return self.forward_at_fidelity(x, self.n_fidelities - 1)

    def set_rho_from_data(self, corr: FidelityCorrelation) -> None:
        """
        Initialise ρ values from empirical scaling factors estimated
        by ``FidelityCorrelation``.
        """
        for t in range(1, self.n_fidelities):
            sf = corr.scaling_factor(t - 1, t)
            if sf is not None:
                self.rho[t - 1].data.fill_(sf)
                logger.info(f"Set ρ_{t} = {sf:.4f} from data.")


# ═══════════════════════════════════════════════════════════════════════
# 4.  FIDELITY SELECTION POLICY
# ═══════════════════════════════════════════════════════════════════════

class FidelitySelector:
    """
    Policy that assigns a fidelity level to each candidate.

    Strategies
    ──────────
    *  ``"cost_ratio"`` — use the cheapest fidelity whose correlation
       ρ with the target exceeds a threshold; upgrade to the target
       fidelity if ρ is low or unknown.
    *  ``"budget_aware"`` — start cheap and progressively shift to
       expensive fidelities as the budget is consumed.
    *  ``"information_gain"`` — estimate the information gain per
       unit cost at each fidelity and pick the best ratio (requires
       the AR1 model for variance estimates).
    *  ``"round_robin"`` — cycle through fidelities in order
       (mostly for baselines / ablations).

    Parameters
    ----------
    fidelity_costs   : Dict mapping fidelity index → cost.
    n_fidelities     : Number of fidelity levels.
    strategy         : One of the strategies above.
    rho_threshold    : Minimum ρ to trust a cheap fidelity.
    budget           : ``CostBudget`` instance for budget-aware policies.
    correlation      : ``FidelityCorrelation`` for data-driven decisions.
    """

    STRATEGIES = ("cost_ratio", "budget_aware", "information_gain", "round_robin")

    def __init__(
        self,
        fidelity_costs: Dict[int, float],
        n_fidelities: int = 3,
        strategy: str = "cost_ratio",
        rho_threshold: float = 0.85,
        budget: Optional[CostBudget] = None,
        correlation: Optional[FidelityCorrelation] = None,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {self.STRATEGIES}")

        self.fidelity_costs = fidelity_costs
        self.n_fidelities = n_fidelities
        self.strategy = strategy
        self.rho_threshold = rho_threshold
        self.budget = budget
        self.correlation = correlation
        self._rr_counter = 0  # for round-robin

    def select(
        self,
        candidates: Optional[torch.Tensor] = None,
        uncertainties: Optional[torch.Tensor] = None,
        n: int = 1,
    ) -> List[int]:
        """
        Select fidelity levels for ``n`` candidates.

        Parameters
        ----------
        candidates    : ``[n, d]`` candidate feature vectors (optional,
                        used by information-gain strategy).
        uncertainties : ``[n]`` model uncertainty per candidate (optional).
        n             : Number of fidelity assignments to return.

        Returns
        -------
        List of ``n`` fidelity indices.
        """
        if self.strategy == "cost_ratio":
            return self._cost_ratio(n)
        elif self.strategy == "budget_aware":
            return self._budget_aware(n)
        elif self.strategy == "information_gain":
            return self._information_gain(n, uncertainties)
        elif self.strategy == "round_robin":
            return self._round_robin(n)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    # ── Strategies ───────────────────────────────────────────────

    def _cost_ratio(self, n: int) -> List[int]:
        """Use cheapest fidelity with ρ above threshold."""
        target = self.n_fidelities - 1  # highest fidelity

        for fid in range(self.n_fidelities - 1):
            if self.correlation is not None:
                rho = self.correlation.correlation(fid, target)
                if rho is not None and rho >= self.rho_threshold:
                    # Check budget
                    cost = self.fidelity_costs.get(fid, 0.0)
                    if self.budget is None or self.budget.can_afford(cost * n):
                        return [fid] * n

        # Default: use highest fidelity
        return [target] * n

    def _budget_aware(self, n: int) -> List[int]:
        """
        Start cheap, shift to expensive as budget is consumed.

        The fraction of expensive evaluations increases linearly
        with the fraction of budget spent.
        """
        if self.budget is None:
            return [self.n_fidelities - 1] * n

        frac = self.budget.fraction_spent  # 0 → 1

        assignments = []
        for _ in range(n):
            # Probability of choosing highest fidelity increases with budget spent
            if np.random.random() < frac:
                fid = self.n_fidelities - 1
            else:
                fid = 0  # cheapest
            # Ensure we can afford it
            cost = self.fidelity_costs.get(fid, 0.0)
            if not self.budget.can_afford(cost):
                fid = 0  # fallback to cheapest
            assignments.append(fid)

        return assignments

    def _information_gain(
        self,
        n: int,
        uncertainties: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """
        Pick the fidelity that maximises information gain per unit
        cost: ``IG(fid) / cost(fid)``.

        Approximation: IG ∝ (1 − noise(fid)) × uncertainty(x).
        """
        target = self.n_fidelities - 1
        assignments = []

        for i in range(n):
            best_fid = target
            best_ratio = -float("inf")

            unc_i = float(uncertainties[i]) if uncertainties is not None else 1.0

            for fid in range(self.n_fidelities):
                cost = self.fidelity_costs.get(fid, 1.0)
                # Rough noise proxy
                if self.correlation is not None:
                    rho = self.correlation.correlation(fid, target)
                    rho = rho if rho is not None else 0.5
                else:
                    rho = 1.0 - 0.3 * (target - fid)  # heuristic

                info_gain = rho * unc_i
                ratio = info_gain / (cost + 1e-8)

                if ratio > best_ratio:
                    if self.budget is None or self.budget.can_afford(cost):
                        best_ratio = ratio
                        best_fid = fid

            assignments.append(best_fid)

        return assignments

    def _round_robin(self, n: int) -> List[int]:
        """Cycle through fidelities."""
        assignments = []
        for _ in range(n):
            fid = self._rr_counter % self.n_fidelities
            self._rr_counter += 1
            assignments.append(fid)
        return assignments


# ═══════════════════════════════════════════════════════════════════════
# 5.  MULTI-FIDELITY DATASET CONTAINER
# ═══════════════════════════════════════════════════════════════════════

class MultiFidelityDataset:
    """
    Container for observations collected at multiple fidelity levels.

    Stores ``(X, Y, fidelity, cost)`` tuples and provides convenience
    methods for filtering, alignment, and conversion to tensors.

    Parameters
    ----------
    n_fidelities : Number of fidelity levels.
    """

    def __init__(self, n_fidelities: int = 3):
        self.n_fidelities = n_fidelities
        self._X: List[np.ndarray] = []
        self._Y: List[np.ndarray] = []
        self._fid: List[int] = []
        self._cost: List[float] = []

    # ── Adding data ──────────────────────────────────────────────

    def add(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        fidelity: int,
        cost: float = 0.0,
    ) -> None:
        """Add observations at a single fidelity level."""
        X = np.atleast_2d(X)
        Y = np.atleast_1d(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        for i in range(len(X)):
            self._X.append(X[i])
            self._Y.append(Y[i])
            self._fid.append(fidelity)
            self._cost.append(cost / max(len(X), 1))

    # ── Queries ──────────────────────────────────────────────────

    @property
    def n_total(self) -> int:
        return len(self._X)

    def n_at_fidelity(self, fidelity: int) -> int:
        return sum(1 for f in self._fid if f == fidelity)

    def get_fidelity(self, fidelity: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(X, Y)`` for a specific fidelity."""
        mask = [i for i, f in enumerate(self._fid) if f == fidelity]
        if not mask:
            return np.empty((0, 0)), np.empty((0, 0))
        X = np.stack([self._X[i] for i in mask])
        Y = np.stack([self._Y[i] for i in mask])
        return X, Y

    def get_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(X, Y, fidelities)`` for all data."""
        if self.n_total == 0:
            return np.empty((0,)), np.empty((0,)), np.empty((0,))
        X = np.stack(self._X)
        Y = np.stack(self._Y)
        fids = np.array(self._fid)
        return X, Y, fids

    def to_tensors(
        self,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Convert all data to PyTorch tensors."""
        X, Y, fids = self.get_all()
        d = {
            "X": torch.from_numpy(X).float(),
            "Y": torch.from_numpy(Y).float(),
            "fidelities": torch.from_numpy(fids).long(),
        }
        if device is not None:
            d = {k: v.to(device) for k, v in d.items()}
        return d

    # ── Paired observations ──────────────────────────────────────

    def get_paired(
        self,
        fid_low: int,
        fid_high: int,
        rtol: float = 1e-5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find observations at both fidelities for the same X
        (within relative tolerance).

        Returns ``(X_common, Y_low, Y_high)``.
        """
        X_low, Y_low = self.get_fidelity(fid_low)
        X_high, Y_high = self.get_fidelity(fid_high)

        if len(X_low) == 0 or len(X_high) == 0:
            return np.empty((0,)), np.empty((0,)), np.empty((0,))

        # Brute-force matching (fine for moderate dataset sizes)
        paired_X, paired_Yl, paired_Yh = [], [], []
        for i, xl in enumerate(X_low):
            diffs = np.abs(X_high - xl).max(axis=-1) if X_high.ndim > 1 else np.abs(X_high - xl)
            matches = np.where(diffs < rtol * (np.abs(xl).max() + 1e-12))[0]
            if len(matches) > 0:
                j = matches[0]
                paired_X.append(xl)
                paired_Yl.append(Y_low[i])
                paired_Yh.append(Y_high[j])

        if not paired_X:
            return np.empty((0,)), np.empty((0,)), np.empty((0,))

        return np.stack(paired_X), np.stack(paired_Yl), np.stack(paired_Yh)

    def total_cost(self) -> float:
        return sum(self._cost)

    # ── Summary ──────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        return {
            "n_total": self.n_total,
            "per_fidelity": {
                f: self.n_at_fidelity(f) for f in range(self.n_fidelities)
            },
            "total_cost": self.total_cost(),
        }

    def __repr__(self) -> str:
        counts = ", ".join(
            f"fid{f}={self.n_at_fidelity(f)}" for f in range(self.n_fidelities)
        )
        return f"MultiFidelityDataset({counts}, cost={self.total_cost():.1f})"


# ═══════════════════════════════════════════════════════════════════════
# 6.  CONVENIENCE: MULTI-FIDELITY ACTIVE LEARNING STEP
# ═══════════════════════════════════════════════════════════════════════

def multi_fidelity_step(
    candidates: torch.Tensor,
    uncertainties: torch.Tensor,
    objective_fn: Callable,
    selector: FidelitySelector,
    budget: CostBudget,
    dataset: MultiFidelityDataset,
    correlation: Optional[FidelityCorrelation] = None,
    n_select: int = 5,
) -> Dict[str, Any]:
    """
    Execute one step of multi-fidelity active learning:

    1.  Select fidelities for the top ``n_select`` candidates.
    2.  Evaluate the objective at the chosen fidelities.
    3.  Update dataset, budget, and correlation estimator.

    Parameters
    ----------
    candidates    : ``[N, d]`` candidate features.
    uncertainties : ``[N]`` uncertainty scores (from acquisition fn).
    objective_fn  : ``(x, fidelity) → (y, cost)``
    selector      : ``FidelitySelector`` instance.
    budget        : ``CostBudget`` instance.
    dataset       : ``MultiFidelityDataset`` to append to.
    correlation   : ``FidelityCorrelation`` (optional, updated in-place).
    n_select      : How many candidates to evaluate.

    Returns
    -------
    Dict with ``"X_new"``, ``"Y_new"``, ``"fidelities"``, ``"costs"``,
    ``"budget_remaining"``.
    """
    # Rank by uncertainty (descending) and take top n_select
    _, top_idx = uncertainties.topk(min(n_select, len(uncertainties)))
    top_X = candidates[top_idx]

    # Assign fidelities
    fids = selector.select(
        candidates=top_X,
        uncertainties=uncertainties[top_idx],
        n=len(top_idx),
    )

    X_new, Y_new, fid_list, cost_list = [], [], [], []

    for i, (x, fid) in enumerate(zip(top_X, fids)):
        cost_est = selector.fidelity_costs.get(fid, 1.0)
        if not budget.can_afford(cost_est):
            logger.info(f"Budget exhausted; stopping after {i} evaluations.")
            break

        x_np = x.cpu().numpy()
        y, actual_cost = objective_fn(x_np, fid)

        budget.record(fid, actual_cost)
        dataset.add(x_np, np.atleast_1d(y), fid, actual_cost)

        X_new.append(x_np)
        Y_new.append(np.atleast_1d(y))
        fid_list.append(fid)
        cost_list.append(actual_cost)

    # Update correlation if we have paired observations
    if correlation is not None and len(X_new) > 0:
        target_fid = selector.n_fidelities - 1
        for fid in set(fid_list):
            if fid != target_fid:
                X_common, Y_low, Y_high = dataset.get_paired(fid, target_fid)
                if len(X_common) > 0:
                    correlation.add_pairs_batch(fid, target_fid, Y_low.ravel(), Y_high.ravel())

    return {
        "X_new": np.array(X_new) if X_new else np.empty((0,)),
        "Y_new": np.array(Y_new) if Y_new else np.empty((0,)),
        "fidelities": fid_list,
        "costs": cost_list,
        "budget_remaining": budget.remaining,
    }


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "CostBudget",
    "FidelityCorrelation",
    "AutoRegressiveModel",
    "FidelitySelector",
    "MultiFidelityDataset",
    "multi_fidelity_step",
]