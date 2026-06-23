"""
Multi-fidelity Bayesian optimisation for MOF discovery.

This module implements the active-learning loop that selects which MOF
structures to simulate next, balancing exploration (high uncertainty)
against exploitation (promising predicted performance) while accounting
for the drastically different costs of low- vs. high-fidelity GCMC
simulations.

Architecture overview
─────────────────────
1.  **Acquisition functions** — stand-alone callables that score
    candidate MOFs.  Both model-agnostic heuristics (uncertainty
    sampling, expected improvement, Thompson sampling) and
    BoTorch-backed multi-objective acquisitions (qEHVI, qNEI) are
    provided.
2.  **FidelityManager** — maps discrete fidelity levels (e.g.
    short GCMC, production GCMC, DFT) to wall-clock costs and decides
    which fidelity to use for a given candidate.
3.  **MultiFidelityBO** — the main optimisation loop.  It maintains a
    Gaussian-process surrogate (via BoTorch when available, or a
    lightweight fallback), proposes batches of candidates, and tracks
    the Pareto front of multi-objective evaluations.
4.  **UncertaintyAcquisition** — TPNO-specific acquisition that
    combines the ensemble's epistemic uncertainty with conformal
    prediction widths to prioritise MOFs in under-explored regions of
    chemical space.

Dependencies
────────────
* **Required:** ``torch``, ``numpy``.
* **Optional (lazy):** ``botorch``, ``gpytorch`` — used for GP
  surrogates and analytic acquisition functions.  When absent, the
  module falls back to simpler heuristics and logs a warning.

References
──────────
[1] Lakshminarayanan et al. (2017). Simple and Scalable Predictive
    Uncertainty Estimation using Deep Ensembles.
[2] Balandat et al. (2020). BoTorch: A Framework for Efficient
    Monte-Carlo Bayesian Optimization. NeurIPS.
[3] Takeno et al. (2020). Multi-fidelity Bayesian Optimization with
    Max-value Entropy Search. ICML.

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
    Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union,
)

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Optional BoTorch imports (lazy) ──────────────────────────────────

_HAS_BOTORCH = False

def _import_botorch():
    """Lazy-import BoTorch + GPyTorch; set module-level flag."""
    global _HAS_BOTORCH
    try:
        import botorch  # noqa: F401
        import gpytorch  # noqa: F401
        _HAS_BOTORCH = True
    except ImportError:
        _HAS_BOTORCH = False
    return _HAS_BOTORCH


# ═══════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BOConfig:
    """
    Bayesian-optimisation and active-learning hyperparameters.

    Attributes
    ──────────
    n_init          : Number of initial random or LHS evaluations.
    n_iterations    : Total BO iterations (outer loop).
    n_candidates    : Batch size — candidates proposed per iteration.
    acquisition     : Acquisition function name.  One of:
                      ``'qEI'``, ``'qNEI'``, ``'qEHVI'``,
                      ``'ucb'``, ``'thompson'``, ``'uncertainty'``.
    multi_fidelity  : Use multi-fidelity GP and fidelity-aware costs.
    fidelity_dim    : Index of the fidelity column in the design
                      matrix (typically the last column).
    cost_aware      : Re-weight acquisitions by 1 / cost.
    ref_point       : Reference point for hypervolume computations
                      (multi-objective only).  If *None*, inferred
                      from observed data.
    ucb_beta        : Exploration weight for UCB acquisition.
    gp_fit_steps    : Adam steps when fitting the GP hyperparameters.
    gp_lr           : Learning rate for GP fitting.
    mc_samples      : Number of MC samples for qMC-based acquisitions.
    num_restarts    : Multi-start restarts for acquisition optimisation.
    raw_samples     : Random raw samples for initialising the optimiser.
    seed            : Random seed for reproducibility.
    """

    n_init: int = 20
    n_iterations: int = 50
    n_candidates: int = 10
    acquisition: str = "qEHVI"
    multi_fidelity: bool = True
    fidelity_dim: int = -1
    cost_aware: bool = True
    ref_point: Optional[List[float]] = None
    ucb_beta: float = 2.0
    gp_fit_steps: int = 100
    gp_lr: float = 0.01
    mc_samples: int = 256
    num_restarts: int = 10
    raw_samples: int = 512
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════
# 2.  FIDELITY MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FidelityLevel:
    """
    Description of a single simulation fidelity.

    Attributes
    ──────────
    name    : Human-readable label (e.g. ``'gcmc_short'``).
    cost    : Estimated wall-clock cost in CPU-hours.
    index   : Numeric index (0 = cheapest, higher = more expensive).
    noise   : Expected observation noise at this fidelity.
    """

    name: str
    cost: float
    index: int
    noise: float = 0.0


class FidelityManager:
    """
    Registry of available fidelity levels with cost look-up.

    Default levels for the UC-TPNO pipeline::

        0  gcmc_short    ~0.1 CPU-h   (1 k cycles, screening)
        1  gcmc_long     ~1.0 CPU-h   (10 k cycles, production)
        2  dft_opt       ~50  CPU-h   (DFT geometry optimisation)

    Parameters
    ----------
    levels : Sequence of ``FidelityLevel`` objects.  If *None*, the
             three default levels above are used.
    """

    def __init__(self, levels: Optional[Sequence[FidelityLevel]] = None):
        if levels is None:
            levels = [
                FidelityLevel("gcmc_short", cost=0.1, index=0, noise=0.10),
                FidelityLevel("gcmc_long", cost=1.0, index=1, noise=0.02),
                FidelityLevel("dft_opt", cost=50.0, index=2, noise=0.005),
            ]
        self.levels = sorted(levels, key=lambda l: l.index)
        self._by_index = {l.index: l for l in self.levels}

    def cost(self, fidelity_index: Union[int, float, torch.Tensor]) -> float:
        """Return the cost of a given fidelity index."""
        idx = int(fidelity_index)
        if idx in self._by_index:
            return self._by_index[idx].cost
        # Interpolate if continuous fidelity
        indices = sorted(self._by_index.keys())
        costs = [self._by_index[i].cost for i in indices]
        return float(np.interp(float(fidelity_index), indices, costs))

    def noise(self, fidelity_index: Union[int, float]) -> float:
        idx = int(fidelity_index)
        if idx in self._by_index:
            return self._by_index[idx].noise
        return 0.0

    @property
    def cheapest(self) -> FidelityLevel:
        return self.levels[0]

    @property
    def most_accurate(self) -> FidelityLevel:
        return self.levels[-1]

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def __repr__(self) -> str:
        return f"FidelityManager({[l.name for l in self.levels]})"


# ═══════════════════════════════════════════════════════════════════════
# 3.  STAND-ALONE ACQUISITION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

class AcquisitionFunction(Protocol):
    """Protocol that all acquisition functions satisfy."""

    def __call__(
        self,
        X: torch.Tensor,
        model: Any,
        Y_observed: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidates ``X`` → ``[n_candidates]``."""
        ...


def upper_confidence_bound(
    X: torch.Tensor,
    model: Any,
    Y_observed: torch.Tensor,
    beta: float = 2.0,
) -> torch.Tensor:
    """
    UCB: μ(x) + β·σ(x).

    Works with any model that has a ``.posterior(X)`` returning an
    object with ``.mean`` and ``.variance``.
    """
    posterior = model.posterior(X)
    mu = posterior.mean.squeeze(-1)
    sigma = posterior.variance.squeeze(-1).sqrt()
    return mu + beta * sigma


def thompson_sampling(
    X: torch.Tensor,
    model: Any,
    Y_observed: torch.Tensor,
    n_samples: int = 1,
) -> torch.Tensor:
    """
    Score candidates by drawing a sample from the GP posterior.
    """
    posterior = model.posterior(X)
    sample = posterior.rsample(torch.Size([n_samples]))  # [S, N, 1]
    return sample.mean(dim=0).squeeze(-1)  # [N]


def pure_uncertainty(
    X: torch.Tensor,
    model: Any,
    Y_observed: torch.Tensor,
) -> torch.Tensor:
    """
    Select the candidate with maximum predictive variance (pure
    exploration).
    """
    posterior = model.posterior(X)
    return posterior.variance.squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
# 4.  TPNO-SPECIFIC ACQUISITION
# ═══════════════════════════════════════════════════════════════════════

class UncertaintyAcquisition:
    """
    TPNO-specific acquisition function that combines:

    * **Epistemic uncertainty** from the deep ensemble (model
      disagreement).
    * **Conformal interval width** from the calibrated conformal
      predictor (distribution-free coverage).
    * **Predicted selectivity** as an exploitation signal (prefer
      MOFs likely to have high CO₂/N₂ selectivity).

    The final score is::

        score = w_epi · σ_epi + w_conf · width_conf + w_sel · selectivity

    Parameters
    ----------
    ensemble     : The ``TPNOEnsemble`` model.
    conformal    : A fitted ``ConformalCalibrator`` (optional).
    w_epistemic  : Weight for epistemic uncertainty.
    w_conformal  : Weight for conformal interval width.
    w_selectivity: Weight for predicted selectivity.
    """

    def __init__(
        self,
        ensemble: nn.Module,
        conformal: Optional[Any] = None,
        w_epistemic: float = 1.0,
        w_conformal: float = 0.5,
        w_selectivity: float = 0.3,
    ):
        self.ensemble = ensemble
        self.conformal = conformal
        self.w_epi = w_epistemic
        self.w_conf = w_conformal
        self.w_sel = w_selectivity

    @torch.no_grad()
    def __call__(
        self,
        graphs: Any,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score a batch of MOF candidates.

        Parameters
        ----------
        graphs     : Batched graph data for the ensemble encoder.
        conditions : ``[B, P, 4]`` thermodynamic conditions.

        Returns
        -------
        ``[B]`` acquisition scores (higher = more informative).
        """
        out = self.ensemble(graphs, conditions, return_all=False)

        # Epistemic: mean std across condition points and components
        epi = out["epistemic"].mean(dim=(1, 2))  # [B]

        score = self.w_epi * epi

        # Conformal width (if calibrator is fitted)
        if self.conformal is not None and self.conformal.is_fitted:
            y_pred_np = out["q_pred"].mean(dim=1).cpu().numpy()  # [B, C]
            sigma_np = out["aleatoric"].mean(dim=1).cpu().numpy()
            iv = self.conformal.predict_intervals(
                {"y_pred": y_pred_np, "y_std": sigma_np}
            )
            conf_width = iv["upper"] - iv["lower"]
            conf_score = torch.from_numpy(conf_width.mean(axis=-1)).to(epi)
            score = score + self.w_conf * conf_score

        # Selectivity exploitation: CO₂/N₂ selectivity
        q_mean = out["q_pred"].mean(dim=1)  # [B, C]
        if q_mean.shape[-1] >= 2:
            sel = q_mean[:, 0] / (q_mean[:, 1] + 1e-8)  # CO₂ / N₂
            # Normalise to [0, 1]
            sel = (sel - sel.min()) / (sel.max() - sel.min() + 1e-8)
            score = score + self.w_sel * sel

        return score


# ═══════════════════════════════════════════════════════════════════════
# 5.  MULTI-FIDELITY BAYESIAN OPTIMISATION LOOP
# ═══════════════════════════════════════════════════════════════════════

class MultiFidelityBO:
    """
    Multi-fidelity Bayesian optimisation loop for MOF screening.

    This class orchestrates:

    1.  Initial random evaluations at the cheapest fidelity.
    2.  GP surrogate fitting (BoTorch ``SingleTaskGP`` or
        ``SingleTaskMultiFidelityGP``).
    3.  Acquisition-function optimisation to propose the next batch.
    4.  Fidelity selection (cheapest for exploration, expensive for
        promising candidates).
    5.  Pareto-front tracking for multi-objective problems.

    Parameters
    ----------
    config        : ``BOConfig`` hyperparameters.
    bounds        : ``[2, d]`` lower/upper bounds for the design space.
    fidelity_mgr  : ``FidelityManager`` (optional; uses defaults).
    objective_fn  : Callable ``(x, fidelity) → (y, cost)`` that
                    evaluates a candidate (may be a simulation wrapper).
    """

    def __init__(
        self,
        config: BOConfig,
        bounds: torch.Tensor,
        fidelity_mgr: Optional[FidelityManager] = None,
        objective_fn: Optional[Callable] = None,
    ):
        self.config = config
        self.bounds = bounds
        self.fidelity_mgr = fidelity_mgr or FidelityManager()
        self.objective_fn = objective_fn

        # Observation storage
        self.X: Optional[torch.Tensor] = None
        self.Y: Optional[torch.Tensor] = None
        self.fidelities: Optional[torch.Tensor] = None
        self.costs: Optional[torch.Tensor] = None

        # GP surrogate
        self._model = None
        self._mll = None

    # ── Observation management ───────────────────────────────────

    @property
    def n_observed(self) -> int:
        return 0 if self.X is None else self.X.shape[0]

    def add_observations(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        fidelities: Optional[torch.Tensor] = None,
        costs: Optional[torch.Tensor] = None,
    ) -> None:
        """Append new observations and refit the GP."""
        if self.X is None:
            self.X = X
            self.Y = Y
            self.fidelities = fidelities
            self.costs = costs
        else:
            self.X = torch.cat([self.X, X], dim=0)
            self.Y = torch.cat([self.Y, Y], dim=0)
            if fidelities is not None and self.fidelities is not None:
                self.fidelities = torch.cat([self.fidelities, fidelities], dim=0)
            if costs is not None and self.costs is not None:
                self.costs = torch.cat([self.costs, costs], dim=0)

        self._fit_model()

    def initialize_random(
        self,
        n: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate ``n`` random initial designs via Sobol or uniform
        sampling within ``self.bounds``.

        Returns the design tensor ``[n, d]`` (does *not* evaluate them).
        """
        n = n or self.config.n_init
        d = self.bounds.shape[1]

        try:
            from torch.quasirandom import SobolEngine
            sobol = SobolEngine(dimension=d, scramble=True, seed=self.config.seed)
            raw = sobol.draw(n).to(self.bounds.dtype)
        except Exception:
            torch.manual_seed(self.config.seed)
            raw = torch.rand(n, d)

        lo, hi = self.bounds[0], self.bounds[1]
        return lo + (hi - lo) * raw

    # ── GP surrogate ─────────────────────────────────────────────

    def _fit_model(self) -> None:
        """Fit (or refit) the GP surrogate on current observations."""
        if not _import_botorch():
            logger.warning(
                "BoTorch not installed — GP surrogate unavailable. "
                "Falling back to random candidate proposal."
            )
            return

        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood

        train_X = self.X.detach().clone()
        train_Y = self.Y.detach().clone()

        # Append fidelity column for multi-fidelity
        if self.config.multi_fidelity and self.fidelities is not None:
            f_col = self.fidelities.unsqueeze(-1).to(train_X)
            train_X = torch.cat([train_X, f_col], dim=-1)

        n_obj = train_Y.shape[-1] if train_Y.dim() > 1 else 1

        try:
            from botorch.models import SingleTaskMultiFidelityGP
            if self.config.multi_fidelity and self.fidelities is not None:
                fid_col = train_X.shape[-1] - 1
                self._model = SingleTaskMultiFidelityGP(
                    train_X, train_Y,
                    data_fidelities=[fid_col],
                    outcome_transform=Standardize(m=n_obj),
                )
            else:
                self._model = SingleTaskGP(
                    train_X, train_Y,
                    outcome_transform=Standardize(m=n_obj),
                )
        except Exception:
            self._model = SingleTaskGP(
                train_X, train_Y,
                outcome_transform=Standardize(m=n_obj),
            )

        self._mll = ExactMarginalLogLikelihood(
            self._model.likelihood, self._model,
        )

        # Fit via Adam
        self._model.train()
        self._mll.train()
        opt = torch.optim.Adam(self._model.parameters(), lr=self.config.gp_lr)

        for _ in range(self.config.gp_fit_steps):
            opt.zero_grad()
            output = self._model(train_X)
            loss = -self._mll(output, train_Y)
            loss.backward()
            opt.step()

        self._model.eval()
        self._mll.eval()

    # ── Acquisition functions ────────────────────────────────────

    def _build_acquisition(self):
        """Build a BoTorch acquisition function from config."""
        if self._model is None:
            return None

        if not _HAS_BOTORCH:
            return None

        from botorch.sampling import SobolQMCNormalSampler
        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.config.mc_samples]),
        )

        acq_name = self.config.acquisition.lower()

        if acq_name == "qei":
            from botorch.acquisition import qExpectedImprovement
            best_f = self.Y.max(dim=0)[0] if self.Y.dim() > 1 else self.Y.max()
            return qExpectedImprovement(
                model=self._model,
                best_f=best_f,
                sampler=sampler,
            )

        if acq_name == "qnei":
            from botorch.acquisition import qNoisyExpectedImprovement
            return qNoisyExpectedImprovement(
                model=self._model,
                X_baseline=self.X,
                sampler=sampler,
            )

        if acq_name == "qehvi":
            from botorch.acquisition.multi_objective import (
                qExpectedHypervolumeImprovement,
            )
            if self.config.ref_point is not None:
                ref = torch.tensor(self.config.ref_point, dtype=self.Y.dtype)
            else:
                y_range = self.Y.max(0)[0] - self.Y.min(0)[0]
                ref = self.Y.min(0)[0] - 0.1 * y_range

            return qExpectedHypervolumeImprovement(
                model=self._model,
                ref_point=ref,
                sampler=sampler,
            )

        if acq_name == "ucb":
            from botorch.acquisition import qUpperConfidenceBound
            return qUpperConfidenceBound(
                model=self._model,
                beta=self.config.ucb_beta,
                sampler=sampler,
            )

        raise ValueError(f"Unknown BoTorch acquisition: {self.config.acquisition}")

    # ── Candidate proposal ───────────────────────────────────────

    def propose_candidates(
        self,
        n: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Propose the next batch of candidates by optimising the
        acquisition function.

        Returns ``[n, d]`` design-space candidates.
        """
        n = n or self.config.n_candidates

        acq = self._build_acquisition()
        if acq is None:
            logger.info("No GP model available; proposing random candidates.")
            return self.initialize_random(n)

        from botorch.optim import optimize_acqf

        bounds = self.bounds
        if self.config.multi_fidelity and self.fidelities is not None:
            # Extend bounds to include fidelity column
            fid_lo = torch.zeros(1, dtype=bounds.dtype)
            fid_hi = torch.tensor(
                [float(self.fidelity_mgr.n_levels - 1)], dtype=bounds.dtype,
            )
            bounds = torch.cat(
                [bounds, torch.stack([fid_lo, fid_hi], dim=0)], dim=1,
            )

        candidates, acq_values = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=n,
            num_restarts=self.config.num_restarts,
            raw_samples=self.config.raw_samples,
            options={"batch_limit": 5, "maxiter": 200},
        )

        # Cost-aware re-ranking
        if self.config.cost_aware and self.config.multi_fidelity:
            candidates = self._cost_rerank(candidates, acq_values)

        return candidates

    def _cost_rerank(
        self,
        candidates: torch.Tensor,
        acq_values: torch.Tensor,
    ) -> torch.Tensor:
        """Re-rank candidates by acquisition / cost."""
        costs = torch.tensor(
            [self.fidelity_mgr.cost(c[self.config.fidelity_dim]) for c in candidates],
            dtype=candidates.dtype,
        )
        weighted = acq_values / (costs + 1e-8)
        idx = weighted.argsort(descending=True)
        return candidates[idx]

    # ── Full optimisation loop ───────────────────────────────────

    def run(
        self,
        objective_fn: Optional[Callable] = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full BO loop.

        Parameters
        ----------
        objective_fn : ``(x, fidelity) → (y, cost)``; overrides the
                       instance-level ``self.objective_fn``.
        callback     : Called after each iteration with
                       ``(iteration, X, Y, candidates)``.

        Returns
        -------
        Dict with ``X``, ``Y``, ``costs``, ``fidelities``,
        ``pareto_X``, ``pareto_Y``.
        """
        obj_fn = objective_fn or self.objective_fn
        if obj_fn is None:
            raise ValueError("No objective function provided.")

        # Initialise
        if self.n_observed == 0:
            X_init = self.initialize_random()
            Y_list, cost_list, fid_list = [], [], []
            fid_cheapest = self.fidelity_mgr.cheapest.index
            for x in X_init:
                y, cost = obj_fn(x, fid_cheapest)
                Y_list.append(y)
                cost_list.append(cost)
                fid_list.append(fid_cheapest)
            self.add_observations(
                X_init,
                torch.stack(Y_list),
                torch.tensor(fid_list, dtype=torch.float),
                torch.tensor(cost_list, dtype=torch.float),
            )

        # Main loop
        for it in range(self.config.n_iterations):
            candidates = self.propose_candidates()

            Y_new, cost_new, fid_new = [], [], []
            for c in candidates:
                if self.config.multi_fidelity:
                    fid = int(c[self.config.fidelity_dim].item())
                else:
                    fid = self.fidelity_mgr.most_accurate.index

                x = c[: self.bounds.shape[1]]
                y, cost = obj_fn(x, fid)
                Y_new.append(y)
                cost_new.append(cost)
                fid_new.append(fid)

            self.add_observations(
                candidates[:, : self.bounds.shape[1]],
                torch.stack(Y_new),
                torch.tensor(fid_new, dtype=torch.float),
                torch.tensor(cost_new, dtype=torch.float),
            )

            if callback is not None:
                callback(it, self.X, self.Y, candidates)

            logger.info(
                f"BO iter {it+1}/{self.config.n_iterations} — "
                f"n_obs={self.n_observed}, "
                f"best_Y={self.Y.max(0)[0].tolist()}"
            )

        # Pareto front
        pareto_X, pareto_Y = self.get_pareto_front()

        return {
            "X": self.X,
            "Y": self.Y,
            "costs": self.costs,
            "fidelities": self.fidelities,
            "pareto_X": pareto_X,
            "pareto_Y": pareto_Y,
        }

    # ── Pareto front ─────────────────────────────────────────────

    def get_pareto_front(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the non-dominated subset of observed ``(X, Y)``."""
        if self.Y is None or self.Y.numel() == 0:
            return torch.empty(0), torch.empty(0)

        Y = self.Y
        if Y.dim() == 1:
            best_idx = Y.argmax()
            return self.X[best_idx : best_idx + 1], Y[best_idx : best_idx + 1]

        mask = _is_non_dominated(Y)
        return self.X[mask], Y[mask]

    # ── Summary ──────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return a dict summarising the current state."""
        pareto_X, pareto_Y = self.get_pareto_front()
        total_cost = float(self.costs.sum()) if self.costs is not None else 0.0
        return {
            "n_observed": self.n_observed,
            "total_cost": total_cost,
            "pareto_size": len(pareto_X),
            "best_Y": self.Y.max(0)[0].tolist() if self.Y is not None else [],
        }


# ═══════════════════════════════════════════════════════════════════════
# 6.  PARETO-DOMINANCE UTILITY
# ═══════════════════════════════════════════════════════════════════════

def _is_non_dominated(Y: torch.Tensor) -> torch.Tensor:
    """
    Return a boolean mask identifying the Pareto-optimal rows of
    ``Y`` (maximisation assumed on all objectives).

    Falls back to a simple O(n²) sweep when BoTorch is absent.
    """
    if _import_botorch():
        try:
            from botorch.utils.multi_objective import is_non_dominated
            return is_non_dominated(Y)
        except Exception:
            pass

    # Fallback: pairwise dominance check
    n = Y.shape[0]
    dominated = torch.zeros(n, dtype=torch.bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (Y[j] >= Y[i]).all() and (Y[j] > Y[i]).any():
                dominated[i] = True
                break
    return ~dominated


# ═══════════════════════════════════════════════════════════════════════
# 7.  CONVENIENCE: TPNO-AWARE SCREENING LOOP
# ═══════════════════════════════════════════════════════════════════════

def tpno_screening_loop(
    ensemble: nn.Module,
    candidate_graphs: List[Any],
    candidate_conditions: torch.Tensor,
    budget: int = 100,
    conformal: Optional[Any] = None,
    w_epistemic: float = 1.0,
    w_conformal: float = 0.5,
    w_selectivity: float = 0.3,
) -> List[int]:
    """
    Lightweight active-screening loop that ranks candidate MOFs using
    the TPNO ensemble + conformal calibrator without requiring a GP
    surrogate or BoTorch.

    Parameters
    ----------
    ensemble            : Trained ``TPNOEnsemble``.
    candidate_graphs    : List of graph-batch inputs (one per MOF).
    candidate_conditions: ``[N, P, 4]`` conditions to evaluate at.
    budget              : How many MOFs to select.
    conformal           : Fitted ``ConformalCalibrator`` (optional).

    Returns
    -------
    List of indices into ``candidate_graphs`` ranked by acquisition
    score (highest first), truncated to ``budget``.
    """
    acq = UncertaintyAcquisition(
        ensemble=ensemble,
        conformal=conformal,
        w_epistemic=w_epistemic,
        w_conformal=w_conformal,
        w_selectivity=w_selectivity,
    )

    scores_list = []
    for i, g in enumerate(candidate_graphs):
        cond = candidate_conditions[i : i + 1] if candidate_conditions.dim() == 3 else candidate_conditions.unsqueeze(0)
        s = acq(g, cond)
        scores_list.append(s.item())

    scores = np.array(scores_list)
    ranking = np.argsort(-scores).tolist()

    return ranking[:budget]


# ═══════════════════════════════════════════════════════════════════════
# 8.  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    # Config
    "BOConfig",
    # Fidelity
    "FidelityLevel",
    "FidelityManager",
    # Acquisition functions
    "upper_confidence_bound",
    "thompson_sampling",
    "pure_uncertainty",
    "UncertaintyAcquisition",
    # BO loop
    "MultiFidelityBO",
    # Convenience
    "tpno_screening_loop",
]