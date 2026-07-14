"""
Multi-fidelity Bayesian optimisation for MOF discovery.

Architecture overview
─────────────────────
1.  Acquisition functions — stand-alone callables that score candidate MOFs.
2.  FidelityManager    — maps fidelity levels to costs.
3.  MultiFidelityBO    — main optimisation loop with GP surrogate.
4.  UncertaintyAcquisition — TPNO-specific acquisition (ensemble + conformal).

Dependencies
────────────
* Required : ``torch``, ``numpy``.
* Optional  : ``botorch``, ``gpytorch`` — lazy-imported; falls back to
              heuristics when absent.

Fixes vs. original
------------------
1. BUG FIXED: @torch.no_grad() on UncertaintyAcquisition.__call__.
   TPNO derives loadings via autograd.grad(omega, mu).  Under no_grad,
   requires_grad_(True) is silently ignored → RuntimeError.  Decorator
   removed; model.eval() is sufficient.

2. BUG FIXED: with torch.no_grad() in score_candidates_simple.
   Same root cause — model call inside a no_grad context crashes TPNO.
   Removed; the caller is responsible for gradient context.

3. BUG FIXED: entropy_acquisition called ensemble members inside
   torch.no_grad() without re-enabling gradients.  The outer no_grad
   block has been removed; model.eval() handles dropout/BN correctly.

4. BUG FIXED: qEHVI acquisition missing required partitioning argument.
   Since BoTorch 0.6, qExpectedHypervolumeImprovement requires a
   NondominatedPartitioning object.  Added.

5. BUG FIXED: Manual Adam GP-fitting loop passed 2D train_Y directly
   to self._mll(), which fails for single-output models that expect a
   1D target.  Replaced with BoTorch's fit_gpytorch_mll() which handles
   all model/output-transform combinations correctly.

References
──────────
[1] Lakshminarayanan et al. (2017). Deep Ensembles. NeurIPS.
[2] Balandat et al. (2020). BoTorch. NeurIPS.
[3] Takeno et al. (2020). Multi-fidelity BO with Max-value Entropy. ICML.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO
License : MIT
"""
from __future__ import annotations

import logging
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


def _import_botorch() -> bool:
    global _HAS_BOTORCH
    try:
        import botorch   # noqa: F401
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
    """Bayesian-optimisation and active-learning hyperparameters."""
    n_init:         int            = 20
    n_iterations:   int            = 50
    n_candidates:   int            = 10
    acquisition:    str            = "qEHVI"
    multi_fidelity: bool           = True
    fidelity_dim:   int            = -1
    cost_aware:     bool           = True
    ref_point:      Optional[List[float]] = None
    ucb_beta:       float          = 2.0
    gp_fit_steps:   int            = 100
    gp_lr:          float          = 0.01
    mc_samples:     int            = 256
    num_restarts:   int            = 10
    raw_samples:    int            = 512
    seed:           int            = 42


# ═══════════════════════════════════════════════════════════════════════
# 2.  FIDELITY MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FidelityLevel:
    """Description of a single simulation fidelity."""
    name:  str
    cost:  float
    index: int
    noise: float = 0.0


class FidelityManager:
    """
    Registry of available fidelity levels with cost look-up.

    Default levels for the UC-TPNO pipeline::

        0  gcmc_short    ~0.1 CPU-h   (1 k cycles, screening)
        1  gcmc_long     ~1.0 CPU-h   (10 k cycles, production)
        2  dft_opt       ~50  CPU-h   (DFT geometry optimisation)
    """

    def __init__(self, levels: Optional[Sequence[FidelityLevel]] = None):
        if levels is None:
            levels = [
                FidelityLevel("gcmc_short", cost=0.1,  index=0, noise=0.10),
                FidelityLevel("gcmc_long",  cost=1.0,  index=1, noise=0.02),
                FidelityLevel("dft_opt",    cost=50.0, index=2, noise=0.005),
            ]
        self.levels    = sorted(levels, key=lambda l: l.index)
        self._by_index = {l.index: l for l in self.levels}

    def cost(self, fidelity_index: Union[int, float, torch.Tensor]) -> float:
        idx = int(fidelity_index)
        if idx in self._by_index:
            return self._by_index[idx].cost
        indices = sorted(self._by_index.keys())
        costs   = [self._by_index[i].cost for i in indices]
        return float(np.interp(float(fidelity_index), indices, costs))

    def noise(self, fidelity_index: Union[int, float]) -> float:
        idx = int(fidelity_index)
        return self._by_index[idx].noise if idx in self._by_index else 0.0

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
        X:          torch.Tensor,
        model:      Any,
        Y_observed: torch.Tensor,
    ) -> torch.Tensor: ...


def upper_confidence_bound(
    X:          torch.Tensor,
    model:      Any,
    Y_observed: torch.Tensor,
    beta:       float = 2.0,
) -> torch.Tensor:
    """UCB: μ(x) + β·σ(x).  Requires model.posterior(X)."""
    posterior = model.posterior(X)
    mu    = posterior.mean.squeeze(-1)
    sigma = posterior.variance.squeeze(-1).sqrt()
    return mu + beta * sigma


def thompson_sampling(
    X:          torch.Tensor,
    model:      Any,
    Y_observed: torch.Tensor,
    n_samples:  int = 1,
) -> torch.Tensor:
    """Score candidates by drawing a sample from the GP posterior."""
    posterior = model.posterior(X)
    sample    = posterior.rsample(torch.Size([n_samples]))
    return sample.mean(dim=0).squeeze(-1)


def pure_uncertainty(
    X:          torch.Tensor,
    model:      Any,
    Y_observed: torch.Tensor,
) -> torch.Tensor:
    """Pure-exploration: maximum predictive variance."""
    posterior = model.posterior(X)
    return posterior.variance.squeeze(-1)


def entropy_acquisition(
    model:      nn.Module,
    candidates: Any,
    conditions: torch.Tensor,
    n_samples:  int = 10,
) -> torch.Tensor:
    """
    Entropy-based acquisition using MC dropout or ensemble.

    FIX: original wrapped ensemble model calls inside torch.no_grad(),
    which prevents TPNO from computing loadings via autograd.grad.
    Removed the outer no_grad block; model.eval() handles BN/dropout.

    Parameters
    ----------
    model      : nn.Module (TPNO or ensemble)
    candidates : batched graph data
    conditions : [B, P, D]
    n_samples  : MC-dropout samples (ignored for ensembles)
    """
    # FIX: model.eval() is sufficient — NO torch.no_grad() wrapper here
    # because TPNO internally calls autograd.grad(omega, mu).
    model.eval()

    ensemble_obj = getattr(model, "ensemble", None) or getattr(model, "models", None)

    if ensemble_obj is not None:
        preds: List[torch.Tensor] = []
        for m in ensemble_obj:
            out = m(candidates, conditions)
            q = out["q_pred"] if isinstance(out, dict) else out[0]
            preds.append(q)
        pred_stack = torch.stack(preds, dim=0)          # [M, B, P, C]
        std        = pred_stack.std(dim=0)               # [B, P, C]
        return std.mean(dim=(1, 2))                      # [B]

    # MC-dropout path — switch to train() to activate dropout
    model.train()
    preds_mc: List[torch.Tensor] = []
    for _ in range(n_samples):
        out = model(candidates, conditions)
        q   = out["q_pred"] if isinstance(out, dict) else out[0]
        preds_mc.append(q.detach())
    model.eval()

    pred_stack = torch.stack(preds_mc, dim=0)
    std        = pred_stack.std(dim=0)
    return std.mean(dim=(1, 2))


def random_acquisition(
    candidates: Any,
    n:          int,
    seed:       int = 42,
) -> torch.Tensor:
    """Random acquisition for baseline comparison."""
    rng = np.random.RandomState(seed)
    return torch.tensor(rng.random(n), dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════
# 4.  TPNO-SPECIFIC ACQUISITION
# ═══════════════════════════════════════════════════════════════════════

class UncertaintyAcquisition:
    """
    TPNO-specific acquisition combining:
    * Epistemic uncertainty   (ensemble disagreement)
    * Conformal interval width (distribution-free coverage)
    * Predicted CO₂/N₂ selectivity (exploitation signal)

    Score = w_epi · σ_epi + w_conf · width_conf + w_sel · selectivity

    FIX: original had @torch.no_grad() decorator which prevents TPNO
    from computing loadings via autograd.grad(omega, mu).  Removed;
    model.eval() is sufficient.
    """

    def __init__(
        self,
        ensemble:      nn.Module,
        conformal:     Optional[Any] = None,
        w_epistemic:   float = 1.0,
        w_conformal:   float = 0.5,
        w_selectivity: float = 0.3,
    ):
        self.ensemble = ensemble
        self.conformal = conformal
        self.w_epi  = w_epistemic
        self.w_conf = w_conformal
        self.w_sel  = w_selectivity

    # FIX: @torch.no_grad() REMOVED — autograd.grad inside TPNO requires
    # gradients to be enabled.  model.eval() disables dropout/BN only.
    def __call__(
        self,
        graphs:     Any,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score a batch of MOF candidates.

        Parameters
        ----------
        graphs     : batched graph data for the encoder.
        conditions : [B, P, 4] thermodynamic conditions.

        Returns
        -------
        [B] acquisition scores (higher = more informative).
        """
        self.ensemble.eval()

        out = self.ensemble(graphs, conditions, return_all=False)

        # Epistemic: mean std across condition-points and components
        epi   = out["epistemic"].mean(dim=(1, 2))  # [B]
        score = self.w_epi * epi

        # Conformal width (optional)
        if self.conformal is not None and self.conformal.is_fitted:
            y_pred_np = out["q_pred"].mean(dim=1).cpu().detach().numpy()
            sigma_np  = out["aleatoric"].mean(dim=1).cpu().detach().numpy()
            iv = self.conformal.predict_intervals(
                {"y_pred": y_pred_np, "y_std": sigma_np}
            )
            conf_width = iv["upper"] - iv["lower"]
            conf_score = torch.from_numpy(
                conf_width.mean(axis=-1).astype(np.float32)
            ).to(epi.device)
            score = score + self.w_conf * conf_score

        # Selectivity exploitation
        q_mean = out["q_pred"].mean(dim=1)              # [B, C]
        if q_mean.shape[-1] >= 2:
            sel = q_mean[:, 0] / (q_mean[:, 1] + 1e-8) # [B]
            sel = (sel - sel.min()) / (sel.max() - sel.min() + 1e-8)
            score = score + self.w_sel * sel

        return score


# ═══════════════════════════════════════════════════════════════════════
# 5.  MULTI-FIDELITY BAYESIAN OPTIMISATION LOOP
# ═══════════════════════════════════════════════════════════════════════

class MultiFidelityBO:
    """
    Multi-fidelity Bayesian optimisation loop for MOF screening.

    Orchestrates:
    1. Initial random evaluations at the cheapest fidelity.
    2. GP surrogate fitting (BoTorch SingleTaskGP or
       SingleTaskMultiFidelityGP).
    3. Acquisition-function optimisation → next batch.
    4. Fidelity selection.
    5. Pareto-front tracking.

    Parameters
    ----------
    config       : BOConfig hyperparameters.
    bounds       : [2, d] lower/upper bounds for the design space.
    fidelity_mgr : FidelityManager (optional; uses defaults).
    objective_fn : Callable (x, fidelity) → (y, cost).
    """

    def __init__(
        self,
        config:       BOConfig,
        bounds:       torch.Tensor,
        fidelity_mgr: Optional[FidelityManager] = None,
        objective_fn: Optional[Callable]        = None,
    ):
        self.config       = config
        self.bounds       = bounds
        self.fidelity_mgr = fidelity_mgr or FidelityManager()
        self.objective_fn = objective_fn

        self.X:          Optional[torch.Tensor] = None
        self.Y:          Optional[torch.Tensor] = None
        self.fidelities: Optional[torch.Tensor] = None
        self.costs:      Optional[torch.Tensor] = None

        self._model = None
        self._mll   = None

    # ── Observation management ────────────────────────────────────────

    @property
    def n_observed(self) -> int:
        return 0 if self.X is None else self.X.shape[0]

    def add_observations(
        self,
        X:          torch.Tensor,
        Y:          torch.Tensor,
        fidelities: Optional[torch.Tensor] = None,
        costs:      Optional[torch.Tensor] = None,
    ) -> None:
        """Append new observations and refit the GP."""
        if self.X is None:
            self.X          = X
            self.Y          = Y
            self.fidelities = fidelities
            self.costs      = costs
        else:
            self.X = torch.cat([self.X, X], dim=0)
            self.Y = torch.cat([self.Y, Y], dim=0)
            if fidelities is not None and self.fidelities is not None:
                self.fidelities = torch.cat([self.fidelities, fidelities], dim=0)
            if costs is not None and self.costs is not None:
                self.costs = torch.cat([self.costs, costs], dim=0)
        self._fit_model()

    def initialize_random(self, n: Optional[int] = None) -> torch.Tensor:
        """Generate ``n`` initial designs via Sobol / uniform sampling."""
        n = n or self.config.n_init
        d = self.bounds.shape[1]
        try:
            from torch.quasirandom import SobolEngine
            sobol = SobolEngine(dimension=d, scramble=True, seed=self.config.seed)
            raw   = sobol.draw(n).to(self.bounds.dtype)
        except Exception:
            torch.manual_seed(self.config.seed)
            raw = torch.rand(n, d)
        lo, hi = self.bounds[0], self.bounds[1]
        return lo + (hi - lo) * raw

    # ── GP surrogate ──────────────────────────────────────────────────

    def _fit_model(self) -> None:
        """
        Fit (or refit) the GP surrogate on current observations.

        FIX: original used a manual Adam loop that passed 2D train_Y to
        self._mll(), failing for single-output models.  Now uses
        fit_gpytorch_mll() which handles all output-transform variants.
        """
        if not _import_botorch():
            logger.warning(
                "BoTorch not installed — GP surrogate unavailable. "
                "Falling back to random candidate proposal."
            )
            return

        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Standardize
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        train_X = self.X.detach().clone().double()
        train_Y = self.Y.detach().clone().double()

        # Append fidelity column for multi-fidelity
        if self.config.multi_fidelity and self.fidelities is not None:
            f_col   = self.fidelities.unsqueeze(-1).to(train_X)
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
            self._model.likelihood, self._model
        )

        # FIX: use fit_gpytorch_mll instead of a fragile manual Adam loop
        try:
            fit_gpytorch_mll(self._mll)
        except Exception as e:
            logger.warning("fit_gpytorch_mll failed (%s); GP may be suboptimal.", e)

        self._model.eval()

    # ── Acquisition functions ─────────────────────────────────────────

    def _build_acquisition(self):
        """Build a BoTorch acquisition function from config."""
        if self._model is None or not _HAS_BOTORCH:
            return None

        from botorch.sampling import SobolQMCNormalSampler
        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.config.mc_samples])
        )

        acq_name = self.config.acquisition.lower()

        if acq_name == "qei":
            from botorch.acquisition import qExpectedImprovement
            best_f = self.Y.max(dim=0)[0] if self.Y.dim() > 1 else self.Y.max()
            return qExpectedImprovement(
                model=self._model, best_f=best_f, sampler=sampler
            )

        if acq_name == "qnei":
            from botorch.acquisition import qNoisyExpectedImprovement
            return qNoisyExpectedImprovement(
                model=self._model, X_baseline=self.X, sampler=sampler
            )

        if acq_name == "qehvi":
            # FIX: NondominatedPartitioning is required since BoTorch 0.6.
            # The original call without partitioning raises TypeError.
            from botorch.acquisition.multi_objective import (
                qExpectedHypervolumeImprovement,
            )
            from botorch.utils.multi_objective.box_decompositions.non_dominated import (
                NondominatedPartitioning,
            )
            if self.config.ref_point is not None:
                ref = torch.tensor(
                    self.config.ref_point, dtype=self.Y.dtype
                )
            else:
                y_range = self.Y.max(0)[0] - self.Y.min(0)[0]
                ref     = self.Y.min(0)[0] - 0.1 * y_range

            partitioning = NondominatedPartitioning(ref_point=ref, Y=self.Y)

            return qExpectedHypervolumeImprovement(
                model=self._model,
                ref_point=ref,
                partitioning=partitioning,
                sampler=sampler,
            )

        if acq_name == "ucb":
            from botorch.acquisition import qUpperConfidenceBound
            return qUpperConfidenceBound(
                model=self._model, beta=self.config.ucb_beta, sampler=sampler
            )

        raise ValueError(
            f"Unknown BoTorch acquisition: '{self.config.acquisition}'. "
            "Choose from: qEI, qNEI, qEHVI, ucb."
        )

    # ── Candidate proposal ────────────────────────────────────────────

    def propose_candidates(self, n: Optional[int] = None) -> torch.Tensor:
        """Propose next batch by optimising the acquisition function."""
        n   = n or self.config.n_candidates
        acq = self._build_acquisition()

        if acq is None:
            logger.info("No GP model available; proposing random candidates.")
            return self.initialize_random(n)

        from botorch.optim import optimize_acqf
        bounds = self.bounds

        if self.config.multi_fidelity and self.fidelities is not None:
            fid_lo = torch.zeros(1, dtype=bounds.dtype)
            fid_hi = torch.tensor(
                [float(self.fidelity_mgr.n_levels - 1)], dtype=bounds.dtype
            )
            bounds = torch.cat(
                [bounds, torch.stack([fid_lo, fid_hi], dim=0)], dim=1
            )

        candidates, acq_values = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=n,
            num_restarts=self.config.num_restarts,
            raw_samples=self.config.raw_samples,
            options={"batch_limit": 5, "maxiter": 200},
        )

        if self.config.cost_aware and self.config.multi_fidelity:
            candidates = self._cost_rerank(candidates, acq_values)

        return candidates

    def _cost_rerank(
        self,
        candidates:  torch.Tensor,
        acq_values:  torch.Tensor,
    ) -> torch.Tensor:
        """Re-rank candidates by acquisition / cost."""
        costs = torch.tensor(
            [
                self.fidelity_mgr.cost(c[self.config.fidelity_dim])
                for c in candidates
            ],
            dtype=candidates.dtype,
        )
        weighted = acq_values / (costs + 1e-8)
        idx      = weighted.argsort(descending=True)
        return candidates[idx]

    # ── Full optimisation loop ────────────────────────────────────────

    def run(
        self,
        objective_fn: Optional[Callable] = None,
        callback:     Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full BO loop.

        Parameters
        ----------
        objective_fn : (x, fidelity) → (y: Tensor, cost: float)
        callback     : called after each iteration with
                       (iteration, X, Y, candidates)

        Returns
        -------
        Dict with X, Y, costs, fidelities, pareto_X, pareto_Y.
        """
        obj_fn = objective_fn or self.objective_fn
        if obj_fn is None:
            raise ValueError("No objective function provided.")

        if self.n_observed == 0:
            X_init       = self.initialize_random()
            Y_list:    List[torch.Tensor] = []
            cost_list: List[float]        = []
            fid_list:  List[int]          = []
            fid_cheapest = self.fidelity_mgr.cheapest.index

            for x in X_init:
                y, cost = obj_fn(x, fid_cheapest)
                Y_list.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
                cost_list.append(float(cost))
                fid_list.append(fid_cheapest)

            self.add_observations(
                X_init,
                torch.stack(Y_list),
                torch.tensor(fid_list,  dtype=torch.float),
                torch.tensor(cost_list, dtype=torch.float),
            )

        for it in range(self.config.n_iterations):
            candidates = self.propose_candidates()
            Y_new:    List[torch.Tensor] = []
            cost_new: List[float]        = []
            fid_new:  List[int]          = []

            for c in candidates:
                if self.config.multi_fidelity:
                    fid = int(c[self.config.fidelity_dim].item())
                else:
                    fid = self.fidelity_mgr.most_accurate.index

                x       = c[: self.bounds.shape[1]]
                y, cost = obj_fn(x, fid)
                Y_new.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
                cost_new.append(float(cost))
                fid_new.append(fid)

            self.add_observations(
                candidates[:, : self.bounds.shape[1]],
                torch.stack(Y_new),
                torch.tensor(fid_new,  dtype=torch.float),
                torch.tensor(cost_new, dtype=torch.float),
            )

            if callback is not None:
                callback(it, self.X, self.Y, candidates)

            logger.info(
                "BO iter %d/%d — n_obs=%d, best_Y=%s",
                it + 1, self.config.n_iterations,
                self.n_observed,
                self.Y.max(0)[0].tolist(),
            )

        pareto_X, pareto_Y = self.get_pareto_front()
        return {
            "X":          self.X,
            "Y":          self.Y,
            "costs":      self.costs,
            "fidelities": self.fidelities,
            "pareto_X":   pareto_X,
            "pareto_Y":   pareto_Y,
        }

    # ── Pareto front ──────────────────────────────────────────────────

    def get_pareto_front(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the non-dominated subset of observed (X, Y)."""
        if self.Y is None or self.Y.numel() == 0:
            return torch.empty(0), torch.empty(0)

        Y = self.Y
        if Y.dim() == 1:
            best_idx = Y.argmax()
            return self.X[best_idx: best_idx + 1], Y[best_idx: best_idx + 1]

        mask = _is_non_dominated(Y)
        return self.X[mask], Y[mask]

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
    Boolean mask of Pareto-optimal rows of Y (maximisation on all
    objectives).  Falls back to O(n²) sweep when BoTorch is absent.
    """
    if _import_botorch():
        try:
            from botorch.utils.multi_objective import is_non_dominated
            return is_non_dominated(Y)
        except Exception:
            pass

    n          = Y.shape[0]
    dominated  = torch.zeros(n, dtype=torch.bool)
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
    ensemble:             nn.Module,
    candidate_graphs:     List[Any],
    candidate_conditions: torch.Tensor,
    budget:               int          = 100,
    conformal:            Optional[Any] = None,
    w_epistemic:          float         = 1.0,
    w_conformal:          float         = 0.5,
    w_selectivity:        float         = 0.3,
) -> List[int]:
    """
    Lightweight active-screening loop that ranks candidate MOFs using
    the TPNO ensemble + conformal calibrator, without a GP surrogate.

    Parameters
    ----------
    ensemble            : Trained TPNOEnsemble.
    candidate_graphs    : List of graph inputs (one per MOF).
    candidate_conditions: [N, P, 4] conditions.
    budget              : Number of MOFs to select.

    Returns
    -------
    Indices into candidate_graphs ranked by score (highest first),
    truncated to budget.
    """
    acq = UncertaintyAcquisition(
        ensemble=ensemble,
        conformal=conformal,
        w_epistemic=w_epistemic,
        w_conformal=w_conformal,
        w_selectivity=w_selectivity,
    )

    scores_list: List[float] = []
    for i, g in enumerate(candidate_graphs):
        if candidate_conditions.dim() == 3:
            cond = candidate_conditions[i: i + 1]   # [1, P, D]
        else:
            cond = candidate_conditions.unsqueeze(0) # [1, P, D]

        s = acq(g, cond)
        # s may be [1] — call mean() to get a scalar safely
        scores_list.append(float(s.mean().item()))

    scores  = np.array(scores_list, dtype=np.float32)
    ranking = np.argsort(-scores).tolist()
    return ranking[:budget]


# ═══════════════════════════════════════════════════════════════════════
# 8.  SIMPLE ACQUISITION WRAPPER FOR ACTIVE LEARNING LOOP
# ═══════════════════════════════════════════════════════════════════════

def score_candidates_simple(
    model:       nn.Module,
    candidates:  Any,
    conditions:  torch.Tensor,
    acquisition: str   = "ucb",
    beta:        float = 1.0,
    return_all:  bool  = False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Simple acquisition function wrapper for the active learning loop.

    FIX: original wrapped the model call in torch.no_grad() which
    prevents TPNO from computing loadings via autograd.grad(omega, mu).
    Removed; model.eval() is sufficient.

    Parameters
    ----------
    model       : ThermodynamicPotentialNO or TPNOEnsemble.
    candidates  : batched graph data.
    conditions  : [B, P, D] thermodynamic conditions.
    acquisition : 'ucb' | 'uncertainty' | 'exploitation' | 'random' | 'entropy'
    beta        : UCB exploration weight.
    return_all  : If True, return dict with all intermediate scores.
    """
    # FIX: NO torch.no_grad() — TPNO uses autograd.grad internally.
    model.eval()

    out    = model(candidates, conditions)
    q_pred = out["q_pred"] if isinstance(out, dict) else out[0]
    sigma  = (
        out.get("sigma", torch.zeros_like(q_pred))
        if isinstance(out, dict)
        else torch.zeros_like(q_pred)
    )

    mean_q     = q_pred.mean(dim=(1, 2))             # [B]
    mean_sigma = sigma.mean(dim=(1, 2))              # [B]
    max_sigma  = sigma.max(dim=1)[0].mean(dim=1)    # [B]

    if acquisition == "uncertainty":
        scores = mean_sigma
    elif acquisition == "exploitation":
        scores = mean_q
    elif acquisition == "random":
        scores = torch.rand_like(mean_q)
    elif acquisition == "entropy":
        scores = entropy_acquisition(model, candidates, conditions)
    else:  # ucb (default)
        scores = mean_q + beta * mean_sigma

    if return_all:
        return {
            "scores":     scores,
            "mean_q":     mean_q,
            "mean_sigma": mean_sigma,
            "max_sigma":  max_sigma,
        }
    return scores


# ═══════════════════════════════════════════════════════════════════════
# 9.  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "BOConfig",
    "FidelityLevel",
    "FidelityManager",
    "upper_confidence_bound",
    "thompson_sampling",
    "pure_uncertainty",
    "entropy_acquisition",
    "random_acquisition",
    "UncertaintyAcquisition",
    "MultiFidelityBO",
    "tpno_screening_loop",
    "score_candidates_simple",
]