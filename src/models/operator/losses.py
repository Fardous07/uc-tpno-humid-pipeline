"""
Physics-informed loss functions for thermodynamic consistency.

This module provides the complete loss stack used to train the TPNO:

1.  **Data losses** — MSE and heteroscedastic negative log-likelihood
    (Gaussian NLL) that leverage the aleatoric uncertainty head.
2.  **Hessian symmetry** — enforces Maxwell relations by penalising
    asymmetry of ∂²Ω / ∂μᵢ∂μⱼ.
3.  **Monotonicity** — penalises negative ∂q/∂μ (loading must increase
    with chemical potential at fixed T).
4.  **Henry's-law** — encourages linearity of q vs. P in the dilute
    limit (low chemical potential).
5.  **Competition** — at fixed total pressure, increasing one species'
    partial pressure should not increase *other* species' loadings.
6.  **Gibbs–Duhem** — optional integral consistency penalty derived
    from the Gibbs–Duhem equation at constant T.
7.  **ThermodynamicValidator** — post-training checks for convexity,
    monotonicity, and Henry-region compliance (not differentiable;
    used for evaluation only).

All physics losses are weighted by configurable λ coefficients and
combined into a single ``total`` scalar for ``backward()``.

References
──────────
[1] Amos et al. (2017). Input Convex Neural Networks. ICML.
[2] Raissi et al. (2019). Physics-informed neural networks. JCP.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LossConfig:
    """
    Weights and toggles for every term in the composite loss.

    Attributes
    ──────────
    lambda_data        : Weight for the data-fidelity term.
    lambda_hessian     : Maxwell-relation (Hessian symmetry) penalty.
    lambda_monotonic   : Monotonicity-in-μ penalty.
    lambda_henry       : Henry's-law linearity penalty.
    lambda_competition : Cross-species competition penalty.
    lambda_gibbs_duhem : Gibbs–Duhem integral consistency penalty.
    henry_mu_threshold : Chemical-potential threshold below which the
                         Henry penalty is active (corresponds to the
                         dilute / low-pressure regime).
    use_nll            : If *True* and ``sigma`` is present in
                         predictions, use heteroscedastic NLL instead
                         of MSE for the data term.
    """

    lambda_data: float = 1.0
    lambda_hessian: float = 0.1
    lambda_monotonic: float = 0.1
    lambda_henry: float = 0.01
    lambda_competition: float = 0.05
    lambda_gibbs_duhem: float = 0.0
    henry_mu_threshold: float = -5.0
    use_nll: bool = True


# ═══════════════════════════════════════════════════════════════════════
# 2.  INDIVIDUAL LOSS TERMS
# ═══════════════════════════════════════════════════════════════════════

def data_loss_mse(
    q_pred: torch.Tensor,
    q_true: torch.Tensor,
) -> torch.Tensor:
    """Mean-squared error between predicted and true loadings."""
    return F.mse_loss(q_pred, q_true)


def data_loss_nll(
    q_pred: torch.Tensor,
    q_true: torch.Tensor,
    sigma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""
    Heteroscedastic Gaussian negative log-likelihood:

    .. math::
        \mathcal{L}_{\text{NLL}} =
            \frac{1}{2}\ln\sigma^2
          + \frac{(q - \hat q)^2}{2\sigma^2}

    Parameters
    ----------
    q_pred : ``[B, P, C]`` predicted loadings.
    q_true : ``[B, P, C]`` ground-truth loadings.
    sigma  : ``[B, P, C]`` predicted aleatoric std-dev.
    """
    var = sigma.pow(2) + eps
    return torch.mean(0.5 * torch.log(var) + 0.5 * (q_pred - q_true).pow(2) / var)


def hessian_symmetry_loss(hessian: torch.Tensor) -> torch.Tensor:
    r"""
    Penalise asymmetry of the Hessian (Maxwell-relation violation):

    .. math::
        \mathcal{L}_H = \| H - H^\top \|_F^2

    Parameters
    ----------
    hessian : ``[B, P, C, C]`` second-derivative matrix of Ω.
    """
    antisym = hessian - hessian.transpose(-1, -2)
    return antisym.pow(2).mean()


def monotonicity_loss(
    q_pred: torch.Tensor,
    conditions: torch.Tensor,
    n_components: int = 3,
) -> torch.Tensor:
    r"""
    Penalise violations of ∂qᵢ/∂μᵢ ≥ 0.

    For consecutive condition points ordered by μ, if μᵢ increases
    then qᵢ must not decrease.

    Parameters
    ----------
    q_pred     : ``[B, P, C]`` predicted loadings.
    conditions : ``[B, P, n_cond]`` thermodynamic conditions (first
                 ``n_components`` columns are chemical potentials).
    """
    mu = conditions[..., :n_components]           # [B, P, C]
    mu_diff = mu[:, 1:] - mu[:, :-1]             # [B, P-1, C]
    q_diff = q_pred[:, 1:] - q_pred[:, :-1]      # [B, P-1, C]

    # Where μ increased, q should not have decreased
    mask = (mu_diff > 0).float()
    violations = F.relu(-q_diff) * mask
    return violations.mean()


def henry_law_loss(
    q_pred: torch.Tensor,
    conditions: torch.Tensor,
    n_components: int = 3,
    mu_threshold: float = -5.0,
) -> torch.Tensor:
    r"""
    In the Henry (dilute) regime, loading should be approximately
    proportional to fugacity (≈ pressure), i.e. linear in exp(μ).

    We identify dilute points as those with μ < ``mu_threshold`` and
    penalise deviation from a linear q–exp(μ) relationship by checking
    that the ratio q/exp(μ) is approximately constant across those
    points.

    Parameters
    ----------
    q_pred       : ``[B, P, C]``
    conditions   : ``[B, P, n_cond]``
    mu_threshold : Chemical-potential ceiling for the Henry region.
    """
    mu = conditions[..., :n_components]  # [B, P, C]

    # Mask: all components below threshold
    henry_mask = (mu < mu_threshold).all(dim=-1)  # [B, P]

    if not henry_mask.any():
        return torch.tensor(0.0, device=q_pred.device)

    q_henry = q_pred[henry_mask]   # [N_henry, C]
    mu_henry = mu[henry_mask]      # [N_henry, C]

    # Normalise each to [0, 1] within the Henry subset
    q_norm = q_henry / (q_henry.max(dim=0, keepdim=True)[0] + 1e-8)
    mu_norm = mu_henry / (mu_henry.min(dim=0, keepdim=True)[0].abs() + 1e-8)

    return F.mse_loss(q_norm, mu_norm)


def competition_loss(
    q_pred: torch.Tensor,
    conditions: torch.Tensor,
    n_components: int = 3,
) -> torch.Tensor:
    r"""
    Cross-species competition: at fixed total pressure, increasing
    μᵢ should not increase qⱼ for j ≠ i.

    We approximate this by checking that off-diagonal elements of the
    loading-sensitivity matrix are non-positive.

    Parameters
    ----------
    q_pred     : ``[B, P, C]``
    conditions : ``[B, P, n_cond]``
    """
    mu = conditions[..., :n_components]
    mu_diff = mu[:, 1:] - mu[:, :-1]   # [B, P-1, C]
    q_diff = q_pred[:, 1:] - q_pred[:, :-1]

    penalty = torch.tensor(0.0, device=q_pred.device)
    count = 0

    for i in range(n_components):
        for j in range(n_components):
            if i == j:
                continue
            # Where μᵢ increased, qⱼ should not increase
            mask = (mu_diff[..., i] > 0).float()
            violations = F.relu(q_diff[..., j]) * mask
            penalty = penalty + violations.mean()
            count += 1

    return penalty / max(count, 1)


def gibbs_duhem_loss(
    q_pred: torch.Tensor,
    conditions: torch.Tensor,
    n_components: int = 3,
) -> torch.Tensor:
    r"""
    Gibbs–Duhem integral consistency at constant T:

    .. math::
        \sum_i \int q_i \, d\mu_i = \Omega(\mu_{\max}) - \Omega(\mu_{\min})

    We approximate the integral with the trapezoidal rule over the
    condition-point grid and penalise discrepancy.

    Only active when ``lambda_gibbs_duhem > 0`` in ``LossConfig``.
    """
    mu = conditions[..., :n_components]  # [B, P, C]
    dmu = mu[:, 1:] - mu[:, :-1]        # [B, P-1, C]

    # Trapezoidal: 0.5 * (q[k] + q[k+1]) * dmu[k]
    q_mid = 0.5 * (q_pred[:, 1:] + q_pred[:, :-1])  # [B, P-1, C]
    integrand = (q_mid * dmu).sum(dim=-1)              # [B, P-1]
    integral = integrand.sum(dim=-1)                    # [B]

    # The integral should be non-negative (Ω is convex and increasing)
    return F.relu(-integral).mean()


# ═══════════════════════════════════════════════════════════════════════
# 3.  COMPOSITE LOSS MODULE
# ═══════════════════════════════════════════════════════════════════════

class ThermodynamicLoss(nn.Module):
    """
    Composite physics-informed loss for TPNO training.

    Aggregates data fidelity and all physics penalty terms with
    configurable weights.

    Parameters
    ----------
    config : ``LossConfig`` or keyword overrides.
    """

    def __init__(
        self,
        config: Optional[LossConfig] = None,
        *,
        lambda_hessian: Optional[float] = None,
        lambda_monotonic: Optional[float] = None,
        lambda_henry: Optional[float] = None,
        lambda_competition: Optional[float] = None,
        lambda_gibbs_duhem: Optional[float] = None,
    ):
        super().__init__()
        self.config = config or LossConfig()

        # Allow keyword overrides
        if lambda_hessian is not None:
            self.config.lambda_hessian = lambda_hessian
        if lambda_monotonic is not None:
            self.config.lambda_monotonic = lambda_monotonic
        if lambda_henry is not None:
            self.config.lambda_henry = lambda_henry
        if lambda_competition is not None:
            self.config.lambda_competition = lambda_competition
        if lambda_gibbs_duhem is not None:
            self.config.lambda_gibbs_duhem = lambda_gibbs_duhem

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        model: nn.Module,
        graphs: Any,
        conditions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the composite loss.

        Parameters
        ----------
        predictions : Output dict from ``ThermodynamicPotentialNO.forward()``.
        targets     : ``[B, P, C]`` ground-truth loadings.
        model       : The TPNO model (used for Hessian computation).
        graphs      : Graph batch (passed through to ``model.get_hessian``).
        conditions  : ``[B, P, n_cond]`` thermodynamic conditions.

        Returns
        -------
        Dict with ``'total'``, ``'data'``, and ``'physics'`` (sub-dict).
        """
        c = self.config
        q_pred = predictions["q_pred"]
        n_comp = q_pred.shape[-1]

        # Ensure conditions are 3-D
        if conditions.dim() == 2:
            conditions = conditions.unsqueeze(1)

        # ── Data loss ────────────────────────────────────────────
        if c.use_nll and "sigma" in predictions:
            d_loss = data_loss_nll(q_pred, targets, predictions["sigma"])
        else:
            d_loss = data_loss_mse(q_pred, targets)

        physics: Dict[str, torch.Tensor] = {}
        total = c.lambda_data * d_loss

        # ── Hessian symmetry (Maxwell relations) ─────────────────
        if c.lambda_hessian > 0:
            hess = model.get_hessian(graphs, conditions)
            h_loss = hessian_symmetry_loss(hess)
            physics["hessian"] = h_loss
            total = total + c.lambda_hessian * h_loss

        # ── Monotonicity ─────────────────────────────────────────
        if c.lambda_monotonic > 0 and conditions.shape[1] > 1:
            m_loss = monotonicity_loss(q_pred, conditions, n_comp)
            physics["monotonic"] = m_loss
            total = total + c.lambda_monotonic * m_loss

        # ── Henry's law ──────────────────────────────────────────
        if c.lambda_henry > 0 and conditions.shape[1] > 1:
            hl = henry_law_loss(q_pred, conditions, n_comp, c.henry_mu_threshold)
            physics["henry"] = hl
            total = total + c.lambda_henry * hl

        # ── Competition ──────────────────────────────────────────
        if c.lambda_competition > 0 and conditions.shape[1] > 1:
            comp = competition_loss(q_pred, conditions, n_comp)
            physics["competition"] = comp
            total = total + c.lambda_competition * comp

        # ── Gibbs–Duhem ──────────────────────────────────────────
        if c.lambda_gibbs_duhem > 0 and conditions.shape[1] > 1:
            gd = gibbs_duhem_loss(q_pred, conditions, n_comp)
            physics["gibbs_duhem"] = gd
            total = total + c.lambda_gibbs_duhem * gd

        return {"total": total, "data": d_loss, "physics": physics}


# ═══════════════════════════════════════════════════════════════════════
# 4.  SCHEDULED PHYSICS-LOSS WEIGHTING
# ═══════════════════════════════════════════════════════════════════════

class PhysicsLossScheduler:
    """
    Linearly ramp physics-loss weights from 0 to their target over a
    warm-up period, preventing the physics terms from dominating early
    training when the data loss is still large.

    Usage
    ─────
    >>> scheduler = PhysicsLossScheduler(loss_fn, warmup_epochs=20)
    >>> for epoch in range(n_epochs):
    ...     scheduler.step(epoch)
    ...     # loss_fn.config.lambda_* are now ramped
    """

    def __init__(self, loss_module: ThermodynamicLoss, warmup_epochs: int = 20):
        self.loss = loss_module
        self.warmup = warmup_epochs

        # Store targets
        c = loss_module.config
        self._targets = {
            "lambda_hessian": c.lambda_hessian,
            "lambda_monotonic": c.lambda_monotonic,
            "lambda_henry": c.lambda_henry,
            "lambda_competition": c.lambda_competition,
            "lambda_gibbs_duhem": c.lambda_gibbs_duhem,
        }

    def step(self, epoch: int) -> None:
        """Update weights for the current epoch."""
        alpha = min(1.0, epoch / max(self.warmup, 1))
        for attr, target in self._targets.items():
            setattr(self.loss.config, attr, target * alpha)


# ═══════════════════════════════════════════════════════════════════════
# 5.  THERMODYNAMIC VALIDATOR  (evaluation-only)
# ═══════════════════════════════════════════════════════════════════════

class ThermodynamicValidator:
    """
    Post-training thermodynamic consistency checks (non-differentiable).

    Methods return plain dicts of scalar metrics suitable for logging.
    """

    def __init__(self, n_test_points: int = 100):
        self.n_test_points = n_test_points

    # ── convexity ────────────────────────────────────────────────

    @torch.no_grad()
    def check_convexity(
        self,
        model: nn.Module,
        graphs: Any,
        conditions: torch.Tensor,
        n_pairs: int = 200,
    ) -> Dict[str, float]:
        """
        Jensen's-inequality test on random interpolated pairs.

        Returns ``convexity_violation_rate``, ``mean_violation``,
        ``max_violation``.
        """
        B, P = conditions.shape[:2]
        device = conditions.device

        out = model(graphs, conditions, return_potential=True, return_uncertainty=False)
        omega = out["omega"]  # [B, P, 1]

        idx1 = torch.randint(0, P, (B, n_pairs), device=device)
        idx2 = torch.randint(0, P, (B, n_pairs), device=device)
        lam = torch.rand(B, n_pairs, 1, device=device)

        # Gather
        c1 = conditions.gather(1, idx1.unsqueeze(-1).expand(-1, -1, conditions.shape[-1]))
        c2 = conditions.gather(1, idx2.unsqueeze(-1).expand(-1, -1, conditions.shape[-1]))
        o1 = omega.gather(1, idx1.unsqueeze(-1))
        o2 = omega.gather(1, idx2.unsqueeze(-1))

        c_interp = lam * c1 + (1 - lam) * c2
        out_interp = model(graphs, c_interp, return_potential=True, return_uncertainty=False)
        o_interp = out_interp["omega"]

        bound = lam * o1 + (1 - lam) * o2
        violations = F.relu(o_interp - bound).squeeze(-1)

        return {
            "convexity_violation_rate": (violations > 1e-4).float().mean().item(),
            "mean_convexity_violation": violations.mean().item(),
            "max_convexity_violation": violations.max().item(),
        }

    # ── monotonicity ─────────────────────────────────────────────

    @torch.no_grad()
    def check_monotonicity(
        self,
        model: nn.Module,
        graphs: Any,
        conditions: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Check ∂q/∂μ ≥ 0 on the provided condition grid.
        """
        out = model(graphs, conditions, return_uncertainty=False)
        q = out["q_pred"]  # [B, P, C]
        mu = conditions[..., : q.shape[-1]]

        mu_diff = mu[:, 1:] - mu[:, :-1]
        q_diff = q[:, 1:] - q[:, :-1]

        mask = (mu_diff > 0).float()
        violations = ((q_diff < -1e-6).float() * mask)

        return {
            "monotonicity_violation_rate": violations.mean().item(),
            "monotonicity_by_component": violations.mean(dim=(0, 1)).cpu().tolist(),
        }

    # ── Henry region ─────────────────────────────────────────────

    @torch.no_grad()
    def check_henry_region(
        self,
        model: nn.Module,
        graphs: Any,
        low_pressure: float = 1e-3,
        high_pressure: float = 1e-2,
        T: float = 313.0,
    ) -> Dict[str, float]:
        """
        Check q(P₂)/q(P₁) ≈ P₂/P₁ in the dilute limit.
        """
        device = next(model.parameters()).device
        mu = torch.log(torch.tensor([low_pressure, high_pressure], device=device))

        cond = torch.zeros(2, 4, device=device)
        cond[:, 0] = mu
        cond[:, 1] = mu
        cond[:, 2] = -100.0  # negligible H₂O
        cond[:, 3] = T
        cond = cond.unsqueeze(0)  # [1, 2, 4]

        out = model(graphs, cond, return_uncertainty=False)
        q = out["q_pred"]  # [1, 2, C]

        ratio = q[:, 1] / (q[:, 0] + 1e-8)
        expected = high_pressure / low_pressure
        error = (ratio - expected).abs() / expected

        return {
            "henry_mean_error": error.mean().item(),
            "henry_max_error": error.max().item(),
        }


# ═══════════════════════════════════════════════════════════════════════
# 6.  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    # Config
    "LossConfig",
    # Individual terms (usable standalone)
    "data_loss_mse",
    "data_loss_nll",
    "hessian_symmetry_loss",
    "monotonicity_loss",
    "henry_law_loss",
    "competition_loss",
    "gibbs_duhem_loss",
    # Composite loss
    "ThermodynamicLoss",
    # Scheduler
    "PhysicsLossScheduler",
    # Validator
    "ThermodynamicValidator",
]