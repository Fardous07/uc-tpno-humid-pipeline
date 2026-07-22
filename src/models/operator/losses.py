from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LossConfig:
    lambda_data:        float = 1.0
    lambda_hessian:     float = 0.01
    lambda_monotonic:   float = 0.05
    lambda_henry:       float = 0.005
    lambda_competition: float = 0.05
    lambda_gibbs_duhem: float = 0.0
    lambda_aux:         float = 0.3
    henry_mu_threshold: float = -5.0   # μ < threshold → Henry region
    use_nll:            bool  = True
    # --- run_006: robust data loss ---------------------------------------
    # Huber ("smooth L1") on the data term: quadratic for |resid| < delta,
    # linear beyond it. Since the data loss is computed in NORMALIZED units
    # (std ≈ 1), delta = 1.0 caps the influence of >1σ outliers — this is
    # the fix for the heavy CO2 error tail (kurtosis ~140, max_abs_err ~1.9).
    # Ignored when use_nll is True and a sigma head is available.
    use_huber:          bool  = True
    huber_delta:        float = 1.0


# ---------------------------------------------------------------------------
# Data losses
# ---------------------------------------------------------------------------

def data_loss_mse(
    q_pred: torch.Tensor,
    q_true: torch.Tensor,
    mask:   Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    MSE loss with optional boolean mask [B, P] to exclude padded positions.
    """
    if mask is not None:
        m = mask.unsqueeze(-1).expand_as(q_pred)   # [B, P, C]
        return F.mse_loss(q_pred[m], q_true[m])
    return F.mse_loss(q_pred, q_true)


def data_loss_huber(
    q_pred: torch.Tensor,
    q_true: torch.Tensor,
    mask:   Optional[torch.Tensor] = None,
    delta:  float = 1.0,
) -> torch.Tensor:
    """
    Huber (smooth-L1) loss with optional boolean mask [B, P].

    Behaves like MSE for |q_pred - q_true| < delta and like MAE beyond it,
    so a small number of large-error points (strong CO2 binders) can no
    longer dominate the gradient the way squared error does.

    Expects inputs in NORMALIZED units so that delta ≈ 1 corresponds to
    roughly one standard deviation of the target.
    """
    if mask is not None:
        m = mask.unsqueeze(-1).expand_as(q_pred)   # [B, P, C]
        return F.huber_loss(q_pred[m], q_true[m], delta=delta)
    return F.huber_loss(q_pred, q_true, delta=delta)


def data_loss_nll(
    q_pred: torch.Tensor,
    q_true: torch.Tensor,
    sigma:  torch.Tensor,
    mask:   Optional[torch.Tensor] = None,
    eps:    float = 1e-6,
) -> torch.Tensor:
    """
    Heteroscedastic NLL: 0.5*(log σ² + (q−q̂)²/σ²)
    mask [B, P] excludes padded positions.
    """
    if mask is not None:
        m       = mask.unsqueeze(-1).expand_as(q_pred)
        q_pred  = q_pred[m]
        q_true  = q_true[m]
        sigma   = sigma[m]
    var = sigma.pow(2) + eps
    return torch.mean(0.5 * torch.log(var) + 0.5 * (q_pred - q_true).pow(2) / var)


# ---------------------------------------------------------------------------
# Physics losses
# ---------------------------------------------------------------------------

def hessian_symmetry_loss(hessian: torch.Tensor) -> torch.Tensor:
    """
    Enforce Maxwell relations: ∂n_i/∂μ_j = ∂n_j/∂μ_i
    Penalise the antisymmetric part of the Hessian.
    hessian : [B, P, n_comp, n_comp]
    """
    antisym = hessian - hessian.transpose(-1, -2)
    return antisym.pow(2).mean()


def monotonicity_loss(
    q_pred:       torch.Tensor,
    conditions:   torch.Tensor,
    n_components: int = 3,
    mask:         Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Enforce ∂n_i/∂μ_i ≥ 0 (each loading increases with its own chemical potential).

    conditions : [B, P, D]
    mask       : [B, P] bool — only adjacent real pairs are compared.
    """
    mu     = conditions[..., :n_components]       # [B, P, n_comp]
    mu_diff = mu[:, 1:] - mu[:, :-1]             # [B, P-1, n_comp]
    q_diff  = q_pred[:, 1:] - q_pred[:, :-1]     # [B, P-1, n_comp]

    pos_mu = (mu_diff > 0).float()               # where μ increased

    if mask is not None:
        # Both positions in a pair must be real (not padded)
        pair_mask = (mask[:, 1:] & mask[:, :-1]).float()  # [B, P-1]
        pos_mu = pos_mu * pair_mask.unsqueeze(-1)

    violations = F.relu(-q_diff) * pos_mu        # penalise q↓ when μ↑
    return violations.mean()


def henry_law_loss(
    q_pred:        torch.Tensor,
    conditions:    torch.Tensor,
    n_components:  int   = 3,
    mu_threshold:  float = -5.0,
) -> torch.Tensor:
    """
    Enforce Henry's law in the dilute limit.

    Henry's law: q_i = K_H,i · P_i  with P_i = exp(μ_i / RT).
    Setting RT = 1 in our μ units:  q_i = K_H,i · exp(μ_i).
    → K_H,i = q_i / exp(μ_i) should be CONSTANT across the Henry region.

    FIX vs. original: original compared q vs. μ linearly (wrong physics).
    Correct formulation: penalise the coefficient of variation of K_H.

    conditions : [B, P, D]
    """
    mu = conditions[..., :n_components]           # [B, P, n_comp]
    henry_mask = (mu < mu_threshold).all(dim=-1)  # [B, P] bool

    if not henry_mask.any():
        return torch.tensor(0.0, device=q_pred.device)

    q_henry  = q_pred[henry_mask]                 # [N, n_comp]
    mu_henry = mu[henry_mask]                     # [N, n_comp]

    # P_eff = exp(μ) — clamp to avoid overflow for very negative μ
    P_eff = torch.exp(mu_henry.clamp(min=-50.0, max=0.0))   # [N, n_comp]
    K_H   = q_henry / (P_eff + 1e-8)                        # [N, n_comp]

    # Penalise variance of K_H relative to its mean
    # (coefficient of variation → dimensionless, scale-invariant)
    K_H_mean = K_H.mean(dim=0, keepdim=True)                # [1, n_comp]
    cv       = (K_H - K_H_mean) / (K_H_mean.abs() + 1e-8)  # [N, n_comp]
    return cv.pow(2).mean()


def competition_loss(
    q_pred:       torch.Tensor,
    conditions:   torch.Tensor,
    n_components: int = 3,
    mask:         Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Enforce competitive adsorption: when μ_i increases, q_j (j≠i) should
    decrease or stay the same (cross-effect ∂n_j/∂μ_i ≤ 0).

    conditions : [B, P, D]
    mask       : [B, P] bool
    """
    mu     = conditions[..., :n_components]
    mu_diff = mu[:, 1:] - mu[:, :-1]             # [B, P-1, n_comp]
    q_diff  = q_pred[:, 1:] - q_pred[:, :-1]     # [B, P-1, n_comp]

    penalty = torch.tensor(0.0, device=q_pred.device)
    count   = 0

    pair_mask: Optional[torch.Tensor] = None
    if mask is not None:
        pair_mask = (mask[:, 1:] & mask[:, :-1]).float()   # [B, P-1]

    for i in range(n_components):
        for j in range(n_components):
            if i == j:
                continue
            pos_mu_i = (mu_diff[..., i] > 0).float()       # [B, P-1]
            if pair_mask is not None:
                pos_mu_i = pos_mu_i * pair_mask
            # q_j should not increase when μ_i increases
            violations = F.relu(q_diff[..., j]) * pos_mu_i
            penalty = penalty + violations.mean()
            count  += 1

    return penalty / max(count, 1)


def gibbs_duhem_loss(
    q_pred:       torch.Tensor,
    conditions:   torch.Tensor,
    n_components: int = 3,
    mask:         Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Gibbs-Duhem consistency: ∫ Σ n_i dμ_i ≥ 0 along any thermodynamic path
    (follows from Ω being convex).

    conditions : [B, P, D]
    mask       : [B, P] bool
    """
    mu    = conditions[..., :n_components]
    dmu   = mu[:, 1:] - mu[:, :-1]               # [B, P-1, n_comp]
    q_mid = 0.5 * (q_pred[:, 1:] + q_pred[:, :-1])

    integrand = (q_mid * dmu).sum(dim=-1)         # [B, P-1]

    if mask is not None:
        pair_mask = (mask[:, 1:] & mask[:, :-1]).float()
        integrand = integrand * pair_mask

    integral = integrand.sum(dim=-1)              # [B]
    return F.relu(-integral).mean()


# ---------------------------------------------------------------------------
# Combined thermodynamic loss
# ---------------------------------------------------------------------------

class ThermodynamicLoss(nn.Module):
    """
    Combined data + physics loss for UC-TPNO training.

    FIX: forward() now accepts a mask [B, P] to exclude padded pressure
    points from all loss computations.  Without this, the collate function's
    zero-padding at μ=0,T=0 was silently included in every loss term.

    run_006: the data term (and the auxiliary-head term) use Huber loss by
    default (config.use_huber) so a few high-error CO2 points can no longer
    dominate training. Set config.use_huber = False to fall back to MSE.
    """

    def __init__(
        self,
        config: Optional[LossConfig] = None,
        *,
        lambda_hessian:     Optional[float] = None,
        lambda_monotonic:   Optional[float] = None,
        lambda_henry:       Optional[float] = None,
        lambda_competition: Optional[float] = None,
        lambda_gibbs_duhem: Optional[float] = None,
    ):
        super().__init__()
        self.config = config or LossConfig()
        # Allow individual overrides
        if lambda_hessian     is not None: self.config.lambda_hessian     = lambda_hessian
        if lambda_monotonic   is not None: self.config.lambda_monotonic   = lambda_monotonic
        if lambda_henry       is not None: self.config.lambda_henry       = lambda_henry
        if lambda_competition is not None: self.config.lambda_competition = lambda_competition
        if lambda_gibbs_duhem is not None: self.config.lambda_gibbs_duhem = lambda_gibbs_duhem

    def _data_term(
        self,
        q_pred_n:  torch.Tensor,
        targets_n: torch.Tensor,
        mask:      Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Robust (Huber) or plain (MSE) data term, in normalized units."""
        c = self.config
        if getattr(c, "use_huber", True):
            return data_loss_huber(
                q_pred_n, targets_n, mask=mask,
                delta=getattr(c, "huber_delta", 1.0),
            )
        return data_loss_mse(q_pred_n, targets_n, mask=mask)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets:     torch.Tensor,
        model:       nn.Module,
        graphs:      Any,
        conditions:  torch.Tensor,
        mask:        Optional[torch.Tensor] = None,   # [B, P] bool
    ) -> Dict[str, torch.Tensor]:
        """
        predictions : dict with keys q_pred [B,P,C], optionally sigma [B,P,C]
        targets     : [B, P, C]  (may contain zeros at padded positions)
        mask        : [B, P] bool — True where data is real, False where padded
        """
        c = self.config

        q_pred = predictions["q_pred"]
        n_comp = q_pred.shape[-1]

        if conditions.dim() == 2:
            conditions = conditions.unsqueeze(1)

        # --- Data loss in NORMALIZED (transformed) units ---
        # run_007: use the model's normalize_q so the loss automatically
        # respects config.target_transform (e.g. log1p) and stays consistent
        # with denormalize_q. For target_transform="identity" this reduces to
        # the previous linear (q - q_mean) / q_std standardization.
        base = model.models[0] if hasattr(model, "models") else model
        q_pred_n  = base.normalize_q(q_pred)
        targets_n = base.normalize_q(targets)

        sigma = predictions.get("sigma")
        if c.use_nll and sigma is not None:
            # Heteroscedastic NLL takes precedence when a sigma head is active.
            d_loss = data_loss_nll(q_pred_n, targets_n, sigma, mask=mask)
        else:
            # run_006: robust Huber (or MSE) data term.
            d_loss = self._data_term(q_pred_n, targets_n, mask=mask)

        physics: Dict[str, torch.Tensor] = {}
        total = c.lambda_data * d_loss

        # Auxiliary direct-head loss (trains the encoder with a
        # first-order signal; the ICNN remains the physics predictor)
        q_aux = predictions.get("q_aux")
        if q_aux is not None and getattr(c, "lambda_aux", 0.0) > 0:
            q_aux_n = base.normalize_q(q_aux)
            aux_loss = self._data_term(q_aux_n, targets_n, mask=mask)
            physics["aux"] = aux_loss
            total = total + c.lambda_aux * aux_loss

        # --- Hessian symmetry (Maxwell relations) ---
        if c.lambda_hessian > 0:
            hess  = model.get_hessian(graphs, conditions)
            h_loss = hessian_symmetry_loss(hess)
            physics["hessian"] = h_loss
            total = total + c.lambda_hessian * h_loss

        # The remaining physics losses need at least 2 pressure points
        if conditions.shape[1] > 1:

            if c.lambda_monotonic > 0:
                m_loss = monotonicity_loss(q_pred, conditions, n_comp, mask=mask)
                physics["monotonic"] = m_loss
                total = total + c.lambda_monotonic * m_loss

            if c.lambda_henry > 0:
                hl = henry_law_loss(q_pred, conditions, n_comp, c.henry_mu_threshold)
                physics["henry"] = hl
                total = total + c.lambda_henry * hl

            if c.lambda_competition > 0:
                comp = competition_loss(q_pred, conditions, n_comp, mask=mask)
                physics["competition"] = comp
                total = total + c.lambda_competition * comp

            if c.lambda_gibbs_duhem > 0:
                gd = gibbs_duhem_loss(q_pred, conditions, n_comp, mask=mask)
                physics["gibbs_duhem"] = gd
                total = total + c.lambda_gibbs_duhem * gd

        return {"total": total, "data": d_loss, "physics": physics}


# ---------------------------------------------------------------------------
# Adaptive loss weighting
# ---------------------------------------------------------------------------

class AdaptiveLossWeighting:
    """
    Automatically rescale physics loss weights so that no single term
    dominates.  Uses an exponential moving average of recent loss magnitudes.

    FIX: denominator now uses len(mean_losses) (only keys that have data)
    instead of len(self.weights) (all keys), which was wrong when some
    physics losses are disabled or not yet logged.
    """

    def __init__(
        self,
        initial_weights: Dict[str, float],
        smoothing:    float = 0.9,
        update_freq:  int   = 10,
        window:       int   = 100,
    ):
        self.weights       = initial_weights.copy()
        self.smoothing     = smoothing
        self.update_freq   = update_freq
        self.window        = window
        self.running_losses: Dict[str, List[float]] = {k: [] for k in initial_weights}
        self._step = 0

    def update_weights(
        self, losses: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        self._step += 1

        for k, v in losses.items():
            if k not in self.running_losses:
                self.running_losses[k] = []
            val = v.detach().item() if isinstance(v, torch.Tensor) else float(v)
            self.running_losses[k].append(val)
            if len(self.running_losses[k]) > self.window:
                self.running_losses[k].pop(0)

        if self._step % self.update_freq == 0:
            mean_losses = {
                k: float(np.mean(v))
                for k, v in self.running_losses.items()
                if v and float(np.mean(v)) > 0
            }
            if mean_losses:
                total = sum(mean_losses.values())
                n     = len(mean_losses)         # FIX: only counted keys
                for k in self.weights:
                    if k in mean_losses and total > 0:
                        new_w = total / (n * mean_losses[k])
                        self.weights[k] = (
                            self.smoothing * self.weights[k]
                            + (1.0 - self.smoothing) * new_w
                        )

        return self.weights.copy()


# ---------------------------------------------------------------------------
# Physics loss warm-up scheduler
# ---------------------------------------------------------------------------

class PhysicsLossScheduler:
    """
    Linearly ramp physics loss weights from 0 → target over warmup_epochs.

    Usage:
        scheduler = PhysicsLossScheduler(loss_fn, warmup_epochs=30)
        for epoch in range(n_epochs):
            scheduler.step(epoch)   # call BEFORE the training loop body
            ...
    """

    def __init__(self, loss_module: ThermodynamicLoss, warmup_epochs: int = 30):
        self.loss    = loss_module
        self.warmup  = warmup_epochs
        c = loss_module.config
        # Capture intended final values at construction time
        self._targets: Dict[str, float] = {
            "lambda_hessian":     c.lambda_hessian,
            "lambda_monotonic":   c.lambda_monotonic,
            "lambda_henry":       c.lambda_henry,
            "lambda_competition": c.lambda_competition,
            "lambda_gibbs_duhem": c.lambda_gibbs_duhem,
        }

    def step(self, epoch: int) -> None:
        alpha = min(1.0, epoch / max(self.warmup, 1))
        for attr, target in self._targets.items():
            setattr(self.loss.config, attr, target * alpha)

    def get_current_weights(self) -> Dict[str, float]:
        c = self.loss.config
        return {
            "lambda_hessian":     c.lambda_hessian,
            "lambda_monotonic":   c.lambda_monotonic,
            "lambda_henry":       c.lambda_henry,
            "lambda_competition": c.lambda_competition,
            "lambda_gibbs_duhem": c.lambda_gibbs_duhem,
        }


# ---------------------------------------------------------------------------
# Thermodynamic validator  (run every N epochs, no_grad)
# ---------------------------------------------------------------------------

class ThermodynamicValidator:
    """
    Post-hoc validation of thermodynamic consistency.

    NOTE: methods do NOT use @torch.no_grad().  TPNO derives loadings via
    autograd.grad(omega, mu) internally; torch.no_grad() would silently
    ignore requires_grad_(True) on mu and crash that call.
    Call model.eval() before using this validator — that is sufficient
    to disable dropout / batchnorm tracking without breaking autograd.
    """

    def __init__(self, n_test_points: int = 100):
        self.n_test_points = n_test_points

    def check_convexity(
        self,
        model:      nn.Module,
        graphs:     Any,
        conditions: torch.Tensor,
        n_pairs:    int = 200,
    ) -> Dict[str, float]:
        """
        Jensen's inequality check: Ω(λc1 + (1-λ)c2) ≤ λΩ(c1) + (1-λ)Ω(c2)
        """
        B, P = conditions.shape[:2]
        D    = conditions.shape[-1]
        device = conditions.device

        out   = model(graphs, conditions, return_potential=True, return_uncertainty=False)
        omega = out["omega"]                          # [B, P, 1]

        idx1 = torch.randint(0, P, (B, n_pairs), device=device)
        idx2 = torch.randint(0, P, (B, n_pairs), device=device)
        lam  = torch.rand(B, n_pairs, 1, device=device)

        c1 = conditions.gather(1, idx1.unsqueeze(-1).expand(-1, -1, D))
        c2 = conditions.gather(1, idx2.unsqueeze(-1).expand(-1, -1, D))
        o1 = omega.gather(1, idx1.unsqueeze(-1))
        o2 = omega.gather(1, idx2.unsqueeze(-1))

        c_interp = lam * c1 + (1 - lam) * c2
        out_interp = model(graphs, c_interp,
                           return_potential=True, return_uncertainty=False)
        o_interp = out_interp["omega"]

        bound      = lam * o1 + (1 - lam) * o2
        violations = F.relu(o_interp - bound - 1e-4).squeeze(-1)

        return {
            "convexity_violation_rate":  (violations > 0).float().mean().item(),
            "mean_convexity_violation":  violations.mean().item(),
            "max_convexity_violation":   violations.max().item(),
        }

    def check_monotonicity(
        self,
        model:      nn.Module,
        graphs:     Any,
        conditions: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Check ∂n_i/∂μ_i ≥ 0 for each component.
        """
        out = model(graphs, conditions, return_uncertainty=False)
        q   = out["q_pred"]
        mu  = conditions[..., : q.shape[-1]]

        mu_diff = mu[:, 1:] - mu[:, :-1]
        q_diff  = q[:, 1:]  - q[:, :-1]
        mask    = (mu_diff > 0).float()
        violations = (q_diff < -1e-6).float() * mask

        return {
            "monotonicity_violation_rate":       violations.mean().item(),
            "monotonicity_by_component":         violations.mean(dim=(0, 1)).cpu().tolist(),
        }

    def check_henry_region(
        self,
        model:          nn.Module,
        graphs:         Any,
        low_pressure:   float = 1e-3,
        high_pressure:  float = 1e-2,
        T:              float = 313.0,
        n_components:   int   = 3,
    ) -> Dict[str, float]:
        """
        Verify Henry's law: q(P_high)/q(P_low) ≈ P_high/P_low.
        Only checks CO2 and N2 (H2O at very low μ is numerically unstable).
        """
        device = next(model.parameters()).device
        mu_lo  = float(torch.tensor(low_pressure).log())
        mu_hi  = float(torch.tensor(high_pressure).log())

        cond = torch.zeros(2, 4, device=device)
        cond[0, :n_components] = mu_lo
        cond[1, :n_components] = mu_hi
        cond[:, -1] = T
        # Set H2O very low to avoid numerical instability in ratio check
        if n_components >= 3:
            cond[:, 2] = -50.0
        cond = cond.unsqueeze(0)                    # [1, 2, 4]

        out = model(graphs, cond, return_uncertainty=False)
        q   = out["q_pred"]                         # [1, 2, n_comp]

        expected = high_pressure / low_pressure
        # Only check CO2 (idx 0) and N2 (idx 1) — both physically meaningful
        ratio = q[0, 1, :2] / (q[0, 0, :2] + 1e-8)   # [2]
        error = (ratio - expected).abs() / expected

        return {
            "henry_mean_error_co2_n2": error.mean().item(),
            "henry_co2_ratio":         ratio[0].item(),
            "henry_n2_ratio":          ratio[1].item(),
            "expected_ratio":          expected,
        }

    def check_maxwell_relations(
        self,
        model:      nn.Module,
        graphs:     Any,
        conditions: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Check Hessian symmetry ∂n_i/∂μ_j ≈ ∂n_j/∂μ_i.
        Requires model to support return_hessian=True.
        """
        out  = model(graphs, conditions,
                     return_hessian=True, return_uncertainty=False)
        hess = out["hessian"]                        # [B, P, C, C]
        asym = (hess - hess.transpose(-1, -2)).abs()
        return {
            "maxwell_mean_asymmetry": asym.mean().item(),
            "maxwell_max_asymmetry":  asym.max().item(),
        }


# ---------------------------------------------------------------------------

__all__ = [
    "LossConfig",
    "data_loss_mse",
    "data_loss_huber",
    "data_loss_nll",
    "hessian_symmetry_loss",
    "monotonicity_loss",
    "henry_law_loss",
    "competition_loss",
    "gibbs_duhem_loss",
    "ThermodynamicLoss",
    "AdaptiveLossWeighting",
    "PhysicsLossScheduler",
    "ThermodynamicValidator",
]