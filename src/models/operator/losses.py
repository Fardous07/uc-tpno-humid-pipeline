from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import math

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
    # run_009: threshold on ln(f / 1 bar) = μ/RT (was on raw μ in kJ/mol).
    henry_mu_threshold: float = -2.3   # ln(0.1 bar): dilute region
    use_nll:            bool  = True
    # --- run_006: robust data loss ---------------------------------------
    # Huber ("smooth L1") on the data term: quadratic for |resid| < delta,
    # linear beyond it. Since the data loss is computed in NORMALIZED units
    # (std ≈ 1), delta = 1.0 caps the influence of >1σ outliers — this is
    # the fix for the heavy CO2 error tail (kurtosis ~140, max_abs_err ~1.9).
    # Ignored when use_nll is True and a sigma head is available.
    use_huber:          bool  = True
    huber_delta:        float = 1.0
    # --- run_009 ----------------------------------------------------------
    # Variance head: trained with a Gaussian NLL on a STOP-GRADIENT copy of
    # the prediction, so σ learns the residual scale without pulling the
    # mean away from the robust (Huber) fit. 0 disables.
    lambda_nll:         float = 1.0
    # Penalty on negative predicted loadings (n = c·∂Φ/∂μ can be < 0;
    # physically n ≥ 0). mean(relu(−n)²) in mol/kg.
    lambda_negative:    float = 1.0
    # Weight data points by the GCMC block-average error bars:
    # w ∝ 1 / (σ_z² + floor²), σ_z = error in normalised log space.
    use_error_weights:  bool  = True
    error_weight_floor: float = 0.05


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


def data_loss_huber_weighted(
    q_pred:  torch.Tensor,
    q_true:  torch.Tensor,
    weights: torch.Tensor,
    mask:    Optional[torch.Tensor] = None,
    delta:   float = 1.0,
) -> torch.Tensor:
    """Point-weighted Huber loss: Σ w·huber / Σ w over valid points."""
    per = F.huber_loss(q_pred, q_true, delta=delta, reduction="none")
    if mask is not None:
        m = mask.unsqueeze(-1).expand_as(q_pred)
        per, weights = per[m], weights[m]
    return (weights * per).sum() / weights.sum().clamp(min=1e-12)


def error_bar_weights(
    targets:    torch.Tensor,            # [B, P, C] mol/kg
    target_err: torch.Tensor,            # [B, P, C] mol/kg, NaN where unknown
    q_std:      torch.Tensor,            # [C] std of the transformed target
    log1p:      bool,
    floor:      float,
    mask:       Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Inverse-variance weights from the GCMC error bars, expressed in the
    loss's normalised space: z = (T(n) − m)/s, so σ_z = σ_n · T'(n) / s with
    T'(n) = 1/(1+n) for log1p. w = 1/(σ_z² + floor²); the floor stops points
    with a (near-)zero error bar (e.g. water at 0 % RH) from dominating.
    Unknown errors get the median weight. Weights are scaled to mean 1.
    """
    deriv = 1.0 / (1.0 + targets.clamp(min=0.0)) if log1p else torch.ones_like(targets)
    sigma_z = target_err * deriv / q_std
    w = 1.0 / (sigma_z.pow(2) + floor ** 2)
    valid = torch.isfinite(w)
    if mask is not None:
        valid = valid & mask.unsqueeze(-1).expand_as(w)
    fill = w[valid].median() if valid.any() else torch.tensor(1.0, device=w.device)
    w = torch.where(torch.isfinite(w), w, fill)
    sel = mask.unsqueeze(-1).expand_as(w) if mask is not None else torch.ones_like(w, dtype=torch.bool)
    return w / w[sel].mean().clamp(min=1e-12)


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

    pair_mask: Optional[torch.Tensor] = None
    if mask is not None:
        # Both positions in a pair must be real (not padded)
        pair_mask = (mask[:, 1:] & mask[:, :-1]).float()  # [B, P-1]
        pos_mu = pos_mu * pair_mask.unsqueeze(-1)

    violations = F.relu(-q_diff) * pos_mu        # penalise q↓ when μ↑
    # FIX: violations.mean() used to divide by the FULL padded tensor size
    # (B*(P-1)*n_components) even when a mask zeroed out padded pairs, so
    # the same physical violation rate produced a smaller loss/gradient the
    # more a batch happened to be padded -- making lambda_monotonic
    # effectively batch-composition-dependent instead of a fixed weight.
    # (Currently latent: configs/pipeline.yaml trains at batch_size=1, where
    # there is no padding at all -- but it would have silently under-
    # weighted this term the moment batch_size was increased.) Divide by
    # the actual valid-element count when a mask is given.
    if pair_mask is not None:
        denom = pair_mask.sum() * n_components
        return violations.sum() / denom.clamp(min=1.0)
    return violations.mean()


def henry_law_loss(
    q_pred:        torch.Tensor,
    conditions:    torch.Tensor,
    n_components:  int   = 3,
    mu_threshold:  float = -2.3,
    mask:          Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Henry's law in the dilute limit: n_i = K_H,i(T) · f_i, with the
    fugacity f_i = exp(μ_i / RT) (μ in kJ/mol, f in bar; see
    src.utils.chemistry). K_H,i = n_i / f_i should therefore be constant
    across dilute points of the SAME MOF at the SAME temperature.

    run_009 FIX: the previous version used exp(μ) (i.e. RT = 1 with μ in
    kJ/mol -- wrong units) and pooled points at different temperatures,
    although K_H depends on T. Now: f = exp(μ/RT), dilute = ln f < mu_threshold
    (default ln 0.1 bar) for that component, trace species (f < 1e-8 bar,
    e.g. water at 0 % RH) excluded, and the coefficient of variation of K_H
    is penalised within each (MOF, T) group.

    conditions : [B, P, D] with T in the last column.
    """
    R_KJ = 8.314462618e-3
    mu = conditions[..., :n_components]                     # [B, P, C]
    T  = conditions[..., -1:].clamp(min=1.0)                # [B, P, 1]
    ln_f = mu / (R_KJ * T)                                  # [B, P, C]
    dilute = (ln_f < mu_threshold) & (ln_f > math.log(1e-8))
    if mask is not None:
        dilute = dilute & mask.unsqueeze(-1)
    K = q_pred / torch.exp(ln_f.clamp(min=-40.0, max=5.0))
    terms = []
    Tr = T.squeeze(-1).round()
    for b in range(q_pred.shape[0]):
        for t in torch.unique(Tr[b]):
            sel_t = Tr[b] == t
            for c in range(n_components):
                m = dilute[b, :, c] & sel_t
                if m.sum() < 2:
                    continue
                k = K[b, m, c]
                cv = (k - k.mean()) / (k.mean().abs() + 1e-8)
                terms.append(cv.pow(2).mean())
    if not terms:
        return torch.zeros((), device=q_pred.device)
    return torch.stack(terms).mean()


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
            # FIX: violations.mean() divided by the full B*(P-1) padded
            # size even when pair_mask zeroed out padded pairs, diluting
            # this term's magnitude by the batch's padding ratio (same
            # class of bug as monotonicity_loss above; also currently
            # latent at batch_size=1). Divide by the actual valid-pair
            # count when a mask is given.
            if pair_mask is not None:
                denom = pair_mask.sum().clamp(min=1.0)
                penalty = penalty + violations.sum() / denom
            else:
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
        target_err:  Optional[torch.Tensor] = None,   # [B, P, C] mol/kg, NaN = unknown
    ) -> Dict[str, torch.Tensor]:
        """
        predictions : model output. For an ensemble it contains the
                      per-member stacks all_predictions [M,B,P,C],
                      all_log_var, all_q_aux; each member is trained on its
                      OWN prediction (deep ensemble). A single model's
                      q_pred / log_var / q_aux are treated as M = 1.
        targets     : [B, P, C] simulated loadings (mol/kg)
        mask        : [B, P] bool -- True where data is real, False where padded
        target_err  : GCMC block-average error bars; used for weighting when
                      config.use_error_weights is True.
        """
        c = self.config
        q_mean_pred = predictions["q_pred"]
        n_comp = q_mean_pred.shape[-1]
        if conditions.dim() == 2:
            conditions = conditions.unsqueeze(1)

        base = model.models[0] if hasattr(model, "models") else model
        log1p = getattr(base.config, "target_transform", "identity") == "log1p"

        # Per-member stacks (M = 1 for a single model)
        preds = predictions.get("all_predictions", q_mean_pred.unsqueeze(0))
        logvs = predictions.get("all_log_var")
        if logvs is None and "log_var" in predictions:
            logvs = predictions["log_var"].unsqueeze(0)
        auxs = predictions.get("all_q_aux")
        if auxs is None and "q_aux" in predictions:
            auxs = predictions["q_aux"].unsqueeze(0)

        targets_n = base.normalize_q(targets)

        weights = None
        if c.use_error_weights and target_err is not None:
            weights = error_bar_weights(
                targets, target_err, base.q_std, log1p,
                c.error_weight_floor, mask=mask,
            )

        def fit_term(pred_n: torch.Tensor) -> torch.Tensor:
            if weights is not None:
                return data_loss_huber_weighted(
                    pred_n, targets_n, weights, mask=mask,
                    delta=getattr(c, "huber_delta", 1.0),
                )
            return self._data_term(pred_n, targets_n, mask=mask)

        physics: Dict[str, torch.Tensor] = {}

        # --- Data term, averaged over ensemble members ---
        d_terms = []
        nll_terms = []
        for m in range(preds.shape[0]):
            pred_n = base.normalize_q(preds[m])
            if c.use_nll and logvs is not None:
                # legacy: NLL drives the mean as well
                sig = torch.exp(0.5 * logvs[m])
                d_terms.append(data_loss_nll(pred_n, targets_n, sig, mask=mask))
            else:
                d_terms.append(fit_term(pred_n))
                if c.lambda_nll > 0 and logvs is not None:
                    # σ head: NLL against a stop-gradient copy of the mean
                    sig = torch.exp(0.5 * logvs[m])
                    nll_terms.append(data_loss_nll(pred_n.detach(), targets_n, sig, mask=mask))
        d_loss = torch.stack(d_terms).mean()
        total = c.lambda_data * d_loss

        if nll_terms:
            nll = torch.stack(nll_terms).mean()
            physics["nll_sigma"] = nll
            total = total + c.lambda_nll * nll

        # --- Auxiliary direct-head term (first-order signal to the encoder) ---
        if auxs is not None and getattr(c, "lambda_aux", 0.0) > 0:
            aux_loss = torch.stack(
                [fit_term(base.normalize_q(auxs[m])) for m in range(auxs.shape[0])]
            ).mean()
            physics["aux"] = aux_loss
            total = total + c.lambda_aux * aux_loss

        # --- Negative loadings (n = c·∂Φ/∂μ is not sign-constrained) ---
        if c.lambda_negative > 0:
            neg = F.relu(-preds)
            if mask is not None:
                mm = mask.unsqueeze(0).unsqueeze(-1).expand_as(neg)
                neg_loss = neg[mm].pow(2).mean() if mm.any() else neg.sum() * 0
            else:
                neg_loss = neg.pow(2).mean()
            physics["negative"] = neg_loss
            total = total + c.lambda_negative * neg_loss

        q_pred = q_mean_pred

        # --- Hessian symmetry (exact by construction since run_009; kept for ablations) ---
        if c.lambda_hessian > 0:
            hess  = model.get_hessian(graphs, conditions)
            h_loss = hessian_symmetry_loss(hess)
            physics["hessian"] = h_loss
            total = total + c.lambda_hessian * h_loss

        # The remaining physics losses need at least 2 points per MOF
        if conditions.shape[1] > 1:
            if c.lambda_monotonic > 0:
                m_loss = monotonicity_loss(q_pred, conditions, n_comp, mask=mask)
                physics["monotonic"] = m_loss
                total = total + c.lambda_monotonic * m_loss

            if c.lambda_henry > 0:
                hl = henry_law_loss(q_pred, conditions, n_comp, c.henry_mu_threshold, mask=mask)
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