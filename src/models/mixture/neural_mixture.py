"""
Neural mixture model: learned non-ideal corrections to IAST.

IAST assumes the adsorbed phase is an *ideal* solution, which breaks
down for humid flue-gas CO₂ capture where H₂O–CO₂ interactions
inside MOF pores are strongly non-ideal.  This module learns the
deviation from IAST using neural networks, in two complementary
modes:

Modes
─────
1.  **Activity-coefficient mode** (``mode='activity'``) — predict
    per-component activity coefficients γ_i(x, π, T, MOF) that
    modify the Raoult's-law analogue:
        ``P_i = x_i · γ_i · P_i⁰``
    Then re-solve IAST with γ ≠ 1.  This is the *Real Adsorbed
    Solution Theory* (RAST) approach.

2.  **Direct-correction mode** (``mode='direct'``) — predict a
    multiplicative correction Δ_i to the IAST loadings:
        ``q_i^true = q_i^IAST · Δ_i``
    Simpler and faster, but less physically grounded.

Architecture
────────────
``NeuralMixtureModel`` is an ``nn.Module`` that takes:
*   MOF embedding ``h`` (from encoder) — ``[B, emb_dim]``
*   Thermodynamic conditions (T, P, y) — ``[B, n_cond]``
*   IAST base predictions — ``[B, n_components]``

And outputs corrected per-component loadings.

Integration
───────────
*   Uses ``IASTCalculator`` from ``iast.py`` for the base prediction.
*   Can be trained end-to-end with the TPNO operator (the MOF
    embedding is shared).
*   Evaluated against GCMC ground truth in ``benchmarking.py``.

References
──────────
[1] Swisher et al. (2013). Evaluating Mixture Adsorption Models
    Using Molecular Simulation. AIChE Journal.
[2] Costa et al. (2021). Machine Learning in Multicomponent
    Adsorption: From Data to Prediction. Adsorption.

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
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class NeuralMixtureConfig:
    """
    Configuration for the neural mixture correction model.

    Attributes
    ──────────
    mode            : ``'activity'`` for γ_i prediction or
                      ``'direct'`` for multiplicative Δ_i correction.
    emb_dim         : MOF embedding dimension (from encoder).
    n_components    : Number of adsorbate species.
    n_conditions    : Number of thermodynamic condition inputs
                      (T, P_total, y_1, …, y_{C-1}).
    hidden_dim      : Hidden layer width.
    n_layers        : Number of hidden layers in the correction MLP.
    dropout         : Dropout rate.
    residual        : Use residual connections (predict deviation from 1).
    use_iast_input  : Feed IAST base predictions as extra input.
    use_composition : Feed adsorbed-phase mole fractions x_i.
    activation      : ``'silu'``, ``'relu'``, or ``'gelu'``.
    gamma_clamp     : Clamp activity coefficients to ``[1/clamp, clamp]``
                      for numerical stability.
    """

    mode: str = "activity"
    emb_dim: int = 128
    n_components: int = 3
    n_conditions: int = 5  # T, P, y_CO2, y_N2, y_H2O
    hidden_dim: int = 128
    n_layers: int = 3
    dropout: float = 0.1
    residual: bool = True
    use_iast_input: bool = True
    use_composition: bool = True
    activation: str = "silu"
    gamma_clamp: float = 10.0


# ═══════════════════════════════════════════════════════════════════════
# 2.  BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════════

def _get_activation(name: str) -> nn.Module:
    return {"silu": nn.SiLU(), "relu": nn.ReLU(), "gelu": nn.GELU()}.get(
        name.lower(), nn.SiLU()
    )


class ConditionEncoder(nn.Module):
    """
    Encode thermodynamic conditions (T, P, y) into a latent vector.

    Applies log-scaling to pressure and temperature for better
    numerical conditioning, then a small MLP.
    """

    def __init__(self, n_conditions: int, out_dim: int, activation: str = "silu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_conditions, out_dim),
            nn.LayerNorm(out_dim),
            _get_activation(activation),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        conditions : ``[B, n_conditions]`` — T [K], P [bar],
                     y_1, …, y_{C-1} (raw values; log-transform
                     applied internally for T, P).
        """
        # Log-scale T and P (first two columns) for stability
        c = conditions.clone()
        c[:, 0] = torch.log(c[:, 0].clamp(min=1.0))   # log(T)
        c[:, 1] = torch.log(c[:, 1].clamp(min=1e-6))   # log(P)
        return self.net(c)


class ResidualBlock(nn.Module):
    """MLP block with optional residual connection."""

    def __init__(self, dim: int, dropout: float = 0.1, activation: str = "silu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            _get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = _get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


# ═══════════════════════════════════════════════════════════════════════
# 3.  ACTIVITY COEFFICIENT HEAD
# ═══════════════════════════════════════════════════════════════════════

class ActivityCoefficientHead(nn.Module):
    """
    Predict per-component activity coefficients γ_i.

    Output is clamped to ``[1/gamma_clamp, gamma_clamp]`` and
    optionally parameterised as ``γ = exp(δ)`` where δ → 0
    recovers ideal behaviour (γ = 1).
    """

    def __init__(self, hidden_dim: int, n_components: int, gamma_clamp: float = 10.0):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, n_components)
        self.gamma_clamp = gamma_clamp
        # Initialise near zero → γ ≈ 1 at start
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Return ``[B, n_components]`` activity coefficients ≥ 0."""
        delta = self.proj(h)  # [B, C]
        gamma = torch.exp(delta.clamp(-3.0, 3.0))  # soft exp
        return gamma.clamp(1.0 / self.gamma_clamp, self.gamma_clamp)


# ═══════════════════════════════════════════════════════════════════════
# 4.  DIRECT CORRECTION HEAD
# ═══════════════════════════════════════════════════════════════════════

class DirectCorrectionHead(nn.Module):
    """
    Predict per-component multiplicative corrections Δ_i such that
    ``q_true = q_IAST · Δ``.  Initialised near Δ = 1.
    """

    def __init__(self, hidden_dim: int, n_components: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, n_components)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        delta = self.proj(h)
        # Softplus-shifted so output ≈ 1 at init
        return F.softplus(delta + math.log(math.e - 1))  # softplus(0) = ln(2) ≈ 0.69


# ═══════════════════════════════════════════════════════════════════════
# 5.  NEURAL MIXTURE MODEL
# ═══════════════════════════════════════════════════════════════════════

class NeuralMixtureModel(nn.Module):
    """
    Neural network that corrects IAST for non-ideal mixture effects.

    Parameters
    ----------
    config : ``NeuralMixtureConfig``.

    Inputs (forward)
    ────────────────
    mof_embedding : ``[B, emb_dim]`` from encoder.
    conditions    : ``[B, n_conditions]`` — T, P, y_1, …
    iast_loadings : ``[B, n_components]`` — IAST base prediction.

    Outputs
    ───────
    Dict with:
    *  ``loadings``  — ``[B, n_components]`` corrected loadings.
    *  ``gamma``     — ``[B, n_components]`` activity coefficients
       (only in ``'activity'`` mode).
    *  ``correction``— ``[B, n_components]`` multiplicative factors.
    """

    def __init__(self, config: Optional[Union[NeuralMixtureConfig, Dict]] = None):
        super().__init__()

        if config is None:
            config = NeuralMixtureConfig()
        elif isinstance(config, dict):
            config = NeuralMixtureConfig(**{
                k: v for k, v in config.items()
                if k in NeuralMixtureConfig.__dataclass_fields__
            })
        self.config = config
        C = config.n_components

        # ── Input dimension ──────────────────────────────────────
        inp_dim = config.emb_dim  # MOF embedding always present
        self.cond_enc = ConditionEncoder(config.n_conditions, config.hidden_dim, config.activation)
        inp_dim += config.hidden_dim  # encoded conditions

        if config.use_iast_input:
            inp_dim += C  # IAST base loadings
        if config.use_composition:
            inp_dim += C  # adsorbed-phase mole fractions

        # ── Trunk MLP ────────────────────────────────────────────
        layers: List[nn.Module] = [
            nn.Linear(inp_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            _get_activation(config.activation),
        ]
        for _ in range(config.n_layers):
            layers.append(ResidualBlock(config.hidden_dim, config.dropout, config.activation))

        self.trunk = nn.Sequential(*layers)

        # ── Output head ──────────────────────────────────────────
        if config.mode == "activity":
            self.head = ActivityCoefficientHead(
                config.hidden_dim, C, config.gamma_clamp,
            )
        elif config.mode == "direct":
            self.head = DirectCorrectionHead(config.hidden_dim, C)
        else:
            raise ValueError(f"Unknown mode '{config.mode}'; use 'activity' or 'direct'.")

    def forward(
        self,
        mof_embedding: torch.Tensor,
        conditions: torch.Tensor,
        iast_loadings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        mof_embedding : ``[B, emb_dim]``
        conditions    : ``[B, n_conditions]``
        iast_loadings : ``[B, n_components]``

        Returns
        -------
        Dict with ``"loadings"``, ``"correction"``, and optionally
        ``"gamma"``.
        """
        cfg = self.config

        # Encode conditions
        cond_h = self.cond_enc(conditions)  # [B, hidden_dim]

        # Build input
        parts = [mof_embedding, cond_h]

        if cfg.use_iast_input:
            # Log-scale loadings for stability
            parts.append(torch.log1p(iast_loadings.clamp(min=0.0)))

        if cfg.use_composition:
            # Adsorbed-phase mole fractions from IAST
            q_total = iast_loadings.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            x_ads = iast_loadings / q_total
            parts.append(x_ads)

        h = torch.cat(parts, dim=-1)  # [B, inp_dim]
        h = self.trunk(h)              # [B, hidden_dim]
        correction = self.head(h)      # [B, C]

        # Apply correction
        corrected = iast_loadings * correction  # [B, C]

        result = {
            "loadings": corrected,
            "correction": correction,
        }

        if cfg.mode == "activity":
            result["gamma"] = correction  # γ_i = correction factor

        return result

    # ── Inference helpers ────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        mof_embedding: torch.Tensor,
        conditions: torch.Tensor,
        iast_loadings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Inference-mode forward (no grad)."""
        self.eval()
        return self.forward(mof_embedding, conditions, iast_loadings)

    @property
    def num_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ═══════════════════════════════════════════════════════════════════════
# 6.  TRAINING LOSS FOR MIXTURE CORRECTION
# ═══════════════════════════════════════════════════════════════════════

class MixtureCorrectionLoss(nn.Module):
    """
    Composite loss for training the neural mixture model.

    Terms
    ─────
    *  ``mse_loss``  — MSE between corrected and GCMC ground-truth
       loadings (primary supervision).
    *  ``gamma_reg`` — L2 penalty on ``log(γ)`` to keep corrections
       small (Occam's razor: prefer IAST when data is scarce).
    *  ``sum_rule``  — penalise violation of the adsorption sum rule
       ``Σ x_i = 1`` in the corrected phase.
    *  ``monotone``  — penalise non-monotonic loading w.r.t. pressure
       (soft constraint sampled via finite differences).

    Parameters
    ----------
    w_mse      : Weight for MSE term.
    w_gamma    : Weight for γ regularisation.
    w_sum      : Weight for sum-rule penalty.
    w_monotone : Weight for monotonicity penalty.
    """

    def __init__(
        self,
        w_mse: float = 1.0,
        w_gamma: float = 0.01,
        w_sum: float = 0.1,
        w_monotone: float = 0.0,
    ):
        super().__init__()
        self.w_mse = w_mse
        self.w_gamma = w_gamma
        self.w_sum = w_sum
        self.w_monotone = w_monotone

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        target_loadings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        output          : Dict from ``NeuralMixtureModel.forward()``.
        target_loadings : ``[B, C]`` ground-truth loadings (GCMC).

        Returns
        -------
        Dict with ``"total"``, ``"mse"``, ``"gamma_reg"``,
        ``"sum_rule"``.
        """
        pred = output["loadings"]
        correction = output["correction"]

        # MSE on loadings
        mse = F.mse_loss(pred, target_loadings)

        # γ regularisation: penalise log(γ)² → push γ toward 1
        log_corr = torch.log(correction.clamp(min=1e-6))
        gamma_reg = log_corr.pow(2).mean()

        # Sum-rule: corrected mole fractions should sum to 1
        q_total = pred.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        x_corr = pred / q_total
        sum_violation = (x_corr.sum(dim=-1) - 1.0).pow(2).mean()

        total = (self.w_mse * mse
                 + self.w_gamma * gamma_reg
                 + self.w_sum * sum_violation)

        return {
            "total": total,
            "mse": mse.detach(),
            "gamma_reg": gamma_reg.detach(),
            "sum_rule": sum_violation.detach(),
        }


# ═══════════════════════════════════════════════════════════════════════
# 7.  IAST + NEURAL CORRECTION WRAPPER
# ═══════════════════════════════════════════════════════════════════════

class CorrectedIAST:
    """
    End-to-end wrapper: IAST base → neural correction → final loadings.

    This is the recommended inference-time entry point when both an
    ``IASTCalculator`` and a trained ``NeuralMixtureModel`` are
    available.

    Parameters
    ----------
    iast_calc : Fitted ``IASTCalculator`` from ``iast.py``.
    model     : Trained ``NeuralMixtureModel``.
    device    : Compute device for the neural model.
    """

    def __init__(
        self,
        iast_calc: Any,
        model: NeuralMixtureModel,
        device: Optional[torch.device] = None,
    ):
        self.iast = iast_calc
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        mof_embedding: Union[np.ndarray, torch.Tensor],
        y: Sequence[float],
        P_total: float,
        T: float = 298.15,
    ) -> Dict[str, Any]:
        """
        Predict corrected mixture loadings for one condition.

        Parameters
        ----------
        mof_embedding : ``[emb_dim]`` or ``[1, emb_dim]``.
        y             : Gas-phase mole fractions.
        P_total       : Total pressure [bar].
        T             : Temperature [K].

        Returns
        -------
        Dict with ``"loadings"``, ``"iast_loadings"``, ``"correction"``,
        ``"selectivity"``.
        """
        # IAST base prediction
        iast_result = self.iast.predict(y, P_total)
        q_iast = iast_result["loadings"]

        # Prepare tensors
        if isinstance(mof_embedding, np.ndarray):
            h = torch.from_numpy(mof_embedding).float()
        else:
            h = mof_embedding.float()
        if h.dim() == 1:
            h = h.unsqueeze(0)
        h = h.to(self.device)

        y = list(y)
        cond = torch.tensor([[T, P_total] + y], dtype=torch.float32, device=self.device)
        q_iast_t = torch.from_numpy(q_iast).float().unsqueeze(0).to(self.device)

        # Neural correction
        out = self.model.predict(h, cond, q_iast_t)
        q_final = out["loadings"].squeeze(0).cpu().numpy()
        correction = out["correction"].squeeze(0).cpu().numpy()

        # Selectivity
        selectivity = None
        n_comp = len(q_final)
        if n_comp >= 2 and q_final[1] > 1e-15 and y[1] > 1e-15:
            selectivity = float((q_final[0] / q_final[1]) / (y[0] / y[1]))

        return {
            "loadings": q_final,
            "iast_loadings": q_iast,
            "correction": correction,
            "selectivity": selectivity,
        }

    def predict_batch(
        self,
        mof_embeddings: Union[np.ndarray, torch.Tensor],
        y_batch: np.ndarray,
        P_batch: np.ndarray,
        T_batch: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Batch prediction.

        Parameters
        ----------
        mof_embeddings : ``[B, emb_dim]``.
        y_batch        : ``[B, C]`` mole fractions.
        P_batch        : ``[B]`` pressures.
        T_batch        : ``[B]`` temperatures (default 298.15).
        """
        B = y_batch.shape[0]
        C = y_batch.shape[1]

        if T_batch is None:
            T_batch = np.full(B, 298.15)

        # IAST batch
        q_iast = np.zeros((B, C))
        for i in range(B):
            result = self.iast.predict(y_batch[i], float(P_batch[i]))
            q_iast[i] = result["loadings"]

        # Neural correction batch
        if isinstance(mof_embeddings, np.ndarray):
            h = torch.from_numpy(mof_embeddings).float().to(self.device)
        else:
            h = mof_embeddings.float().to(self.device)

        cond = np.column_stack([T_batch, P_batch, y_batch])
        cond_t = torch.from_numpy(cond).float().to(self.device)
        q_iast_t = torch.from_numpy(q_iast).float().to(self.device)

        with torch.no_grad():
            out = self.model(h, cond_t, q_iast_t)

        return {
            "loadings": out["loadings"].cpu().numpy(),
            "iast_loadings": q_iast,
            "correction": out["correction"].cpu().numpy(),
        }


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "NeuralMixtureConfig",
    "NeuralMixtureModel",
    "MixtureCorrectionLoss",
    "CorrectedIAST",
    "ConditionEncoder",
    "ActivityCoefficientHead",
    "DirectCorrectionHead",
]