"""
Neural mixture model: learned non-ideal corrections to IAST.
[docstring unchanged]
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
    mode: str = "activity"
    emb_dim: int = 128
    n_components: int = 3
    n_conditions: int = 5
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
    def __init__(self, n_conditions: int, out_dim: int, activation: str = "silu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_conditions, out_dim),
            nn.LayerNorm(out_dim),
            _get_activation(activation),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        c = conditions.clone()
        c[:, 0] = torch.log(c[:, 0].clamp(min=1.0))    # log(T)
        c[:, 1] = torch.log(c[:, 1].clamp(min=1e-6))   # log(P)
        return self.net(c)


class ResidualBlock(nn.Module):
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
    def __init__(self, hidden_dim: int, n_components: int, gamma_clamp: float = 10.0):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, n_components)
        self.gamma_clamp = gamma_clamp
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        delta = self.proj(h)
        gamma = torch.exp(delta.clamp(-3.0, 3.0))
        return gamma.clamp(1.0 / self.gamma_clamp, self.gamma_clamp)


# ═══════════════════════════════════════════════════════════════════════
# 4.  DIRECT CORRECTION HEAD
# ═══════════════════════════════════════════════════════════════════════

class DirectCorrectionHead(nn.Module):
    def __init__(self, hidden_dim: int, n_components: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, n_components)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        delta = self.proj(h)
        # softplus(log(e-1)) = log(1 + (e-1)) = log(e) = 1 at init
        return F.softplus(delta + math.log(math.e - 1))


# ═══════════════════════════════════════════════════════════════════════
# 5.  NEURAL MIXTURE MODEL
# ═══════════════════════════════════════════════════════════════════════

class NeuralMixtureModel(nn.Module):
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

        inp_dim = config.emb_dim
        self.cond_enc = ConditionEncoder(config.n_conditions, config.hidden_dim, config.activation)
        inp_dim += config.hidden_dim

        if config.use_iast_input:
            inp_dim += C
        if config.use_composition:
            inp_dim += C

        layers: List[nn.Module] = [
            nn.Linear(inp_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            _get_activation(config.activation),
        ]
        for _ in range(config.n_layers):
            layers.append(ResidualBlock(config.hidden_dim, config.dropout, config.activation))

        self.trunk = nn.Sequential(*layers)

        if config.mode == "activity":
            self.head = ActivityCoefficientHead(config.hidden_dim, C, config.gamma_clamp)
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
        cfg = self.config
        cond_h = self.cond_enc(conditions)
        parts = [mof_embedding, cond_h]

        if cfg.use_iast_input:
            parts.append(torch.log1p(iast_loadings.clamp(min=0.0)))

        if cfg.use_composition:
            q_total = iast_loadings.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            x_ads = iast_loadings / q_total
            parts.append(x_ads)

        h = torch.cat(parts, dim=-1)
        h = self.trunk(h)
        correction = self.head(h)
        corrected = iast_loadings * correction

        result = {"loadings": corrected, "correction": correction}
        if cfg.mode == "activity":
            result["gamma"] = correction
        return result

    @torch.no_grad()
    def predict(
        self,
        mof_embedding: torch.Tensor,
        conditions: torch.Tensor,
        iast_loadings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
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
    *  ``mse_loss``  — MSE between corrected and GCMC ground-truth loadings.
    *  ``gamma_reg`` — L2 penalty on log(correction) to keep corrections small.
    *  ``sum_rule``  — MSE between adsorbed-phase mole fractions predicted by
                       the model and those derived from GCMC ground truth.

                       FIX: original code computed
                           x_corr = pred / pred.sum(dim=-1)
                           loss   = (x_corr.sum(dim=-1) - 1)²
                       which is identically 0 by construction (any vector
                       divided by its own sum always sums to 1).  Replaced
                       with MSE between corrected mole fractions and GCMC
                       mole fractions, which is a meaningful constraint.

    *  ``monotone``  — weight reserved for monotonicity penalty (not yet
                       implemented; kept at w_monotone=0 by default).
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
        pred = output["loadings"]
        correction = output["correction"]

        # MSE on absolute loadings
        mse = F.mse_loss(pred, target_loadings)

        # γ regularisation: penalise log(γ)² → push γ toward 1
        log_corr = torch.log(correction.clamp(min=1e-6))
        gamma_reg = log_corr.pow(2).mean()

        # FIX: sum-rule — MSE between corrected adsorbed mole fractions and
        # GCMC-derived mole fractions.
        #
        # Old code (always 0):
        #   x_corr = pred / pred.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        #   sum_violation = (x_corr.sum(dim=-1) - 1.0).pow(2).mean()
        #
        # New code: compare mole-fraction distributions.
        x_pred = pred.clamp(min=0.0)
        x_pred = x_pred / x_pred.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        x_true = target_loadings.clamp(min=0.0)
        x_true = x_true / x_true.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        sum_violation = F.mse_loss(x_pred, x_true)

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
        iast_result = self.iast.predict(y, P_total)
        q_iast = iast_result["loadings"]

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

        out = self.model.predict(h, cond, q_iast_t)
        q_final = out["loadings"].squeeze(0).cpu().numpy()
        correction = out["correction"].squeeze(0).cpu().numpy()

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
        B = y_batch.shape[0]
        C = y_batch.shape[1]

        if T_batch is None:
            T_batch = np.full(B, 298.15)

        q_iast = np.zeros((B, C))
        for i in range(B):
            result = self.iast.predict(y_batch[i], float(P_batch[i]))
            q_iast[i] = result["loadings"]

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