"""
Stochastic Weight Averaging Gaussian (SWAG) for Bayesian UQ.

SWAG (Maddox et al., 2019) provides a scalable approximation to the
posterior distribution over neural-network weights by collecting
running statistics during the **tail** of standard SGD/Adam training.
It yields calibrated predictive uncertainty from a **single** training
run — no ensemble of models required.

How it works
────────────
1.  **SWA phase** — after normal training converges, continue for
    ``swa_epochs`` with a cyclical or constant learning rate.  At the
    end of each epoch (or every ``collect_freq`` steps), snapshot the
    current weights.
2.  **Statistics** — from the K collected snapshots, compute:
        *  θ̄  — running mean of weights (= SWA solution).
        *  Σ_diag — running variance (diagonal covariance).
        *  D  — low-rank deviation matrix ``[n_params, rank]``
           storing the K most recent (θ_k − θ̄) columns.
    The approximate posterior is:
        q(θ) = N(θ̄,  ½ · (Σ_diag + (1/K) D Dᵀ))
3.  **Inference** — draw S weight samples from q(θ), evaluate the
    model with each sample, then compute mean prediction ± std as
    the uncertainty estimate.

Integration with UC-TPNO
────────────────────────
*   ``SWAGWrapper`` wraps any ``nn.Module`` (including our TPNO or
    ensemble) and exposes ``collect()``, ``sample_and_predict()``.
*   The resulting σ_swag can be fed into the conformal calibrator as
    the ``y_std`` input for normalised or studentised scores, giving
    distribution-free coverage guarantees on top of the Bayesian UQ.

Dependencies
────────────
``torch``, ``numpy`` — no additional packages required.

References
──────────
[1] Maddox et al. (2019). A Simple Baseline for Bayesian Inference
    in Deep Learning. NeurIPS.
[2] Izmailov et al. (2018). Averaging Weights Leads to Wider Optima
    and Better Generalization. UAI.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SWAGConfig:
    """
    SWAG hyperparameters.

    Attributes
    ──────────
    swa_epochs      : Number of additional epochs in the SWA phase.
    swa_lr          : Learning rate during SWA collection (constant or
                      peak of cyclical schedule).
    collect_freq    : Collect a weight snapshot every N epochs.
    max_rank        : Maximum columns kept in the low-rank deviation
                      matrix D.  Memory cost is O(rank × n_params).
                      Typically 20–30 suffices.
    n_samples       : Number of posterior weight samples drawn at
                      inference time for Monte-Carlo averaging.
    scale           : Global scale factor applied to the covariance
                      (default 0.5 as in the original paper).
    cyclical        : Use cyclical LR schedule during SWA phase.
    cycle_length    : Epochs per LR cycle (only if ``cyclical=True``).
    var_clamp       : Minimum variance to prevent numerical issues.
    """

    swa_epochs: int = 30
    swa_lr: float = 1e-3
    collect_freq: int = 1
    max_rank: int = 20
    n_samples: int = 30
    scale: float = 0.5
    cyclical: bool = True
    cycle_length: int = 5
    var_clamp: float = 1e-30


# ═══════════════════════════════════════════════════════════════════════
# 2.  SWAG WRAPPER
# ═══════════════════════════════════════════════════════════════════════

class SWAGWrapper(nn.Module):
    """
    Wraps any ``nn.Module`` and maintains SWAG statistics over its
    parameters.

    Workflow
    ───────
    >>> model = ThermodynamicPotentialNO(encoder, config)
    >>> swag = SWAGWrapper(model, SWAGConfig())
    >>> # --- normal training until convergence ---
    >>> # --- SWA phase ---
    >>> for epoch in range(swag.config.swa_epochs):
    ...     train_one_epoch(model, ...)
    ...     swag.collect()
    >>> # --- inference ---
    >>> mean, std = swag.predict_with_uncertainty(test_data)

    Parameters
    ----------
    base_model : The model whose weights will be tracked.
    config     : ``SWAGConfig`` hyperparameters.
    device     : Compute device for sampled weights.
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: Optional[Union[SWAGConfig, Dict]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.base_model = base_model

        if config is None:
            self.config = SWAGConfig()
        elif isinstance(config, dict):
            self.config = SWAGConfig(**{
                k: v for k, v in config.items()
                if k in SWAGConfig.__dataclass_fields__
            })
        else:
            self.config = config

        if device is None:
            device = next(base_model.parameters()).device
        self.device = torch.device(device) if isinstance(device, str) else device

        # Flatten parameter names for consistent ordering
        self._param_names: List[str] = []
        self._param_shapes: List[torch.Size] = []
        for name, p in base_model.named_parameters():
            if p.requires_grad:
                self._param_names.append(name)
                self._param_shapes.append(p.shape)

        n_params = sum(s.numel() for s in self._param_shapes)
        self._n_params = n_params

        # ── Running statistics (registered as buffers) ───────────
        # Mean of weights  θ̄
        self.register_buffer("_swa_mean", torch.zeros(n_params))
        # Running second moment for diagonal variance
        self.register_buffer("_swa_sq_mean", torch.zeros(n_params))
        # Low-rank deviation columns D: [n_params, max_rank]
        self.register_buffer(
            "_dev_matrix",
            torch.zeros(n_params, self.config.max_rank),
        )
        # Number of snapshots collected so far
        self.register_buffer("_n_collected", torch.tensor(0, dtype=torch.long))
        # Column pointer for circular buffer in D
        self.register_buffer("_dev_col", torch.tensor(0, dtype=torch.long))

    # ── Parameter vector utilities ───────────────────────────────

    def _flatten_params(self, model: Optional[nn.Module] = None) -> torch.Tensor:
        """Flatten all trainable parameters into a single 1-D vector."""
        model = model or self.base_model
        parts = []
        for name, p in model.named_parameters():
            if name in self._param_names:
                parts.append(p.detach().reshape(-1))
        return torch.cat(parts)

    def _load_flat_params(
        self,
        flat: torch.Tensor,
        model: Optional[nn.Module] = None,
    ) -> None:
        """Load a flat parameter vector back into the model."""
        model = model or self.base_model
        offset = 0
        for name, p in model.named_parameters():
            if name in self._param_names:
                numel = p.numel()
                p.data.copy_(flat[offset : offset + numel].reshape(p.shape))
                offset += numel

    # ── Collection ───────────────────────────────────────────────

    @property
    def n_collected(self) -> int:
        """Number of weight snapshots collected."""
        return int(self._n_collected.item())

    def collect(self) -> None:
        """
        Snapshot the current base_model weights and update SWAG
        running statistics.

        Call this at the end of each SWA-phase epoch (or every
        ``collect_freq`` steps).
        """
        n = self.n_collected
        theta = self._flatten_params().to(self._swa_mean.device)

        # Online update of mean:  θ̄_new = (n·θ̄_old + θ) / (n+1)
        self._swa_mean.mul_(n / (n + 1)).add_(theta / (n + 1))

        # Online update of second moment:  same rule
        self._swa_sq_mean.mul_(n / (n + 1)).add_(theta.pow(2) / (n + 1))

        # Low-rank deviation column:  d = θ − θ̄  (using updated mean)
        dev = theta - self._swa_mean
        col = int(self._dev_col.item()) % self.config.max_rank
        self._dev_matrix[:, col] = dev
        self._dev_col.add_(1)

        self._n_collected.add_(1)
        logger.debug(
            f"SWAG collect #{self.n_collected}: "
            f"‖θ̄‖={self._swa_mean.norm():.4f}"
        )

    @property
    def is_fitted(self) -> bool:
        """At least 2 snapshots are needed for variance."""
        return self.n_collected >= 2

    # ── Covariance components ────────────────────────────────────

    @property
    def swa_mean(self) -> torch.Tensor:
        """Return θ̄ (the SWA solution)."""
        return self._swa_mean.clone()

    @property
    def diagonal_variance(self) -> torch.Tensor:
        """
        Σ_diag = E[θ²] − (E[θ])²,  clamped to ``var_clamp``.
        """
        var = (self._swa_sq_mean - self._swa_mean.pow(2)).clamp(
            min=self.config.var_clamp,
        )
        return var

    @property
    def deviation_matrix(self) -> torch.Tensor:
        """
        D: the low-rank deviation matrix ``[n_params, K]`` where
        K = min(n_collected, max_rank).
        """
        k = min(self.n_collected, self.config.max_rank)
        return self._dev_matrix[:, :k]

    # ── Sampling ─────────────────────────────────────────────────

    def sample_parameters(self, scale: Optional[float] = None) -> torch.Tensor:
        """
        Draw one parameter vector from the SWAG posterior:

            θ ~ N(θ̄, scale · (½ Σ_diag + (1/2K) D Dᵀ))

        Returns
        -------
        ``[n_params]`` sampled parameter vector.
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"SWAG needs ≥ 2 collected snapshots; got {self.n_collected}."
            )

        scale = scale if scale is not None else self.config.scale
        device = self._swa_mean.device

        # Diagonal term:  z_1 ~ N(0, I) → √(scale · Σ_diag) ⊙ z_1
        z_diag = torch.randn(self._n_params, device=device)
        diag_part = (scale * self.diagonal_variance).sqrt() * z_diag

        # Low-rank term:  z_2 ~ N(0, I_K) → √(scale / 2K) D z_2
        D = self.deviation_matrix
        K = D.shape[1]
        z_lr = torch.randn(K, device=device)
        lr_part = math.sqrt(scale / (2.0 * K)) * (D @ z_lr)

        # Combine:  θ_sample = θ̄ + (1/√2)(diag_part) + lr_part
        # The 1/√2 ensures the total covariance is
        # scale · (½ Σ_diag + (1/2K) D Dᵀ)
        theta_sample = self._swa_mean + (diag_part / math.sqrt(2.0)) + lr_part

        return theta_sample

    # ── SWA solution (point estimate) ────────────────────────────

    def apply_swa_mean(self) -> None:
        """
        Load the SWA mean θ̄ into ``base_model`` (i.e. replace
        the current weights with the running average).
        """
        if self.n_collected == 0:
            logger.warning("No snapshots collected; cannot apply SWA mean.")
            return
        self._load_flat_params(self._swa_mean)
        logger.info("Applied SWA mean weights to base model.")

    # ── Inference with uncertainty ───────────────────────────────

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        *args: Any,
        n_samples: Optional[int] = None,
        scale: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Draw S weight samples, run forward pass with each, and return
        the mean prediction ± epistemic std.

        Parameters
        ----------
        *args, **kwargs : Forwarded to ``base_model.forward()``.
        n_samples       : Number of MC weight samples (default from config).
        scale           : Covariance scale override.

        Returns
        -------
        Dict with keys:

        *  ``"mean"``      — ``[B, ...]`` mean prediction across samples.
        *  ``"std"``       — ``[B, ...]`` std (epistemic uncertainty).
        *  ``"samples"``   — ``[S, B, ...]`` raw per-sample outputs.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "SWAG is not fitted (need ≥ 2 collected snapshots). "
                "Run the SWA collection phase first."
            )

        n_samples = n_samples or self.config.n_samples
        self.base_model.eval()

        # Save original weights to restore later
        original_params = self._flatten_params().clone()

        all_outputs: List[torch.Tensor] = []

        for s in range(n_samples):
            # Sample weights and load
            theta_s = self.sample_parameters(scale=scale)
            self._load_flat_params(theta_s)

            # Forward pass
            output = self.base_model(*args, **kwargs)

            # Handle dict output (our TPNO returns dicts)
            if isinstance(output, dict):
                out_tensor = output.get("q_pred", output.get("mean", None))
                if out_tensor is None:
                    out_tensor = next(iter(output.values()))
            else:
                out_tensor = output

            all_outputs.append(out_tensor)

        # Restore original weights
        self._load_flat_params(original_params)

        samples = torch.stack(all_outputs, dim=0)  # [S, B, ...]
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)

        return {
            "mean": mean,
            "std": std,
            "samples": samples,
        }

    # ── Forward (delegates to base model with SWA mean) ──────────

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Standard forward pass using the SWA mean weights (if
        collected) or the current base_model weights.

        For uncertainty, use ``predict_with_uncertainty()`` instead.
        """
        return self.base_model(*args, **kwargs)

    # ── SWA learning-rate schedule ───────────────────────────────

    def get_swa_lr(self, epoch: int) -> float:
        """
        Return the learning rate for a given SWA-phase epoch.

        *   **Constant**: returns ``swa_lr``.
        *   **Cyclical**: cosine annealing within each cycle, between
            ``swa_lr`` (peak) and ``swa_lr / 10`` (trough).
        """
        if not self.config.cyclical:
            return self.config.swa_lr

        cycle_len = max(self.config.cycle_length, 1)
        t = (epoch % cycle_len) / cycle_len
        lr_min = self.config.swa_lr / 10.0
        return lr_min + 0.5 * (self.config.swa_lr - lr_min) * (1.0 + math.cos(math.pi * t))

    # ── SWA training loop (convenience) ──────────────────────────

    def run_swa_phase(
        self,
        train_fn: Callable[[nn.Module, float], None],
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """
        Convenience method to run the full SWA collection phase.

        Parameters
        ----------
        train_fn : ``(model, lr) → None``; trains the model for one
                   epoch at the given learning rate.
        callback : ``(epoch, n_collected) → None``; called after each
                   collection step.
        """
        cfg = self.config
        for epoch in range(cfg.swa_epochs):
            lr = self.get_swa_lr(epoch)
            train_fn(self.base_model, lr)

            if (epoch + 1) % cfg.collect_freq == 0:
                self.collect()
                if callback is not None:
                    callback(epoch, self.n_collected)

        logger.info(
            f"SWA phase complete: {self.n_collected} snapshots collected "
            f"over {cfg.swa_epochs} epochs."
        )

    # ── Batch-normalisation update ───────────────────────────────

    @torch.no_grad()
    def update_bn(
        self,
        loader: Any,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Re-estimate BatchNorm running statistics for the SWA mean
        weights by doing a single pass over the training data.

        Should be called after ``apply_swa_mean()`` and before
        evaluation, because the SWA weights differ from the training
        trajectory and BN stats may be stale.
        """
        device = device or self.device
        self.base_model.train()

        # Reset BN statistics
        for module in self.base_model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.reset_running_stats()

        for batch in loader:
            if isinstance(batch, dict):
                # Our data-loader format
                if "graphs" in batch:
                    graphs = batch["graphs"]
                    if isinstance(graphs, dict):
                        graphs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                  for k, v in graphs.items()}
                    else:
                        graphs = graphs.to(device)
                    conditions = batch["conditions"].to(device)
                    self.base_model(graphs, conditions)
                else:
                    self.base_model(batch)
            else:
                if hasattr(batch, "to"):
                    batch = batch.to(device)
                self.base_model(batch)

        self.base_model.eval()
        logger.info("BatchNorm statistics updated for SWA mean weights.")

    # ── Serialisation ────────────────────────────────────────────

    def state_dict_swag(self) -> Dict[str, Any]:
        """
        Return a dict containing all SWAG state (statistics +
        config), suitable for ``torch.save()``.
        """
        return {
            "config": vars(self.config),
            "swa_mean": self._swa_mean.cpu(),
            "swa_sq_mean": self._swa_sq_mean.cpu(),
            "dev_matrix": self._dev_matrix.cpu(),
            "n_collected": self._n_collected.cpu(),
            "dev_col": self._dev_col.cpu(),
            "param_names": self._param_names,
            "param_shapes": [list(s) for s in self._param_shapes],
            "base_model_state": self.base_model.state_dict(),
        }

    def load_state_dict_swag(self, state: Dict[str, Any]) -> None:
        """Restore SWAG state from a dict produced by ``state_dict_swag``."""
        self.config = SWAGConfig(**state["config"])
        self._swa_mean.copy_(state["swa_mean"].to(self._swa_mean.device))
        self._swa_sq_mean.copy_(state["swa_sq_mean"].to(self._swa_sq_mean.device))
        self._dev_matrix.copy_(state["dev_matrix"].to(self._dev_matrix.device))
        self._n_collected.copy_(state["n_collected"].to(self._n_collected.device))
        self._dev_col.copy_(state["dev_col"].to(self._dev_col.device))
        self.base_model.load_state_dict(state["base_model_state"])
        logger.info(f"Loaded SWAG state ({self.n_collected} snapshots).")

    # ── Introspection ────────────────────────────────────────────

    @property
    def n_params(self) -> int:
        """Total number of tracked (trainable) parameters."""
        return self._n_params

    @property
    def rank(self) -> int:
        """Current effective rank of the deviation matrix D."""
        return min(self.n_collected, self.config.max_rank)

    def summary(self) -> Dict[str, Any]:
        """Human-readable summary of SWAG state."""
        return {
            "n_params": self.n_params,
            "n_collected": self.n_collected,
            "max_rank": self.config.max_rank,
            "effective_rank": self.rank,
            "swa_mean_norm": float(self._swa_mean.norm()),
            "diag_var_mean": float(self.diagonal_variance.mean()) if self.is_fitted else None,
            "is_fitted": self.is_fitted,
        }

    def __repr__(self) -> str:
        return (
            f"SWAGWrapper("
            f"n_params={self.n_params:,}, "
            f"collected={self.n_collected}/{self.config.swa_epochs}, "
            f"rank={self.rank}/{self.config.max_rank})"
        )


# ═══════════════════════════════════════════════════════════════════════
# 3.  CONVENIENCE: SWAG + CONFORMAL INTEGRATION
# ═══════════════════════════════════════════════════════════════════════

def swag_conformal_predict(
    swag: SWAGWrapper,
    conformal: Any,
    *forward_args: Any,
    n_samples: Optional[int] = None,
    **forward_kwargs: Any,
) -> Dict[str, Any]:
    """
    Combined SWAG + conformal prediction.

    1.  Draw weight samples from SWAG posterior → epistemic std.
    2.  Feed (mean, std) into the conformal calibrator for
        distribution-free coverage intervals.

    Parameters
    ----------
    swag       : Fitted ``SWAGWrapper``.
    conformal  : Fitted ``ConformalCalibrator`` (from conformal.py).
    *args, **kwargs : Forwarded to SWAG's ``predict_with_uncertainty``.

    Returns
    -------
    Dict with ``"mean"``, ``"std"``, ``"lower"``, ``"upper"``,
    ``"coverage_probability"``.
    """
    # Step 1: SWAG uncertainty
    result = swag.predict_with_uncertainty(
        *forward_args, n_samples=n_samples, **forward_kwargs,
    )

    y_pred = result["mean"].cpu().numpy()
    y_std = result["std"].cpu().numpy()

    # Step 2: Conformal intervals using SWAG std as normaliser
    intervals = conformal.predict_intervals({
        "y_pred": y_pred,
        "y_std": y_std,
    })

    return {
        "mean": result["mean"],
        "std": result["std"],
        "lower": torch.from_numpy(intervals["lower"]).to(result["mean"]),
        "upper": torch.from_numpy(intervals["upper"]).to(result["mean"]),
        "coverage_probability": intervals.get("coverage_probability", None),
        "samples": result["samples"],
    }


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "SWAGConfig",
    "SWAGWrapper",
    "swag_conformal_predict",
]