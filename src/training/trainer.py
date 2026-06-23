#!/usr/bin/env python3
"""
Main training pipeline for the Thermodynamic Potential Neural Operator.

This version fixes:
1. API compatibility:
   - supports trainer.fit(...)
   - supports old trainer.train(...) as an alias
   - restores TrainerConfig as an alias of TrainConfig
2. Mask-aware validation metrics:
   - padded condition rows are excluded from MAE/RMSE/R²
3. More robust criterion calling:
   - works with several possible loss-module signatures
4. Better terminal progress:
   - batch updates are logged at the configured interval
5. Safer ensemble bootstrapping:
   - preserves the original loader collate_fn
6. Validation/autograd compatibility:
   - validate() is NOT wrapped in torch.no_grad()
   - validation runs with torch.set_grad_enabled(True)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """
    Training hyperparameters.
    """

    n_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine_warm_restarts"
    scheduler_T0: int = 10
    scheduler_T_mult: int = 2
    step_size: int = 30
    step_gamma: float = 0.5
    warmup_epochs: int = 5
    physics_warmup: int = 20
    grad_clip: float = 1.0
    use_amp: bool = False
    early_stopping: bool = True
    patience: int = 20
    checkpoint_dir: str = "checkpoints"
    checkpoint_freq: int = 10
    use_wandb: bool = False
    wandb_project: str = "uc-tpno"
    wandb_run_name: Optional[str] = None
    use_tensorboard: bool = False
    tb_log_dir: str = "runs"
    log_interval: int = 50


# backward compatibility
TrainerConfig = TrainConfig


# ═══════════════════════════════════════════════════════════════════════
# 2. METRIC UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def _compute_regression_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, float]:
    """
    MAE, RMSE, R² on already-unpadded arrays.
    """
    if y_pred.size == 0 or y_true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _per_component_mae(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    names: Sequence[str] = ("CO2", "N2", "H2O"),
) -> Dict[str, float]:
    """
    Per-component MAE for arrays with shape [N, C].
    """
    out: Dict[str, float] = {}
    if y_pred.size == 0 or y_true.size == 0:
        for name in names:
            out[f"mae_{name}"] = float("nan")
        return out

    n_comp = y_pred.shape[-1]
    for i in range(min(n_comp, len(names))):
        out[f"mae_{names[i]}"] = float(np.mean(np.abs(y_pred[:, i] - y_true[:, i])))
    return out


def _masked_points_to_numpy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert [B, P, C] predictions/targets into [N, C] arrays,
    removing padded rows using mask if provided.
    """
    if mask is None:
        pred_np = preds.detach().cpu().numpy().reshape(-1, preds.shape[-1])
        true_np = targets.detach().cpu().numpy().reshape(-1, targets.shape[-1])
        return pred_np.astype(np.float64), true_np.astype(np.float64)

    valid = mask.bool()
    pred_np = preds[valid].detach().cpu().numpy().astype(np.float64)
    true_np = targets[valid].detach().cpu().numpy().astype(np.float64)
    return pred_np, true_np


# ═══════════════════════════════════════════════════════════════════════
# 3. TRAINER
# ═══════════════════════════════════════════════════════════════════════

class TPNOTrainer:
    """
    Training orchestrator for ThermodynamicPotentialNO or TPNOEnsemble.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        config: Union[TrainConfig, Dict[str, Any]],
        criterion: Optional[nn.Module] = None,
        device: Union[str, torch.device] = "cuda",
    ):
        if isinstance(config, dict):
            self.config = TrainConfig(
                **{
                    k: v
                    for k, v in config.items()
                    if k in TrainConfig.__dataclass_fields__
                }
            )
        else:
            self.config = config

        cfg = self.config

        if isinstance(device, str):
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            device = torch.device(device)
        self.device = device

        self.model = model.to(self.device)

        if criterion is not None:
            self.criterion = criterion
        else:
            from src.models.operator.losses import ThermodynamicLoss
            self.criterion = ThermodynamicLoss()

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self._physics_scheduler: Optional[Any] = None
        if cfg.physics_warmup > 0:
            try:
                from src.models.operator.losses import PhysicsLossScheduler
                self._physics_scheduler = PhysicsLossScheduler(
                    self.criterion,
                    warmup_epochs=cfg.physics_warmup,
                )
            except Exception:
                self._physics_scheduler = None

        self._scaler: Optional[torch.cuda.amp.GradScaler] = None
        if cfg.use_amp and self.device.type == "cuda":
            self._scaler = torch.cuda.amp.GradScaler()

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.epoch: int = 0
        self.global_step: int = 0
        self.best_val_metric: float = float("inf")
        self.epochs_without_improvement: int = 0
        self.history: List[Dict[str, Any]] = []

        self._wandb_run = None
        self._tb_writer = None
        self._init_loggers()

    # ── builder helpers ─────────────────────────────────────────

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.config
        name = cfg.optimizer.lower()
        params = self.model.parameters()

        if name == "adamw":
            return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if name == "adam":
            return torch.optim.Adam(params, lr=cfg.lr)
        if name == "sgd":
            return torch.optim.SGD(
                params,
                lr=cfg.lr,
                momentum=0.9,
                weight_decay=cfg.weight_decay,
            )
        raise ValueError(f"Unknown optimiser: {cfg.optimizer}")

    def _build_scheduler(self):
        cfg = self.config
        name = cfg.scheduler.lower()

        if name == "cosine_warm_restarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=cfg.scheduler_T0,
                T_mult=cfg.scheduler_T_mult,
            )
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.n_epochs,
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.step_size,
                gamma=cfg.step_gamma,
            )
        if name == "none":
            return None
        raise ValueError(f"Unknown scheduler: {cfg.scheduler}")

    # ── logger init ─────────────────────────────────────────────

    def _init_loggers(self) -> None:
        cfg = self.config

        if cfg.use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name,
                    config=vars(cfg),
                    reinit=True,
                )
            except ImportError:
                logger.warning("wandb not installed; skipping W&B logging.")

        if cfg.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=cfg.tb_log_dir)
            except ImportError:
                logger.warning("tensorboard not installed; skipping TensorBoard logging.")

    def _log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        if self._wandb_run is not None:
            import wandb
            wandb.log(metrics, step=step)

        if self._tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, step)

    # ── LR warmup ───────────────────────────────────────────────

    def _warmup_lr(self, epoch: int) -> None:
        if epoch >= self.config.warmup_epochs:
            return
        warmup_factor = (epoch + 1) / max(self.config.warmup_epochs, 1)
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.config.lr * warmup_factor

    # ── batch unpacking ─────────────────────────────────────────

    @staticmethod
    def _move_graphs_to_device(graphs: Any, device: torch.device) -> Any:
        if isinstance(graphs, dict):
            return {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in graphs.items()
            }
        return graphs.to(device)

    @classmethod
    def _to_device(
        cls,
        batch: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Unpack batch into (graphs, conditions, targets, mask).
        """
        graphs = cls._move_graphs_to_device(batch["graphs"], device)
        conditions = batch["conditions"].to(device)
        targets = batch["loadings"].to(device)
        mask = batch.get("mask", None)
        if mask is not None:
            mask = mask.to(device)
        return graphs, conditions, targets, mask

    # ── criterion dispatch ──────────────────────────────────────

    def _call_criterion(
        self,
        predictions: Dict[str, Any],
        targets: torch.Tensor,
        graphs: Any,
        conditions: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Call the loss module robustly across slightly different signatures.
        """
        try:
            loss_out = self.criterion(
                predictions,
                targets,
                self.model,
                graphs,
                conditions,
                mask=mask,
            )
        except TypeError:
            try:
                loss_out = self.criterion(
                    predictions,
                    targets,
                    self.model,
                    graphs,
                    conditions,
                )
            except TypeError:
                try:
                    loss_out = self.criterion(
                        predictions,
                        targets,
                        mask=mask,
                    )
                except TypeError:
                    loss_out = self.criterion(predictions, targets)

        if isinstance(loss_out, torch.Tensor):
            return {
                "total": loss_out,
                "data": loss_out,
                "physics": {},
            }

        if isinstance(loss_out, dict):
            total = loss_out.get("total", None)
            data = loss_out.get("data", None)

            if total is None:
                total = (
                    loss_out.get("loss")
                    or loss_out.get("total_loss")
                    or loss_out.get("data_loss")
                )
            if data is None:
                data = (
                    loss_out.get("data_loss")
                    or loss_out.get("loss")
                    or total
                )

            physics = loss_out.get("physics", {})

            if not physics:
                physics = {
                    k: v
                    for k, v in loss_out.items()
                    if k not in {"total", "loss", "total_loss", "data", "data_loss"}
                }

            return {
                "total": total,
                "data": data,
                "physics": physics,
            }

        raise TypeError(
            f"Unsupported criterion output type: {type(loss_out).__name__}"
        )

    # ── single epoch ────────────────────────────────────────────

    def train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch and return averaged training losses.
        """
        self.model.train()
        cfg = self.config

        total_loss = 0.0
        total_data = 0.0
        physics_accum: Dict[str, float] = {}
        n_batches = 0

        for i, batch in enumerate(self.train_loader):
            graphs, conditions, targets, mask = self._to_device(batch, self.device)

            with torch.cuda.amp.autocast(enabled=self._scaler is not None):
                predictions = self.model(
                    graphs,
                    conditions,
                    return_uncertainty=True,
                    return_potential=False,
                    return_hessian=False,
                )
                losses = self._call_criterion(
                    predictions=predictions,
                    targets=targets,
                    graphs=graphs,
                    conditions=conditions,
                    mask=mask,
                )
                loss = losses["total"]

            self.optimizer.zero_grad(set_to_none=True)

            if self._scaler is not None:
                self._scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    self._scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.optimizer.step()

            total_loss += float(loss.item())
            total_data += float(
                losses["data"].item() if isinstance(losses["data"], torch.Tensor) else losses["data"]
            )

            for k, v in losses.get("physics", {}).items():
                if isinstance(v, torch.Tensor):
                    value = float(v.item())
                elif isinstance(v, (int, float)):
                    value = float(v)
                else:
                    continue  # skip nested dicts or non-scalar physics terms
                physics_accum[k] = physics_accum.get(k, 0.0) + value

            n_batches += 1
            self.global_step += 1

            if cfg.log_interval > 0 and (i + 1) % cfg.log_interval == 0:
                logger.info(
                    "  batch %d/%d | loss=%.6f | data=%.6f",
                    i + 1,
                    len(self.train_loader),
                    float(loss.item()),
                    float(
                        losses["data"].item()
                        if isinstance(losses["data"], torch.Tensor)
                        else losses["data"]
                    ),
                )

        n = max(n_batches, 1)
        result: Dict[str, float] = {
            "train/total_loss": total_loss / n,
            "train/data_loss": total_data / n,
        }
        for k, v in physics_accum.items():
            result[f"train/physics_{k}"] = v / n
        return result

    # ── validation ──────────────────────────────────────────────

    def validate(self) -> Dict[str, float]:
        """
        Run one validation pass with mask-aware metrics.

        Important:
        We do NOT use torch.no_grad() here because the model forward may call
        autograd.grad internally for physics terms / derivatives.
        """
        self.model.eval()

        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        with torch.set_grad_enabled(True):
            for batch in self.val_loader:
                graphs, conditions, targets, mask = self._to_device(batch, self.device)

                predictions = self.model(
                    graphs,
                    conditions,
                    return_uncertainty=True,
                    return_potential=False,
                    return_hessian=False,
                )
                q = predictions["q_pred"]

                pred_np, true_np = _masked_points_to_numpy(q, targets, mask)
                all_preds.append(pred_np)
                all_targets.append(true_np)

        if not all_preds:
            raise RuntimeError("Validation loader produced no batches.")

        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_targets, axis=0)

        reg = _compute_regression_metrics(preds, trues)
        comp = _per_component_mae(preds, trues)

        metrics: Dict[str, float] = {
            "val/mae": reg["mae"],
            "val/rmse": reg["rmse"],
            "val/r2": reg["r2"],
        }
        for k, v in comp.items():
            metrics[f"val/{k}"] = v

        return metrics

    # ── full training loop ──────────────────────────────────────

    def fit(
        self,
        n_epochs: Optional[int] = None,
        callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute the full training loop.
        """
        n_epochs = n_epochs or self.config.n_epochs
        cfg = self.config

        logger.info(
            "Starting training for %d epochs (device=%s, amp=%s)",
            n_epochs,
            self.device,
            cfg.use_amp,
        )

        start_epoch = self.epoch
        end_epoch = start_epoch + n_epochs

        for ep in range(start_epoch, end_epoch):
            self.epoch = ep
            t0 = time.perf_counter()

            self._warmup_lr(ep)

            if self._physics_scheduler is not None:
                try:
                    self._physics_scheduler.step(ep)
                except Exception:
                    pass

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            if self.scheduler is not None and ep >= cfg.warmup_epochs:
                self.scheduler.step()

            lr = float(self.optimizer.param_groups[0]["lr"])
            epoch_time = float(time.perf_counter() - t0)

            metrics: Dict[str, Any] = {
                **train_metrics,
                **val_metrics,
                "epoch": ep,
                "lr": lr,
                "epoch_time_s": epoch_time,
            }

            self.history.append(metrics)
            self._log_metrics(metrics, step=ep)

            logger.info(
                "Epoch %d/%d | train_loss=%.6f | val_mae=%.6f | val_rmse=%.6f | val_r2=%.4f | lr=%.2e | %.1fs",
                ep + 1,
                end_epoch,
                train_metrics["train/total_loss"],
                val_metrics["val/mae"],
                val_metrics["val/rmse"],
                val_metrics["val/r2"],
                lr,
                epoch_time,
            )

            val_mae = val_metrics["val/mae"]
            if val_mae < self.best_val_metric:
                self.best_val_metric = val_mae
                self.epochs_without_improvement = 0
                self.save_checkpoint(ep, metrics, is_best=True)
            else:
                self.epochs_without_improvement += 1

            if cfg.checkpoint_freq > 0 and (ep + 1) % cfg.checkpoint_freq == 0:
                self.save_checkpoint(ep, metrics, is_best=False)

            if callback is not None:
                callback(ep, metrics)

            if cfg.early_stopping and self.epochs_without_improvement >= cfg.patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs).",
                    ep + 1,
                    cfg.patience,
                )
                break

        if self.history:
            self.save_checkpoint(self.epoch, self.history[-1], is_best=False, tag="final")

        self._close_loggers()
        return self.history

    # backward-compatible alias
    def train(
        self,
        callbacks: Optional[Any] = None,
        n_epochs: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Backward-compatible alias for older scripts that call trainer.train(...).

        If `callbacks` has an `on_epoch_end(epoch, metrics, trainer)` method,
        it will be bridged into the new fit(callback=...) style.
        """
        callback_fn = None
        if callbacks is not None and hasattr(callbacks, "on_epoch_end"):
            def callback_fn(epoch: int, metrics: Dict[str, Any]) -> None:
                callbacks.on_epoch_end(epoch, metrics, self)

        return self.fit(n_epochs=n_epochs, callback=callback_fn)

    # ── checkpointing ───────────────────────────────────────────

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        is_best: bool = False,
        tag: Optional[str] = None,
    ) -> Path:
        """
        Save a full checkpoint.
        """
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_metric": self.best_val_metric,
            "metrics": {
                k: v
                for k, v in metrics.items()
                if isinstance(v, (int, float, str))
            },
            "config": vars(self.config),
        }

        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        if self._scaler is not None:
            state["scaler_state_dict"] = self._scaler.state_dict()

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        elif tag is not None:
            path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"

        torch.save(state, path)
        logger.info("Saved checkpoint -> %s", path)

        json_path = self.checkpoint_dir / "latest_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(state["metrics"], f, indent=2)

        return path

    def load_checkpoint(
        self,
        path: Union[str, Path],
        load_optimizer: bool = True,
    ) -> Dict[str, Any]:
        """
        Restore from a checkpoint. Returns stored metrics.
        """
        path = Path(path)
        state = torch.load(path, map_location=self.device, weights_only=False)

        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)

        if load_optimizer and "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        if self._scaler is not None and "scaler_state_dict" in state:
            self._scaler.load_state_dict(state["scaler_state_dict"])

        self.epoch = int(state.get("epoch", 0)) + 1
        self.global_step = int(state.get("global_step", 0))
        self.best_val_metric = float(state.get("best_val_metric", float("inf")))

        logger.info(
            "Loaded checkpoint from %s (epoch=%s, best_val=%.6f)",
            path,
            state.get("epoch", None),
            self.best_val_metric,
        )
        return state.get("metrics", {})

    # ── logger teardown ─────────────────────────────────────────

    def _close_loggers(self) -> None:
        if self._wandb_run is not None:
            import wandb
            wandb.finish()
        if self._tb_writer is not None:
            self._tb_writer.close()

    # ── introspection ───────────────────────────────────────────

    @property
    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    @property
    def num_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "total": int(total),
            "trainable": int(trainable),
            "frozen": int(total - trainable),
        }

    def get_best_metrics(self) -> Optional[Dict[str, Any]]:
        if not self.history:
            return None
        return min(self.history, key=lambda m: m.get("val/mae", float("inf")))


# ═══════════════════════════════════════════════════════════════════════
# 4. ENSEMBLE TRAINER
# ═══════════════════════════════════════════════════════════════════════

class EnsembleTrainer:
    """
    Train each member of a TPNOEnsemble independently.
    """

    def __init__(
        self,
        ensemble: nn.Module,
        train_loader: Any,
        val_loader: Any,
        config: Union[TrainConfig, Dict[str, Any]],
        bootstrap: bool = True,
        device: Union[str, torch.device] = "cuda",
    ):
        self.ensemble = ensemble
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = (
            config
            if isinstance(config, TrainConfig)
            else TrainConfig(
                **{
                    k: v
                    for k, v in config.items()
                    if k in TrainConfig.__dataclass_fields__
                }
            )
        )
        self.bootstrap = bootstrap
        self.device = device

    def fit(self) -> List[List[Dict[str, Any]]]:
        """
        Train every ensemble member and return one history per member.
        """
        all_histories = []
        n_models = len(self.ensemble.models)

        for idx, member in enumerate(self.ensemble.models):
            logger.info("═══ Training ensemble member %d/%d ═══", idx + 1, n_models)

            loader = self._maybe_bootstrap(idx)

            member_cfg = TrainConfig(**vars(self.config))
            member_cfg.checkpoint_dir = str(
                Path(self.config.checkpoint_dir) / f"member_{idx}"
            )
            if self.config.wandb_run_name:
                member_cfg.wandb_run_name = f"{self.config.wandb_run_name}_m{idx}"

            trainer = TPNOTrainer(
                model=member,
                train_loader=loader,
                val_loader=self.val_loader,
                config=member_cfg,
                device=self.device,
            )
            history = trainer.fit()
            all_histories.append(history)

        return all_histories

    def _maybe_bootstrap(self, seed_offset: int):
        """
        Return a bootstrapped data loader or the original loader.
        """
        if not self.bootstrap:
            return self.train_loader

        from torch.utils.data import DataLoader, Subset

        dataset = self.train_loader.dataset
        n = len(dataset)
        rng = np.random.RandomState(self.config.n_epochs + seed_offset)
        indices = rng.choice(n, size=n, replace=True).tolist()

        subset = Subset(dataset, indices)
        return DataLoader(
            subset,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            num_workers=getattr(self.train_loader, "num_workers", 0),
            pin_memory=getattr(self.train_loader, "pin_memory", False),
            collate_fn=getattr(self.train_loader, "collate_fn", None),
        )


# ═══════════════════════════════════════════════════════════════════════
# 5. PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "TrainConfig",
    "TrainerConfig",
    "TPNOTrainer",
    "EnsembleTrainer",
]