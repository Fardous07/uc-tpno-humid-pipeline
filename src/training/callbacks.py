#!/usr/bin/env python3
"""
Composable callback system for the UC-TPNO training loop.

This version is aligned with the repaired trainer API and fixes:
1. Metric-name drift (`val/mae` instead of old `val_loss` style).
2. Trainer attribute drift (`trainer.config` vs old `trainer.cfg`).
3. Checkpoint format compatibility with `trainer.load_checkpoint(...)`.
4. Safer callback dispatch (one broken callback does not crash all others).
5. Timing callback compatibility when only `on_epoch_end(...)` is called.
6. Cleaner CSV / JSON-safe metric logging.
"""

from __future__ import annotations

import csv
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1. BASE CALLBACK
# ═══════════════════════════════════════════════════════════════════════

class Callback:
    """
    Base class for training callbacks.

    All hooks are optional no-ops.
    """

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass

    def on_epoch_begin(self, epoch: int, trainer: Any) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], trainer: Any) -> None:
        pass

    def on_batch_begin(self, batch_idx: int, trainer: Any) -> None:
        pass

    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any) -> None:
        pass

    @property
    def should_stop(self) -> bool:
        return False


# ═══════════════════════════════════════════════════════════════════════
# 2. CALLBACK RUNNER
# ═══════════════════════════════════════════════════════════════════════

class CallbackRunner:
    """
    Aggregates multiple callbacks and dispatches hook calls.

    Also callable as:
        runner(epoch, metrics, trainer)
    """

    def __init__(self, callbacks: Optional[Sequence[Callback]] = None):
        self.callbacks: List[Callback] = list(callbacks or [])

    def add(self, cb: Callback) -> None:
        self.callbacks.append(cb)

    def _safe_call(self, cb: Callback, hook_name: str, *args, **kwargs) -> None:
        hook = getattr(cb, hook_name, None)
        if hook is None:
            return
        try:
            hook(*args, **kwargs)
        except Exception as e:
            logger.exception(
                "Callback %s failed in %s: %s",
                cb.__class__.__name__,
                hook_name,
                e,
            )

    def __call__(self, epoch: int, metrics: Dict[str, Any], trainer: Any = None) -> None:
        self.on_epoch_end(epoch, metrics, trainer)

    def on_train_begin(self, trainer: Any = None) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_train_begin", trainer)

    def on_train_end(self, trainer: Any = None) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_train_end", trainer)

    def on_epoch_begin(self, epoch: int, trainer: Any = None) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_epoch_begin", epoch, trainer)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], trainer: Any = None) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_epoch_end", epoch, metrics, trainer)

    def on_batch_begin(self, batch_idx: int, trainer: Any = None) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_batch_begin", batch_idx, trainer)

    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any = None) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_batch_end", batch_idx, loss, trainer)

    @property
    def should_stop(self) -> bool:
        return any(cb.should_stop for cb in self.callbacks)


# ═══════════════════════════════════════════════════════════════════════
# 3. EARLY STOPPING
# ═══════════════════════════════════════════════════════════════════════

class EarlyStopping(Callback):
    """
    Stop training when a monitored metric stops improving.
    """

    def __init__(
        self,
        patience: int = 20,
        monitor: str = "val/mae",
        mode: str = "min",
        min_delta: float = 1e-6,
        restore_best: bool = True,
    ):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best = restore_best

        self._best = float("inf") if mode == "min" else -float("inf")
        self._wait = 0
        self._best_epoch = -1
        self._best_state: Optional[Dict[str, torch.Tensor]] = None
        self._stopped = False

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < self._best - self.min_delta
        return current > self._best + self.min_delta

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], trainer: Any = None) -> None:
        val = metrics.get(self.monitor)
        if val is None:
            return

        if self._is_improvement(float(val)):
            self._best = float(val)
            self._wait = 0
            self._best_epoch = epoch

            if self.restore_best and trainer is not None and hasattr(trainer, "model"):
                try:
                    self._best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in trainer.model.state_dict().items()
                    }
                except Exception:
                    self._best_state = None
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self._stopped = True
                logger.info(
                    "EarlyStopping triggered: no improvement in %s for %d epochs "
                    "(best=%.6f at epoch %d).",
                    self.monitor,
                    self.patience,
                    self._best,
                    self._best_epoch,
                )

    def on_train_end(self, trainer: Any = None) -> None:
        if self.restore_best and self._best_state is not None and trainer is not None:
            if hasattr(trainer, "model"):
                try:
                    trainer.model.load_state_dict(self._best_state)
                    logger.info("Restored best model from epoch %d.", self._best_epoch)
                except Exception as e:
                    logger.warning("Could not restore best model: %s", e)

    @property
    def should_stop(self) -> bool:
        return self._stopped

    @property
    def best_score(self) -> float:
        return self._best

    @property
    def best_epoch(self) -> int:
        return self._best_epoch


# ═══════════════════════════════════════════════════════════════════════
# 4. LR SCHEDULER CALLBACK
# ═══════════════════════════════════════════════════════════════════════

class LRScheduler(Callback):
    """
    Callback-managed learning-rate schedule.

    Note:
    -----
    The repaired trainer already has built-in LR scheduling. This callback
    is mainly useful for custom loops or older training entrypoints.
    """

    def __init__(
        self,
        scheduler_type: str = "cosine",
        T_max: int = 100,
        eta_min: float = 1e-6,
        step_size: int = 30,
        gamma: float = 0.1,
        monitor: str = "val/mae",
        plateau_patience: int = 10,
        plateau_factor: float = 0.5,
    ):
        self.scheduler_type = scheduler_type
        self.T_max = T_max
        self.eta_min = eta_min
        self.step_size = step_size
        self.gamma = gamma
        self.monitor = monitor
        self.plateau_patience = plateau_patience
        self.plateau_factor = plateau_factor

        self._base_lr: Optional[float] = None
        self._current_lr: Optional[float] = None
        self._plateau_best = float("inf")
        self._plateau_wait = 0

    def on_train_begin(self, trainer: Any = None) -> None:
        if trainer is not None and hasattr(trainer, "optimizer"):
            self._base_lr = float(trainer.optimizer.param_groups[0]["lr"])
            self._current_lr = self._base_lr

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], trainer: Any = None) -> None:
        if trainer is None or not hasattr(trainer, "optimizer"):
            return
        if self._base_lr is None:
            return

        new_lr = self._compute_lr(epoch, metrics)
        self._current_lr = float(new_lr)

        for pg in trainer.optimizer.param_groups:
            pg["lr"] = self._current_lr

    def _compute_lr(self, epoch: int, metrics: Dict[str, Any]) -> float:
        base = float(self._base_lr)

        if self.scheduler_type == "cosine":
            return self.eta_min + 0.5 * (base - self.eta_min) * (
                1.0 + math.cos(math.pi * epoch / max(self.T_max, 1))
            )

        if self.scheduler_type == "step":
            return base * (self.gamma ** (epoch // max(self.step_size, 1)))

        if self.scheduler_type == "exponential":
            return base * (self.gamma ** epoch)

        if self.scheduler_type == "one_cycle":
            warmup_end = int(0.3 * self.T_max)
            if epoch < warmup_end:
                return self.eta_min + (base - self.eta_min) * epoch / max(warmup_end, 1)
            progress = (epoch - warmup_end) / max(self.T_max - warmup_end, 1)
            return self.eta_min + 0.5 * (base - self.eta_min) * (1.0 + math.cos(math.pi * progress))

        if self.scheduler_type == "plateau":
            val = metrics.get(self.monitor)
            if val is not None:
                val = float(val)
                if val < self._plateau_best - 1e-6:
                    self._plateau_best = val
                    self._plateau_wait = 0
                else:
                    self._plateau_wait += 1
                    if self._plateau_wait >= self.plateau_patience:
                        self._plateau_wait = 0
                        current = self._current_lr or base
                        return max(current * self.plateau_factor, self.eta_min)
            return self._current_lr or base

        return base

    @property
    def current_lr(self) -> Optional[float]:
        return self._current_lr


# ═══════════════════════════════════════════════════════════════════════
# 5. WARMUP CALLBACK
# ═══════════════════════════════════════════════════════════════════════

class WarmupCallback(Callback):
    """
    Linear LR warmup for custom loops.

    Note:
    -----
    The repaired trainer already performs warmup internally, so this callback
    is mostly for compatibility with older loops.
    """

    def __init__(
        self,
        warmup_epochs: int = 5,
        peak_lr: float = 1e-3,
        start_lr: Optional[float] = None,
    ):
        self.warmup_epochs = warmup_epochs
        self.peak_lr = peak_lr
        self.start_lr = start_lr if start_lr is not None else peak_lr / 100.0

    def on_epoch_begin(self, epoch: int, trainer: Any = None) -> None:
        if epoch >= self.warmup_epochs:
            return
        if trainer is None or not hasattr(trainer, "optimizer"):
            return

        frac = epoch / max(self.warmup_epochs, 1)
        lr = self.start_lr + frac * (self.peak_lr - self.start_lr)
        for pg in trainer.optimizer.param_groups:
            pg["lr"] = float(lr)


# ═══════════════════════════════════════════════════════════════════════
# 6. MODEL CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════

class ModelCheckpoint(Callback):
    """
    Save model checkpoints based on a monitored metric.
    """

    def __init__(
        self,
        save_dir: str = "checkpoints",
        monitor: str = "val/mae",
        mode: str = "min",
        save_best: bool = True,
        save_freq: int = 0,
        max_kept: int = 5,
    ):
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_freq = save_freq
        self.max_kept = max_kept

        self._best = float("inf") if mode == "min" else -float("inf")
        self._saved_paths: List[Path] = []

    def _is_improvement(self, val: float) -> bool:
        if self.mode == "min":
            return val < self._best
        return val > self._best

    def on_train_begin(self, trainer: Any = None) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], trainer: Any = None) -> None:
        if trainer is None:
            return

        val = metrics.get(self.monitor)

        if self.save_best and val is not None and self._is_improvement(float(val)):
            self._best = float(val)
            path = self.save_dir / "best_model.pt"
            self._save(trainer, path, epoch, metrics)
            logger.info(
                "ModelCheckpoint: saved best model at epoch %d (%s=%.6f).",
                epoch,
                self.monitor,
                self._best,
            )

        if self.save_freq > 0 and (epoch + 1) % self.save_freq == 0:
            path = self.save_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            self._save(trainer, path, epoch, metrics)
            self._saved_paths.append(path)

            while len(self._saved_paths) > self.max_kept:
                old = self._saved_paths.pop(0)
                if old.exists():
                    old.unlink()

    def _save(self, trainer: Any, path: Path, epoch: int, metrics: Dict[str, Any]) -> None:
        try:
            state: Dict[str, Any] = {
                "epoch": epoch,
                "metrics": {
                    k: (float(v) if isinstance(v, (np.floating, float)) else v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float, str, np.floating))
                },
            }

            if hasattr(trainer, "global_step"):
                state["global_step"] = trainer.global_step
            if hasattr(trainer, "best_val_metric"):
                state["best_val_metric"] = trainer.best_val_metric
            if hasattr(trainer, "config"):
                state["config"] = vars(trainer.config)

            if hasattr(trainer, "model"):
                state["model_state_dict"] = trainer.model.state_dict()
            if hasattr(trainer, "optimizer"):
                state["optimizer_state_dict"] = trainer.optimizer.state_dict()
            if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
                state["scheduler_state_dict"] = trainer.scheduler.state_dict()
            if hasattr(trainer, "_scaler") and trainer._scaler is not None:
                state["scaler_state_dict"] = trainer._scaler.state_dict()

            torch.save(state, path)
        except Exception as e:
            logger.warning("Checkpoint save failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════
# 7. METRIC LOGGER
# ═══════════════════════════════════════════════════════════════════════

class MetricLogger(Callback):
    """
    Log metrics to console and optionally to CSV.
    """

    def __init__(
        self,
        log_every: int = 1,
        csv_path: Optional[str] = None,
        keys: Optional[List[str]] = None,
    ):
        self.log_every = log_every
        self.csv_path = Path(csv_path) if csv_path else None
        self.keys = keys
        self._header_written = False

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], trainer: Any = None) -> None:
        filtered = self._filter(metrics)

        if (epoch + 1) % max(self.log_every, 1) == 0:
            parts = [f"Epoch {epoch + 1:4d}"]
            for k, v in filtered.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.6f}")
                else:
                    parts.append(f"{k}={v}")
            logger.info("  │  ".join(parts))

        if self.csv_path is not None:
            self._write_csv(epoch, filtered)

    def _filter(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        if self.keys is None:
            return metrics
        return {k: v for k, v in metrics.items() if k in self.keys}

    def _write_csv(self, epoch: int, metrics: Dict[str, Any]) -> None:
        row = {"epoch": epoch, **metrics}
        write_header = not self._header_written or not self.csv_path.exists()

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)

    def on_train_end(self, trainer: Any = None) -> None:
        if self.csv_path is not None:
            logger.info("Metrics saved to %s", self.csv_path)


# ═══════════════════════════════════════════════════════════════════════
# 8. GRADIENT MONITOR
# ═══════════════════════════════════════════════════════════════════════

class GradientMonitor(Callback):
    """
    Monitor gradient statistics and warn about explosion/vanishing.
    """

    def __init__(
        self,
        clip_value: float = 10.0,
        vanish_threshold: float = 1e-7,
        log_every: int = 10,
    ):
        self.clip_value = clip_value
        self.vanish_threshold = vanish_threshold
        self.log_every = log_every
        self._batch_norms: List[float] = []
        self.history: List[Dict[str, float]] = []

    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any = None) -> None:
        if trainer is None or not hasattr(trainer, "model"):
            return

        try:
            total_norm_sq = 0.0
            for p in trainer.model.parameters():
                if p.grad is not None:
                    gnorm = float(p.grad.data.norm(2).item())
                    total_norm_sq += gnorm ** 2
            total_norm = total_norm_sq ** 0.5
            self._batch_norms.append(total_norm)
        except Exception:
            pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], trainer: Any = None) -> None:
        if not self._batch_norms:
            return

        norms = np.asarray(self._batch_norms, dtype=np.float64)
        stats = {
            "grad_norm_mean": float(norms.mean()),
            "grad_norm_max": float(norms.max()),
            "grad_norm_min": float(norms.min()),
        }
        self.history.append(stats)

        if (epoch + 1) % max(self.log_every, 1) == 0:
            logger.info(
                "Gradient stats epoch %d: mean=%.4f max=%.4f",
                epoch + 1,
                stats["grad_norm_mean"],
                stats["grad_norm_max"],
            )

        if stats["grad_norm_max"] > self.clip_value:
            logger.warning(
                "Exploding gradient detected at epoch %d: max_norm=%.4f > %.4f",
                epoch + 1,
                stats["grad_norm_max"],
                self.clip_value,
            )

        if stats["grad_norm_mean"] < self.vanish_threshold:
            logger.warning(
                "Vanishing gradient detected at epoch %d: mean_norm=%.2e < %.2e",
                epoch + 1,
                stats["grad_norm_mean"],
                self.vanish_threshold,
            )

        self._batch_norms.clear()


# ═══════════════════════════════════════════════════════════════════════
# 9. TIMING CALLBACK
# ═══════════════════════════════════════════════════════════════════════

class TimingCallback(Callback):
    """
    Track wall-clock time per epoch and estimate remaining time.

    Works both when `on_epoch_begin(...)` is called and when the trainer
    only provides `epoch_time_s` inside metrics.
    """

    def __init__(self):
        self._epoch_start: float = 0.0
        self._train_start: float = 0.0
        self._epoch_times: List[float] = []

    def on_train_begin(self, trainer: Any = None) -> None:
        self._train_start = time.time()

    def on_epoch_begin(self, epoch: int, trainer: Any = None) -> None:
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], trainer: Any = None) -> None:
        if "epoch_time_s" in metrics:
            elapsed = float(metrics["epoch_time_s"])
        elif self._epoch_start > 0:
            elapsed = time.time() - self._epoch_start
            metrics["epoch_time_s"] = elapsed
        else:
            elapsed = 0.0

        if elapsed > 0:
            self._epoch_times.append(elapsed)

        total_epochs = None
        if trainer is not None:
            if hasattr(trainer, "config"):
                total_epochs = getattr(trainer.config, "n_epochs", None)
            elif hasattr(trainer, "cfg"):
                total_epochs = getattr(trainer.cfg, "n_epochs", None)

        if total_epochs is not None and self._epoch_times:
            remaining = int(total_epochs) - epoch - 1
            if remaining > 0:
                avg = float(np.mean(self._epoch_times[-10:]))
                metrics["eta_s"] = avg * remaining

    @property
    def total_time(self) -> float:
        return time.time() - self._train_start if self._train_start else 0.0

    @property
    def mean_epoch_time(self) -> float:
        return float(np.mean(self._epoch_times)) if self._epoch_times else 0.0

    def on_train_end(self, trainer: Any = None) -> None:
        logger.info(
            "Training complete in %.1fs (%.2fs/epoch avg).",
            self.total_time,
            self.mean_epoch_time,
        )


# ═══════════════════════════════════════════════════════════════════════
# 10. LAMBDA CALLBACK
# ═══════════════════════════════════════════════════════════════════════

class LambdaCallback(Callback):
    """
    Wrap arbitrary hook functions as a callback.
    """

    def __init__(
        self,
        on_epoch_end_fn: Optional[Callable] = None,
        on_train_end_fn: Optional[Callable] = None,
        on_train_begin_fn: Optional[Callable] = None,
        on_batch_end_fn: Optional[Callable] = None,
    ):
        self._on_epoch_end = on_epoch_end_fn
        self._on_train_end = on_train_end_fn
        self._on_train_begin = on_train_begin_fn
        self._on_batch_end = on_batch_end_fn

    def on_train_begin(self, trainer: Any = None) -> None:
        if self._on_train_begin:
            self._on_train_begin(trainer)

    def on_train_end(self, trainer: Any = None) -> None:
        if self._on_train_end:
            self._on_train_end(trainer)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], trainer: Any = None) -> None:
        if self._on_epoch_end:
            self._on_epoch_end(epoch, metrics, trainer)

    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any = None) -> None:
        if self._on_batch_end:
            self._on_batch_end(batch_idx, loss, trainer)


# ═══════════════════════════════════════════════════════════════════════
# 11. DEFAULT STACK
# ═══════════════════════════════════════════════════════════════════════

def default_callbacks(
    patience: int = 20,
    monitor: str = "val/mae",
    log_every: int = 5,
    save_dir: str = "checkpoints",
    csv_path: Optional[str] = "metrics.csv",
    warmup_epochs: int = 5,
    peak_lr: float = 1e-3,
    scheduler: str = "cosine",
    T_max: int = 100,
) -> CallbackRunner:
    """
    Build a sensible default callback stack.

    Note:
    -----
    The repaired trainer already handles warmup / scheduling / early stopping.
    These callbacks are kept for compatibility and auxiliary logging/checkpointing.
    """
    return CallbackRunner(
        [
            EarlyStopping(patience=patience, monitor=monitor),
            WarmupCallback(warmup_epochs=warmup_epochs, peak_lr=peak_lr),
            LRScheduler(scheduler_type=scheduler, T_max=T_max, monitor=monitor),
            ModelCheckpoint(save_dir=save_dir, monitor=monitor),
            MetricLogger(log_every=log_every, csv_path=csv_path),
            GradientMonitor(),
            TimingCallback(),
        ]
    )


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "Callback",
    "CallbackRunner",
    "EarlyStopping",
    "LRScheduler",
    "WarmupCallback",
    "ModelCheckpoint",
    "MetricLogger",
    "GradientMonitor",
    "TimingCallback",
    "LambdaCallback",
    "default_callbacks",
]