"""
Logging, metric tracking, and experiment monitoring for the UC-TPNO pipeline.

This module provides a unified logging and metric-tracking stack that
every other module uses.  It removes the ad-hoc ``print()`` / ``wandb.log()``
calls scattered through the original codebase and replaces them with a
single, composable API:

    1.  **Structured Python logging** — ``setup_logger`` configures
        console + rotating file handlers with a consistent format across
        every module.
    2.  **MetricTracker** — accumulates per-batch scalars (loss, MAE, …),
        computes running/epoch averages, and provides history for
        learning-curve plots.
    3.  **MetricLogger** — fan-out sink that simultaneously writes to
        the Python logger, CSV file, JSON-lines file, and (optionally)
        Weights & Biases.
    4.  **Training progress helpers** — epoch summary formatting, ETA
        estimation, and a thin ``tqdm``-compatible progress wrapper.
    5.  **W&B convenience layer** — lazy init, safe ``log`` / ``finish``
        calls that silently no-op if W&B is disabled or not installed.
    6.  **Console formatters** — colour-coded severity levels, aligned
        metric columns, and optional ``rich`` integration.

Design goals
────────────
* Zero-dependency path: the module works with nothing but the stdlib.
  ``wandb``, ``rich``, and ``tqdm`` are imported lazily and their
  absence is handled gracefully.
* Every function is safe to call from any process (main or DataLoader
  worker) — file handlers use append mode and W&B calls are guarded.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from .constants import PIPELINE_NAME, PIPELINE_VERSION

PathLike = Union[str, Path]

# ═══════════════════════════════════════════════════════════════════════
# 1.  STRUCTURED PYTHON LOGGING
# ═══════════════════════════════════════════════════════════════════════

_DEFAULT_FMT = "%(asctime)s │ %(levelname)-8s │ %(name)-24s │ %(message)s"
_DEFAULT_DATE = "%Y-%m-%d %H:%M:%S"

# ANSI colour codes for console output (no-op on Windows unless
# coloured-terminal support is detected).
_COLOURS = {
    "DEBUG": "\033[36m",    # cyan
    "INFO": "\033[32m",     # green
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",    # red
    "CRITICAL": "\033[35m", # magenta
    "RESET": "\033[0m",
}

_USE_COLOUR: bool = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


class _ColourFormatter(logging.Formatter):
    """Formatter that adds ANSI colours to the level name."""

    def __init__(self, fmt: str = _DEFAULT_FMT, datefmt: str = _DEFAULT_DATE):
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        if _USE_COLOUR:
            colour = _COLOURS.get(record.levelname, "")
            reset = _COLOURS["RESET"]
            record.levelname = f"{colour}{record.levelname}{reset}"
        return super().format(record)


def setup_logger(
    name: str = PIPELINE_NAME,
    level: Union[int, str] = logging.INFO,
    log_file: Optional[PathLike] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MiB
    backup_count: int = 3,
    console: bool = True,
    propagate: bool = False,
) -> logging.Logger:
    """
    Configure and return a logger with console and optional file output.

    Parameters
    ----------
    name         : Logger name (typically the pipeline or module name).
    level        : Logging level (``DEBUG``, ``INFO``, ``WARNING``, …).
    log_file     : If given, write logs to this file with rotation.
    max_bytes    : Maximum file size before rotation (default 10 MiB).
    backup_count : Number of rotated backup files to keep.
    console      : Attach a coloured ``StreamHandler`` to stderr.
    propagate    : Whether to propagate to parent loggers.

    Returns
    -------
    ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # ── Console handler ──────────────────────────────────────────
    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        ch.setFormatter(_ColourFormatter())
        logger.addHandler(ch)

    # ── File handler (rotating) ──────────────────────────────────
    if log_file is not None:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            str(p),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(_DEFAULT_FMT, _DEFAULT_DATE))
        logger.addHandler(fh)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the pipeline root.

    Equivalent to ``logging.getLogger(f'{PIPELINE_NAME}.{name}')``.
    Ensures the root logger has been set up at least once.
    """
    root = logging.getLogger(PIPELINE_NAME)
    if not root.handlers:
        setup_logger()
    return logging.getLogger(f"{PIPELINE_NAME}.{name}")


# ═══════════════════════════════════════════════════════════════════════
# 2.  METRIC TRACKER  (per-epoch accumulator)
# ═══════════════════════════════════════════════════════════════════════

class MetricTracker:
    """
    Lightweight accumulator for training / validation scalars.

    Usage
    ─────
    >>> tracker = MetricTracker()
    >>> for batch in dataloader:
    ...     tracker.update(loss=0.5, mae=0.1)
    >>> avg = tracker.average()        # {'loss': …, 'mae': …}
    >>> tracker.reset()                # call at epoch start

    The ``history`` attribute stores a list of epoch-averaged dicts
    for learning-curve plotting.
    """

    def __init__(self) -> None:
        self._sums: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
        self.history: List[Dict[str, float]] = []

    # ── core API ─────────────────────────────────────────────────

    def update(self, n: int = 1, **metrics: float) -> None:
        """
        Add one batch of metric values.

        Parameters
        ----------
        n        : Number of samples this batch represents (for
                   weighted averaging).
        **metrics: Keyword metric values, e.g. ``loss=0.3, mae=0.12``.
        """
        for k, v in metrics.items():
            if v is not None and np.isfinite(v):
                self._sums[k] += float(v) * n
                self._counts[k] += n

    def average(self) -> Dict[str, float]:
        """Return the running average of all tracked metrics."""
        return {
            k: self._sums[k] / max(self._counts[k], 1)
            for k in sorted(self._sums)
        }

    def sum(self) -> Dict[str, float]:
        """Return the running sum of all tracked metrics."""
        return dict(self._sums)

    def reset(self, *, commit: bool = True) -> Dict[str, float]:
        """
        Reset accumulators.  If *commit* is True (default), append
        the current averages to ``self.history`` before clearing.

        Returns the averages from the completed epoch.
        """
        avg = self.average()
        if commit and self._counts:
            self.history.append(avg)
        self._sums.clear()
        self._counts.clear()
        return avg

    # ── convenience ──────────────────────────────────────────────

    @property
    def last(self) -> Dict[str, float]:
        """Most recent committed epoch averages."""
        return self.history[-1] if self.history else {}

    @property
    def best(self) -> Dict[str, Optional[float]]:
        """Best (minimum) value seen for each metric across epochs."""
        if not self.history:
            return {}
        keys = self.history[0].keys()
        return {k: min(h.get(k, float("inf")) for h in self.history) for k in keys}

    def best_epoch(self, metric: str, mode: str = "min") -> int:
        """Return the 0-indexed epoch with the best value of *metric*."""
        vals = [h.get(metric, float("inf") if mode == "min" else float("-inf"))
                for h in self.history]
        if mode == "min":
            return int(np.argmin(vals))
        return int(np.argmax(vals))

    def get_history_array(self, metric: str) -> np.ndarray:
        """Return the per-epoch history of a single metric as a NumPy array."""
        return np.array([h.get(metric, np.nan) for h in self.history])

    def __len__(self) -> int:
        return len(self.history)

    def __repr__(self) -> str:
        cur = self.average()
        items = ", ".join(f"{k}={v:.4f}" for k, v in cur.items())
        return f"MetricTracker(epochs={len(self.history)}, current=[{items}])"


# ═══════════════════════════════════════════════════════════════════════
# 3.  CSV & JSONL METRIC SINKS
# ═══════════════════════════════════════════════════════════════════════

class CSVMetricSink:
    """
    Append-only CSV writer for per-epoch metrics.

    Creates the file and writes the header row on first call to
    :meth:`write`.  Subsequent calls append rows, adding new columns
    as they appear.
    """

    def __init__(self, path: PathLike) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._columns: Optional[List[str]] = None
        self._file = None
        self._writer = None

    def write(self, metrics: Dict[str, Any]) -> None:
        """Append a row of metrics."""
        if self._columns is None:
            # First write — create file and header
            self._columns = sorted(metrics.keys())
            self._file = open(self.path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._file, fieldnames=self._columns, extrasaction="ignore"
            )
            self._writer.writeheader()

        # Handle new columns by rewriting header (rare edge case)
        new_keys = set(metrics.keys()) - set(self._columns)
        if new_keys:
            self._columns = sorted(set(self._columns) | new_keys)
            self.close()
            # Re-open and rewrite
            self._rewrite_with_new_columns()

        self._writer.writerow(
            {k: f"{v:.6g}" if isinstance(v, float) else v for k, v in metrics.items()}
        )
        self._file.flush()

    def _rewrite_with_new_columns(self) -> None:
        """Rewrite the CSV with updated column set."""
        import pandas as pd

        df = pd.read_csv(self.path) if self.path.exists() else pd.DataFrame()
        for col in self._columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self._columns]
        df.to_csv(self.path, index=False)
        # Re-open for appending
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file, fieldnames=self._columns, extrasaction="ignore"
        )

    def close(self) -> None:
        f = getattr(self, "_file", None)
        if f is not None and not f.closed:
            f.close()

    def __del__(self) -> None:
        self.close()


class JSONLMetricSink:
    """
    Append-only JSON-lines writer.  Each call to :meth:`write` appends
    one JSON object per line.  This format is easy to parse incrementally
    and tolerates new keys without header issues.
    """

    def __init__(self, path: PathLike) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")

    def write(self, metrics: Dict[str, Any]) -> None:
        row = {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in metrics.items()
        }
        self._file.write(json.dumps(row, default=str) + "\n")
        self._file.flush()

    def read_all(self) -> List[Dict[str, Any]]:
        """Read all rows back as a list of dicts."""
        rows = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def close(self) -> None:
        f = getattr(self, "_file", None)
        if f is not None and not f.closed:
            f.close()

    def __del__(self) -> None:
        self.close()


# ═══════════════════════════════════════════════════════════════════════

class WandbLogger:
    """
    Thin wrapper around W&B that silently no-ops when ``wandb`` is not
    installed or when the logger is disabled.

    Usage
    ─────
    >>> wb = WandbLogger(enabled=True, project='tpno-mof', config=cfg)
    >>> wb.log({'loss': 0.5}, step=100)
    >>> wb.log_artifact('checkpoints/best.pt', name='best-model', type='model')
    >>> wb.finish()
    """

    def __init__(
        self,
        enabled: bool = False,
        project: str = "tpno-mof",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        group: Optional[str] = None,
        resume: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self._run = None

        if not enabled:
            return

        try:
            import wandb

            self._wandb = wandb
            self._run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config or {},
                tags=tags,
                group=group,
                resume=resume,
                reinit=True,
            )
        except ImportError:
            logging.getLogger(__name__).warning(
                "wandb not installed — disabling W&B logging."
            )
            self.enabled = False
        except Exception as e:
            logging.getLogger(__name__).warning(
                "wandb.init() failed: %s — disabling W&B logging.", e
            )
            self.enabled = False

    # ── core methods ─────────────────────────────────────────────

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """Log metrics to W&B."""
        if not self.enabled or self._run is None:
            return
        self._wandb.log(metrics, step=step, commit=commit)

    def log_summary(self, metrics: Dict[str, Any]) -> None:
        """Write to the W&B run summary (final metrics)."""
        if not self.enabled or self._run is None:
            return
        for k, v in metrics.items():
            self._run.summary[k] = v

    def log_artifact(
        self,
        path: PathLike,
        name: str,
        type: str = "model",
        metadata: Optional[Dict] = None,
    ) -> None:
        """Log a file as a W&B artifact."""
        if not self.enabled or self._run is None:
            return
        artifact = self._wandb.Artifact(name, type=type, metadata=metadata)
        artifact.add_file(str(path))
        self._run.log_artifact(artifact)

    def watch(self, model: Any, log: str = "gradients", log_freq: int = 100) -> None:
        """Watch a model for gradient / weight logging."""
        if not self.enabled or self._run is None:
            return
        self._wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self) -> None:
        """Finish the W&B run."""
        if not self.enabled or self._run is None:
            return
        self._wandb.finish()
        self._run = None

    @property
    def run_id(self) -> Optional[str]:
        if self._run is not None:
            return self._run.id
        return None

    @property
    def run_url(self) -> Optional[str]:
        if self._run is not None:
            return self._run.get_url()
        return None


# ═══════════════════════════════════════════════════════════════════════
# 5.  METRIC LOGGER  (unified fan-out)
# ═══════════════════════════════════════════════════════════════════════

class MetricLogger:
    """
    Unified logging sink that fans out metrics to:

    * Python ``logging.Logger`` (human-readable summary)
    * CSV file  (tabular, easy to load in pandas)
    * JSON-lines file  (structured, schema-flexible)
    * Weights & Biases  (interactive dashboards)

    The trainer calls ``logger.log(metrics, step=epoch)`` once per
    epoch and this class handles the rest.

    Parameters
    ----------
    log_dir      : Directory for CSV / JSONL / log files.
    experiment   : Experiment name (used for file naming).
    use_wandb    : Enable W&B integration.
    wandb_kwargs : Extra keyword arguments passed to ``WandbLogger``.
    console      : Print summary to console via Python logger.
    log_level    : Python logging level.
    """

    def __init__(
        self,
        log_dir: PathLike = "logs",
        experiment: str = "default",
        use_wandb: bool = False,
        wandb_kwargs: Optional[Dict[str, Any]] = None,
        console: bool = True,
        log_level: Union[int, str] = logging.INFO,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.experiment = experiment

        # ── Python logger ────────────────────────────────────────
        self.logger = setup_logger(
            name=f"{PIPELINE_NAME}.{experiment}",
            level=log_level,
            log_file=self.log_dir / f"{experiment}.log",
            console=console,
        )

        # ── File sinks ──────────────────────────────────────────
        self.csv_sink = CSVMetricSink(self.log_dir / f"{experiment}_metrics.csv")
        self.jsonl_sink = JSONLMetricSink(self.log_dir / f"{experiment}_metrics.jsonl")

        # ── W&B ─────────────────────────────────────────────────
        wb_kw = wandb_kwargs or {}
        self.wandb = WandbLogger(enabled=use_wandb, **wb_kw)

        # ── Timing ──────────────────────────────────────────────
        self._epoch_start: Optional[float] = None
        self._train_start: Optional[float] = None

    # ── core API ─────────────────────────────────────────────────

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """
        Log a dict of metrics to all sinks.

        Parameters
        ----------
        metrics : Metric name → value mapping.
        step    : Global step (typically epoch number).
        prefix  : Optional prefix to prepend to metric names
                  (e.g. ``'train/'``, ``'val/'``).
        """
        # Prefix keys if requested
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        # Add step if not already present
        if step is not None and "step" not in metrics and "epoch" not in metrics:
            metrics["epoch"] = step

        # Add timestamp
        metrics.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

        # ── Fan out ──────────────────────────────────────────────
        self.csv_sink.write(metrics)
        self.jsonl_sink.write(metrics)
        self.wandb.log(metrics, step=step)

        # ── Console summary ──────────────────────────────────────
        summary = self._format_metrics(metrics)
        self.logger.info(summary)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Convenience method for end-of-epoch logging.

        Merges train / val metrics with standard prefixes and logs once.
        """
        combined: Dict[str, Any] = {"epoch": epoch}

        for k, v in train_metrics.items():
            key = k if k.startswith("train/") else f"train/{k}"
            combined[key] = v

        if val_metrics is not None:
            for k, v in val_metrics.items():
                key = k if k.startswith("val/") else f"val/{k}"
                combined[key] = v

        if lr is not None:
            combined["lr"] = lr

        if extra is not None:
            combined.update(extra)

        # Timing
        if self._epoch_start is not None:
            combined["epoch_time_s"] = round(time.time() - self._epoch_start, 2)
        if self._train_start is not None:
            combined["elapsed_s"] = round(time.time() - self._train_start, 2)

        self.log(combined, step=epoch)

    # ── timing helpers ───────────────────────────────────────────

    def start_training(self) -> None:
        """Call at the beginning of the training loop."""
        self._train_start = time.time()
        self.logger.info(
            "Training started — %s v%s — experiment: %s",
            PIPELINE_NAME, PIPELINE_VERSION, self.experiment,
        )

    def start_epoch(self) -> None:
        """Call at the beginning of each epoch."""
        self._epoch_start = time.time()

    def eta(self, current_epoch: int, total_epochs: int) -> str:
        """Return a human-readable ETA string."""
        if self._train_start is None or current_epoch == 0:
            return "N/A"
        elapsed = time.time() - self._train_start
        per_epoch = elapsed / current_epoch
        remaining = per_epoch * (total_epochs - current_epoch)
        return _format_seconds(remaining)

    # ── cleanup ──────────────────────────────────────────────────

    def finish(self, final_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Finalise all sinks."""
        if final_metrics is not None:
            self.log(final_metrics, prefix="final/")
            self.wandb.log_summary(final_metrics)

        self.csv_sink.close()
        self.jsonl_sink.close()
        self.wandb.finish()

        self.logger.info("Logging finalised for experiment '%s'.", self.experiment)

    # ── formatting ───────────────────────────────────────────────

    @staticmethod
    def _format_metrics(metrics: Dict[str, Any]) -> str:
        """Format metrics dict into a human-readable one-liner."""
        parts = []
        for k, v in sorted(metrics.items()):
            if k in ("timestamp",):
                continue
            if isinstance(v, float):
                # Adaptive formatting
                if abs(v) < 0.001 and v != 0:
                    parts.append(f"{k}={v:.2e}")
                else:
                    parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        return " │ ".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# 6.  TRAINING PROGRESS HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _format_seconds(seconds: float) -> str:
    """Convert seconds to ``HH:MM:SS`` string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def epoch_summary(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_metrics: Optional[Dict[str, float]] = None,
    lr: Optional[float] = None,
    eta: Optional[str] = None,
) -> str:
    """
    Format a single-line epoch summary.

    Example output::

        Epoch  12/100 │ loss=0.0342 │ val/mae=0.118 │ val/rmse=0.245 │ lr=3.0e-04 │ ETA 1h 23m 04s
    """
    parts = [f"Epoch {epoch + 1:4d}/{total_epochs}"]
    parts.append(f"loss={train_loss:.4f}")

    if val_metrics:
        for k, v in sorted(val_metrics.items()):
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")

    if lr is not None:
        parts.append(f"lr={lr:.1e}")

    if eta is not None:
        parts.append(f"ETA {eta}")

    return " │ ".join(parts)


def log_model_summary(
    model: Any,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Log a summary of model architecture and parameter count.

    Uses ``torchinfo`` if available, otherwise falls back to a simple
    parameter count.
    """
    if logger is None:
        logger = get_logger("model")

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    lines = [
        f"Model: {model.__class__.__name__}",
        f"  Parameters — total: {total:,}  trainable: {trainable:,}  frozen: {frozen:,}",
    ]

    # Try torchinfo for a detailed table
    try:
        from torchinfo import summary as ti_summary

        info = ti_summary(model, verbose=0)
        lines.append(f"  Estimated size: {info.total_output_bytes / 1e6:.1f} MB")
    except (ImportError, Exception):
        pass

    summary_str = "\n".join(lines)
    logger.info(summary_str)
    return summary_str


# ═══════════════════════════════════════════════════════════════════════
# 7.  PROGRESS BAR WRAPPER
# ═══════════════════════════════════════════════════════════════════════

def progress_bar(
    iterable,
    desc: str = "",
    total: Optional[int] = None,
    disable: bool = False,
    **kwargs,
):
    """
    Thin wrapper around ``tqdm`` that falls back to a plain iterator
    if ``tqdm`` is not installed.

    Accepts the same keyword arguments as ``tqdm.tqdm``.
    """
    if disable:
        return iterable

    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, total=total, **kwargs)
    except ImportError:
        return iterable


@contextmanager
def log_phase(name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager that logs the start and end (with elapsed time)
    of a named phase.

    >>> with log_phase("Data loading"):
    ...     dataset = load_data()
    # INFO: ┌ Data loading
    # INFO: └ Data loading — 12.34 s
    """
    if logger is None:
        logger = get_logger("phase")

    logger.info("┌ %s", name)
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    logger.info("└ %s — %s", name, _format_seconds(elapsed))


# ═══════════════════════════════════════════════════════════════════════
# 8.  EXCEPTION LOGGING
# ═══════════════════════════════════════════════════════════════════════

def log_exception(
    exc: BaseException,
    logger: Optional[logging.Logger] = None,
    context: str = "",
) -> None:
    """
    Log an exception with traceback at ERROR level.

    Useful in ``except`` blocks to ensure exceptions are recorded in
    log files even if they are caught and handled.
    """
    if logger is None:
        logger = get_logger("exception")
    msg = f"Exception in {context}: " if context else "Exception: "
    logger.error(msg + str(exc), exc_info=True)


# ═══════════════════════════════════════════════════════════════════════
# 9.  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    # Logger setup
    "setup_logger",
    "get_logger",
    # Metric tracking
    "MetricTracker",
    # Metric sinks
    "CSVMetricSink",
    "JSONLMetricSink",
    # W&B
    "WandbLogger",
    # Unified logger
    "MetricLogger",
    # Training helpers
    "epoch_summary",
    "log_model_summary",
    # Progress
    "progress_bar",
    "log_phase",
    # Exception
    "log_exception",
]