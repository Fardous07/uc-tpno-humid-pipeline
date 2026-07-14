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

Fixes vs previous version
──────────────────────────
1. ``from .constants import PIPELINE_NAME, PIPELINE_VERSION`` crashed at
   import because constants.py contains physics constants only — those
   names were never defined.  Fixed: define them as module-level strings
   here so the module is self-contained.

2. MetricLogger.log() mutated the caller's metrics dict when prefix=""
   (the default).  ``metrics["epoch"] = step`` and
   ``metrics.setdefault("timestamp", …)`` modified the original object.
   Fixed: always copy() before any mutation.

3. CSVMetricSink._rewrite_with_new_columns() imported pandas for the
   rare new-column case.  If pandas is not installed (lightweight
   analysis scripts) this raised ImportError.  Fixed: rewrite using
   stdlib csv reader/writer only.

4. MetricTracker.best only iterated history[0].keys(), silently omitting
   metrics that first appeared after epoch 0.  Fixed: union all keys.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

# ── FIX 1: define pipeline identity here, not imported from constants ──
# constants.py holds physics constants (R, K_B, …) — it never had these.
PIPELINE_NAME: str    = "uc_tpno"
PIPELINE_VERSION: str = "0.1.0"

PathLike = Union[str, Path]


# ═══════════════════════════════════════════════════════════════════════
# 1.  STRUCTURED PYTHON LOGGING
# ═══════════════════════════════════════════════════════════════════════

_DEFAULT_FMT  = "%(asctime)s │ %(levelname)-8s │ %(name)-24s │ %(message)s"
_DEFAULT_DATE = "%Y-%m-%d %H:%M:%S"

_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
    "RESET":    "\033[0m",
}

_USE_COLOUR: bool = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


class _ColourFormatter(logging.Formatter):
    """Formatter that adds ANSI colours to the level name."""

    def __init__(self, fmt: str = _DEFAULT_FMT, datefmt: str = _DEFAULT_DATE):
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        if _USE_COLOUR:
            colour = _COLOURS.get(record.levelname, "")
            reset  = _COLOURS["RESET"]
            record = logging.makeLogRecord(record.__dict__)  # don't mutate original
            record.levelname = f"{colour}{record.levelname}{reset}"
        return super().format(record)


def setup_logger(
    name: str = PIPELINE_NAME,
    level: Union[int, str] = logging.INFO,
    log_file: Optional[PathLike] = None,
    max_bytes: int = 10 * 1024 * 1024,
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
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        ch.setFormatter(_ColourFormatter())
        logger.addHandler(ch)

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
    >>> avg = tracker.average()
    >>> tracker.reset()                # call at epoch start
    """

    def __init__(self) -> None:
        self._sums:   Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int]   = defaultdict(int)
        self.history: List[Dict[str, float]] = []

    def update(self, n: int = 1, **metrics: float) -> None:
        """
        Add one batch of metric values.

        Parameters
        ----------
        n        : Number of samples (for weighted averaging).
        **metrics: Keyword metric values, e.g. ``loss=0.3, mae=0.12``.
        """
        for k, v in metrics.items():
            if v is not None and np.isfinite(float(v)):
                self._sums[k]   += float(v) * n
                self._counts[k] += n

    def average(self) -> Dict[str, float]:
        """Return the running average of all tracked metrics."""
        return {
            k: self._sums[k] / max(self._counts[k], 1)
            for k in sorted(self._sums)
        }

    def sum(self) -> Dict[str, float]:
        return dict(self._sums)

    def reset(self, *, commit: bool = True) -> Dict[str, float]:
        """
        Reset accumulators.  If *commit* (default True), append current
        averages to ``self.history`` before clearing.
        """
        avg = self.average()
        if commit and self._counts:
            self.history.append(avg)
        self._sums.clear()
        self._counts.clear()
        return avg

    @property
    def last(self) -> Dict[str, float]:
        """Most recent committed epoch averages."""
        return self.history[-1] if self.history else {}

    @property
    def best(self) -> Dict[str, Optional[float]]:
        """
        Best (minimum) value seen for each metric across all epochs.

        FIX: previous version only iterated history[0].keys(), which
        silently omitted metrics that first appeared after epoch 0
        (e.g. physics losses that activate after warmup).
        Now unions all keys across the full history.
        """
        if not self.history:
            return {}
        all_keys = set()
        for h in self.history:
            all_keys.update(h.keys())
        return {
            k: min(h.get(k, float("inf")) for h in self.history)
            for k in all_keys
        }

    def best_epoch(self, metric: str, mode: str = "min") -> int:
        """Return the 0-indexed epoch with the best value of *metric*."""
        sentinel = float("inf") if mode == "min" else float("-inf")
        vals = [h.get(metric, sentinel) for h in self.history]
        if mode == "min":
            return int(np.argmin(vals))
        return int(np.argmax(vals))

    def get_history_array(self, metric: str) -> np.ndarray:
        """Return the per-epoch history of a single metric as a NumPy array."""
        return np.array([h.get(metric, np.nan) for h in self.history])

    def __len__(self) -> int:
        return len(self.history)

    def __repr__(self) -> str:
        cur   = self.average()
        items = ", ".join(f"{k}={v:.4f}" for k, v in cur.items())
        return f"MetricTracker(epochs={len(self.history)}, current=[{items}])"


# ═══════════════════════════════════════════════════════════════════════
# 3.  CSV & JSONL METRIC SINKS
# ═══════════════════════════════════════════════════════════════════════

class CSVMetricSink:
    """
    Append-only CSV writer for per-epoch metrics.

    Creates the file and writes the header row on first call to
    :meth:`write`.  Subsequent calls append rows.  If new metric keys
    appear mid-training the file is rewritten using stdlib ``csv`` only
    (no pandas dependency).
    """

    def __init__(self, path: PathLike) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._columns: Optional[List[str]] = None
        self._rows_cache: List[Dict[str, Any]] = []   # for rewrite
        self._file = None
        self._writer = None

    def write(self, metrics: Dict[str, Any]) -> None:
        """Append a row of metrics, rewriting header if new keys appear."""
        row = {k: f"{v:.6g}" if isinstance(v, float) else v
               for k, v in metrics.items()}

        if self._columns is None:
            # First write — open file and write header
            self._columns = sorted(metrics.keys())
            self._open_append()
            self._writer.writeheader()

        new_keys = sorted(set(metrics.keys()) - set(self._columns))
        if new_keys:
            # FIX: rewrite using stdlib csv, not pandas
            self._columns = sorted(set(self._columns) | set(new_keys))
            self.close()
            self._rewrite_with_new_columns()

        self._rows_cache.append(row)
        self._writer.writerow(row)
        self._file.flush()

    def _open_append(self) -> None:
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self._columns,
            extrasaction="ignore",
        )

    def _rewrite_with_new_columns(self) -> None:
        """
        Rewrite the CSV with the updated column set.

        FIX: uses stdlib csv only — previous version imported pandas,
        which crashed with ImportError in lightweight environments.
        Falls back to the in-memory row cache if the file does not exist.
        """
        # Read existing file rows (if any) into memory
        existing_rows: List[Dict[str, str]] = []
        if self.path.exists() and self.path.stat().st_size > 0:
            try:
                with open(self.path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    existing_rows = list(reader)
            except Exception:
                existing_rows = []

        # If file was unreadable, use in-memory cache
        if not existing_rows and self._rows_cache:
            existing_rows = [
                {k: str(v) for k, v in r.items()} for r in self._rows_cache
            ]

        # Rewrite the file with all columns
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self._columns,
                extrasaction="ignore",
            )
            writer.writeheader()
            for r in existing_rows:
                writer.writerow(r)

        # Re-open in append mode (header already written)
        self._open_append()

    def close(self) -> None:
        f = getattr(self, "_file", None)
        if f is not None and not f.closed:
            f.close()

    def __del__(self) -> None:
        self.close()


class JSONLMetricSink:
    """
    Append-only JSON-lines writer.  Each call to :meth:`write` appends
    one JSON object per line.  Tolerates new keys without header issues.
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
# 4.  W&B LOGGER
# ═══════════════════════════════════════════════════════════════════════

class WandbLogger:
    """
    Thin wrapper around W&B that silently no-ops when ``wandb`` is not
    installed or when the logger is disabled.
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
        self._run    = None

        if not enabled:
            return

        try:
            import wandb

            self._wandb = wandb
            self._run   = wandb.init(
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

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        if not self.enabled or self._run is None:
            return
        self._wandb.log(metrics, step=step, commit=commit)

    def log_summary(self, metrics: Dict[str, Any]) -> None:
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
        if not self.enabled or self._run is None:
            return
        artifact = self._wandb.Artifact(name, type=type, metadata=metadata)
        artifact.add_file(str(path))
        self._run.log_artifact(artifact)

    def watch(
        self,
        model: Any,
        log_gradients: str = "gradients",
        log_freq: int = 100,
    ) -> None:
        """Watch a model for gradient / weight logging."""
        if not self.enabled or self._run is None:
            return
        self._wandb.watch(model, log=log_gradients, log_freq=log_freq)

    def finish(self) -> None:
        if not self.enabled or self._run is None:
            return
        self._wandb.finish()
        self._run = None

    @property
    def run_id(self) -> Optional[str]:
        return self._run.id if self._run is not None else None

    @property
    def run_url(self) -> Optional[str]:
        return self._run.get_url() if self._run is not None else None


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
        self.log_dir    = Path(log_dir)
        self.experiment = experiment

        self.logger = setup_logger(
            name=f"{PIPELINE_NAME}.{experiment}",
            level=log_level,
            log_file=self.log_dir / f"{experiment}.log",
            console=console,
        )

        self.csv_sink   = CSVMetricSink(self.log_dir / f"{experiment}_metrics.csv")
        self.jsonl_sink = JSONLMetricSink(self.log_dir / f"{experiment}_metrics.jsonl")

        wb_kw       = wandb_kwargs or {}
        self.wandb  = WandbLogger(enabled=use_wandb, **wb_kw)

        self._epoch_start: Optional[float] = None
        self._train_start: Optional[float] = None

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """
        Log a dict of metrics to all sinks.

        FIX: always works on a *copy* of metrics so the caller's dict is
        never mutated.  Previous version did ``metrics["epoch"] = step``
        and ``metrics.setdefault("timestamp", …)`` directly on the
        original when prefix was empty.
        """
        # Always copy to avoid mutating caller's dict
        m: Dict[str, Any] = (
            {f"{prefix}{k}": v for k, v in metrics.items()} if prefix
            else dict(metrics)
        )

        if step is not None and "step" not in m and "epoch" not in m:
            m["epoch"] = step

        m.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

        self.csv_sink.write(m)
        self.jsonl_sink.write(m)
        self.wandb.log(m, step=step)

        summary = self._format_metrics(m)
        self.logger.info(summary)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Convenience: merge train / val metrics and log once."""
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

        if self._epoch_start is not None:
            combined["epoch_time_s"] = round(time.time() - self._epoch_start, 2)
        if self._train_start is not None:
            combined["elapsed_s"] = round(time.time() - self._train_start, 2)

        self.log(combined, step=epoch)

    def start_training(self) -> None:
        self._train_start = time.time()
        self.logger.info(
            "Training started — %s v%s — experiment: %s",
            PIPELINE_NAME, PIPELINE_VERSION, self.experiment,
        )

    def start_epoch(self) -> None:
        self._epoch_start = time.time()

    def eta(self, current_epoch: int, total_epochs: int) -> str:
        if self._train_start is None or current_epoch == 0:
            return "N/A"
        elapsed   = time.time() - self._train_start
        per_epoch = elapsed / current_epoch
        remaining = per_epoch * (total_epochs - current_epoch)
        return _format_seconds(remaining)

    def finish(self, final_metrics: Optional[Dict[str, Any]] = None) -> None:
        if final_metrics is not None:
            self.log(final_metrics, prefix="final/")
            self.wandb.log_summary(final_metrics)

        self.csv_sink.close()
        self.jsonl_sink.close()
        self.wandb.finish()

        self.logger.info("Logging finalised for experiment '%s'.", self.experiment)

    @staticmethod
    def _format_metrics(metrics: Dict[str, Any]) -> str:
        parts = []
        for k, v in sorted(metrics.items()):
            if k == "timestamp":
                continue
            if isinstance(v, float):
                parts.append(f"{k}={v:.2e}" if (abs(v) < 0.001 and v != 0) else f"{k}={v:.4f}")
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

        Epoch  12/100 │ loss=0.0342 │ val/mae=0.118 │ lr=3.0e-04 │ ETA 1h 23m 04s
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
    Log model architecture and parameter count.

    Uses ``torchinfo`` if available; falls back to a simple count.
    """
    if logger is None:
        logger = get_logger("model")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    lines = [
        f"Model: {model.__class__.__name__}",
        f"  Parameters — total: {total:,}  trainable: {trainable:,}  frozen: {frozen:,}",
    ]

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
    """Log an exception with full traceback at ERROR level."""
    if logger is None:
        logger = get_logger("exception")
    msg = f"Exception in {context}: " if context else "Exception: "
    logger.error(msg + str(exc), exc_info=True)


# ═══════════════════════════════════════════════════════════════════════
# 9.  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    # Identity
    "PIPELINE_NAME",
    "PIPELINE_VERSION",
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