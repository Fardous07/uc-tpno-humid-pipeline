"""
Reproducibility and experiment provenance utilities for the UC-TPNO pipeline.

Scientific computing requires bit-exact or statistically-equivalent
reproducibility across runs, machines, and time.  This module centralises
*every* source of non-determinism that affects the UC-TPNO training and
evaluation pipeline:

    1.  **RNG seeding** — Python, NumPy, PyTorch (CPU + all CUDA devices),
        and PYTHONHASHSEED in a single call.
    2.  **Deterministic algorithm enforcement** — cuDNN, CUBLAS, and the
        PyTorch global deterministic-algorithms flag.
    3.  **RNG state snapshots** — save / restore the complete RNG state
        (Python + NumPy + Torch CPU + Torch CUDA) so that a training run
        can be resumed from any checkpoint and produce identical results.
    4.  **DataLoader worker seeding** — a ``worker_init_fn`` that gives each
        DataLoader worker a deterministic but unique seed derived from the
        global seed + worker id + epoch.
    5.  **Environment fingerprinting** — capture Python version, package
        versions, CUDA toolkit, GPU model, OS, CPU count, hostname, and
        git revision into a single JSON-serialisable dict.
    6.  **Checkpoint integrity** — SHA-256 hashing of saved ``.pt`` files so
        that downstream consumers can verify that a checkpoint has not been
        corrupted or silently modified.
    7.  **Experiment manifest** — a single JSON file that records everything
        needed to reproduce a run: config, seed, environment fingerprint,
        git revision, data splits hash, and model architecture summary.
    8.  **Context managers** — ``ReproducibleBlock`` for scoped seeding of
        isolated sub-computations without contaminating the global RNG state.

Fixes vs previous version
──────────────────────────
1. ``from .constants import DEFAULT_SEED, PIPELINE_VERSION`` crashed at
   import because constants.py contains CODATA physics constants only —
   neither name was defined there.  Fixed: both are defined in this module.
   PIPELINE_VERSION = "0.1.0", DEFAULT_SEED = 42.

2. ExperimentManifest.load() did ``cls(**d)`` which raises TypeError if the
   JSON has fields unknown to the current dataclass (e.g., from an older
   or newer manifest format).  Fixed: filter to known fields before passing.

3. worker_init_fn() seeded ``random`` and ``np.random`` but not
   ``torch.random``.  Workers using torch augmentation got identical
   sequences across all workers.  Fixed: added ``torch.manual_seed``.

4. get_environment_fingerprint() used ``__import__(pkg)`` with PyPI names
   (e.g. "scikit-learn") where the import name differs ("sklearn").
   __import__("scikit-learn") raises ModuleNotFoundError.
   Fixed: explicit import-name mapping.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# FIX 1: define these here — they do NOT exist in constants.py (which
# contains CODATA physics constants only).
DEFAULT_SEED:     int = 42
PIPELINE_VERSION: str = "0.1.0"

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1.  GLOBAL SEEDING
# ═══════════════════════════════════════════════════════════════════════

_SEED_HISTORY: List[Tuple[int, str]] = []
"""Audit trail: each ``(seed, timestamp)`` pair set during the process."""


def set_seed(
    seed: int = DEFAULT_SEED,
    *,
    deterministic_algorithms: bool = True,
    warn_only: bool = True,
    benchmark: bool = False,
) -> None:
    """
    Set **all** random seeds and configure deterministic behaviour.

    After this call:
    * ``random``, ``numpy``, ``torch`` (CPU + CUDA) are seeded.
    * ``PYTHONHASHSEED`` is fixed.
    * cuDNN is put into deterministic mode.
    * ``torch.use_deterministic_algorithms(True)`` is activated if requested.
    * ``CUBLAS_WORKSPACE_CONFIG`` is set for deterministic cuBLAS.

    Parameters
    ----------
    seed                    : Master seed.
    deterministic_algorithms: Activate PyTorch deterministic-algorithms mode.
    warn_only               : Emit warning (not raise) on non-deterministic ops.
    benchmark               : Value for ``torch.backends.cudnn.benchmark``.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = benchmark

        if deterministic_algorithms and hasattr(torch, "use_deterministic_algorithms"):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=warn_only)

    except ImportError:
        logger.debug("PyTorch not available — skipping torch seeding.")

    ts = datetime.now(timezone.utc).isoformat()
    _SEED_HISTORY.append((seed, ts))
    logger.info(
        "Global seed set to %d  (deterministic=%s, benchmark=%s)",
        seed, deterministic_algorithms, benchmark,
    )


def get_seed_history() -> List[Tuple[int, str]]:
    """Return the list of ``(seed, utc_timestamp)`` values set so far."""
    return list(_SEED_HISTORY)


# ═══════════════════════════════════════════════════════════════════════
# 2.  RNG STATE SAVE / RESTORE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RNGState:
    """Snapshot of all RNG states needed for exact resumption."""

    python_state:        Any                  = None
    numpy_state:         Any                  = None
    torch_cpu_state:     Any                  = None
    torch_cuda_states:   Optional[List[Any]]  = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "python_state":      self.python_state,
            "numpy_state":       self.numpy_state,
            "torch_cpu_state":   self.torch_cpu_state,
            "torch_cuda_states": self.torch_cuda_states,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RNGState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def capture_rng_state() -> RNGState:
    """Capture a snapshot of **all** RNG states."""
    state              = RNGState()
    state.python_state = random.getstate()
    state.numpy_state  = np.random.get_state()

    try:
        import torch

        state.torch_cpu_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            state.torch_cuda_states = [
                torch.cuda.get_rng_state(device=i)
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass

    return state


def restore_rng_state(state: RNGState) -> None:
    """Restore all RNG states from a snapshot."""
    if state.python_state is not None:
        random.setstate(state.python_state)
    if state.numpy_state is not None:
        np.random.set_state(state.numpy_state)

    try:
        import torch

        if state.torch_cpu_state is not None:
            torch.random.set_rng_state(state.torch_cpu_state)
        if state.torch_cuda_states is not None and torch.cuda.is_available():
            for i, s in enumerate(state.torch_cuda_states):
                if i < torch.cuda.device_count():
                    torch.cuda.set_rng_state(s, device=i)
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════
# 3.  CONTEXT MANAGER FOR SCOPED REPRODUCIBLE BLOCKS
# ═══════════════════════════════════════════════════════════════════════

class ReproducibleBlock:
    """
    Context manager that seeds all RNGs on entry and restores the
    previous state on exit.

    Example
    -------
    >>> set_seed(42)
    >>> x = np.random.rand()          # advance global RNG
    >>> with ReproducibleBlock(seed=99):
    ...     y = np.random.rand()      # deterministic within block
    >>> z = np.random.rand()          # continues from *x*, not *y*
    """

    def __init__(self, seed: int) -> None:
        self.seed          = seed
        self._saved_state: Optional[RNGState] = None

    def __enter__(self) -> "ReproducibleBlock":
        self._saved_state = capture_rng_state()
        set_seed(self.seed, deterministic_algorithms=False)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._saved_state is not None:
            restore_rng_state(self._saved_state)


@contextmanager
def reproducible_block(seed: int):
    """Functional alternative to :class:`ReproducibleBlock`."""
    saved = capture_rng_state()
    set_seed(seed, deterministic_algorithms=False)
    try:
        yield
    finally:
        restore_rng_state(saved)


# ═══════════════════════════════════════════════════════════════════════
# 4.  DATALOADER WORKER SEEDING
# ═══════════════════════════════════════════════════════════════════════

def worker_init_fn(worker_id: int) -> None:
    """
    Seed each DataLoader worker deterministically.

    PyTorch forks workers with the *same* base seed.  Without explicit
    re-seeding, every worker produces the same augmentation sequence.
    This function computes a unique-but-deterministic seed from:

        worker_seed = torch.initial_seed() % 2**32 + worker_id

    FIX: now also calls ``torch.manual_seed()`` — previous version only
    seeded ``random`` and ``np.random``, so workers using torch
    augmentation (e.g. ``torch.randn``) got identical sequences.

    Pass to ``DataLoader(worker_init_fn=worker_init_fn)``.
    """
    try:
        import torch
        base = torch.initial_seed() % (2 ** 32)
    except ImportError:
        base = DEFAULT_SEED

    seed = (base + worker_id) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def make_worker_init_fn(base_seed: int, epoch: int = 0):
    """
    Factory that returns a ``worker_init_fn`` incorporating a specific
    base seed *and* epoch number.  Guarantees that workers in different
    epochs see different-but-reproducible augmentations.

    Usage::

        for epoch in range(n_epochs):
            loader = DataLoader(dataset,
                                worker_init_fn=make_worker_init_fn(42, epoch))
    """

    def _init(worker_id: int) -> None:
        seed = (base_seed + epoch * 1_000 + worker_id) % (2 ** 32)
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass

    return _init


# ═══════════════════════════════════════════════════════════════════════
# 5.  GIT REVISION
# ═══════════════════════════════════════════════════════════════════════

def get_git_revision() -> str:
    """Return the current ``HEAD`` commit hash, or ``'unknown'``."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def get_git_diff_stat() -> str:
    """Return ``git diff --stat`` output (uncommitted changes summary)."""
    try:
        diff = (
            subprocess.check_output(
                ["git", "diff", "--stat"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        return diff if diff else "(clean working tree)"
    except Exception:
        return "unknown"


def is_git_clean() -> bool:
    """Return *True* if the working tree has no uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════
# 6.  ENVIRONMENT FINGERPRINT
# ═══════════════════════════════════════════════════════════════════════

# FIX 4: map PyPI package names → actual Python import names.
# __import__("scikit-learn") raises ModuleNotFoundError because the
# importable name is "sklearn", not "scikit-learn".
_PKG_IMPORT_NAMES: Dict[str, str] = {
    "scikit-learn": "sklearn",
    "torch_geometric": "torch_geometric",
    "torch-geometric": "torch_geometric",
    "pymatgen": "pymatgen",
    "e3nn": "e3nn",
    "wandb": "wandb",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "numpy": "numpy",
    "torch": "torch",
    "ase": "ase",
    "pandas": "pandas",
}


def _get_version(pkg_name: str) -> str:
    """Return version string for a package, or 'NOT INSTALLED'."""
    import_name = _PKG_IMPORT_NAMES.get(pkg_name, pkg_name)
    try:
        mod = __import__(import_name)
        return getattr(mod, "__version__", "installed (version unknown)")
    except ImportError:
        pass
    # Fallback: importlib.metadata (works for packages without __version__)
    try:
        from importlib.metadata import version, PackageNotFoundError
        return version(pkg_name)
    except Exception:
        return "NOT INSTALLED"


def get_environment_fingerprint() -> Dict[str, Any]:
    """
    Collect a comprehensive snapshot of the software and hardware
    environment.  The returned dict is JSON-serialisable and should be
    saved alongside every experiment for audit purposes.
    """
    info: Dict[str, Any] = {
        "timestamp_utc":    datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "python_version":   sys.version,
        "platform":         platform.platform(),
        "hostname":         platform.node(),
        "cpu_count":        os.cpu_count(),
        "architecture":     platform.machine(),
    }

    packages = [
        "numpy", "scipy", "torch", "torch_geometric",
        "e3nn", "ase", "pymatgen", "wandb", "pandas",
        "scikit-learn", "matplotlib",
    ]
    info["packages"] = {pkg: _get_version(pkg) for pkg in packages}

    # ── CUDA / GPU ───────────────────────────────────────────────
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"]  = torch.version.cuda
            info["cudnn_version"] = str(torch.backends.cudnn.version())
            info["gpu_count"]     = torch.cuda.device_count()
            info["gpus"] = [
                {
                    "index":     i,
                    "name":      torch.cuda.get_device_name(i),
                    "memory_gb": round(
                        torch.cuda.get_device_properties(i).total_mem / 1e9, 2
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
        else:
            info["cuda_version"] = None
            info["gpu_count"]    = 0
            info["gpus"]         = []
    except ImportError:
        info["cuda_available"] = False
        info["gpu_count"]      = 0
        info["gpus"]           = []

    # ── Git ──────────────────────────────────────────────────────
    info["git_revision"] = get_git_revision()
    info["git_clean"]    = is_git_clean()

    return info


# ═══════════════════════════════════════════════════════════════════════
# 7.  CHECKPOINT INTEGRITY (SHA-256)
# ═══════════════════════════════════════════════════════════════════════

def compute_file_hash(
    path: Union[str, Path],
    algorithm: str = "sha256",
    chunk_size: int = 1 << 20,
) -> str:
    """Compute a hex-digest hash of a file."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint_integrity(
    path: Union[str, Path],
    expected_hash: str,
    algorithm: str = "sha256",
) -> bool:
    """
    Verify that a checkpoint file matches an expected hash.

    Returns *True* if the hashes match, *False* otherwise.
    """
    actual = compute_file_hash(path, algorithm=algorithm)
    match  = actual == expected_hash
    if not match:
        logger.warning(
            "Checkpoint integrity check FAILED for %s:\n"
            "  expected: %s\n"
            "  actual:   %s",
            path, expected_hash, actual,
        )
    return match


def hash_dict(d: Dict[str, Any]) -> str:
    """Deterministic SHA-256 of a JSON-serialisable dict."""
    canonical = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════
# 8.  EXPERIMENT MANIFEST
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentManifest:
    """
    A complete record of everything needed to reproduce a run.

    Fields
    ──────
    experiment_name : Human-readable identifier.
    seed            : Global seed used.
    config          : Full training / model / data config dict.
    environment     : Output of :func:`get_environment_fingerprint`.
    data_hash       : SHA-256 of the sorted training-set MOF IDs.
    model_summary   : String summary of the model architecture.
    notes           : Free-form notes.
    checkpoint_hash : SHA-256 of the best checkpoint file.
    metrics         : Final evaluation metrics dict.
    created_utc     : ISO-8601 UTC timestamp of manifest creation.
    """

    experiment_name: str              = "unnamed"
    seed:            int              = DEFAULT_SEED
    config:          Dict[str, Any]   = field(default_factory=dict)
    environment:     Dict[str, Any]   = field(default_factory=dict)
    data_hash:       str              = ""
    model_summary:   str              = ""
    notes:           str              = ""
    checkpoint_hash: str              = ""
    metrics:         Dict[str, Any]   = field(default_factory=dict)
    created_utc:     str              = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        """Write the manifest to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("Experiment manifest saved to %s", p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExperimentManifest":
        """
        Load a manifest from JSON.

        FIX: previous version did ``cls(**d)`` which raises TypeError
        if the JSON contains fields unknown to the current dataclass
        (e.g. a manifest from a different codebase version).
        Now filters to known fields before passing.
        """
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        known = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in known}
        unknown  = set(d.keys()) - known
        if unknown:
            logger.warning(
                "ExperimentManifest.load: ignoring unknown fields: %s",
                sorted(unknown),
            )
        return cls(**filtered)


def save_experiment_config(
    config: Dict[str, Any],
    save_path: Union[str, Path],
    *,
    seed: int = DEFAULT_SEED,
    include_environment: bool = True,
) -> None:
    """
    Save an experiment configuration enriched with git revision,
    timestamp, and optionally the full environment fingerprint.
    """
    out = copy.deepcopy(config)
    out["seed"]             = seed
    out["pipeline_version"] = PIPELINE_VERSION
    out["timestamp_utc"]    = datetime.now(timezone.utc).isoformat()
    out["git_revision"]     = get_git_revision()
    out["git_clean"]        = is_git_clean()

    if include_environment:
        out["environment"] = get_environment_fingerprint()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("Experiment config saved to %s", p)


def load_experiment_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a previously saved experiment config."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
# 9.  MODEL PARAMETER HASH
# ═══════════════════════════════════════════════════════════════════════

def model_parameter_hash(model: Any) -> str:
    """
    Compute a SHA-256 hash of all trainable parameters of a
    ``torch.nn.Module``.  Useful for verifying weight identity after
    loading a checkpoint.
    """
    h   = hashlib.sha256()
    buf = io.BytesIO()
    for name, param in sorted(model.named_parameters()):
        np_arr = param.detach().cpu().numpy()
        buf.seek(0)
        buf.truncate()
        np.save(buf, np_arr)
        h.update(name.encode())
        h.update(buf.getvalue())
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════
# 10.  TIMING UTILITY
# ═══════════════════════════════════════════════════════════════════════

@contextmanager
def timer(label: str = "Block"):
    """
    Simple context-manager timer that logs wall-clock elapsed time.

    >>> with timer("Data loading"):
    ...     dataset = load_data()
    # INFO: Data loading completed in 12.34 s
    """
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    logger.info("%s completed in %.2f s", label, elapsed)


# ═══════════════════════════════════════════════════════════════════════
# 11.  ENHANCED CHECKPOINT I/O
# ═══════════════════════════════════════════════════════════════════════

def save_reproducible_checkpoint(
    path: Union[str, Path],
    *,
    epoch: int,
    model: Any,
    optimizer: Any,
    scheduler: Any = None,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: int = DEFAULT_SEED,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a checkpoint that includes RNG states and a SHA-256 hash for
    integrity verification.

    Returns
    -------
    sha256 : str
        Hex-digest of the saved file (store in manifest for later verification).
    """
    import torch

    rng_state = capture_rng_state()

    payload: Dict[str, Any] = {
        "epoch":                epoch,
        "seed":                 seed,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state":            rng_state.to_dict(),
        "pipeline_version":     PIPELINE_VERSION,
        "timestamp_utc":        datetime.now(timezone.utc).isoformat(),
    }

    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if metrics is not None:
        payload["metrics"] = metrics
    if config is not None:
        payload["config"]  = config
    if extra is not None:
        payload["extra"]   = extra

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)

    sha = compute_file_hash(p)
    logger.info(
        "Checkpoint saved to %s  (epoch=%d, sha256=%s…)",
        p, epoch, sha[:16],
    )
    return sha


def load_reproducible_checkpoint(
    path: Union[str, Path],
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any = None,
    device: str = "cpu",
    restore_rng: bool = True,
    expected_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint saved by :func:`save_reproducible_checkpoint`,
    optionally verifying integrity and restoring RNG states.

    Returns
    -------
    Dict with ``epoch``, ``metrics``, ``config``, ``extra``, ``seed``.
    """
    import torch

    p = Path(path)
    if expected_hash is not None:
        if not verify_checkpoint_integrity(p, expected_hash):
            raise RuntimeError(
                f"Checkpoint {p} failed integrity check — expected sha256={expected_hash}"
            )

    ckpt = torch.load(p, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if restore_rng and "rng_state" in ckpt:
        rng = RNGState.from_dict(ckpt["rng_state"])
        restore_rng_state(rng)
        logger.info("RNG state restored from checkpoint.")

    logger.info(
        "Loaded checkpoint from %s  (epoch=%d, pipeline_version=%s)",
        p,
        ckpt.get("epoch", -1),
        ckpt.get("pipeline_version", "unknown"),
    )

    return {
        "epoch":   ckpt.get("epoch",   0),
        "metrics": ckpt.get("metrics", {}),
        "config":  ckpt.get("config",  {}),
        "extra":   ckpt.get("extra",   {}),
        "seed":    ckpt.get("seed",    DEFAULT_SEED),
    }


# ═══════════════════════════════════════════════════════════════════════
# 12.  DATA SPLIT HASHING
# ═══════════════════════════════════════════════════════════════════════

def hash_data_split(
    train_ids: List[str],
    val_ids:   List[str],
    test_ids:  List[str],
) -> str:
    """
    Compute a deterministic SHA-256 of the data split, independent of
    list ordering.
    """
    payload = {
        "train": sorted(train_ids),
        "val":   sorted(val_ids),
        "test":  sorted(test_ids),
    }
    return hash_dict(payload)


# ═══════════════════════════════════════════════════════════════════════
# 13.  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    # Constants
    "DEFAULT_SEED",
    "PIPELINE_VERSION",
    # Seeding
    "set_seed",
    "get_seed_history",
    # RNG state
    "RNGState",
    "capture_rng_state",
    "restore_rng_state",
    # Scoped reproducibility
    "ReproducibleBlock",
    "reproducible_block",
    # DataLoader workers
    "worker_init_fn",
    "make_worker_init_fn",
    # Git
    "get_git_revision",
    "get_git_diff_stat",
    "is_git_clean",
    # Environment
    "get_environment_fingerprint",
    # Checkpoint integrity
    "compute_file_hash",
    "verify_checkpoint_integrity",
    "hash_dict",
    "model_parameter_hash",
    # Experiment manifest
    "ExperimentManifest",
    "save_experiment_config",
    "load_experiment_config",
    # Enhanced checkpoint I/O
    "save_reproducible_checkpoint",
    "load_reproducible_checkpoint",
    # Data split hashing
    "hash_data_split",
    # Timing
    "timer",
]