#!/usr/bin/env python3
"""
src/simulation/gctmmc/runner.py
────────────────────────────────
GC-TMMC output parser for the UC-TPNO pipeline.

NOTE on project layout
──────────────────────
In this project the conventional names are swapped:
    gctmmc/parser.py  → GCTMMCConfig / generate_tmmc_input / GCTMMCRunner (RUNNER)
    gctmmc/runner.py  → parse_tmmc_output / collection_matrix_to_ln_pi … (PARSER)

Fixes applied
─────────────
1. Re-export GCTMMCConfig, generate_tmmc_input, GCTMMCRunner from parser.py so
   ``from src.simulation.gctmmc.runner import GCTMMCConfig`` works in tests.
2. ln_pi_to_isotherm: renamed parameter temperature → T so
   ``ln_pi_to_isotherm(ln_pi, T=313.15, pressures=p)`` works.
3. parse_tmmc_output: renamed parameter temperature → T so
   ``parse_tmmc_output(tmp_path, T=313.15)`` works.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Constants
R_GAS = 8.314462618   # J/(mol·K)
KB    = 1.380649e-23  # J/K
NA    = 6.02214076e23 # mol^-1


# ═══════════════════════════════════════════════════════════════════════
# RE-EXPORTS FROM parser.py
# (test imports GCTMMCConfig / generate_tmmc_input / GCTMMCRunner from here)
# ═══════════════════════════════════════════════════════════════════════

from .parser import GCTMMCConfig, generate_tmmc_input, GCTMMCRunner  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# 1. COLLECTION MATRIX FILE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════

def _find_collection_matrix_file(job_dir: Path) -> Optional[Path]:
    """Locate the GC-TMMC collection matrix file."""
    candidates = [
        job_dir / "Output" / "System_0" / "CollectionMatrix.dat",
        job_dir / "Output" / "System_0" / "TMMC_CollectionMatrix.dat",
        job_dir / "Output" / "CollectionMatrix.dat",
        job_dir / "CollectionMatrix.dat",
        job_dir / "TMMC_CollectionMatrix.dat",
    ]
    for path in candidates:
        if path.exists():
            return path
    for pattern in ("CollectionMatrix*.dat", "TMMC*Collection*.dat", "*CollectionMatrix*.txt"):
        matches = sorted(job_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


# ═══════════════════════════════════════════════════════════════════════
# 2. COLLECTION MATRIX PARSING
# ═══════════════════════════════════════════════════════════════════════

def parse_collection_matrix(filepath: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """
    Parse a RASPA GC-TMMC collection matrix file.

    Expected format (one row per molecule number N):
        N  C(N->N-1)  C(N->N)  C(N->N+1)

    Returns
    -------
    C     : array of shape [N_max+1, 3]
    N_max : maximum particle number represented
    """
    filepath = Path(filepath)
    rows: List[List[float]] = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            nums = re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", stripped)
            if len(nums) < 3:
                continue
            values = [float(x) for x in nums]
            rows.append(values[:4] if len(values) >= 4 else values)

    if not rows:
        raise ValueError(f"No numeric collection-matrix rows found in: {filepath}")

    data = np.asarray(rows, dtype=float)

    if data.shape[1] >= 4:
        n_vals = data[:, 0].astype(int)
        n_max  = int(np.max(n_vals))
        C      = np.zeros((n_max + 1, 3), dtype=float)
        for row in data:
            n = int(row[0])
            if 0 <= n <= n_max:
                C[n, :] = row[1:4]
    else:
        C     = data[:, :3].copy()
        n_max = C.shape[0] - 1

    return C, n_max


def collection_matrix_to_ln_pi(C: np.ndarray) -> np.ndarray:
    """
    Convert collection matrix to macrostate distribution ln Π(N).

    Uses the transition-probability ratio:
        Π(N+1)/Π(N) = P(N→N+1) / P(N+1→N)

    Returns
    -------
    ln_pi : array of shape [N_max+1], normalised so ln_pi[0] = 0
    """
    if C.ndim != 2 or C.shape[1] != 3:
        raise ValueError(f"C must have shape [N, 3], got {C.shape}")

    n_max = C.shape[0] - 1
    ln_pi = np.zeros(n_max + 1, dtype=float)
    eps   = 1e-300

    for n in range(n_max):
        c_insert = max(float(C[n,     2]), eps)
        c_delete = max(float(C[n + 1, 0]), eps)
        row_n    = max(float(np.sum(C[n,     :])), eps)
        row_np1  = max(float(np.sum(C[n + 1, :])), eps)
        p_insert = c_insert / row_n
        p_delete = c_delete / row_np1
        ln_pi[n + 1] = ln_pi[n] + np.log(max(p_insert / max(p_delete, eps), eps))

    ln_pi -= ln_pi[0]   # anchor: ln_pi[0] = 0
    return ln_pi


# ═══════════════════════════════════════════════════════════════════════
# 3. AUXILIARY PARSING
# ═══════════════════════════════════════════════════════════════════════

def _extract_framework_mass_kg(job_dir: Path) -> Optional[float]:
    candidate_files = (
        list(job_dir.rglob("*.data"))
        + list(job_dir.rglob("*.log"))
        + [job_dir / "stdout.log", job_dir / "stderr.log"]
    )
    patterns = [
        r"framework mass\s*[:=]?\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)",
        r"simulation box mass\s*[:=]?\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)",
    ]
    seen: set = set()
    for fp in candidate_files:
        if fp in seen or not fp.exists():
            continue
        seen.add(fp)
        try:
            lowered = fp.read_text(errors="replace").lower()
        except Exception:
            continue
        for pat in patterns:
            m = re.search(pat, lowered)
            if m:
                try:
                    val = float(m.group(1))
                    if np.isfinite(val) and val > 0:
                        return val
                except ValueError:
                    pass
    return None


def _collect_warnings(job_dir: Path) -> List[str]:
    warnings: List[str] = []
    candidate_files = list(job_dir.rglob("*.log")) + [
        job_dir / "stdout.log", job_dir / "stderr.log"
    ]
    seen: set = set()
    bad_keywords = ("warning", "error", "failed", "fatal", "overflow",
                    "underflow", "nan", "did not converge", "not converged")
    for fp in candidate_files:
        if fp in seen or not fp.exists():
            continue
        seen.add(fp)
        try:
            lowered = fp.read_text(errors="replace").lower()
        except Exception:
            continue
        for kw in bad_keywords:
            if kw in lowered:
                warnings.append(f"{fp.name}: contains '{kw}'")
                break
    return warnings


# ═══════════════════════════════════════════════════════════════════════
# 4. ISOTHERM CONSTRUCTION
# FIX: parameter renamed temperature → T to match test call signature:
#      ln_pi_to_isotherm(ln_pi, T=313.15, pressures=pressures)
# ═══════════════════════════════════════════════════════════════════════

def ln_pi_to_isotherm(
    ln_pi: np.ndarray,
    T: float,                           # FIX: was temperature
    pressures: Optional[np.ndarray] = None,
    n_pressures: int = 50,
    P_min: float = 1e-3,
    P_max: float = 100.0,
    framework_mass_kg: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Convert a macrostate distribution to a relative isotherm.

    Parameters
    ----------
    ln_pi    : log macrostate weights, ln_pi[0] = 0.
    T        : Temperature [K].
    pressures: Pressure points [bar]; if None, a log-spaced grid is used.

    Returns
    -------
    dict with ``pressures``, ``loadings``, ``ln_pi``.
    """
    if pressures is None:
        pressures = np.logspace(np.log10(P_min), np.log10(P_max), n_pressures)
    else:
        pressures = np.asarray(pressures, dtype=float)

    n_arr    = np.arange(len(ln_pi), dtype=float)
    loadings = np.zeros(len(pressures), dtype=float)

    for i, P_bar in enumerate(pressures):
        P_pa  = max(float(P_bar) * 1e5, 1e-300)
        shift = np.log(P_pa) * n_arr
        lp    = ln_pi + shift
        lp   -= np.max(lp)
        wts   = np.exp(lp)
        Z     = np.sum(wts)
        loadings[i] = float(np.sum(wts * n_arr) / Z) if Z > 0 else 0.0

    if framework_mass_kg is not None and np.isfinite(framework_mass_kg) and framework_mass_kg > 0:
        loadings = (loadings / NA) / framework_mass_kg

    return {
        "pressures": pressures,
        "loadings":  loadings,
        "ln_pi":     np.asarray(ln_pi, dtype=float),
    }


# ═══════════════════════════════════════════════════════════════════════
# 5. TOP-LEVEL PARSER
# FIX: parameter renamed temperature → T to match test call signature:
#      parse_tmmc_output(tmp_path, T=313.15)
# ═══════════════════════════════════════════════════════════════════════

def parse_tmmc_output(
    job_dir: Union[str, Path],
    T: float = 313.15,                  # FIX: was temperature
    pressures: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Parse GC-TMMC output and construct a complete isotherm.

    Parameters
    ----------
    job_dir : Root TMMC run directory.
    T       : Temperature [K].

    Returns
    -------
    dict with ``success``, ``pressures``, ``loadings``, ``ln_pi``, …
    """
    job_dir = Path(job_dir)

    result: Dict[str, Any] = {
        "success":           False,
        "pressures":         np.array([], dtype=float),
        "loadings":          np.array([], dtype=float),
        "ln_pi":             np.array([], dtype=float),
        "N_max":             None,
        "framework_mass_kg": None,
        "warnings":          [],
        "error":             None,
    }

    cm_file = _find_collection_matrix_file(job_dir)
    if cm_file is None:
        result["error"]    = "Collection matrix file not found."
        result["warnings"] = _collect_warnings(job_dir)
        logger.warning("No GC-TMMC collection matrix found in %s", job_dir)
        return result

    try:
        C, n_max           = parse_collection_matrix(cm_file)
        ln_pi              = collection_matrix_to_ln_pi(C)
        framework_mass_kg  = _extract_framework_mass_kg(job_dir)

        iso = ln_pi_to_isotherm(
            ln_pi=ln_pi, T=float(T),
            pressures=pressures,
            framework_mass_kg=framework_mass_kg,
        )

        result.update(iso)
        result["N_max"]             = int(n_max)
        result["framework_mass_kg"] = framework_mass_kg
        result["warnings"]          = _collect_warnings(job_dir)
        result["success"]           = True

    except Exception as e:
        result["error"]    = str(e)
        result["warnings"] = _collect_warnings(job_dir)
        logger.exception("GC-TMMC parse error in %s", job_dir)

    return result


# ═══════════════════════════════════════════════════════════════════════
# 6. SYNTHETIC TEST UTILITY
# ═══════════════════════════════════════════════════════════════════════

def synthetic_collection_matrix(
    N_max: int = 100,
    q_sat: float = 5.0,
    K: float = 0.5,
    noise: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic collection matrix for testing."""
    rng = np.random.RandomState(seed)

    N         = np.arange(N_max + 1, dtype=float)
    site_cap  = max(q_sat * 10.0, float(N_max))
    theta     = np.clip(N / max(site_cap, 1e-12), 1e-12, 1.0 - 1e-12)
    ln_pi     = N * np.log(max(K, 1e-12)) - (
        N * np.log(theta) + (site_cap - N) * np.log(1.0 - theta)
    )
    ln_pi    -= np.max(ln_pi)

    C = np.zeros((N_max + 1, 3), dtype=float)
    for i in range(N_max + 1):
        C[i, 1] = 1000.0 + rng.poisson(500)
        if i < N_max:
            delta    = ln_pi[i + 1] - ln_pi[i]
            rate     = np.exp(min(delta, 10.0))
            C[i, 2] = max(1.0, 1000.0 * rate * (1.0 + noise * rng.randn()))
        if i > 0:
            delta    = ln_pi[i - 1] - ln_pi[i]
            rate     = np.exp(min(delta, 10.0))
            C[i, 0] = max(1.0, 1000.0 * rate * (1.0 + noise * rng.randn()))

    return C


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    # Re-exported from parser.py
    "GCTMMCConfig",
    "generate_tmmc_input",
    "GCTMMCRunner",
    # Defined here
    "parse_tmmc_output",
    "parse_collection_matrix",
    "collection_matrix_to_ln_pi",
    "ln_pi_to_isotherm",
    "synthetic_collection_matrix",
]