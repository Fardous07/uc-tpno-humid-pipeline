#!/usr/bin/env python3
"""
GC-TMMC output parser.

IMPORTANT
---------
In this project layout:
    - src/simulation/gctmmc/parser.py  -> RUNNER
    - src/simulation/gctmmc/runner.py  -> OUTPUT PARSER
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Constants
R_GAS = 8.314462618       # J/(mol·K)
KB = 1.380649e-23         # J/K
NA = 6.02214076e23        # mol^-1


# ═══════════════════════════════════════════════════════════════════════
# 1. COLLECTION MATRIX FILE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════

def _find_collection_matrix_file(job_dir: Path) -> Optional[Path]:
    """
    Locate the GC-TMMC collection matrix file.

    Searches common RASPA layouts first, then falls back to recursive search.
    """
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

    patterns = (
        "CollectionMatrix*.dat",
        "TMMC*Collection*.dat",
        "*CollectionMatrix*.txt",
    )
    for pattern in patterns:
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

    Expected file structure is typically one of:
        N  C(N->N-1)  C(N->N)  C(N->N+1)
    or just three numeric columns.

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
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue

            nums = re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", stripped)
            if len(nums) < 3:
                continue

            values = [float(x) for x in nums]

            if len(values) >= 4:
                rows.append(values[:4])
            elif len(values) == 3:
                rows.append(values)

    if not rows:
        raise ValueError(f"No numeric collection-matrix rows found in: {filepath}")

    data = np.asarray(rows, dtype=float)

    if data.shape[1] >= 4:
        n_vals = data[:, 0].astype(int)
        n_max = int(np.max(n_vals))

        C = np.zeros((n_max + 1, 3), dtype=float)
        for row in data:
            n = int(row[0])
            if 0 <= n <= n_max:
                C[n, :] = row[1:4]
    elif data.shape[1] == 3:
        C = data[:, :3].copy()
        n_max = C.shape[0] - 1
    else:
        raise ValueError(f"Unexpected collection matrix shape: {data.shape}")

    return C, n_max


def collection_matrix_to_ln_pi(C: np.ndarray) -> np.ndarray:
    """
    Convert collection matrix to macrostate distribution ln Π(N).

    Uses the standard transition-probability ratio idea:
        Π(N+1)/Π(N) = P(N->N+1) / P(N+1->N)

    Returns
    -------
    ln_pi : array of shape [N_max+1], normalized so ln_pi[0] = 0
    """
    if C.ndim != 2 or C.shape[1] != 3:
        raise ValueError(f"C must have shape [N, 3], got {C.shape}")

    n_max = C.shape[0] - 1
    ln_pi = np.zeros(n_max + 1, dtype=float)

    eps = 1e-300

    for n in range(n_max):
        c_insert = max(float(C[n, 2]), eps)      # C(N -> N+1)
        c_delete = max(float(C[n + 1, 0]), eps)  # C(N+1 -> N)

        row_n = max(float(np.sum(C[n, :])), eps)
        row_np1 = max(float(np.sum(C[n + 1, :])), eps)

        p_insert = c_insert / row_n
        p_delete = c_delete / row_np1

        ratio = max(p_insert / max(p_delete, eps), eps)
        ln_pi[n + 1] = ln_pi[n] + np.log(ratio)

    ln_pi -= ln_pi[0]
    return ln_pi


# ═══════════════════════════════════════════════════════════════════════
# 3. AUXILIARY PARSING
# ═══════════════════════════════════════════════════════════════════════

def _extract_framework_mass_kg(job_dir: Path) -> Optional[float]:
    """
    Try to extract framework / simulation-box mass from text outputs.
    """
    candidate_files = list(job_dir.rglob("*.data")) + list(job_dir.rglob("*.log")) + [
        job_dir / "stdout.log",
        job_dir / "stderr.log",
    ]

    patterns = [
        r"framework mass\s*[:=]?\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)",
        r"simulation box mass\s*[:=]?\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)",
    ]

    seen = set()
    for file_path in candidate_files:
        if file_path in seen or not file_path.exists():
            continue
        seen.add(file_path)

        try:
            text = file_path.read_text(errors="replace")
        except Exception:
            continue

        lowered = text.lower()
        for pattern in patterns:
            m = re.search(pattern, lowered)
            if m:
                try:
                    value = float(m.group(1))
                    if np.isfinite(value) and value > 0:
                        return value
                except ValueError:
                    pass

    return None


def _collect_warnings(job_dir: Path) -> List[str]:
    """
    Scan common log files for warnings and obvious failure signals.
    """
    warnings: List[str] = []
    candidate_files = list(job_dir.rglob("*.log")) + [
        job_dir / "stdout.log",
        job_dir / "stderr.log",
    ]

    seen = set()
    patterns = [
        "warning",
        "error",
        "failed",
        "fatal",
        "overflow",
        "underflow",
        "nan",
        "did not converge",
        "not converged",
    ]

    for file_path in candidate_files:
        if file_path in seen or not file_path.exists():
            continue
        seen.add(file_path)

        try:
            text = file_path.read_text(errors="replace").lower()
        except Exception:
            continue

        for pat in patterns:
            if pat in text:
                warnings.append(f"{file_path.name}: contains '{pat}'")
                break

    return warnings


# ═══════════════════════════════════════════════════════════════════════
# 4. ISOTHERM CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

def ln_pi_to_isotherm(
    ln_pi: np.ndarray,
    temperature: float,
    pressures: Optional[np.ndarray] = None,
    n_pressures: int = 50,
    P_min: float = 1e-3,
    P_max: float = 100.0,
    framework_mass_kg: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Convert a macrostate distribution to a relative isotherm.

    Returns
    -------
    Dict with:
        - pressures : [n_pressures]
        - loadings  : [n_pressures]
        - ln_pi     : [N_max+1]
    """
    if pressures is None:
        pressures = np.logspace(np.log10(P_min), np.log10(P_max), n_pressures)
    else:
        pressures = np.asarray(pressures, dtype=float)

    n_arr = np.arange(len(ln_pi), dtype=float)
    loadings = np.zeros(len(pressures), dtype=float)

    for i, P_bar in enumerate(pressures):
        P_pa = max(float(P_bar) * 1e5, 1e-300)
        shift = np.log(P_pa) * n_arr

        ln_pi_shifted = ln_pi + shift
        ln_pi_shifted -= np.max(ln_pi_shifted)

        weights = np.exp(ln_pi_shifted)
        Z = np.sum(weights)

        if Z > 0:
            mean_n = np.sum(weights * n_arr) / Z
        else:
            mean_n = 0.0

        loadings[i] = mean_n

    if framework_mass_kg is not None and np.isfinite(framework_mass_kg) and framework_mass_kg > 0:
        loadings = (loadings / NA) / framework_mass_kg

    return {
        "pressures": pressures,
        "loadings": loadings,
        "ln_pi": np.asarray(ln_pi, dtype=float),
    }


# ═══════════════════════════════════════════════════════════════════════
# 5. TOP-LEVEL PARSER
# ═══════════════════════════════════════════════════════════════════════

def parse_tmmc_output(
    job_dir: Union[str, Path],
    temperature: float = 313.15,
    pressures: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Parse GC-TMMC output and construct a complete isotherm.
    """
    job_dir = Path(job_dir)

    result: Dict[str, Any] = {
        "success": False,
        "pressures": np.array([], dtype=float),
        "loadings": np.array([], dtype=float),
        "ln_pi": np.array([], dtype=float),
        "N_max": None,
        "framework_mass_kg": None,
        "warnings": [],
        "error": None,
    }

    cm_file = _find_collection_matrix_file(job_dir)
    if cm_file is None:
        result["error"] = "Collection matrix file not found."
        result["warnings"] = _collect_warnings(job_dir)
        logger.warning("No GC-TMMC collection matrix found in %s", job_dir)
        return result

    try:
        C, n_max = parse_collection_matrix(cm_file)
        ln_pi = collection_matrix_to_ln_pi(C)
        framework_mass_kg = _extract_framework_mass_kg(job_dir)

        iso = ln_pi_to_isotherm(
            ln_pi=ln_pi,
            temperature=float(temperature),
            pressures=pressures,
            framework_mass_kg=framework_mass_kg,
        )

        result.update(iso)
        result["N_max"] = int(n_max)
        result["framework_mass_kg"] = framework_mass_kg
        result["warnings"] = _collect_warnings(job_dir)
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
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
    """
    Generate a synthetic collection matrix for testing.
    """
    rng = np.random.RandomState(seed)

    N = np.arange(N_max + 1, dtype=float)
    site_cap = max(q_sat * 10.0, float(N_max))
    theta = np.clip(N / max(site_cap, 1e-12), 1e-12, 1.0 - 1e-12)

    ln_pi = N * np.log(max(K, 1e-12)) - (
        N * np.log(theta) + (site_cap - N) * np.log(1.0 - theta)
    )
    ln_pi -= np.max(ln_pi)

    C = np.zeros((N_max + 1, 3), dtype=float)

    for i in range(N_max + 1):
        C[i, 1] = 1000.0 + rng.poisson(500)

        if i < N_max:
            delta = ln_pi[i + 1] - ln_pi[i]
            rate = np.exp(min(delta, 10.0))
            C[i, 2] = max(1.0, 1000.0 * rate * (1.0 + noise * rng.randn()))

        if i > 0:
            delta = ln_pi[i - 1] - ln_pi[i]
            rate = np.exp(min(delta, 10.0))
            C[i, 0] = max(1.0, 1000.0 * rate * (1.0 + noise * rng.randn()))

    return C


__all__ = [
    "parse_tmmc_output",
    "parse_collection_matrix",
    "collection_matrix_to_ln_pi",
    "ln_pi_to_isotherm",
    "synthetic_collection_matrix",
]