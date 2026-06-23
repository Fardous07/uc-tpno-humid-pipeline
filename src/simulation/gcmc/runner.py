#!/usr/bin/env python3
"""
RASPA2 output parser for GCMC jobs.

IMPORTANT
---------
In this project layout:
    - src/simulation/gcmc/parser.py  -> RUNNER
    - src/simulation/gcmc/runner.py  -> OUTPUT PARSER

This file provides:
    - parse_raspa_output(...)
    - parse_batch(...)
    - results_to_arrays(...)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

# RASPA reports some energetic quantities in K; convert via R (kJ/mol/K)
_KB_KJMOL = 8.314462618e-3


# ═══════════════════════════════════════════════════════════════════════
# 1. MAIN PARSER
# ═══════════════════════════════════════════════════════════════════════

def parse_raspa_output(job_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse all available RASPA outputs from a job directory.

    Parameters
    ----------
    job_dir : Path to a RASPA working directory.

    Returns
    -------
    Dict with keys such as:
        - loadings
        - loadings_uc
        - energies
        - enthalpy
        - converged
        - warnings
        - unit_cell_mass_kg
    """
    job_dir = Path(job_dir)

    result: Dict[str, Any] = {
        "loadings": {},
        "loadings_uc": {},
        "loadings_mg_g": {},
        "energies": {},
        "enthalpy": None,
        "converged": None,
        "warnings": [],
        "unit_cell_mass_kg": None,
    }

    candidate_dirs = [
        job_dir / "Output" / "System_0",
        job_dir / "Output",
        job_dir,
    ]

    output_dir = None
    for d in candidate_dirs:
        if d.exists():
            output_dir = d
            break

    if output_dir is None:
        result["warnings"].append("No output directory found.")
        logger.warning("No RASPA output directory found in %s", job_dir)
        return result

    data_files = sorted(output_dir.rglob("*.data"))
    if not data_files:
        result["warnings"].append("No .data output files found.")
        logger.warning("No RASPA .data files found in %s", job_dir)

    for data_file in data_files:
        try:
            text = data_file.read_text(errors="replace")
            _parse_data_file(text, result)
        except Exception as e:
            msg = f"Parse error in {data_file.name}: {e}"
            result["warnings"].append(msg)
            logger.warning(msg)

    log_candidates = list(output_dir.rglob("*.log")) + [
        job_dir / "stdout.log",
        job_dir / "stderr.log",
    ]

    seen_logs = set()
    for log_file in log_candidates:
        if log_file in seen_logs or not log_file.exists():
            continue
        seen_logs.add(log_file)

        try:
            log_text = log_file.read_text(errors="replace")
            _parse_data_file(log_text, result)
            _parse_log_text(log_text, result)
        except Exception:
            pass

    _backfill_loadings(result)
    _check_convergence(result)

    return result


def _parse_data_file(text: str, result: Dict[str, Any]) -> None:
    """
    Parse a single RASPA `.data` file and update the result dict in-place.
    """
    lines = text.splitlines()

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()

        # ── Average loadings ───────────────────────────────────────────
        if "average loading absolute" in line_lower and "molecules/unit cell" in line_lower:
            component = _extract_component_name(lines, i)
            value = _extract_value_with_error(line)
            if component and value is not None:
                result["loadings_uc"][component] = value

        if "average loading absolute" in line_lower and "mol/kg" in line_lower:
            component = _extract_component_name(lines, i)
            value = _extract_value_with_error(line)
            if component and value is not None:
                result["loadings"][component] = value

        if "average loading absolute" in line_lower and "milligram/gram" in line_lower:
            component = _extract_component_name(lines, i)
            value = _extract_value_with_error(line)
            if component and value is not None:
                result["loadings_mg_g"][component] = value

        # ── Energies ───────────────────────────────────────────────────
        if "average adsorbate-adsorbate energy" in line_lower:
            value = _extract_value_with_error(line)
            if value is not None:
                result["energies"]["guest_guest_K"] = value
                result["energies"]["guest_guest_kJ_mol"] = value * _KB_KJMOL

        if "average host-adsorbate energy" in line_lower or "average framework-adsorbate energy" in line_lower:
            value = _extract_value_with_error(line)
            if value is not None:
                result["energies"]["host_guest_K"] = value
                result["energies"]["host_guest_kJ_mol"] = value * _KB_KJMOL

        if "enthalpy of adsorption" in line_lower:
            value = _extract_value_with_error(line)
            if value is not None:
                result["enthalpy"] = value * _KB_KJMOL

        # ── Framework / box mass ──────────────────────────────────────
        if "framework mass" in line_lower or "simulation box mass" in line_lower:
            value = _extract_float(line)
            if value is not None:
                result["unit_cell_mass_kg"] = value


def _parse_log_text(text: str, result: Dict[str, Any]) -> None:
    """
    Parse stdout/stderr/log text for warning and failure diagnostics.
    """
    lowered = text.lower()

    warning_patterns = [
        "warning",
        "nan",
        "overflow",
        "underflow",
        "did not converge",
        "not converged",
        "insufficient statistics",
        "failed",
        "fatal",
        "error",
    ]

    for pat in warning_patterns:
        if pat in lowered:
            result["warnings"].append(f"Log contains: {pat}")
            break


# ═══════════════════════════════════════════════════════════════════════
# 2. HELPER EXTRACTORS
# ═══════════════════════════════════════════════════════════════════════

def _extract_component_name(lines: List[str], idx: int) -> Optional[str]:
    """
    Look backward from line idx to find the nearest component identifier.

    Common RASPA patterns include:
        Component 0 [CO2]
        Component 0 MoleculeName CO2
    """
    for j in range(idx - 1, max(-1, idx - 30), -1):
        line = lines[j]

        m = re.search(r"Component\s+\d+\s+\[([^\]]+)\]", line)
        if m:
            return m.group(1).strip()

        m = re.search(r"Component\s+\d+\s+MoleculeName\s+([A-Za-z0-9_\-+/\.]+)", line)
        if m:
            return m.group(1).strip()

    return None


def _extract_value_with_error(line: str) -> Optional[float]:
    """
    Extract the first numeric value after the descriptive part of a line.
    """
    search_region = line.split("]")[-1]
    m = re.search(r"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)", search_region)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _extract_float(line: str) -> Optional[float]:
    """
    Extract the last float on a line.
    """
    floats = re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", line)
    if not floats:
        return None
    try:
        return float(floats[-1])
    except ValueError:
        return None


def _backfill_loadings(result: Dict[str, Any]) -> None:
    """
    If direct mol/kg loadings are missing but molecules/unit-cell are present,
    attempt a conservative backfill using reported box/framework mass.
    """
    if result["loadings"]:
        return

    mass_kg = result.get("unit_cell_mass_kg")
    if mass_kg is None or not np.isfinite(mass_kg) or mass_kg <= 0:
        return

    na = 6.02214076e23

    for species, n_uc in result.get("loadings_uc", {}).items():
        if not np.isfinite(n_uc):
            continue

        mol = n_uc / na
        mol_kg = mol / mass_kg
        result["loadings"][species] = mol_kg


def _check_convergence(result: Dict[str, Any]) -> None:
    """
    Simple heuristic convergence check.
    """
    loadings = result.get("loadings", {})
    if not loadings:
        result["converged"] = None
        return

    all_valid = all(np.isfinite(v) and v >= 0 for v in loadings.values())
    has_bad_warning = any(
        any(key in w.lower() for key in ["nan", "overflow", "fatal", "did not converge"])
        for w in result.get("warnings", [])
    )

    result["converged"] = bool(all_valid and not has_bad_warning)


# ═══════════════════════════════════════════════════════════════════════
# 3. BATCH UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def parse_batch(job_dirs: Sequence[Union[str, Path]]) -> List[Dict[str, Any]]:
    """
    Parse multiple RASPA job directories.
    """
    return [parse_raspa_output(jd) for jd in job_dirs]


def results_to_arrays(
    results: Sequence[Dict[str, Any]],
    species: Sequence[str] = ("CO2", "N2", "H2O"),
) -> Dict[str, np.ndarray]:
    """
    Convert parsed results into numpy arrays.

    Returns
    -------
    Dict with:
        - loadings: shape [N, C]
        - success:  shape [N]
    """
    n = len(results)
    c = len(species)

    loadings = np.full((n, c), np.nan, dtype=float)
    success = np.zeros(n, dtype=bool)

    for i, r in enumerate(results):
        r_loadings = r.get("loadings", {})
        for j, sp in enumerate(species):
            if sp in r_loadings:
                loadings[i, j] = float(r_loadings[sp])

        success[i] = bool(r_loadings)

    return {
        "loadings": loadings,
        "success": success,
    }


__all__ = [
    "parse_raspa_output",
    "parse_batch",
    "results_to_arrays",
]