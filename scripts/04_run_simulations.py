#!/usr/bin/env python3
"""
04_run_simulations.py
Robust RASPA runner for humid CO2/N2/H2O adsorption on MOFs.

Pipeline
--------
Stage 1  Run single-component RASPA at a sweep of pressures for every MOF
         (cheap, acts as our low-fidelity TMMC proxy).
         → data/processed/adsorption/tmmc_isotherms.parquet

Stage 2  Select the best K MOFs from Stage 1 using a Pareto front over
         (CO2/N2 selectivity, CO2 working capacity) plus diversity sampling.
         → data/processed/adsorption/gcmc_selected_mofs.txt

Stage 3  Run multicomponent GCMC (CO2 + N2 + H2O) at all humid conditions
         for the selected MOFs only.
         → data/processed/adsorption/gcmc_adsorption.parquet
         → data/processed/adsorption/adsorption_training.parquet

Fixes vs. earlier version
--------------------------
1.  BUG FIXED: retry logic now checks returncode (subprocess.run with no
    check=True never raises CalledProcessError — the old code never retried).
2.  BUG FIXED: run_low_fidelity_isotherms now sweeps all pressures for
    each (MOF, component, temperature) so that we get a real isotherm to
    compute selectivity and working capacity from, not a single point.
3.  ADDED: select_mofs_for_gcmc() implements a Pareto-front + diversity
    strategy to pick the best K MOFs from Stage 1 for expensive GCMC.
4.  ADDED: build_training_dataset() merges GCMC rows and saves them in the
    column layout expected by AdsorptionDataset.
5.  CIF syntax error fixed: determine_unit_cells now calls cif_path correctly.
6.  Molecule-definition lookup now falls through to any available match so
    the script works with both TraPPE and ExampleMOFsForceField layouts.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from ase.io import read as ase_read
except Exception:
    ase_read = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
R_KJ_MOL_K: float = 8.314462618e-3   # kJ / (mol·K)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class Condition:
    temperature_k: float
    pressure_bar: float
    rh: float
    y_co2_dry: float
    y_n2_dry: float
    x_co2_total: float
    x_n2_total: float
    x_h2o_total: float
    p_h2o_bar: float
    p_co2_bar: float
    p_n2_bar: float
    mu_co2: float
    mu_n2: float
    mu_h2o: float


@dataclass
class RASPAResult:
    ok: bool
    combined_text: str
    log_path: Path
    returncode: int


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------
def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def tail_text(text: str, n: int = 80) -> str:
    lines = (text or "").splitlines()
    return "\n".join(lines[-n:]) if lines else "(no output)"


def available_subdirs(p: Path) -> List[str]:
    if not p.exists():
        return []
    return sorted(d.name for d in p.iterdir() if d.is_dir())


# ---------------------------------------------------------------------------
# Thermodynamics
# ---------------------------------------------------------------------------
def water_saturation_pressure_bar(temperature_k: float) -> float:
    """Antoine equation → bar (valid 1–100 °C, adequate for 25–60 °C range)."""
    t_c = temperature_k - 273.15
    log_p_mmHg = 8.07131 - 1730.63 / (233.426 + t_c)
    return float(10.0 ** log_p_mmHg * 133.322368e-5)


def mu_ideal_gas(partial_pressure_bar: float, temperature_k: float) -> float:
    """μ = RT ln(P/P_ref)  with P_ref = 1 bar, result in kJ/mol."""
    p = max(float(partial_pressure_bar), 1e-12)
    return float(R_KJ_MOL_K * temperature_k * math.log(p))


def build_conditions(
    temperatures_k: Sequence[float],
    pressures_bar: Sequence[float],
    rhs: Sequence[float],
    dry_co2_frac: float,
) -> List[Condition]:
    """Build all valid (T, P, RH) thermodynamic conditions."""
    out: List[Condition] = []
    for t in temperatures_k:
        p_sat = water_saturation_pressure_bar(t)
        for p_total in pressures_bar:
            for rh in rhs:
                p_h2o = rh * p_sat
                if p_h2o >= p_total:          # physically impossible
                    continue
                dry = p_total - p_h2o
                x_h2o = p_h2o / p_total
                x_co2 = (dry / p_total) * dry_co2_frac
                x_n2  = (dry / p_total) * (1.0 - dry_co2_frac)
                p_co2 = x_co2 * p_total
                p_n2  = x_n2  * p_total
                out.append(Condition(
                    temperature_k=float(t),
                    pressure_bar=float(p_total),
                    rh=float(rh),
                    y_co2_dry=float(dry_co2_frac),
                    y_n2_dry=float(1.0 - dry_co2_frac),
                    x_co2_total=float(x_co2),
                    x_n2_total=float(x_n2),
                    x_h2o_total=float(x_h2o),
                    p_h2o_bar=float(p_h2o),
                    p_co2_bar=float(p_co2),
                    p_n2_bar=float(p_n2),
                    mu_co2=mu_ideal_gas(p_co2, t),
                    mu_n2=mu_ideal_gas(p_n2,  t),
                    mu_h2o=mu_ideal_gas(p_h2o, t),
                ))
    return out


# ---------------------------------------------------------------------------
# CIF helpers
# ---------------------------------------------------------------------------
_CELL_TAGS = [
    ("_cell_length_a", "a"), ("_cell_length_b", "b"), ("_cell_length_c", "c"),
    ("_cell_angle_alpha", "al"), ("_cell_angle_beta", "be"), ("_cell_angle_gamma", "ga"),
]


def read_cell_lengths(cif_path: Path) -> Optional[Tuple[float, float, float]]:
    """Return (a, b, c) in Å from CIF text, or None on failure."""
    try:
        text = cif_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    vals: Dict[str, float] = {}
    for tag, key in _CELL_TAGS:
        m = re.search(rf"^\s*{re.escape(tag)}\s+([^\s]+)", text, re.MULTILINE)
        if not m:
            continue
        token = re.sub(r"\([^)]+\)$", "", m.group(1).strip().strip("'\""))
        try:
            vals[key] = float(token)
        except ValueError:
            pass
    if {"a", "b", "c"}.issubset(vals):
        return vals["a"], vals["b"], vals["c"]
    return None


def determine_unit_cells(cif_path: Path, cutoff_angstrom: float) -> Tuple[int, int, int]:
    """Minimum integer replication so every dimension ≥ 2×cutoff."""
    lengths = read_cell_lengths(cif_path)   # BUG WAS HERE: was cif_ <path>
    if lengths is None:
        logger.warning(f"Could not read cell for {cif_path.name}; defaulting to 2×2×2")
        return (2, 2, 2)
    min_len = 2.0 * cutoff_angstrom
    reps = [max(1, math.ceil(min_len / max(L, 1e-8))) for L in lengths]
    return (reps[0], reps[1], reps[2])


def canonicalize_cif(input_cif: Path, output_cif: Path) -> None:
    """Write a minimal P1 CIF that RASPA can always parse."""
    if ase_read is None:
        raise RuntimeError("ASE required: pip install ase")
    atoms = ase_read(str(input_cif))
    if isinstance(atoms, list):
        if not atoms:
            raise ValueError(f"Empty CIF: {input_cif}")
        atoms = atoms[0]
    cl  = atoms.cell.lengths()
    ca  = atoms.cell.angles()
    sp  = atoms.get_scaled_positions(wrap=True)
    sym = atoms.get_chemical_symbols()
    counts: Dict[str, int] = {}
    lines = [
        f"data_{input_cif.stem}",
        f"_cell_length_a {cl[0]:.6f}",
        f"_cell_length_b {cl[1]:.6f}",
        f"_cell_length_c {cl[2]:.6f}",
        f"_cell_angle_alpha {ca[0]:.6f}",
        f"_cell_angle_beta  {ca[1]:.6f}",
        f"_cell_angle_gamma {ca[2]:.6f}",
        "_symmetry_space_group_name_Hall 'P 1'",
        "_symmetry_space_group_name_H-M  'P 1'",
        "_symmetry_Int_Tables_number 1",
        "loop_", "_symmetry_equiv_pos_as_xyz", "x,y,z",
        "loop_",
        "_atom_site_label", "_atom_site_type_symbol",
        "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]
    for s, pos in zip(sym, sp):
        counts[s] = counts.get(s, 0) + 1
        lines.append(f"{s}{counts[s]} {s} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} 1.0")
    lines.append("")
    ensure_dir(output_cif.parent)
    output_cif.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# RASPA setup helpers
# ---------------------------------------------------------------------------
def detect_raspa_share(raspa_data_dir: Path) -> Path:
    raspa_data_dir = raspa_data_dir.expanduser().resolve()
    candidate = raspa_data_dir / "share" / "raspa"
    if candidate.exists():
        return candidate
    if raspa_data_dir.name == "raspa" and raspa_data_dir.exists():
        return raspa_data_dir
    raise FileNotFoundError(
        f"Cannot find RASPA share under {raspa_data_dir}. "
        "Expected <dir>/share/raspa or a direct raspa dir."
    )


def resolve_forcefield(raspa_share: Path, requested: str) -> str:
    ff_root = raspa_share / "forcefield"
    available = available_subdirs(ff_root)
    if not available:
        raise FileNotFoundError(f"No forcefield dirs under {ff_root}")
    if requested in available:
        return requested
    lower = {n.lower(): n for n in available}
    if requested.lower() in lower:
        return lower[requested.lower()]
    # Common alias
    if requested == "GenericMOFs" and "ExampleMOFsForceField" in available:
        logger.info("GenericMOFs not found → using ExampleMOFsForceField")
        return "ExampleMOFsForceField"
    raise FileNotFoundError(
        f"Forcefield '{requested}' not found. Available: {available}"
    )


def find_molecule_dir(raspa_share: Path, molecule: str, preferred: Optional[str] = None) -> str:
    mol_root = raspa_share / "molecules"
    if not mol_root.exists():
        raise FileNotFoundError(f"RASPA molecules dir not found: {mol_root}")
    target = f"{molecule}.def"
    if preferred:
        d = mol_root / preferred
        if d.is_dir() and (d / target).exists():
            return preferred
    matches = [
        d.name for d in sorted(mol_root.iterdir())
        if d.is_dir() and (d / target).exists()
    ]
    if matches:
        if preferred and preferred not in matches:
            logger.info(f"Preferred dir '{preferred}' for {molecule} not found → using '{matches[0]}'")
        return matches[0]
    raise FileNotFoundError(
        f"No {target} found anywhere under {mol_root}. "
        f"Available dirs: {available_subdirs(mol_root)}"
    )


def install_user_def(runtime_share: Path, molecule: str, def_file: Path) -> str:
    def_file = def_file.expanduser().resolve()
    if not def_file.exists():
        raise FileNotFoundError(def_file)
    if def_file.name != f"{molecule}.def":
        raise ValueError(f"File must be named exactly {molecule}.def")
    dst = ensure_dir(runtime_share / "molecules" / "UserDefinitions")
    shutil.copy2(def_file, dst / def_file.name)
    return "UserDefinitions"


def prepare_runtime(output_dir: Path, raspa_share: Path) -> Tuple[Path, Path, Path]:
    """Copy RASPA share into a local runtime tree."""
    runtime_root  = (output_dir / "_raspa_runtime").resolve()
    runtime_share = runtime_root / "share" / "raspa"
    ensure_dir(runtime_root)
    if runtime_share.exists():
        shutil.rmtree(runtime_share)
    shutil.copytree(raspa_share, runtime_share, dirs_exist_ok=True)
    cif_dir = ensure_dir(runtime_share / "structures" / "cif")
    return runtime_root, runtime_share, cif_dir


def check_ff_assets(runtime_share: Path, ff_name: str) -> None:
    ff_dir = runtime_share / "forcefield" / ff_name
    if not ff_dir.exists():
        raise FileNotFoundError(f"Forcefield dir missing: {ff_dir}")
    required = ["force_field.def", "pseudo_atoms.def"]
    missing = [f for f in required if not (ff_dir / f).exists()]
    mixing = any((ff_dir / f).exists() for f in
                 ["force_field_mixing_rules.def", "mixing_rules.def"])
    if not mixing:
        missing.append("force_field_mixing_rules.def")
    if missing:
        raise FileNotFoundError(f"FF '{ff_name}' missing files: {missing}")


def check_mol_def(runtime_share: Path, mol_dir: str, molecule: str) -> None:
    p = runtime_share / "molecules" / mol_dir / f"{molecule}.def"
    if not p.exists():
        raise FileNotFoundError(f"Molecule def not found: {p}")


# ---------------------------------------------------------------------------
# RASPA input builders
# ---------------------------------------------------------------------------
def _common_header(
    framework: str,
    unit_cells: Tuple[int, int, int],
    temperature_k: float,
    pressure_bar: float,
    n_cycles: int,
    forcefield: str,
    use_charges: bool,
    cutoff: float,
) -> List[str]:
    n_init = max(500, n_cycles // 2)
    return [
        "SimulationType MonteCarlo",
        f"NumberOfCycles {n_cycles}",
        f"NumberOfInitializationCycles {n_init}",
        f"PrintEvery {max(100, n_cycles // 10)}",
        "RestartFile no",
        "ContinueAfterCrash no",
        f"Forcefield {forcefield}",
        f"UseChargesFromCIFFile {'yes' if use_charges else 'no'}",
        "ChargeMethod Ewald",
        "EwaldPrecision 1e-6",
        f"CutOffVDW {cutoff:.3f}",
        f"CutOffChargeCharge {cutoff:.3f}",
        "RemoveAtomNumberCodeFromLabel yes",
        "Framework 0",
        f"FrameworkName {framework}",
        f"UnitCells {unit_cells[0]} {unit_cells[1]} {unit_cells[2]}",
        f"ExternalTemperature {temperature_k:.6f}",
        f"ExternalPressure {pressure_bar * 1e5:.6f}",
    ]


def build_single_component_input(
    framework: str, unit_cells: Tuple[int, int, int],
    temperature_k: float, pressure_bar: float,
    molecule: str, mol_dir: str,
    n_cycles: int, forcefield: str, use_charges: bool, cutoff: float,
) -> str:
    lines = _common_header(framework, unit_cells, temperature_k, pressure_bar,
                           n_cycles, forcefield, use_charges, cutoff)
    lines += [
        f"Component 0 MoleculeName {molecule}",
        f"            MoleculeDefinition {mol_dir}",
        "            TranslationProbability 1.0",
        "            RotationProbability 1.0",
        "            ReinsertionProbability 1.0",
        "            SwapProbability 1.0",
        "            CreateNumberOfMolecules 0",
    ]
    return "\n".join(lines) + "\n"


def build_multicomponent_input(
    framework: str, unit_cells: Tuple[int, int, int],
    cond: Condition,
    n_cycles: int, forcefield: str, use_charges: bool, cutoff: float,
    co2_dir: str, n2_dir: str, h2o_dir: str,
) -> str:
    lines = _common_header(framework, unit_cells, cond.temperature_k,
                           cond.pressure_bar, n_cycles, forcefield, use_charges, cutoff)
    for i, (name, mol_dir, frac) in enumerate([
        ("CO2", co2_dir, cond.x_co2_total),
        ("N2",  n2_dir,  cond.x_n2_total),
        ("H2O", h2o_dir, cond.x_h2o_total),
    ]):
        lines += [
            f"Component {i} MoleculeName {name}",
            f"            MoleculeDefinition {mol_dir}",
            f"            MolFraction {frac:.12f}",
            "            TranslationProbability 1.0",
            "            RotationProbability 1.0",
            "            ReinsertionProbability 1.0",
            "            SwapProbability 1.0",
            "            CreateNumberOfMolecules 0",
        ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# RASPA runner  ← CRITICAL BUG FIXED HERE
# ---------------------------------------------------------------------------
def _collect_output_files(workdir: Path) -> str:
    pieces: List[str] = []
    for p in sorted(workdir.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("output") or p.suffix.lower() in {".data", ".txt"}:
            try:
                pieces.append(f"\n===== {p.relative_to(workdir)} =====\n")
                pieces.append(p.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                pass
    return "\n".join(pieces)


def _run_once(
    raspa_path: Path, runtime_root: Path, framework_cif_dir: Path,
    source_cif: Path, workspace: Path, input_text: str,
) -> RASPAResult:
    """Single RASPA execution — no retry logic here."""
    name = source_cif.stem
    runtime_cif = framework_cif_dir / f"{name}.cif"
    canonicalize_cif(source_cif.resolve(), runtime_cif)

    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / f"{name}.canonical.cif").write_bytes(runtime_cif.read_bytes())
    (workspace / "simulation.input").write_text(input_text, encoding="utf-8")

    env = os.environ.copy()
    env["RASPA_DIR"] = str(runtime_root.resolve())

    try:
        proc = subprocess.run(
            [str(raspa_path.resolve())],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            env=env,
            timeout=7200,
        )
    except subprocess.TimeoutExpired:
        log = workspace / "raspa_full_output.log"
        log.write_text("TIMEOUT after 7200 s", encoding="utf-8")
        return RASPAResult(ok=False, combined_text="TIMEOUT", log_path=log, returncode=-9)

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    file_out = _collect_output_files(workspace)
    combined = "\n".join(p for p in [stdout, stderr, file_out] if p.strip())

    log = workspace / "raspa_full_output.log"
    log.write_text(combined, encoding="utf-8", errors="replace")

    ok = (proc.returncode == 0)
    return RASPAResult(ok=ok, combined_text=combined, log_path=log, returncode=proc.returncode)


def run_raspa_job(
    raspa_path: Path, runtime_root: Path, framework_cif_dir: Path,
    source_cif: Path, workspace: Path, input_text: str,
    max_retries: int = 3, retry_delay: int = 60,
) -> RASPAResult:
    """
    Run a RASPA job with retry on failure.

    FIX: Previous version caught CalledProcessError which subprocess.run
    (without check=True) never raises.  We now check result.ok and retry
    on any non-zero returncode or timeout.
    """
    last: Optional[RASPAResult] = None
    for attempt in range(max_retries):
        result = _run_once(raspa_path, runtime_root, framework_cif_dir,
                           source_cif, workspace, input_text)
        if result.ok:
            return result
        last = result
        if attempt < max_retries - 1:
            wait = retry_delay * (2 ** attempt)   # 60s, 120s, 240s …
            logger.warning(
                f"  Attempt {attempt + 1}/{max_retries} failed "
                f"(rc={result.returncode}). Retrying in {wait}s …"
            )
            time.sleep(wait)
    return last  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------
def _extract_float(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _parse_component(text: str, molecule: str) -> Dict[str, Optional[float]]:
    pat = re.compile(
        rf"Component\s+\d+\s+\[{re.escape(molecule)}\](.*?)"
        r"(?=Component\s+\d+\s+\[|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    matches = list(pat.finditer(text))
    block = matches[-1].group(1) if matches else ""
    return {
        "loading_molkg": _extract_float(
            r"Average loading absolute \[mol/kg framework\]\s+"
            r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", block),
        "loading_moluc": _extract_float(
            r"Average loading absolute \[molecules/unit cell\]\s+"
            r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", block),
        "enthalpy_kjmol": _extract_float(
            r"Enthalpy of adsorption \[kJ/mol\]\s+"
            r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", block),
    }


def parse_single(text: str, molecule: str) -> Dict[str, Optional[float]]:
    d = _parse_component(text, molecule)
    return {
        "loading_molkg":  d["loading_molkg"],
        "loading_moluc":  d["loading_moluc"],
        "enthalpy_kjmol": d["enthalpy_kjmol"],
    }


def parse_multi(text: str) -> Dict[str, Optional[float]]:
    co2 = _parse_component(text, "CO2")
    n2  = _parse_component(text, "N2")
    h2o = _parse_component(text, "H2O")
    sel = None
    if co2["loading_molkg"] is not None and n2["loading_molkg"] and n2["loading_molkg"] > 0:
        sel = co2["loading_molkg"] / n2["loading_molkg"]
    return {
        "co2_loading_molkg":    co2["loading_molkg"],
        "n2_loading_molkg":     n2["loading_molkg"],
        "h2o_loading_molkg":    h2o["loading_molkg"],
        "co2_loading_moluc":    co2["loading_moluc"],
        "n2_loading_moluc":     n2["loading_moluc"],
        "h2o_loading_moluc":    h2o["loading_moluc"],
        "co2_enthalpy_kjmol":   co2["enthalpy_kjmol"],
        "n2_enthalpy_kjmol":    n2["enthalpy_kjmol"],
        "h2o_enthalpy_kjmol":   h2o["enthalpy_kjmol"],
        "co2_over_n2_selectivity": sel,
    }


# ---------------------------------------------------------------------------
# Stage 1 — Low-fidelity single-component pressure sweep
# ---------------------------------------------------------------------------
def run_stage1_isotherms(
    cif_paths: Sequence[Path], output_dir: Path,
    runtime_root: Path, framework_cif_dir: Path, raspa_path: Path,
    temperatures_k: Sequence[float], pressures_bar: Sequence[float],
    n_cycles: int, keep_workspaces: bool,
    forcefield: str, use_charges: bool, cutoff: float,
    co2_dir: str, n2_dir: str,
    max_retries: int = 3, retry_delay: int = 60,
) -> pd.DataFrame:
    """
    Run single-component RASPA at every (MOF, component, T, P) point.

    FIX: Previous version only ran at max(pressures_bar) — a single point
    per isotherm — which cannot compute working capacity.  We now sweep ALL
    pressures so that Stage 2 can compute ΔN = N(P_high) − N(P_low).
    """
    rows: List[Dict] = []
    ws_root = ensure_dir(output_dir / "stage1_workspace")
    components = [("CO2", co2_dir), ("N2", n2_dir)]
    total = len(cif_paths) * len(components) * len(temperatures_k) * len(pressures_bar)
    job = 0

    logger.info(f"\n[Stage 1] Single-component isotherms: {total} jobs across {len(cif_paths)} MOFs")

    for cif in cif_paths:
        name = cif.stem
        uc = determine_unit_cells(cif, cutoff)
        for mol, mol_dir in components:
            for t in temperatures_k:
                for p in pressures_bar:
                    job += 1
                    ws = ws_root / f"{name}__{mol}__T{t:.0f}__P{p:g}"
                    inp = build_single_component_input(
                        name, uc, t, p, mol, mol_dir,
                        n_cycles, forcefield, use_charges, cutoff,
                    )
                    logger.info(f"  [{job}/{total}] {name} | {mol} | T={t:.0f}K P={p:g}bar")
                    res = run_raspa_job(raspa_path, runtime_root, framework_cif_dir,
                                       cif, ws, inp, max_retries, retry_delay)
                    if res.ok:
                        parsed = parse_single(res.combined_text, mol)
                        rows.append({"mof_id": name, "component": mol,
                                     "temperature_k": t, "pressure_bar": p,
                                     "log": str(res.log_path), **parsed})
                    else:
                        logger.warning(f"    FAILED rc={res.returncode}: {tail_text(res.combined_text, 5)}")
                    if not keep_workspaces and ws.exists():
                        shutil.rmtree(ws, ignore_errors=True)

    df = pd.DataFrame(rows)
    out = output_dir / "stage1_isotherms.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"[Stage 1] {len(df)}/{total} succeeded → {out}")
    return df


# ---------------------------------------------------------------------------
# Stage 2 — TMMC→GCMC selection
# ---------------------------------------------------------------------------
def _pareto_front(costs: np.ndarray) -> np.ndarray:
    """
    Return boolean mask of Pareto-optimal rows (maximise all columns).
    A point is dominated if another point is ≥ in every column and > in one.
    """
    n = costs.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(costs[j] >= costs[i]) and np.any(costs[j] > costs[i]):
                dominated[i] = True
                break
    return ~dominated


def select_mofs_for_gcmc(
    stage1_df: pd.DataFrame,
    gcmc_budget: int,
    adsorption_pressure_bar: float,
    desorption_pressure_bar: float,
    temperature_k: float = 298.15,
    exploration_fraction: float = 0.2,
) -> List[str]:
    """
    Pick the best MOFs for expensive GCMC using Stage 1 single-component data.

    Strategy
    --------
    For each MOF we compute two KPIs from the CO2 and N2 single-component
    isotherms at the reference temperature:

      selectivity_proxy  = KH_CO2 / KH_N2
                         (Henry constants from low-pressure loading / pressure)

      working_capacity   = q_CO2(P_ads) − q_CO2(P_des)
                         (usable CO2 loading over a PSA/VSA swing)

    We then find the Pareto front over (selectivity, working_capacity),
    fill remaining budget with the highest-selectivity non-Pareto candidates,
    and finally replace exploration_fraction of slots with diverse candidates
    (highest selectivity that are NOT already selected) to avoid clustering.

    Parameters
    ----------
    stage1_df               : output of run_stage1_isotherms
    gcmc_budget             : how many MOFs to label with GCMC (e.g. 500)
    adsorption_pressure_bar : high pressure point for working capacity (e.g. 5.0)
    desorption_pressure_bar : low pressure point for working capacity  (e.g. 0.1)
    temperature_k           : reference temperature for KPI computation
    exploration_fraction    : fraction of budget reserved for diversity
    """
    df = stage1_df.copy()
    df = df[df["temperature_k"].between(temperature_k - 1.0, temperature_k + 1.0)]

    if df.empty:
        logger.warning("Stage 1 data has no rows near reference T — selecting all MOFs")
        return list(stage1_df["mof_id"].unique())

    mof_ids = df["mof_id"].unique().tolist()
    records: List[Dict] = []

    for mof in mof_ids:
        sub = df[df["mof_id"] == mof]
        co2 = sub[sub["component"] == "CO2"].sort_values("pressure_bar")
        n2  = sub[sub["component"] == "N2" ].sort_values("pressure_bar")
        if co2.empty or n2.empty:
            continue

        # Henry constants: slope at lowest pressure point (linear regime)
        kh_co2 = float(co2.iloc[0]["loading_molkg"]) / float(co2.iloc[0]["pressure_bar"]) \
            if co2.iloc[0]["loading_molkg"] is not None and co2.iloc[0]["pressure_bar"] > 0 else 0.0
        kh_n2  = float(n2.iloc[0]["loading_molkg"])  / float(n2.iloc[0]["pressure_bar"]) \
            if n2.iloc[0]["loading_molkg"]  is not None and n2.iloc[0]["pressure_bar"]  > 0 else 0.0

        sel = kh_co2 / max(kh_n2, 1e-9)

        # Working capacity: interpolate at adsorption / desorption pressures
        def interp_loading(rows: pd.DataFrame, p_target: float) -> float:
            if rows.empty or rows["loading_molkg"].isna().all():
                return 0.0
            rows = rows.dropna(subset=["loading_molkg"])
            if len(rows) == 1:
                return float(rows.iloc[0]["loading_molkg"])
            return float(np.interp(p_target,
                                   rows["pressure_bar"].values,
                                   rows["loading_molkg"].values))

        q_ads = interp_loading(co2, adsorption_pressure_bar)
        q_des = interp_loading(co2, desorption_pressure_bar)
        wc = max(0.0, q_ads - q_des)

        records.append({"mof_id": mof, "selectivity": sel, "working_capacity": wc})

    if not records:
        logger.warning("Could not compute KPIs for any MOF — returning all")
        return mof_ids

    kpi_df = pd.DataFrame(records)

    # Normalise to [0, 1] for Pareto
    for col in ["selectivity", "working_capacity"]:
        rng = kpi_df[col].max() - kpi_df[col].min()
        kpi_df[f"{col}_norm"] = (kpi_df[col] - kpi_df[col].min()) / max(rng, 1e-9)

    costs = kpi_df[["selectivity_norm", "working_capacity_norm"]].values
    pareto_mask = _pareto_front(costs)
    pareto_mofs = kpi_df.loc[pareto_mask, "mof_id"].tolist()

    n_explore  = max(1, int(gcmc_budget * exploration_fraction))
    n_exploit  = gcmc_budget - n_explore

    # Exploitation: Pareto first, then top selectivity
    exploit_candidates = (
        kpi_df.loc[pareto_mask].sort_values("selectivity", ascending=False)["mof_id"].tolist()
        + kpi_df.loc[~pareto_mask].sort_values("selectivity", ascending=False)["mof_id"].tolist()
    )
    selected = exploit_candidates[:n_exploit]

    # Exploration: highest working capacity NOT already selected
    explore_pool = kpi_df[~kpi_df["mof_id"].isin(selected)] \
        .sort_values("working_capacity", ascending=False)["mof_id"].tolist()
    selected += explore_pool[:n_explore]

    selected = selected[:gcmc_budget]

    logger.info(
        f"[Stage 2] Selected {len(selected)}/{len(mof_ids)} MOFs for GCMC\n"
        f"  Pareto-front size : {pareto_mask.sum()}\n"
        f"  Exploitation slots: {n_exploit}\n"
        f"  Exploration slots : {n_explore}\n"
        f"  Top selectivity   : {kpi_df['selectivity'].max():.1f}  "
        f"(CO2/N2 Henry ratio)\n"
        f"  Top working cap.  : {kpi_df['working_capacity'].max():.3f} mol/kg"
    )
    return selected


# ---------------------------------------------------------------------------
# Stage 3 — High-fidelity multicomponent GCMC
# ---------------------------------------------------------------------------
def run_stage3_gcmc(
    cif_paths: Sequence[Path], output_dir: Path,
    runtime_root: Path, framework_cif_dir: Path, raspa_path: Path,
    conditions: Sequence[Condition],
    n_cycles: int, keep_workspaces: bool,
    forcefield: str, use_charges: bool, cutoff: float,
    co2_dir: str, n2_dir: str, h2o_dir: str,
    max_retries: int = 3, retry_delay: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict] = []
    failures: List[Dict] = []
    ws_root = ensure_dir(output_dir / "stage3_workspace")
    total = len(cif_paths) * len(conditions)
    job = 0

    logger.info(f"\n[Stage 3] Multicomponent GCMC: {total} jobs across {len(cif_paths)} MOFs")

    for cif in cif_paths:
        name = cif.stem
        uc = determine_unit_cells(cif, cutoff)
        for cond in conditions:
            job += 1
            logger.info(
                f"  [{job}/{total}] {name} | "
                f"T={cond.temperature_k:.0f}K P={cond.pressure_bar:g}bar RH={cond.rh*100:.0f}%"
            )
            ws = ws_root / (
                f"{name}__T{cond.temperature_k:.0f}"
                f"__P{cond.pressure_bar:g}__RH{cond.rh:.3f}"
            )
            inp = build_multicomponent_input(
                name, uc, cond, n_cycles, forcefield, use_charges, cutoff,
                co2_dir, n2_dir, h2o_dir,
            )
            res = run_raspa_job(raspa_path, runtime_root, framework_cif_dir,
                                cif, ws, inp, max_retries, retry_delay)
            if res.ok:
                parsed = parse_multi(res.combined_text)
                rows.append({
                    "mof_id":        name,
                    "temperature_k": cond.temperature_k,
                    "pressure_bar":  cond.pressure_bar,
                    "rh":            cond.rh,
                    "p_co2_bar":     cond.p_co2_bar,
                    "p_n2_bar":      cond.p_n2_bar,
                    "p_h2o_bar":     cond.p_h2o_bar,
                    "x_co2_total":   cond.x_co2_total,
                    "x_n2_total":    cond.x_n2_total,
                    "x_h2o_total":   cond.x_h2o_total,
                    "y_co2_dry":     cond.y_co2_dry,
                    "y_n2_dry":      cond.y_n2_dry,
                    # Columns expected by AdsorptionDataset:
                    "mu_CO2":        cond.mu_co2,
                    "mu_N2":         cond.mu_n2,
                    "mu_H2O":        cond.mu_h2o,
                    "T":             cond.temperature_k,
                    "log":           str(res.log_path),
                    **parsed,
                })
            else:
                failures.append({
                    "mof_id": name, "temperature_k": cond.temperature_k,
                    "pressure_bar": cond.pressure_bar, "rh": cond.rh,
                    "returncode": res.returncode,
                    "error_tail": tail_text(res.combined_text, 40),
                })
                logger.warning(f"    FAILED rc={res.returncode}")
            if not keep_workspaces and ws.exists():
                shutil.rmtree(ws, ignore_errors=True)

    gcmc_df    = pd.DataFrame(rows)
    failure_df = pd.DataFrame(failures)

    gcmc_out = output_dir / "gcmc_adsorption.parquet"
    fail_out = output_dir / "gcmc_failures.parquet"
    gcmc_df.to_parquet(gcmc_out,    index=False)
    failure_df.to_parquet(fail_out, index=False)

    logger.info(f"[Stage 3] {len(gcmc_df)}/{total} succeeded → {gcmc_out}")
    if not failure_df.empty:
        logger.info(f"  {len(failure_df)} failures → {fail_out}")
    return gcmc_df, failure_df


# ---------------------------------------------------------------------------
# Build training dataset (AdsorptionDataset layout)
# ---------------------------------------------------------------------------
def build_training_dataset(gcmc_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Ensure the saved parquet has the column names that AdsorptionDataset
    expects: mof_id, mu_CO2, mu_N2, mu_H2O, T,
             co2_loading_molkg, n2_loading_molkg, h2o_loading_molkg.
    Drop rows where any loading is None (simulation produced NaN).
    """
    df = gcmc_df.dropna(subset=["co2_loading_molkg", "n2_loading_molkg", "h2o_loading_molkg"])
    out = output_dir / "adsorption_training.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"\n[Training data] {len(df)} clean rows → {out}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Run RASPA humid CO2/N2/H2O simulations with TMMC→GCMC selection."
    )
    # Paths
    p.add_argument("--cif-dir",      default="data/intermediate/cifs_sanitized")
    p.add_argument("--output-dir",   default="data/processed/adsorption")
    p.add_argument("--raspa-path",   required=True, help="Path to RASPA2 simulate binary")
    p.add_argument("--raspa-data-dir", required=True, help="RASPA2 data/share root")
    p.add_argument("--config",       help="Optional pipeline.yaml config file")

    # MOF selection
    p.add_argument("--max-mofs",    type=int, default=0,   help="0=all CIFs")
    p.add_argument("--gcmc-budget", type=int, default=500, help="MOFs to label with GCMC")
    p.add_argument("--exploration-fraction", type=float, default=0.2,
                   help="Fraction of GCMC budget for diversity exploration")

    # Simulation parameters
    p.add_argument("--n-cycles",    type=int,   default=5000)
    p.add_argument("--temperatures", default="298.15,313.15,333.15")
    p.add_argument("--pressures",    default="0.1,0.5,1.0,5.0,10.0")
    p.add_argument("--rhs",          default="0.0,0.05,0.15")
    p.add_argument("--dry-co2-frac", type=float, default=0.15)
    p.add_argument("--cutoff",       type=float, default=12.0)
    p.add_argument("--forcefield",   default="GenericMOFs")
    p.add_argument("--use-framework-charges", action="store_true")
    p.add_argument("--keep-workspaces",       action="store_true")

    # Molecule definitions
    p.add_argument("--co2-definition", default="TraPPE")
    p.add_argument("--n2-definition",  default="TraPPE")
    p.add_argument("--h2o-definition", default="TraPPE")
    p.add_argument("--co2-def-file",   default=None)
    p.add_argument("--n2-def-file",    default=None)
    p.add_argument("--h2o-def-file",   default=None)

    # Retry
    p.add_argument("--max-retries",  type=int, default=3)
    p.add_argument("--retry-delay",  type=int, default=60, help="Initial delay seconds")

    # Stage control (useful for resuming)
    p.add_argument("--skip-stage1",  action="store_true", help="Load existing stage1 parquet")
    p.add_argument("--skip-stage2",  action="store_true", help="Run GCMC on all (skip selection)")

    args = p.parse_args()

    # Optional YAML override
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        sim = cfg.get("simulation", {})
        for key, val in sim.items():
            attr = key.replace("-", "_")
            if hasattr(args, attr):
                setattr(args, attr, val)

    # Resolve paths
    cif_dir     = Path(args.cif_dir).expanduser().resolve()
    output_dir  = ensure_dir(Path(args.output_dir).expanduser().resolve())
    raspa_path  = Path(args.raspa_path).expanduser().resolve()
    raspa_dd    = Path(args.raspa_data_dir).expanduser().resolve()

    if not cif_dir.exists():
        raise FileNotFoundError(f"CIF dir not found: {cif_dir}")
    if not raspa_path.exists():
        raise FileNotFoundError(f"RASPA binary not found: {raspa_path}")

    # Load CIFs
    all_cifs = sorted(cif_dir.glob("*.cif"))
    if args.max_mofs > 0:
        all_cifs = all_cifs[:args.max_mofs]
    all_cifs = [c.resolve() for c in all_cifs]
    if not all_cifs:
        raise FileNotFoundError(f"No CIF files in {cif_dir}")

    temps    = parse_float_list(args.temperatures)
    pressures = parse_float_list(args.pressures)
    rhs      = parse_float_list(args.rhs)

    # RASPA setup
    raspa_share  = detect_raspa_share(raspa_dd)
    ff_name      = resolve_forcefield(raspa_share, args.forcefield)
    runtime_root, runtime_share, fw_cif_dir = prepare_runtime(output_dir, raspa_share)

    # Molecule directories
    def _resolve_mol(def_file_arg, mol_name, preferred):
        if def_file_arg:
            return install_user_def(runtime_share, mol_name, Path(def_file_arg))
        return find_molecule_dir(raspa_share, mol_name, preferred)

    co2_dir = _resolve_mol(args.co2_def_file, "CO2", args.co2_definition)
    n2_dir  = _resolve_mol(args.n2_def_file,  "N2",  args.n2_definition)
    h2o_dir = _resolve_mol(args.h2o_def_file, "H2O", args.h2o_definition)

    check_ff_assets(runtime_share, ff_name)
    check_mol_def(runtime_share, co2_dir, "CO2")
    check_mol_def(runtime_share, n2_dir,  "N2")
    check_mol_def(runtime_share, h2o_dir, "H2O")

    conditions = build_conditions(temps, pressures, rhs, args.dry_co2_frac)

    logger.info("=" * 65)
    logger.info(f"UC-TPNO SIMULATION PIPELINE")
    logger.info(f"  CIFs            : {len(all_cifs)}")
    logger.info(f"  Temperatures (K): {temps}")
    logger.info(f"  Pressures (bar) : {pressures}")
    logger.info(f"  RH values       : {rhs}")
    logger.info(f"  Dry CO2 frac    : {args.dry_co2_frac}")
    logger.info(f"  Valid cond/MOF  : {len(conditions)}")
    logger.info(f"  GCMC budget     : {args.gcmc_budget} MOFs")
    logger.info(f"  Forcefield      : {ff_name}")
    logger.info(f"  CO2 mol dir     : {co2_dir}")
    logger.info(f"  N2  mol dir     : {n2_dir}")
    logger.info(f"  H2O mol dir     : {h2o_dir}")
    logger.info("=" * 65)

    # ------------------------------------------------------------------
    # Stage 1 — single-component pressure sweep (ALL MOFs)
    # ------------------------------------------------------------------
    stage1_parquet = output_dir / "stage1_isotherms.parquet"
    if args.skip_stage1 and stage1_parquet.exists():
        logger.info(f"\n[Stage 1] Skipped — loading {stage1_parquet}")
        stage1_df = pd.read_parquet(stage1_parquet)
    else:
        stage1_df = run_stage1_isotherms(
            all_cifs, output_dir, runtime_root, fw_cif_dir, raspa_path,
            temps, pressures,
            args.n_cycles, args.keep_workspaces,
            ff_name, args.use_framework_charges, args.cutoff,
            co2_dir, n2_dir,
            args.max_retries, args.retry_delay,
        )

    # ------------------------------------------------------------------
    # Stage 2 — select best MOFs for GCMC
    # ------------------------------------------------------------------
    if args.skip_stage2:
        gcmc_cifs = all_cifs
        logger.info(f"\n[Stage 2] Skipped — using all {len(gcmc_cifs)} MOFs for GCMC")
    else:
        selected_ids = select_mofs_for_gcmc(
            stage1_df,
            gcmc_budget=args.gcmc_budget,
            adsorption_pressure_bar=max(pressures),
            desorption_pressure_bar=min(pressures),
            temperature_k=temps[0],
            exploration_fraction=args.exploration_fraction,
        )
        # Save selection list
        sel_file = output_dir / "gcmc_selected_mofs.txt"
        sel_file.write_text("\n".join(selected_ids), encoding="utf-8")
        logger.info(f"[Stage 2] Selection saved → {sel_file}")

        id_set   = set(selected_ids)
        gcmc_cifs = [c for c in all_cifs if c.stem in id_set]

    # ------------------------------------------------------------------
    # Stage 3 — multicomponent GCMC on selected MOFs
    # ------------------------------------------------------------------
    gcmc_df, _ = run_stage3_gcmc(
        gcmc_cifs, output_dir, runtime_root, fw_cif_dir, raspa_path,
        conditions,
        args.n_cycles, args.keep_workspaces,
        ff_name, args.use_framework_charges, args.cutoff,
        co2_dir, n2_dir, h2o_dir,
        args.max_retries, args.retry_delay,
    )

    train_df = build_training_dataset(gcmc_df, output_dir)

    logger.info("\n" + "=" * 65)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Stage 1 rows     : {len(stage1_df)}")
    logger.info(f"  GCMC MOFs        : {len(gcmc_cifs)}")
    logger.info(f"  Training rows    : {len(train_df)}")
    logger.info(f"  Output dir       : {output_dir}")
    logger.info("Next: python scripts/05_train_model.py")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()