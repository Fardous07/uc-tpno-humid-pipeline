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
import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# FIX: this script used to compute mu via a local, ideal-gas-only
# mu_ideal_gas() that silently disagreed with src.utils.chemistry's
# Peng-Robinson mixture EOS (the version 09_benchmark.py already queries the
# model with). Standardizing on chemistry.py so training labels and
# inference-time conditions use identical thermodynamics.
from src.utils.chemistry import (
    mixture_pressure_to_chemical_potentials,
    water_saturation_pressure_antoine,
)
from src.utils.cif_charges import read_site_charges

# Force fields shipped with this repo (see scripts/build_raspa_forcefield.py),
# installed into every RASPA runtime tree by prepare_runtime().
REPO_FORCEFIELD_DIR = Path(__file__).resolve().parent.parent / "forcefields"
DEFAULT_FORCEFIELD = "UFF_TIP4P2005"

# Wall-clock limit per RASPA job (set from --job-timeout-hours in main()).
# FIX: was a hard-coded 7200 s. With framework charges, high-pressure humid
# jobs on large MOFs take 2-6 h, so they were killed at 2 h and retried from
# scratch up to 3x (~6 h wasted per job) before being recorded as failures.
RASPA_TIMEOUT_S = 48 * 3600

# ADDED: GC-TMMC existed in src/simulation/gctmmc/ but nothing in this
# pipeline ever drove it -- Stage 1 only ever ran ordinary GCMC pressure
# sweeps. See run_stage1_isotherms_gctmmc() below and --stage1-method.
from src.simulation.gctmmc.parser import GCTMMCConfig, GCTMMCRunner

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


def append_checkpoint(path: Path, row: Dict) -> None:
    """
    Append one result row to a checkpoint CSV, keeping columns aligned.

    ADDED: a plain to_csv(mode="a", header=False) silently shifts values
    into the wrong columns if the row's columns differ from the file's
    header -- e.g. a unit started before the *_err columns existed and
    resumed after. Rows are written in the header's order; if the row
    brings new columns, the file is rewritten once with the union.
    """
    if not path.exists() or path.stat().st_size == 0:
        pd.DataFrame([row]).to_csv(path, index=False)
        return
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split(",")
    if set(row) <= set(header):
        pd.DataFrame([row]).reindex(columns=header).to_csv(path, mode="a", header=False, index=False)
    else:
        old = pd.read_csv(path)
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


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
def build_conditions(
    temperatures_k: Sequence[float],
    pressures_bar: Sequence[float],
    rhs: Sequence[float],
    dry_co2_frac: float,
) -> List[Condition]:
    """Build all valid (T, P, RH) thermodynamic conditions."""
    out: List[Condition] = []
    for t in temperatures_k:
        p_sat = water_saturation_pressure_antoine(t)
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
                mu = mixture_pressure_to_chemical_potentials(
                    y={"CO2": x_co2, "N2": x_n2, "H2O": x_h2o},
                    pressure=p_total,
                    temperature=t,
                )
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
                    mu_co2=float(mu["CO2"]),
                    mu_n2=float(mu["N2"]),
                    mu_h2o=float(mu["H2O"]),
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


def _read_atoms(input_cif: Path):
    if ase_read is None:
        raise RuntimeError("ASE required: pip install ase")
    atoms = ase_read(str(input_cif))
    if isinstance(atoms, list):
        if not atoms:
            raise ValueError(f"Empty CIF: {input_cif}")
        atoms = atoms[0]
    return atoms


def canonicalize_cif(input_cif: Path, output_cif: Path, with_charges: bool = True) -> None:
    """
    Write a minimal P1 CIF that RASPA can always parse.

    FIX: this used to write positions only. ASE never returns the CIF's
    _atom_site_charge values (get_initial_charges() is all zeros for every
    sanitised CIF), and the charge column was not written at all, so RASPA
    ran with UseChargesFromCIFFile yes against a CIF with no charges and set
    every framework charge to 0 -- no host-guest electrostatics in any
    simulation, which is why water uptake was ~0 in all 476 MOFs. Charges
    are now read with the same reader the graph builder uses (file order ==
    ASE atom order for the P1 sanitised CIFs) and written back out.
    """
    atoms = _read_atoms(input_cif)
    cl  = atoms.cell.lengths()
    ca  = atoms.cell.angles()
    sp  = atoms.get_scaled_positions(wrap=True)
    sym = atoms.get_chemical_symbols()
    charges = None
    if with_charges:
        charges = read_site_charges(input_cif, len(atoms))
        if charges is None:
            raise ValueError(f"No per-atom charge column matching {len(atoms)} atoms in {input_cif.name}")
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
    if charges is not None:
        lines.append("_atom_site_charge")
    for k, (s, pos) in enumerate(zip(sym, sp)):
        counts[s] = counts.get(s, 0) + 1
        row = f"{s}{counts[s]} {s} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} 1.0"
        if charges is not None:
            row += f" {float(charges[k]):.6f}"
        lines.append(row)
    lines.append("")
    ensure_dir(output_cif.parent)
    output_cif.write_text("\n".join(lines), encoding="utf-8")


# ADDED: pre-flight checks, run once per MOF before any RASPA job. Every
# failure here used to be silent -- RASPA happily simulates an uncharged
# framework, an element with no LJ parameters, or a disordered CIF with
# atoms 0.1 A apart, and returns plausible-looking but meaningless numbers.
MIN_ATOM_DISTANCE_A = 0.7   # below any real bond (O-H 0.96, C-H 1.08 A): overlapping/disordered sites
MAX_NET_CHARGE_E    = 0.05  # per unit cell; ARC REPEAT and PACMAN charges are neutral to ~1e-5


def forcefield_pseudo_atoms(runtime_share: Path, ff_name: str) -> set:
    lines = (runtime_share / "forcefield" / ff_name / "pseudo_atoms.def").read_text().splitlines()
    return {ln.split()[0] for ln in lines[3:] if ln.strip() and not ln.startswith("#")}


def validate_framework(cif: Path, ff_atoms: set, use_charges: bool) -> List[str]:
    """Return a list of problems that make this MOF unsafe to simulate (empty = OK)."""
    from ase.neighborlist import neighbor_list
    problems: List[str] = []
    try:
        atoms = _read_atoms(cif)
    except Exception as e:
        return [f"unreadable CIF ({e})"]

    missing = sorted(set(atoms.get_chemical_symbols()) - ff_atoms)
    if missing:
        problems.append(f"elements without force-field parameters: {missing}")

    d = neighbor_list("d", atoms, MIN_ATOM_DISTANCE_A)
    if len(d):
        problems.append(f"{len(d) // 2} atom pairs closer than {MIN_ATOM_DISTANCE_A} A "
                        f"(min {d.min():.2f} A) -- overlapping/disordered sites")

    if use_charges:
        q = read_site_charges(cif, len(atoms))
        if q is None:
            problems.append("no per-atom charge column")
        elif not np.any(np.abs(q) > 1e-4):
            problems.append("all framework charges are zero -- run scripts/assign_charges.py first")
        elif abs(float(q.sum())) > MAX_NET_CHARGE_E:
            problems.append(f"framework net charge {q.sum():+.3f} e (limit {MAX_NET_CHARGE_E})")
    return problems


def validate_frameworks(cif_paths: Sequence[Path], ff_atoms: set, use_charges: bool,
                        output_dir: Path) -> List[Path]:
    """Validate every MOF once; log and record rejects, return the usable ones."""
    ok: List[Path] = []
    rejects: List[Dict] = []
    for cif in cif_paths:
        problems = validate_framework(cif, ff_atoms, use_charges)
        if problems:
            logger.warning(f"  REJECTED {cif.stem}: {'; '.join(problems)}")
            rejects.append({"mof_id": cif.stem, "problems": "; ".join(problems)})
        else:
            ok.append(cif)
    if rejects:
        out = output_dir / "rejected_frameworks.csv"
        pd.DataFrame(rejects).to_csv(out, index=False)
        logger.warning(f"[Pre-flight] {len(rejects)}/{len(cif_paths)} MOFs rejected -> {out}")
    logger.info(f"[Pre-flight] {len(ok)}/{len(cif_paths)} MOFs passed framework checks")
    return ok


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
    # FIX: a missing "GenericMOFs" used to fall back silently to RASPA's
    # ExampleMOFsForceField, whose "C" (TraPPE alkane) entry and "_" prefix
    # wildcards mis-typed framework carbon and ~30 other elements (see
    # scripts/build_raspa_forcefield.py). A missing force field is now an error.
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
    if REPO_FORCEFIELD_DIR.is_dir():
        for ff in REPO_FORCEFIELD_DIR.iterdir():
            if ff.is_dir():
                shutil.copytree(ff, runtime_share / "forcefield" / ff.name, dirs_exist_ok=True)
    cif_dir = ensure_dir(runtime_share / "structures" / "cif")
    return runtime_root, runtime_share, cif_dir


def check_ff_assets(runtime_share: Path, ff_name: str) -> None:
    """
    FIX: force_field.def used to be required, but it's genuinely optional
    in RASPA2 -- it only holds explicit pairwise overrides beyond the
    general mixing rule. Verified directly: a real, official RASPA2 build's
    bundled ExampleMOFsForceField has no force_field.def at all (only
    pseudo_atoms.def + force_field_mixing_rules.def); RASPA logs
    "'force_field.def' file not found and therefore not used" and runs a
    complete GCMC simulation correctly without it. Requiring it here would
    have made check_ff_assets() raise FileNotFoundError on the very first
    real invocation of this script against a standard RASPA2 install.
    """
    ff_dir = runtime_share / "forcefield" / ff_name
    if not ff_dir.exists():
        raise FileNotFoundError(f"Forcefield dir missing: {ff_dir}")
    required = ["pseudo_atoms.def"]
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
    use_charges: bool = True,
) -> RASPAResult:
    """Single RASPA execution — no retry logic here."""
    name = source_cif.stem
    runtime_cif = framework_cif_dir / f"{name}.cif"
    canonicalize_cif(source_cif.resolve(), runtime_cif, with_charges=use_charges)

    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / f"{name}.canonical.cif").write_bytes(runtime_cif.read_bytes())
    # FIX: RASPA2 builds the framework path in a fixed 256-char buffer. The
    # runtime-share path + a long MOF id reached 251 chars in production and
    # crashed ("*** buffer overflow detected ***", rc=-6) at 259 in a test
    # dir. RASPA checks the working directory first, so a copy here keeps the
    # path short regardless of where the output dir lives.
    (workspace / f"{name}.cif").write_bytes(runtime_cif.read_bytes())
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
            timeout=RASPA_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        log = workspace / "raspa_full_output.log"
        log.write_text(f"TIMEOUT after {RASPA_TIMEOUT_S} s", encoding="utf-8")
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
    use_charges: bool = True,
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
                           source_cif, workspace, input_text, use_charges)
        if result.ok:
            return result
        last = result
        # A timeout is deterministic -- rerunning the same job from scratch
        # just burns the same wall time again. Only retry real crashes.
        if result.combined_text == "TIMEOUT":
            logger.warning(f"  Job exceeded {RASPA_TIMEOUT_S / 3600:.0f} h limit -- not retrying")
            break
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
        # ADDED: RASPA's block-average error bar ("value +/- err") -- was
        # parsed away. Lets training weight labels by statistical reliability.
        "loading_molkg_err": _extract_float(
            r"Average loading absolute \[mol/kg framework\]\s+\S+\s+\+/-\s+"
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
        "loading_molkg_err": d["loading_molkg_err"],
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
        "co2_loading_molkg_err": co2["loading_molkg_err"],
        "n2_loading_molkg_err":  n2["loading_molkg_err"],
        "h2o_loading_molkg_err": h2o["loading_molkg_err"],
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
    co2_dir: str, n2_dir: str, h2o_dir: str,
    max_retries: int = 3, retry_delay: int = 60,
    h2o_rhs: Sequence[float] = (0.05, 0.15),
) -> pd.DataFrame:
    """
    Run single-component RASPA at every (MOF, component, T, P) point.

    FIX: H2O used to be swept over the same total pressures as CO2/N2
    (0.1/1/10 bar), all ABOVE water's saturation pressure (0.074 bar at
    313 K) -- pure supersaturated water that condenses in the pores once
    framework charges are on, giving a meaningless "Henry constant" for the
    Stage 2 water filter and extremely slow jobs. H2O now runs at the
    Stage 3 relative humidities instead: p = RH * p_sat(T).

    FIX: Previous version only ran at max(pressures_bar) — a single point
    per isotherm — which cannot compute working capacity.  We now sweep ALL
    pressures so that Stage 2 can compute ΔN = N(P_high) − N(P_low).

    FIX: previous version only screened CO2 and N2 here, so the Stage 2 MOF
    selection that determines which MOFs even get the expensive humid
    multicomponent GCMC in Stage 3 was blind to water. H2O is now included
    as a third single-component isotherm so Stage 2 can filter on CO2-vs-H2O
    affinity (see select_mofs_for_gcmc), following the water-aware screening
    funnel used on this same CoRE-MOF-2019 database in Kancharlapalli &
    Snurr, ACS Appl. Mater. Interfaces 2023, 15, 28084-28092.
    """
    rows: List[Dict] = []
    ws_root = ensure_dir(output_dir / "stage1_workspace")
    components = [("CO2", co2_dir), ("N2", n2_dir), ("H2O", h2o_dir)]
    def _pressures(mol: str, t: float) -> List[float]:
        if mol == "H2O":
            return [rh * water_saturation_pressure_antoine(t) for rh in h2o_rhs if rh > 0]
        return list(pressures_bar)

    total = len(cif_paths) * sum(len(_pressures(m, t)) for m, _ in components for t in temperatures_k)
    job = 0

    # ADDED: per-job checkpoint, same scheme as Stage 3 -- Stage 1 used to
    # keep everything in memory until the very end (~47 h single-process for
    # 998 MOFs), so any crash or shutdown lost all of it.
    checkpoint_path = output_dir / "stage1_isotherms.checkpoint.csv"
    done_keys: set = set()
    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
        rows = pd.read_csv(checkpoint_path).to_dict("records")
        done_keys = {(r["mof_id"], r["component"], round(float(r["temperature_k"]), 3),
                      round(float(r["pressure_bar"]), 6)) for r in rows}
        logger.info(f"[Stage 1] Resuming: {len(done_keys)} jobs already completed")

    logger.info(f"\n[Stage 1] Single-component isotherms: {total} jobs across {len(cif_paths)} MOFs")

    for cif in cif_paths:
        name = cif.stem
        uc = determine_unit_cells(cif, cutoff)
        for mol, mol_dir in components:
            for t in temperatures_k:
                for p in _pressures(mol, t):
                    job += 1
                    if (name, mol, round(t, 3), round(p, 6)) in done_keys:
                        continue
                    ws = ws_root / f"{name}__{mol}__T{t:.0f}__P{p:g}"
                    inp = build_single_component_input(
                        name, uc, t, p, mol, mol_dir,
                        n_cycles, forcefield, use_charges, cutoff,
                    )
                    logger.info(f"  [{job}/{total}] {name} | {mol} | T={t:.0f}K P={p:g}bar")
                    res = run_raspa_job(raspa_path, runtime_root, framework_cif_dir,
                                       cif, ws, inp, max_retries, retry_delay,
                                       use_charges)
                    if res.ok:
                        parsed = parse_single(res.combined_text, mol)
                        row = {"mof_id": name, "component": mol,
                               "temperature_k": t, "pressure_bar": p,
                               "log": str(res.log_path), **parsed}
                        rows.append(row)
                        append_checkpoint(checkpoint_path, row)
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
# Stage 1 (alternative) — real GC-TMMC single-component screening
# ---------------------------------------------------------------------------
# ADDED: this project's docstring always called Stage 1 a "low-fidelity TMMC
# proxy", but nothing ever actually drove src/simulation/gctmmc/ -- Stage 1
# was ordinary GCMC repeated at each pressure. GC-TMMC gets a MOF's full
# isotherm from ONE simulation (via the collection matrix), instead of one
# GCMC run per pressure point, so it is genuinely cheaper for screening the
# whole library. This is single-component only (GCTMMCRunner has no
# multicomponent mode in this codebase) -- Stage 3's humid multicomponent
# GCMC is unaffected either way and remains the source of real training
# labels.
def run_stage1_isotherms_gctmmc(
    cif_paths: Sequence[Path], output_dir: Path,
    runtime_root: Path, raspa_path: Path,
    temperatures_k: Sequence[float],
    n_cycles: int, keep_workspaces: bool,
    forcefield: str, use_charges: bool, cutoff: float,
) -> pd.DataFrame:
    """
    Run one GC-TMMC simulation per (MOF, component, T) and expand each
    resulting isotherm into the same row schema run_stage1_isotherms()
    produces (mof_id, component, temperature_k, pressure_bar, loading_molkg),
    so select_mofs_for_gcmc() works unchanged regardless of which Stage 1
    method produced stage1_isotherms.parquet.

    Molecule definitions (CO2/N2 -> TraPPE, H2O -> TIP4P/2005) are fixed by
    src/simulation/gctmmc/parser.py's _MOLECULE_DEFS, not by this script's
    --co2-definition/--n2-definition/--h2o-definition flags -- edit that
    dict if your RASPA install names those folders differently.
    """
    components = ("CO2", "N2", "H2O")
    total = len(cif_paths) * len(components) * len(temperatures_k)
    job = 0

    ws_root = ensure_dir(output_dir / "stage1_gctmmc_workspace")
    # FIX: GCTMMCConfig.n_init defaults to 50,000 (src/simulation/gctmmc/
    # parser.py) and was never overridden here, so EVERY GC-TMMC job ran
    # 50,000+ initialization cycles regardless of --n-cycles -- confirmed
    # in a real test: a "--n-cycles 200" run generated
    # "NumberOfInitializationCycles 50000" in the actual RASPA input file,
    # making a single test job (meant to be near-instant) take 30+ minutes.
    # At production scale (998 MOFs x 9 jobs) this would have made Stage 1
    # "screening" many times more expensive than Stage 3. Scale n_init with
    # n_cycles, matching the convention _common_header() already uses for
    # regular GCMC elsewhere in this file.
    config = GCTMMCConfig(
        raspa_path=str(raspa_path),
        raspa_data_dir=str(runtime_root),
        work_dir=str(ws_root),
        forcefield=forcefield,
        cutoff_vdw=cutoff,
        n_cycles=n_cycles,
        n_init=max(200, n_cycles // 2),
        use_charges=use_charges,
        timeout=RASPA_TIMEOUT_S,
    )
    runner = GCTMMCRunner(config)

    logger.info(f"\n[Stage 1/GC-TMMC] {total} jobs across {len(cif_paths)} MOFs "
                f"({len(components)} components x {len(temperatures_k)} temperatures, "
                f"one simulation per job yields a full isotherm)")

    rows: List[Dict] = []
    for cif in cif_paths:
        name = cif.stem
        # Canonicalize once per MOF (P1, RASPA-parseable) -- GCTMMCRunner
        # would otherwise copy the raw source CIF as-is, which is exactly
        # the messy-CIF failure mode canonicalize_cif() was written to fix.
        canon_cif = ws_root / f"{name}.canonical.cif"
        try:
            canonicalize_cif(cif, canon_cif, with_charges=use_charges)
        except Exception as e:
            logger.warning(f"    Skipping {name}: canonicalisation failed ({e})")
            job += len(components) * len(temperatures_k)
            continue

        for mol in components:
            for t in temperatures_k:
                job += 1
                logger.info(f"  [{job}/{total}] {name} | {mol} | T={t:.0f}K (GC-TMMC)")
                result = runner.run_single(
                    mof_cif=canon_cif, molecule=mol, temperature=t,
                    clean_after=not keep_workspaces,
                )
                if not result.get("success"):
                    logger.warning(f"    FAILED: {result.get('error')}")
                    continue
                for p_bar, q in zip(result["pressures"], result["loadings"]):
                    rows.append({
                        "mof_id": name, "component": mol,
                        "temperature_k": t, "pressure_bar": float(p_bar),
                        "loading_molkg": float(q),
                    })

    df = pd.DataFrame(rows)
    out = output_dir / "stage1_isotherms.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"[Stage 1/GC-TMMC] {job}/{total} jobs run -> {len(df)} isotherm points -> {out}")
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
    require_co2_over_h2o: bool = True,
) -> List[str]:
    """
    Pick the best MOFs for expensive GCMC using Stage 1 single-component data.

    Strategy
    --------
    For each MOF we compute three KPIs from the CO2, N2 and H2O
    single-component isotherms at the reference temperature:

      selectivity_proxy    = KH_CO2 / KH_N2
                           (Henry constants from low-pressure loading / pressure)

      working_capacity     = q_CO2(P_ads) − q_CO2(P_des)
                           (usable CO2 loading over a PSA/VSA swing)

      co2_h2o_selectivity  = KH_CO2 / KH_H2O
                           (water-competition indicator)

    ADDED (water-aware screening): before the Pareto step, MOFs whose water
    Henry constant exceeds their CO2 Henry constant (KH_H2O > KH_CO2, i.e.
    water outcompetes CO2 for adsorption sites) are filtered out, matching
    the "stronger CO2 interaction than H2O/N2" filter used to screen this
    same CoRE-MOF-2019 database in:
        Kancharlapalli, S.; Snurr, R. Q. "High-Throughput Screening of the
        CoRE-MOF-2019 Database for CO2 Capture from Wet Flue Gas: A
        Multi-Scale Modeling Strategy." ACS Appl. Mater. Interfaces 2023,
        15(23), 28084-28092. DOI: 10.1021/acsami.3c04079
    and the CO2-over-H2O selectivity ranking in:
        Li, S.; Chung, Y. G.; Snurr, R. Q. "High-Throughput Screening of
        Metal-Organic Frameworks for CO2 Capture in the Presence of Water."
        Langmuir 2016, 32(40), 10368-10376. DOI: 10.1021/acs.langmuir.6b02803
    Previously this function only ever saw CO2/N2 data, so the MOFs handed
    to the expensive humid Stage 3 GCMC were selected with zero regard for
    water competition — the whole point of this "humid" pipeline.

    From the water-filtered survivors we then find the Pareto front over
    (selectivity, working_capacity), fill remaining budget with the
    highest-selectivity non-Pareto candidates, and finally replace
    exploration_fraction of slots with diverse candidates (highest working
    capacity that are NOT already selected) to avoid clustering.

    Parameters
    ----------
    stage1_df               : output of run_stage1_isotherms
    gcmc_budget             : how many MOFs to label with GCMC (e.g. 500)
    adsorption_pressure_bar : high pressure point for working capacity (e.g. 5.0)
    desorption_pressure_bar : low pressure point for working capacity  (e.g. 0.1)
    temperature_k           : reference temperature for KPI computation
    exploration_fraction    : fraction of budget reserved for diversity
    require_co2_over_h2o    : apply the CO2-over-H2O Henry-constant filter
                              before Pareto selection (set False to disable
                              if H2O Stage 1 data is unavailable/incomplete)
    """
    # FIX: if every single Stage 1 job failed, stage1_df is a completely
    # empty, columnless DataFrame (pd.DataFrame([]) with no successful
    # rows to infer columns from), and the .between() filter below used
    # to raise an opaque "KeyError: 'temperature_k'" instead of pointing
    # at the real problem. Caught in testing: this happened when
    # GC-TMMC's molecule definitions pointed at a folder that doesn't
    # exist, silently failing 100% of jobs -- the crash three lines later
    # gave no hint why. Fail immediately with a clear, actionable message.
    if stage1_df.empty or "temperature_k" not in stage1_df.columns:
        raise RuntimeError(
            "Stage 1 produced ZERO successful simulation results (every "
            "job failed). This is not a 'no MOFs near this temperature' "
            "situation -- something is broken before any physics ran. "
            "Check the Stage 1 log output above for the actual RASPA "
            "error (e.g. a missing molecule/forcefield file), fix that, "
            "and rerun Stage 1 before selection can proceed."
        )

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
        h2o = sub[sub["component"] == "H2O"].sort_values("pressure_bar")
        if co2.empty or n2.empty:
            continue

        # Henry constants: slope at lowest pressure point (linear regime)
        def _henry(rows: pd.DataFrame) -> float:
            if rows.empty:
                return 0.0
            first = rows.iloc[0]
            if first["loading_molkg"] is None or first["pressure_bar"] <= 0:
                return 0.0
            return float(first["loading_molkg"]) / float(first["pressure_bar"])

        kh_co2 = _henry(co2)
        kh_n2  = _henry(n2)
        kh_h2o = _henry(h2o)   # 0.0 if H2O Stage 1 data is missing for this MOF

        sel        = kh_co2 / max(kh_n2, 1e-9)
        sel_co2_h2o = kh_co2 / max(kh_h2o, 1e-9) if kh_h2o > 0 else float("inf")

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

        records.append({
            "mof_id": mof, "selectivity": sel, "working_capacity": wc,
            "kh_co2": kh_co2, "kh_h2o": kh_h2o,
            "co2_over_h2o_selectivity": sel_co2_h2o,
            "has_h2o_data": not h2o.empty,
        })

    if not records:
        logger.warning("Could not compute KPIs for any MOF — returning all")
        return mof_ids

    kpi_df = pd.DataFrame(records)
    n_before_filter = len(kpi_df)

    # ADDED: water-competition filter (Kancharlapalli & Snurr 2023 /
    # Li, Chung & Snurr 2016 precedent) — drop MOFs where water outcompetes
    # CO2 at the Henry-constant level, before spending GCMC budget on them.
    if require_co2_over_h2o and kpi_df["has_h2o_data"].any():
        survives = (~kpi_df["has_h2o_data"]) | (kpi_df["kh_co2"] > kpi_df["kh_h2o"])
        filtered_df = kpi_df[survives]
        if filtered_df.empty:
            logger.warning(
                "[Stage 2] CO2-over-H2O filter removed every MOF — "
                "disabling it for this run and falling back to CO2/N2-only selection."
            )
        else:
            logger.info(
                f"[Stage 2] Water-competition filter: {len(filtered_df)}/{n_before_filter} "
                f"MOFs retained (KH_CO2 > KH_H2O)"
            )
            # FIX: with real framework charges most MOFs have KH_H2O > KH_CO2,
            # so a hard filter can leave far fewer MOFs than the budget (it
            # removed every MOF in a 2-MOF test). Survivors now get priority;
            # the budget is topped up with the removed MOFs that bind water
            # least strongly relative to CO2.
            if len(filtered_df) < gcmc_budget:
                n_top_up = min(gcmc_budget - len(filtered_df), len(kpi_df) - len(filtered_df))
                top_up = (kpi_df[~survives]
                          .sort_values("co2_over_h2o_selectivity", ascending=False)
                          .head(n_top_up))
                logger.warning(
                    f"[Stage 2] Only {len(filtered_df)} MOFs pass the water filter (< budget "
                    f"{gcmc_budget}); topping up with {len(top_up)} MOFs ranked by KH_CO2/KH_H2O."
                )
                filtered_df = pd.concat([filtered_df, top_up])
            kpi_df = filtered_df

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
        f"  Water filter passed : {len(kpi_df)}/{n_before_filter}\n"
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
    ws_root = ensure_dir(output_dir / "stage3_workspace")
    total = len(cif_paths) * len(conditions)
    job = 0

    # FIX: results used to live only in memory (`rows`) and hit disk once,
    # in a single gcmc_adsorption.parquet write, after the *entire* loop
    # finished. A crash/reboot partway through a worker's run (happened for
    # real: a hard reboot mid-run lost 100% of a 15-worker Stage 3 run,
    # hours of compute, not just the in-flight job) lost everything because
    # nothing was ever written incrementally. Now each successful job is
    # appended to a checkpoint CSV immediately, and a restart reloads it
    # and skips already-completed (mof, condition) pairs instead of
    # redoing them.
    checkpoint_path = output_dir / "gcmc_adsorption.checkpoint.csv"
    rows: List[Dict] = []
    done_keys: set = set()
    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
        prior = pd.read_csv(checkpoint_path)
        rows = prior.to_dict("records")
        done_keys = {
            (r["mof_id"], round(float(r["temperature_k"]), 3),
             round(float(r["pressure_bar"]), 6), round(float(r["rh"]), 6))
            for r in rows
        }
        logger.info(f"[Stage 3] Resuming: {len(done_keys)} jobs already "
                    f"completed in a previous run (from {checkpoint_path.name})")

    failures: List[Dict] = []

    logger.info(f"\n[Stage 3] Multicomponent GCMC: {total} jobs across {len(cif_paths)} MOFs")

    for cif in cif_paths:
        name = cif.stem
        uc = determine_unit_cells(cif, cutoff)
        for cond in conditions:
            job += 1
            key = (name, round(cond.temperature_k, 3),
                   round(cond.pressure_bar, 6), round(cond.rh, 6))
            if key in done_keys:
                continue
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
                                cif, ws, inp, max_retries, retry_delay,
                                use_charges)
            if res.ok:
                parsed = parse_multi(res.combined_text)
                row = {
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
                }
                rows.append(row)
                append_checkpoint(checkpoint_path, row)
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
    # ADDED: the KH_CO2 > KH_H2O filter was a no-op while frameworks were
    # uncharged (water barely adsorbed). With real charges it can remove
    # most hydrophilic MOFs -- and with them most of the humidity effect a
    # model is supposed to learn -- and leave fewer than --gcmc-budget MOFs.
    p.add_argument("--water-filter", action=argparse.BooleanOptionalAction, default=True,
                   help="Stage 2: drop MOFs whose H2O Henry constant exceeds CO2's "
                        "(default on; --no-water-filter keeps them for a more diverse dataset).")

    # Simulation parameters
    p.add_argument("--n-cycles",    type=int,   default=5000,
                   help="Cycles for Stage 1 (GCMC or GC-TMMC screening).")
    # ADDED: separate cycle count for Stage 3 -- screening (Stage 1) only
    # needs to rank MOFs relatively, so it can run cheap; Stage 3 produces
    # the actual training labels, where noisier/under-converged loadings
    # directly become noisier training data. None (default) reuses
    # --n-cycles, so existing invocations are unaffected.
    p.add_argument("--stage3-n-cycles", type=int, default=None,
                   help="Cycles for Stage 3 (humid multicomponent GCMC, "
                        "the real training-label generator). Defaults to "
                        "--n-cycles if not given.")
    p.add_argument("--temperatures", default="298.15,313.15,333.15",
                   help="Stage 3 (real training data) temperature grid.")
    p.add_argument("--pressures",    default="0.1,0.5,1.0,5.0,10.0",
                   help="Stage 3 (real training data) pressure grid.")
    # ADDED: select_mofs_for_gcmc() only ever reads ONE temperature
    # (temps[0]) and only the lowest pressure point (Henry constant) plus
    # min/max pressure (working-capacity interpolation) -- the rest of a
    # full --temperatures x --pressures sweep is computed by Stage 1 and
    # then silently never used. A separate, smaller Stage 1 grid gets the
    # cheap-screening benefit GC-TMMC would have given (9 jobs/MOF at the
    # defaults below: 1 temp x 3 pressures x 3 components) using the plain
    # GCMC method that's actually proven to work, instead of GC-TMMC (which
    # doesn't -- see --stage1-method's help).
    p.add_argument("--stage1-temperatures", default=None,
                   help="Stage 1 (screening) temperature grid. Defaults to "
                        "just the first value in --temperatures if not given.")
    p.add_argument("--stage1-pressures", default="0.1,1.0,10.0",
                   help="Stage 1 (screening) pressure grid -- only the "
                        "lowest point (Henry constant) and min/max "
                        "(working-capacity interpolation) are actually "
                        "used by MOF selection, so this can be much "
                        "smaller than --pressures.")
    p.add_argument("--rhs",          default="0.0,0.05,0.15")
    p.add_argument("--dry-co2-frac", type=float, default=0.15)
    p.add_argument("--cutoff",       type=float, default=12.0)
    # FIX: was "GenericMOFs", which isn't in a standard RASPA2 install and
    # silently fell back to ExampleMOFsForceField (mis-typed framework atoms).
    p.add_argument("--forcefield",   default=DEFAULT_FORCEFIELD,
                   help=f"RASPA force field (default: {DEFAULT_FORCEFIELD}, shipped in forcefields/)")
    # FIX: defaulted to False (no framework electrostatics) while the
    # active-learning GCMCConfig/GCTMMCConfig defaulted use_charges=True and
    # the model was trained with use_charges=True as an input feature — a
    # three-way inconsistency. Now defaults on; pass
    # --no-use-framework-charges to reproduce the old uncharged behaviour.
    p.add_argument(
        "--use-framework-charges", action=argparse.BooleanOptionalAction, default=True,
        help="Use CIF partial charges for host-guest electrostatics in RASPA "
             "(default: True). Use --no-use-framework-charges to disable.",
    )
    p.add_argument("--keep-workspaces",       action="store_true")

    # Molecule definitions
    # The guest pseudo atoms in forcefields/UFF_TIP4P2005 match the
    # ExampleDefinitions CO2/N2/H2O .def files (there is no TraPPE dir in a
    # standard RASPA2 install; "TraPPE" only ever fell through to these).
    p.add_argument("--co2-definition", default="ExampleDefinitions")
    p.add_argument("--n2-definition",  default="ExampleDefinitions")
    p.add_argument("--h2o-definition", default="ExampleDefinitions")
    p.add_argument("--co2-def-file",   default=None)
    p.add_argument("--n2-def-file",    default=None)
    p.add_argument("--h2o-def-file",   default=None)

    # Retry
    p.add_argument("--max-retries",  type=int, default=3)
    p.add_argument("--job-timeout-hours", type=float, default=48.0,
                   help="Wall-clock limit per RASPA job (default 48 h; was a hard-coded 2 h).")
    p.add_argument("--retry-delay",  type=int, default=60, help="Initial delay seconds")

    # Stage control (useful for resuming)
    p.add_argument("--skip-stage1",  action="store_true", help="Load existing stage1 parquet")
    p.add_argument("--skip-stage2",  action="store_true", help="Run GCMC on all (skip selection)")
    # ADDED: for a two-phase parallel setup -- run this once, globally, to
    # do Stage 1 (screening) + Stage 2 (selection) only, producing
    # gcmc_selected_mofs.txt, then exit before the expensive Stage 3. Then
    # parallel workers each run with --skip-stage1 --skip-stage2 on their
    # own pre-assigned subset of the selected MOFs for Stage 3 only.
    # ADDED: lets run_parallel_gcmc.sh run Stage 1 on N CIF chunks in
    # parallel, then merge them and run Stage 2 once globally.
    p.add_argument("--stop-after-stage1", action="store_true",
                   help="Exit right after Stage 1 (parallel screening chunks).")
    p.add_argument("--stop-after-stage2", action="store_true",
                   help="Exit after Stage 2 selection, before Stage 3 (GCMC). "
                        "For running Stage 1+2 once globally before "
                        "parallelizing Stage 3 across workers.")
    # ADDED: GC-TMMC existed in src/simulation/gctmmc/ but nothing ever drove
    # it. 'gctmmc' gets a full isotherm per (MOF, component, T) from one
    # simulation instead of one GCMC run per (MOF, component, T, P) --
    # single-component screening only; Stage 3 is unaffected either way.
    p.add_argument("--stage1-method", choices=["gcmc", "gctmmc"], default="gcmc",
                   help="Stage 1 single-component screening method (default: gcmc).")

    args = p.parse_args()
    global RASPA_TIMEOUT_S
    RASPA_TIMEOUT_S = int(args.job_timeout_hours * 3600)
    if args.stage3_n_cycles is None:
        args.stage3_n_cycles = args.n_cycles

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
        # FIX: pipeline.yaml stores use_charges under simulation.gcmc, not as
        # a top-level simulation key, so the generic loop above never applied
        # it — --config's charge setting was silently ignored. Apply explicitly.
        gcmc_cfg = sim.get("gcmc", {})
        if "use_charges" in gcmc_cfg:
            args.use_framework_charges = bool(gcmc_cfg["use_charges"])

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

    # Stage 1's own (smaller) condition grid -- see --stage1-temperatures/
    # --stage1-pressures help text for why this is deliberately separate
    # from Stage 3's temps/pressures above.
    stage1_temps = (
        parse_float_list(args.stage1_temperatures)
        if args.stage1_temperatures else [temps[0]]
    )
    stage1_pressures = parse_float_list(args.stage1_pressures)

    # RASPA setup
    raspa_share  = detect_raspa_share(raspa_dd)
    runtime_root, runtime_share, fw_cif_dir = prepare_runtime(output_dir, raspa_share)
    # Resolved against the runtime tree, which also holds this repo's
    # forcefields/ (installed by prepare_runtime).
    ff_name      = resolve_forcefield(runtime_share, args.forcefield)

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

    # Pre-flight: never simulate a MOF with missing force-field parameters,
    # overlapping atoms, or (when charges are on) no framework charges.
    all_cifs = validate_frameworks(
        all_cifs, forcefield_pseudo_atoms(runtime_share, ff_name),
        args.use_framework_charges, output_dir,
    )
    if not all_cifs:
        raise RuntimeError(f"No MOF passed the pre-flight checks -- see {output_dir / 'rejected_frameworks.csv'}")

    logger.info("=" * 65)
    logger.info(f"UC-TPNO SIMULATION PIPELINE")
    logger.info(f"  CIFs            : {len(all_cifs)}")
    logger.info(f"  Stage 3 temps(K): {temps}")
    logger.info(f"  Stage 3 press.  : {pressures}")
    logger.info(f"  Stage 1 temps(K): {stage1_temps}")
    logger.info(f"  Stage 1 press.  : {stage1_pressures}")
    logger.info(f"  RH values       : {rhs}")
    logger.info(f"  Dry CO2 frac    : {args.dry_co2_frac}")
    logger.info(f"  Valid cond/MOF  : {len(conditions)}")
    logger.info(f"  GCMC budget     : {args.gcmc_budget} MOFs")
    logger.info(f"  Stage 1 cycles  : {args.n_cycles}")
    logger.info(f"  Stage 3 cycles  : {args.stage3_n_cycles}")
    logger.info(f"  Forcefield      : {ff_name}")
    logger.info(f"  Framework charges: {'from CIF' if args.use_framework_charges else 'OFF'}")
    logger.info(f"  CO2 mol dir     : {co2_dir}")
    logger.info(f"  N2  mol dir     : {n2_dir}")
    logger.info(f"  H2O mol dir     : {h2o_dir}")
    logger.info("=" * 65)

    # ------------------------------------------------------------------
    # Stage 1 — single-component pressure sweep (ALL MOFs)
    # ------------------------------------------------------------------
    # ADDED: when both stages are skipped (the "Stage 3 only" mode used by
    # parallel workers that already received a pre-selected MOF list from a
    # prior global Stage 1+2 run), Stage 1 output is never used for
    # anything -- Stage 2 is also skipped, so nothing reads stage1_df
    # except a row-count log line. Don't waste time recomputing it.
    stage1_parquet = output_dir / "stage1_isotherms.parquet"
    if args.skip_stage1 and args.skip_stage2:
        logger.info("\n[Stage 1] Skipped entirely (Stage 3-only mode: Stage 2 is also skipped, "
                     "so Stage 1 output would be unused).")
        stage1_df = pd.DataFrame()
    elif args.skip_stage1 and stage1_parquet.exists():
        logger.info(f"\n[Stage 1] Skipped — loading {stage1_parquet}")
        stage1_df = pd.read_parquet(stage1_parquet)
    elif args.stage1_method == "gctmmc":
        stage1_df = run_stage1_isotherms_gctmmc(
            all_cifs, output_dir, runtime_root, raspa_path,
            stage1_temps,
            args.n_cycles, args.keep_workspaces,
            ff_name, args.use_framework_charges, args.cutoff,
        )
    else:
        stage1_df = run_stage1_isotherms(
            all_cifs, output_dir, runtime_root, fw_cif_dir, raspa_path,
            stage1_temps, stage1_pressures,
            args.n_cycles, args.keep_workspaces,
            ff_name, args.use_framework_charges, args.cutoff,
            co2_dir, n2_dir, h2o_dir,
            args.max_retries, args.retry_delay,
            h2o_rhs=rhs,
        )

    if args.stop_after_stage1:
        logger.info(f"\nSTOPPING AFTER STAGE 1 (--stop-after-stage1): {len(stage1_df)} rows")
        return

    # ------------------------------------------------------------------
    # Stage 2 — select best MOFs for GCMC
    # ------------------------------------------------------------------
    if args.skip_stage2:
        gcmc_cifs = all_cifs
        logger.info(f"\n[Stage 2] Skipped — using all {len(gcmc_cifs)} MOFs for GCMC")
    else:
        # FIX: this used to reference Stage 3's temps/pressures (temps[0],
        # max/min(pressures)) even though Stage 1 now runs its own,
        # separate, smaller condition grid. If the two grids don't share
        # a reference temperature, the "between(temperature_k +/- 1.0)"
        # filter inside select_mofs_for_gcmc would match zero rows and
        # (per the earlier fix) raise immediately instead of silently
        # selecting nothing. Must reference Stage 1's own grid here.
        selected_ids = select_mofs_for_gcmc(
            stage1_df,
            gcmc_budget=args.gcmc_budget,
            adsorption_pressure_bar=max(stage1_pressures),
            desorption_pressure_bar=min(stage1_pressures),
            temperature_k=stage1_temps[0],
            exploration_fraction=args.exploration_fraction,
            require_co2_over_h2o=args.water_filter,
        )
        # Save selection list
        sel_file = output_dir / "gcmc_selected_mofs.txt"
        sel_file.write_text("\n".join(selected_ids), encoding="utf-8")
        logger.info(f"[Stage 2] Selection saved → {sel_file}")

        id_set   = set(selected_ids)
        gcmc_cifs = [c for c in all_cifs if c.stem in id_set]

    if args.stop_after_stage2:
        logger.info("\n" + "=" * 65)
        logger.info("STOPPING AFTER STAGE 2 (--stop-after-stage2)")
        logger.info(f"  Stage 1 rows       : {len(stage1_df)}")
        logger.info(f"  MOFs selected      : {len(gcmc_cifs)}")
        logger.info(f"  Selection file     : {output_dir / 'gcmc_selected_mofs.txt'}")
        logger.info("Next: split the selected MOFs across parallel workers "
                     "for Stage 3 (--skip-stage1 --skip-stage2 per worker).")
        logger.info("=" * 65)
        return

    # ------------------------------------------------------------------
    # Stage 3 — multicomponent GCMC on selected MOFs
    # ------------------------------------------------------------------
    gcmc_df, _ = run_stage3_gcmc(
        gcmc_cifs, output_dir, runtime_root, fw_cif_dir, raspa_path,
        conditions,
        args.stage3_n_cycles, args.keep_workspaces,
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