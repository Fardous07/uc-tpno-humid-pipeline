#!/usr/bin/env python3
"""
04_run_simulations.py

Robust RASPA runner for humid CO2/N2/H2O adsorption on MOFs.

Main fixes in this version
--------------------------
1. Uses absolute paths everywhere so RASPA can find the runtime tree from any workspace.
2. Builds a local runtime root under the output directory without touching the Conda install.
3. Canonicalises each input CIF into a minimal P1 CIF before simulation.
4. Resolves a usable forcefield and molecule-definition folders before any jobs start.
5. Treats H2O as mandatory for humid GCMC, but allows a user-supplied H2O.def file.
6. Saves full stdout/stderr/output files for every job and prints the tail of failures.
7. Parses loading values from the RASPA log text you actually generate.
8. Adds thermodynamic condition columns: mu_CO2, mu_N2, mu_H2O, T.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from ase.io import read as ase_read
except Exception:
    ase_read = None


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def parse_float_list(values: str) -> List[float]:
    return [float(x.strip()) for x in values.split(",") if x.strip()]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def tail_text(text: str, n_lines: int = 80) -> str:
    lines = (text or "").splitlines()
    return "\n".join(lines[-n_lines:]) if lines else "(no output)"


def detect_raspa_share(raspa_data_dir: Path) -> Path:
    raspa_data_dir = raspa_data_dir.expanduser().resolve()

    candidate = raspa_data_dir / "share" / "raspa"
    if candidate.exists():
        return candidate

    if raspa_data_dir.name == "raspa" and raspa_data_dir.exists():
        return raspa_data_dir

    raise FileNotFoundError(
        f"Could not locate RASPA share directory under: {raspa_data_dir}\n"
        "Expected either <raspa_data_dir>/share/raspa or a direct raspa directory."
    )


def copytree_merge(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst = dst.resolve()
    ensure_dir(dst.parent)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def available_subdirs(path: Path) -> List[str]:
    if not path.exists():
        return []
    return sorted([p.name for p in path.iterdir() if p.is_dir()])


# ---------------------------------------------------------------------
# RASPA runtime helpers
# ---------------------------------------------------------------------

def resolve_forcefield_name(raspa_share: Path, requested: str) -> str:
    forcefield_root = raspa_share / "forcefield"
    available = available_subdirs(forcefield_root)

    if not available:
        raise FileNotFoundError(f"No forcefield directories found under: {forcefield_root}")

    if requested in available:
        return requested

    lower_map = {name.lower(): name for name in available}
    if requested.lower() in lower_map:
        return lower_map[requested.lower()]

    if requested == "GenericMOFs" and "ExampleMOFsForceField" in available:
        print(
            "[info] Requested forcefield 'GenericMOFs' was not found. "
            "Using 'ExampleMOFsForceField' instead."
        )
        return "ExampleMOFsForceField"

    raise FileNotFoundError(
        f"RASPA forcefield directory not found for requested forcefield '{requested}'.\n"
        f"Available forcefields: {', '.join(available)}"
    )


def find_definition_dir_by_file(
    raspa_share: Path,
    molecule_name: str,
    preferred_dir: Optional[str] = None,
) -> str:
    molecules_root = raspa_share / "molecules"
    if not molecules_root.exists():
        raise FileNotFoundError(f"RASPA molecules directory not found: {molecules_root}")

    target_file = f"{molecule_name}.def"

    if preferred_dir:
        pref_dir = molecules_root / preferred_dir
        if pref_dir.is_dir() and (pref_dir / target_file).exists():
            return preferred_dir

    matches: List[str] = []
    for subdir in sorted([p for p in molecules_root.iterdir() if p.is_dir()]):
        if (subdir / target_file).exists():
            matches.append(subdir.name)

    if matches:
        if preferred_dir and preferred_dir not in matches:
            print(
                f"[info] Preferred molecule-definition dir '{preferred_dir}' for {molecule_name} "
                f"was not found. Using '{matches[0]}' instead."
            )
        return matches[0]

    available = available_subdirs(molecules_root)
    raise FileNotFoundError(
        f"Could not find a molecule definition for {target_file} under: {molecules_root}\n"
        f"Available molecule-definition folders: {', '.join(available) if available else '(none)'}"
    )


def install_user_definition_file(
    runtime_share: Path,
    molecule_name: str,
    def_file: Path,
    runtime_dir_name: str = "UserDefinitions",
) -> str:
    def_file = def_file.expanduser().resolve()
    if not def_file.exists():
        raise FileNotFoundError(f"User-supplied definition file not found: {def_file}")
    if def_file.suffix.lower() != ".def":
        raise ValueError(f"User-supplied definition file must end with .def: {def_file}")
    if def_file.name != f"{molecule_name}.def":
        raise ValueError(
            f"User-supplied definition file name must be exactly {molecule_name}.def: {def_file}"
        )

    dst_dir = ensure_dir(runtime_share / "molecules" / runtime_dir_name)
    shutil.copy2(def_file, dst_dir / def_file.name)
    return runtime_dir_name


def prepare_runtime_root(runtime_root: Path, raspa_share: Path) -> Tuple[Path, Path, Path]:
    runtime_root = runtime_root.expanduser().resolve()
    runtime_share = runtime_root / "share" / "raspa"

    ensure_dir(runtime_root)
    if runtime_share.exists():
        shutil.rmtree(runtime_share)

    copytree_merge(raspa_share, runtime_share)

    framework_cif_dir = ensure_dir(runtime_share / "structures" / "cif")
    return runtime_root, runtime_share, framework_cif_dir


def check_forcefield_assets(runtime_share: Path, forcefield_name: str) -> None:
    ff_dir = runtime_share / "forcefield" / forcefield_name

    if not ff_dir.exists():
        raise FileNotFoundError(f"Resolved forcefield directory does not exist: {ff_dir}")

    required_strict = ["force_field.def", "pseudo_atoms.def"]
    missing = [name for name in required_strict if not (ff_dir / name).exists()]

    mixing_ok = (
        (ff_dir / "force_field_mixing_rules.def").exists()
        or (ff_dir / "mixing_rules.def").exists()
    )
    if not mixing_ok:
        missing.append("force_field_mixing_rules.def (or mixing_rules.def)")

    if missing:
        raise FileNotFoundError(
            f"Forcefield '{forcefield_name}' is missing required files in {ff_dir}: "
            f"{', '.join(missing)}"
        )


def check_molecule_definition(runtime_share: Path, definition_dir: str, molecule_name: str) -> None:
    path = runtime_share / "molecules" / definition_dir / f"{molecule_name}.def"
    if not path.exists():
        raise FileNotFoundError(
            f"Molecule definition file not found: {path}\n"
            f"Check your RASPA installation or provide a user definition file."
        )


# ---------------------------------------------------------------------
# Thermodynamics helpers
# ---------------------------------------------------------------------

R_KJ_MOL_K = 8.314462618e-3


def water_saturation_pressure_bar(temperature_k: float) -> float:
    t_c = temperature_k - 273.15
    a = 8.07131
    b = 1730.63
    c = 233.426
    p_mmHg = 10 ** (a - b / (c + t_c))
    p_bar = p_mmHg * 133.322368 / 1e5
    return float(p_bar)


def ideal_gas_mu_from_partial_pressure_bar(partial_pressure_bar: float, temperature_k: float) -> float:
    p_ref_bar = 1.0
    p_eff = max(float(partial_pressure_bar), 1e-12)
    return float(R_KJ_MOL_K * temperature_k * math.log(p_eff / p_ref_bar))


def build_valid_conditions(
    temperatures_k: Sequence[float],
    pressures_bar: Sequence[float],
    rhs: Sequence[float],
    dry_co2_frac: float,
) -> List[Condition]:
    conditions: List[Condition] = []

    for t in temperatures_k:
        p_sat = water_saturation_pressure_bar(t)

        for p_total in pressures_bar:
            for rh in rhs:
                p_h2o = rh * p_sat

                if p_h2o >= p_total:
                    continue

                dry_total = p_total - p_h2o
                x_h2o = p_h2o / p_total
                x_co2 = (dry_total / p_total) * dry_co2_frac
                x_n2 = (dry_total / p_total) * (1.0 - dry_co2_frac)

                p_co2 = x_co2 * p_total
                p_n2 = x_n2 * p_total

                mu_co2 = ideal_gas_mu_from_partial_pressure_bar(p_co2, t)
                mu_n2 = ideal_gas_mu_from_partial_pressure_bar(p_n2, t)
                mu_h2o = ideal_gas_mu_from_partial_pressure_bar(max(p_h2o, 1e-12), t)

                conditions.append(
                    Condition(
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
                        mu_co2=float(mu_co2),
                        mu_n2=float(mu_n2),
                        mu_h2o=float(mu_h2o),
                    )
                )

    return conditions


# ---------------------------------------------------------------------
# CIF helpers
# ---------------------------------------------------------------------

_CELL_TAGS = [
    ("_cell_length_a", "a"),
    ("_cell_length_b", "b"),
    ("_cell_length_c", "c"),
    ("_cell_angle_alpha", "alpha"),
    ("_cell_angle_beta", "beta"),
    ("_cell_angle_gamma", "gamma"),
]


def read_cell_lengths_from_text(cif_path: Path) -> Optional[Tuple[float, float, float]]:
    text = cif_path.read_text(encoding="utf-8", errors="replace")
    vals: Dict[str, float] = {}
    for tag, key in _CELL_TAGS:
        m = re.search(rf"^\s*{re.escape(tag)}\s+([^\s]+)", text, flags=re.MULTILINE)
        if not m:
            continue
        token = re.sub(r"\([^)]+\)$", "", m.group(1).strip().strip("'").strip('"'))
        try:
            vals[key] = float(token)
        except Exception:
            pass

    if {"a", "b", "c"}.issubset(vals):
        return vals["a"], vals["b"], vals["c"]
    return None


def determine_unit_cells(cif_path: Path, cutoff_angstrom: float) -> Tuple[int, int, int]:
    lengths = read_cell_lengths_from_text(cif_path)
    if lengths is None:
        return (2, 2, 2)

    reps: List[int] = []
    min_required = 2.0 * cutoff_angstrom
    for L in lengths:
        reps.append(max(1, int(math.ceil(min_required / max(L, 1e-8)))))

    return tuple(reps)  # type: ignore[return-value]


def canonicalize_cif_for_raspa(input_cif: Path, output_cif: Path) -> None:
    if ase_read is None:
        raise RuntimeError("ASE is not installed in this environment. Install it first: pip install ase")

    atoms = ase_read(str(input_cif))
    if isinstance(atoms, list):
        if not atoms:
            raise ValueError(f"No structure could be read from CIF: {input_cif}")
        atoms = atoms[0]

    cell_lengths = atoms.cell.lengths()
    cell_angles = atoms.cell.angles()
    scaled = atoms.get_scaled_positions(wrap=True)
    symbols = atoms.get_chemical_symbols()

    counts: Dict[str, int] = {}
    lines = [
        f"data_{input_cif.stem}",
        f"_cell_length_a {float(cell_lengths[0]):.6f}",
        f"_cell_length_b {float(cell_lengths[1]):.6f}",
        f"_cell_length_c {float(cell_lengths[2]):.6f}",
        f"_cell_angle_alpha {float(cell_angles[0]):.6f}",
        f"_cell_angle_beta {float(cell_angles[1]):.6f}",
        f"_cell_angle_gamma {float(cell_angles[2]):.6f}",
        "_symmetry_space_group_name_Hall 'P 1'",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_symmetry_Int_Tables_number 1",
        "loop_",
        "_symmetry_equiv_pos_as_xyz",
        "x,y,z",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]

    for sym, pos in zip(symbols, scaled):
        counts[sym] = counts.get(sym, 0) + 1
        label = f"{sym}{counts[sym]}"
        lines.append(
            f"{label} {sym} "
            f"{float(pos[0]):.6f} {float(pos[1]):.6f} {float(pos[2]):.6f} 1.0"
        )

    lines.append("")
    ensure_dir(output_cif.parent)
    output_cif.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# RASPA input builders
# ---------------------------------------------------------------------

def build_common_header(
    framework_name: str,
    unit_cells: Tuple[int, int, int],
    temperature_k: float,
    pressure_bar: float,
    n_cycles: int,
    forcefield: str,
    use_framework_charges: bool,
    cutoff: float,
) -> List[str]:
    n_init = max(500, int(max(n_cycles, 1) * 0.5))
    return [
        "SimulationType MonteCarlo",
        f"NumberOfCycles {int(n_cycles)}",
        f"NumberOfInitializationCycles {int(n_init)}",
        f"PrintEvery {max(100, int(max(n_cycles, 1) * 0.1))}",
        "RestartFile no",
        "ContinueAfterCrash no",
        f"Forcefield {forcefield}",
        f"UseChargesFromCIFFile {'yes' if use_framework_charges else 'no'}",
        "ChargeMethod Ewald",
        "EwaldPrecision 1e-6",
        f"CutOffVDW {float(cutoff):.3f}",
        f"CutOffChargeCharge {float(cutoff):.3f}",
        "RemoveAtomNumberCodeFromLabel yes",
        "Framework 0",
        f"FrameworkName {framework_name}",
        f"UnitCells {unit_cells[0]} {unit_cells[1]} {unit_cells[2]}",
        f"ExternalTemperature {float(temperature_k):.6f}",
        f"ExternalPressure {float(pressure_bar) * 1.0e5:.6f}",
    ]


def build_single_component_input(
    framework_name: str,
    unit_cells: Tuple[int, int, int],
    temperature_k: float,
    pressure_bar: float,
    component_name: str,
    molecule_definition: str,
    n_cycles: int,
    forcefield: str,
    use_framework_charges: bool,
    cutoff: float,
) -> str:
    lines = build_common_header(
        framework_name=framework_name,
        unit_cells=unit_cells,
        temperature_k=temperature_k,
        pressure_bar=pressure_bar,
        n_cycles=n_cycles,
        forcefield=forcefield,
        use_framework_charges=use_framework_charges,
        cutoff=cutoff,
    )
    lines.extend(
        [
            f"Component 0 MoleculeName {component_name}",
            f"            MoleculeDefinition {molecule_definition}",
            "            TranslationProbability 1.0",
            "            RotationProbability 1.0",
            "            ReinsertionProbability 1.0",
            "            SwapProbability 1.0",
            "            CreateNumberOfMolecules 0",
        ]
    )
    return "\n".join(lines) + "\n"


def build_multicomponent_input(
    framework_name: str,
    unit_cells: Tuple[int, int, int],
    condition: Condition,
    n_cycles: int,
    forcefield: str,
    use_framework_charges: bool,
    cutoff: float,
    co2_definition: str,
    n2_definition: str,
    h2o_definition: str,
) -> str:
    lines = build_common_header(
        framework_name=framework_name,
        unit_cells=unit_cells,
        temperature_k=condition.temperature_k,
        pressure_bar=condition.pressure_bar,
        n_cycles=n_cycles,
        forcefield=forcefield,
        use_framework_charges=use_framework_charges,
        cutoff=cutoff,
    )

    comps = [
        ("CO2", co2_definition, condition.x_co2_total),
        ("N2", n2_definition, condition.x_n2_total),
        ("H2O", h2o_definition, condition.x_h2o_total),
    ]

    for i, (name, definition, frac) in enumerate(comps):
        lines.extend(
            [
                f"Component {i} MoleculeName {name}",
                f"            MoleculeDefinition {definition}",
                f"            MolFraction {float(frac):.12f}",
                "            TranslationProbability 1.0",
                "            RotationProbability 1.0",
                "            ReinsertionProbability 1.0",
                "            SwapProbability 1.0",
                "            CreateNumberOfMolecules 0",
            ]
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------
# Running and parsing RASPA
# ---------------------------------------------------------------------

def collect_text_output_files(workdir: Path) -> str:
    pieces: List[str] = []
    for path in sorted(workdir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("output") or path.suffix.lower() in {".data", ".txt"}:
            try:
                pieces.append(f"\n===== FILE: {path.relative_to(workdir)} =====\n")
                pieces.append(path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
    return "\n".join(pieces)


def run_raspa_job(
    raspa_path: Path,
    runtime_root: Path,
    framework_cif_dir: Path,
    source_cif: Path,
    workspace: Path,
    input_text: str,
) -> Tuple[bool, str, Path, int]:
    workspace = ensure_dir(workspace.expanduser().resolve())

    framework_name = source_cif.stem
    runtime_framework_cif = framework_cif_dir.expanduser().resolve() / f"{framework_name}.cif"

    canonicalize_cif_for_raspa(source_cif.expanduser().resolve(), runtime_framework_cif)

    if not runtime_framework_cif.exists():
        raise FileNotFoundError(f"Canonical framework CIF was not created: {runtime_framework_cif}")

    canonical_copy = workspace / f"{framework_name}.canonical.cif"
    shutil.copy2(runtime_framework_cif, canonical_copy)

    sim_input = workspace / "simulation.input"
    sim_input.write_text(input_text, encoding="utf-8")

    env = os.environ.copy()
    env["RASPA_DIR"] = str(runtime_root.expanduser().resolve())

    proc = subprocess.run(
        [str(raspa_path.expanduser().resolve())],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        env=env,
    )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    file_output = collect_text_output_files(workspace)
    combined = "\n".join(part for part in [stdout, stderr, file_output] if part and part.strip())

    full_log = workspace / "raspa_full_output.log"
    full_log.write_text(combined, encoding="utf-8", errors="replace")

    ok = proc.returncode == 0
    return ok, combined, full_log, proc.returncode


def _extract_float(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _parse_component_block(output_text: str, component_name: str) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {
        "loading_molkg": None,
        "loading_molecules_uc": None,
        "enthalpy_kjmol": None,
    }

    block_pattern = re.compile(
        rf"Component\s+\d+\s+\[{re.escape(component_name)}\](.*?)(?=Component\s+\d+\s+\[|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    all_matches = list(block_pattern.finditer(output_text))
    m = all_matches[-1] if all_matches else None
    if not m:
        return result

    block = m.group(1)

    result["loading_molkg"] = _extract_float(
        r"Average loading absolute \[mol/kg framework\]\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        block,
    )

    result["loading_molecules_uc"] = _extract_float(
        r"Average loading absolute \[molecules/unit cell\]\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        block,
    )

    result["enthalpy_kjmol"] = _extract_float(
        r"Enthalpy of adsorption \[kJ/mol\]\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        block,
    )

    return result


def parse_single_component_result(output_text: str, component_name: str) -> Dict[str, Optional[float]]:
    parsed = _parse_component_block(output_text, component_name)
    return {
        "loading_molkg": parsed["loading_molkg"],
        "loading_molecules_uc": parsed["loading_molecules_uc"],
        "enthalpy_kjmol": parsed["enthalpy_kjmol"],
    }


def parse_multicomponent_result(output_text: str) -> Dict[str, Optional[float]]:
    co2 = _parse_component_block(output_text, "CO2")
    n2 = _parse_component_block(output_text, "N2")
    h2o = _parse_component_block(output_text, "H2O")

    co2_load = co2["loading_molkg"]
    n2_load = n2["loading_molkg"]

    selectivity = None
    if co2_load is not None and n2_load is not None and n2_load > 0:
        selectivity = co2_load / n2_load

    return {
        "co2_loading_molkg": co2["loading_molkg"],
        "n2_loading_molkg": n2["loading_molkg"],
        "h2o_loading_molkg": h2o["loading_molkg"],
        "co2_loading_molecules_uc": co2["loading_molecules_uc"],
        "n2_loading_molecules_uc": n2["loading_molecules_uc"],
        "h2o_loading_molecules_uc": h2o["loading_molecules_uc"],
        "co2_enthalpy_kjmol": co2["enthalpy_kjmol"],
        "n2_enthalpy_kjmol": n2["enthalpy_kjmol"],
        "h2o_enthalpy_kjmol": h2o["enthalpy_kjmol"],
        "co2_over_n2_selectivity": selectivity,
    }


# ---------------------------------------------------------------------
# Main simulation logic
# ---------------------------------------------------------------------

def load_cif_paths(cif_dir: Path, max_mofs: int) -> List[Path]:
    cifs = sorted(cif_dir.glob("*.cif"))
    if max_mofs > 0:
        cifs = cifs[:max_mofs]
    return [p.resolve() for p in cifs]


def run_low_fidelity_isotherms(
    cif_paths: Sequence[Path],
    output_dir: Path,
    runtime_root: Path,
    framework_cif_dir: Path,
    raspa_path: Path,
    temperatures_k: Sequence[float],
    pressures_bar: Sequence[float],
    n_cycles: int,
    keep_workspaces: bool,
    forcefield: str,
    use_framework_charges: bool,
    cutoff: float,
    co2_definition: str,
    n2_definition: str,
) -> pd.DataFrame:
    rows: List[Dict] = []
    workspace_root = ensure_dir((output_dir / "tmmc_workspace").resolve())

    print("[1/2] Running GC-TMMC (low-fidelity single-component isotherms)...")

    for cif_path in cif_paths:
        framework_name = cif_path.stem
        unit_cells = determine_unit_cells(cif_path, cutoff_angstrom=cutoff)

        for component_name, definition in [("CO2", co2_definition), ("N2", n2_definition)]:
            for temperature_k in temperatures_k:
                workspace = workspace_root / f"{framework_name}__{component_name}__{temperature_k:.2f}K"
                pressure_bar = max(pressures_bar)

                sim_input = build_single_component_input(
                    framework_name=framework_name,
                    unit_cells=unit_cells,
                    temperature_k=temperature_k,
                    pressure_bar=pressure_bar,
                    component_name=component_name,
                    molecule_definition=definition,
                    n_cycles=n_cycles,
                    forcefield=forcefield,
                    use_framework_charges=use_framework_charges,
                    cutoff=cutoff,
                )

                ok, combined, full_log, returncode = run_raspa_job(
                    raspa_path=raspa_path,
                    runtime_root=runtime_root,
                    framework_cif_dir=framework_cif_dir,
                    source_cif=cif_path,
                    workspace=workspace,
                    input_text=sim_input,
                )

                if not ok:
                    print(
                        f"GC-TMMC failed for {framework_name}/{component_name} "
                        f"(returncode={returncode}): {tail_text(combined, 12)}"
                    )
                else:
                    parsed = parse_single_component_result(combined, component_name)
                    row = {
                        "mof_id": framework_name,
                        "component": component_name,
                        "temperature_k": float(temperature_k),
                        "pressure_bar": float(pressure_bar),
                        "log_path": str(full_log),
                        **parsed,
                    }
                    rows.append(row)

                if not keep_workspaces and workspace.exists():
                    shutil.rmtree(workspace, ignore_errors=True)

    df = pd.DataFrame(rows)
    out_path = (output_dir / "tmmc_isotherms.parquet").resolve()
    df.to_parquet(out_path, index=False)
    print(f"  GC-TMMC: {len(df)} rows -> {out_path}")
    return df


def run_high_fidelity_gcmc(
    cif_paths: Sequence[Path],
    output_dir: Path,
    runtime_root: Path,
    framework_cif_dir: Path,
    raspa_path: Path,
    conditions: Sequence[Condition],
    n_cycles: int,
    keep_workspaces: bool,
    forcefield: str,
    use_framework_charges: bool,
    cutoff: float,
    co2_definition: str,
    n2_definition: str,
    h2o_definition: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict] = []
    failures: List[Dict] = []
    workspace_root = ensure_dir((output_dir / "gcmc_workspace").resolve())

    total_jobs = len(cif_paths) * len(conditions)
    job_idx = 0

    print("\n[2/2] Running GCMC (high-fidelity multicomponent simulations)...")

    for cif_path in cif_paths:
        framework_name = cif_path.stem
        unit_cells = determine_unit_cells(cif_path, cutoff_angstrom=cutoff)

        for condition in conditions:
            job_idx += 1
            print(
                f"  GCMC [{job_idx}/{total_jobs}] {framework_name} | "
                f"T={condition.temperature_k:.2f} K, "
                f"P={condition.pressure_bar:g} bar, "
                f"RH={100.0 * condition.rh:.2f}%"
            )

            workspace = workspace_root / (
                f"{framework_name}__"
                f"T{condition.temperature_k:.2f}__"
                f"P{condition.pressure_bar:g}__"
                f"RH{condition.rh:.3f}"
            )

            sim_input = build_multicomponent_input(
                framework_name=framework_name,
                unit_cells=unit_cells,
                condition=condition,
                n_cycles=n_cycles,
                forcefield=forcefield,
                use_framework_charges=use_framework_charges,
                cutoff=cutoff,
                co2_definition=co2_definition,
                n2_definition=n2_definition,
                h2o_definition=h2o_definition,
            )

            ok, combined, full_log, returncode = run_raspa_job(
                raspa_path=raspa_path,
                runtime_root=runtime_root,
                framework_cif_dir=framework_cif_dir,
                source_cif=cif_path,
                workspace=workspace,
                input_text=sim_input,
            )

            if not ok:
                failures.append(
                    {
                        "mof_id": framework_name,
                        "temperature_k": condition.temperature_k,
                        "pressure_bar": condition.pressure_bar,
                        "rh": condition.rh,
                        "x_co2_total": condition.x_co2_total,
                        "x_n2_total": condition.x_n2_total,
                        "x_h2o_total": condition.x_h2o_total,
                        "log_path": str(full_log),
                        "returncode": int(returncode),
                        "error_tail": tail_text(combined, 80),
                    }
                )
                print(
                    f"GCMC failed for {framework_name} (returncode={returncode}): "
                    f"{tail_text(combined, 12)}"
                )
            else:
                parsed = parse_multicomponent_result(combined)
                rows.append(
                    {
                        "mof_id": framework_name,
                        "temperature_k": condition.temperature_k,
                        "pressure_bar": condition.pressure_bar,
                        "rh": condition.rh,
                        "p_h2o_bar": condition.p_h2o_bar,
                        "p_co2_bar": condition.p_co2_bar,
                        "p_n2_bar": condition.p_n2_bar,
                        "x_co2_total": condition.x_co2_total,
                        "x_n2_total": condition.x_n2_total,
                        "x_h2o_total": condition.x_h2o_total,
                        "y_co2_dry": condition.y_co2_dry,
                        "y_n2_dry": condition.y_n2_dry,
                        "mu_CO2": condition.mu_co2,
                        "mu_N2": condition.mu_n2,
                        "mu_H2O": condition.mu_h2o,
                        "T": condition.temperature_k,
                        "log_path": str(full_log),
                        **parsed,
                    }
                )

            if not keep_workspaces and workspace.exists():
                shutil.rmtree(workspace, ignore_errors=True)

    gcmc_df = pd.DataFrame(rows)
    failures_df = pd.DataFrame(failures)

    gcmc_out = (output_dir / "gcmc_adsorption.parquet").resolve()
    fail_out = (output_dir / "gcmc_failures.parquet").resolve()

    gcmc_df.to_parquet(gcmc_out, index=False)
    failures_df.to_parquet(fail_out, index=False)

    print(f"  GCMC: {len(gcmc_df)}/{total_jobs} succeeded -> {gcmc_out}")
    print(f"  Failed jobs saved -> {fail_out}")

    return gcmc_df, failures_df


def build_training_dataset(gcmc_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    print("\nMerging successful GCMC points into training dataset...")
    training_df = gcmc_df.copy()
    out_path = (output_dir / "adsorption_training.parquet").resolve()
    training_df.to_parquet(out_path, index=False)
    print(f"  Training data: {len(training_df)} rows -> {out_path}")
    return training_df


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run RASPA humid adsorption simulations robustly.")
    parser.add_argument("--cif-dir", default="data/intermediate/cifs_sanitized")
    parser.add_argument("--output-dir", default="data/processed/adsorption")
    parser.add_argument("--max-mofs", type=int, default=0, help="0 = all")
    parser.add_argument("--n-cycles", type=int, default=5000)
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument("--raspa-path", required=True)
    parser.add_argument("--raspa-data-dir", required=True)

    parser.add_argument("--temperatures", default="298.15,313.15,333.15")
    parser.add_argument("--pressures", default="0.1,0.5,1.0,5.0,10.0")
    parser.add_argument("--rhs", default="0.0,0.05,0.15")
    parser.add_argument("--dry-co2-frac", type=float, default=0.15)

    parser.add_argument("--forcefield", default="GenericMOFs")
    parser.add_argument("--cutoff", type=float, default=12.0)
    parser.add_argument(
        "--use-framework-charges",
        action="store_true",
        help="Read framework charges from CIF. Off by default in this debug-safe version.",
    )

    parser.add_argument("--co2-definition", default="TraPPE")
    parser.add_argument("--n2-definition", default="TraPPE")
    parser.add_argument("--h2o-definition", default="TraPPE")

    parser.add_argument("--co2-def-file", default=None, help="Optional path to CO2.def")
    parser.add_argument("--n2-def-file", default=None, help="Optional path to N2.def")
    parser.add_argument("--h2o-def-file", default=None, help="Optional path to H2O.def")

    args = parser.parse_args()

    cif_dir = Path(args.cif_dir).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    raspa_path = Path(args.raspa_path).expanduser().resolve()
    raspa_data_dir = Path(args.raspa_data_dir).expanduser().resolve()

    if not cif_dir.exists():
        raise FileNotFoundError(f"CIF directory not found: {cif_dir}")
    if not raspa_path.exists():
        raise FileNotFoundError(f"RASPA executable not found: {raspa_path}")

    temperatures_k = parse_float_list(args.temperatures)
    pressures_bar = parse_float_list(args.pressures)
    rhs = parse_float_list(args.rhs)

    cif_paths = load_cif_paths(cif_dir, args.max_mofs)
    if not cif_paths:
        raise FileNotFoundError(f"No CIF files found in: {cif_dir}")

    conditions = build_valid_conditions(
        temperatures_k=temperatures_k,
        pressures_bar=pressures_bar,
        rhs=rhs,
        dry_co2_frac=args.dry_co2_frac,
    )

    raspa_share = detect_raspa_share(raspa_data_dir)
    resolved_forcefield = resolve_forcefield_name(raspa_share, args.forcefield)

    runtime_root, runtime_share, framework_cif_dir = prepare_runtime_root(
        runtime_root=output_dir / "_raspa_runtime",
        raspa_share=raspa_share,
    )

    co2_definition = args.co2_definition
    n2_definition = args.n2_definition
    h2o_definition = args.h2o_definition

    if args.co2_def_file:
        co2_definition = install_user_definition_file(runtime_share, "CO2", Path(args.co2_def_file))
    else:
        co2_definition = find_definition_dir_by_file(raspa_share, "CO2", args.co2_definition)

    if args.n2_def_file:
        n2_definition = install_user_definition_file(runtime_share, "N2", Path(args.n2_def_file))
    else:
        n2_definition = find_definition_dir_by_file(raspa_share, "N2", args.n2_definition)

    if args.h2o_def_file:
        h2o_definition = install_user_definition_file(runtime_share, "H2O", Path(args.h2o_def_file))
    else:
        h2o_definition = find_definition_dir_by_file(raspa_share, "H2O", args.h2o_definition)

    check_forcefield_assets(runtime_share, resolved_forcefield)
    check_molecule_definition(runtime_share, co2_definition, "CO2")
    check_molecule_definition(runtime_share, n2_definition, "N2")
    check_molecule_definition(runtime_share, h2o_definition, "H2O")

    print("=" * 60)
    print(f"SIMULATION SETUP: {len(cif_paths)} MOFs")
    print(f"  Temperatures (K):   {temperatures_k}")
    print(f"  Pressures (bar):    {pressures_bar}")
    print(f"  RH values:          {rhs}")
    print(f"  Dry gas CO2 frac:   {args.dry_co2_frac:.3f}")
    print(f"  Valid cond./MOF:    {len(conditions)}")
    print(f"  Total GCMC jobs:    {len(cif_paths) * len(conditions)}")
    print(f"  RASPA executable:   {raspa_path}")
    print(f"  RASPA share dir:    {raspa_share}")
    print(f"  Runtime root:       {runtime_root}")
    print(f"  Forcefield used:    {resolved_forcefield}")
    print(f"  CO2 definition dir: {co2_definition}")
    print(f"  N2 definition dir:  {n2_definition}")
    print(f"  H2O definition dir: {h2o_definition}")
    print("=" * 60)
    print()

    run_low_fidelity_isotherms(
        cif_paths=cif_paths,
        output_dir=output_dir,
        runtime_root=runtime_root,
        framework_cif_dir=framework_cif_dir,
        raspa_path=raspa_path,
        temperatures_k=temperatures_k,
        pressures_bar=pressures_bar,
        n_cycles=args.n_cycles,
        keep_workspaces=args.keep_workspaces,
        forcefield=resolved_forcefield,
        use_framework_charges=args.use_framework_charges,
        cutoff=args.cutoff,
        co2_definition=co2_definition,
        n2_definition=n2_definition,
    )

    gcmc_df, _ = run_high_fidelity_gcmc(
        cif_paths=cif_paths,
        output_dir=output_dir,
        runtime_root=runtime_root,
        framework_cif_dir=framework_cif_dir,
        raspa_path=raspa_path,
        conditions=conditions,
        n_cycles=args.n_cycles,
        keep_workspaces=args.keep_workspaces,
        forcefield=resolved_forcefield,
        use_framework_charges=args.use_framework_charges,
        cutoff=args.cutoff,
        co2_definition=co2_definition,
        n2_definition=n2_definition,
        h2o_definition=h2o_definition,
    )

    build_training_dataset(gcmc_df, output_dir)

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("Next: python scripts/05_train_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()