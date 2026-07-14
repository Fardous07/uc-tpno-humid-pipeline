#!/usr/bin/env python3
"""
src/simulation/gctmmc/parser.py
────────────────────────────────
GC-TMMC runner (RASPA2) for the UC-TPNO pipeline.

NOTE on project layout
──────────────────────
In this project the conventional names are swapped:
    gctmmc/parser.py  → GCTMMCConfig / generate_tmmc_input / GCTMMCRunner (RUNNER)
    gctmmc/runner.py  → parse_tmmc_output / collection_matrix_to_ln_pi … (PARSER)

Fixes applied
─────────────
1. parse_tmmc_output bridge: signature was (job_dir, temperature) but test calls
   parse_tmmc_output(tmp_path, T=313.15) → TypeError.  Fixed: accept **T** kwarg.
2. Re-export synthetic_collection_matrix, collection_matrix_to_ln_pi,
   ln_pi_to_isotherm from runner.py so test imports from gctmmc.parser work.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GCTMMCConfig:
    """GC-TMMC simulation parameters."""
    raspa_path:     str           = "simulate"
    raspa_data_dir: Optional[str] = None
    work_dir:       str           = "tmmc_workspace"
    forcefield:     str           = "ExampleMOFsForceField"
    cutoff_vdw:     float         = 12.0
    n_cycles:       int           = 500_000
    n_init:         int           = 50_000
    N_max:          int           = 200
    temperature:    float         = 313.15
    bias_update:    int           = 10_000
    use_charges:    bool          = True
    timeout:        int           = 3600
    unit_cells_min: int           = 1


# ═══════════════════════════════════════════════════════════════════════
# 2. HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _resolve_raspa_executable(raspa_path: str) -> Optional[str]:
    if not raspa_path:
        return None
    p = Path(raspa_path).expanduser()
    if p.exists():
        return str(p.resolve())
    return shutil.which(raspa_path)


def _resolve_raspa_data_dir(
    raspa_data_dir: Optional[str],
    exe_path: Optional[str],
) -> Optional[str]:
    """
    Resolve the RASPA prefix/data root (PREFIX ROOT, not prefix/share/raspa).
    RASPA appends share/raspa internally; passing the full path doubles it.
    """
    if raspa_data_dir:
        p = Path(raspa_data_dir).expanduser().resolve()
        return str(p) if p.exists() else None
    if exe_path:
        exe = Path(exe_path).expanduser().resolve()
        if exe.exists():
            prefix = exe.parent.parent
            if prefix.exists():
                return str(prefix)
    return None


def _estimate_unit_cells(
    cif_path: Path,
    cutoff: float = 12.0,
    min_cells: int = 1,
) -> Tuple[int, int, int]:
    try:
        text    = cif_path.read_text(errors="ignore")
        lengths: List[float] = []
        for tag in ("_cell_length_a", "_cell_length_b", "_cell_length_c"):
            m = re.search(rf"{tag}\s+([+-]?\d+(?:\.\d+)?)", text)
            if m:
                lengths.append(float(m.group(1)))
        if len(lengths) == 3 and all(L > 0 for L in lengths):
            return tuple(max(int(np.ceil((2.0 * cutoff) / L)), min_cells) for L in lengths)
    except Exception:
        pass
    return (2, 2, 2)


# ═══════════════════════════════════════════════════════════════════════
# 3. INPUT GENERATION
# ═══════════════════════════════════════════════════════════════════════

_TEMPLATE_TMMC = """\
SimulationType                MonteCarlo
NumberOfCycles                {n_cycles}
NumberOfInitializationCycles  {n_init}
PrintEvery                    {print_every}

Forcefield                    {forcefield}
CutOffVDW                     {cutoff_vdw}
ChargeMethod                  Ewald
EwaldPrecision                1e-6

Framework 0
FrameworkName                 {mof_name}
FrameworkFolders              {job_dir}
UnitCells                     {uc_a} {uc_b} {uc_c}
ExternalTemperature           {temperature}
{charges_line}

Component 0 MoleculeName             {molecule}
            MoleculeDefinition       {molecule_def}
            TranslationProbability   0.5
            RotationProbability      0.5
            ReinsertionProbability   0.5
            SwapProbability          1.0
            CreateNumberOfMolecules  0

TMC_Method                    yes
TMC_MaxNumberOfMolecules      {N_max}
TMC_BiasUpdate                {bias_update}
TMC_CollectionMatrix          yes
"""

_MOLECULE_DEFS: Dict[str, str] = {
    "CO2": "TraPPE",
    "N2":  "TraPPE",
    "H2O": "TIP4P/2005",
    "CH4": "TraPPE",
    "H2":  "Buch",
}


def generate_tmmc_input(
    mof_cif: Union[str, Path],
    molecule: str,
    temperature: float,
    config: Optional[GCTMMCConfig] = None,
    job_dir: Union[str, Path, None] = None,
) -> str:
    """Generate a RASPA2 GC-TMMC input file string."""
    if config is None:
        config = GCTMMCConfig()

    cif_path     = Path(mof_cif)
    mof_name     = cif_path.stem
    molecule     = str(molecule).strip()
    molecule_def = _MOLECULE_DEFS.get(molecule, "TraPPE")
    uc_a, uc_b, uc_c = _estimate_unit_cells(
        cif_path, cutoff=config.cutoff_vdw, min_cells=config.unit_cells_min,
    )
    charges_line = "UseChargesFromCIFFile         yes" if config.use_charges else ""

    return _TEMPLATE_TMMC.format(
        n_cycles=int(config.n_cycles),
        n_init=int(config.n_init),
        print_every=max(int(config.n_cycles // 20), 1000),
        forcefield=config.forcefield,
        cutoff_vdw=float(config.cutoff_vdw),
        mof_name=mof_name,
        job_dir=str(job_dir) if job_dir is not None else ".",
        uc_a=uc_a, uc_b=uc_b, uc_c=uc_c,
        temperature=float(temperature),
        charges_line=charges_line,
        molecule=molecule,
        molecule_def=molecule_def,
        N_max=int(config.N_max),
        bias_update=int(config.bias_update),
    )


# ═══════════════════════════════════════════════════════════════════════
# 4. PARSER BRIDGE + re-exports
# FIX: accept T kwarg (was 'temperature') to match test call:
#      parse_tmmc_output(tmp_path, T=313.15)
# FIX: re-export synthetic_collection_matrix, collection_matrix_to_ln_pi,
#      ln_pi_to_isotherm from runner.py so test imports from gctmmc.parser work.
# ═══════════════════════════════════════════════════════════════════════

def parse_tmmc_output(
    job_dir: Union[str, Path],
    T: float = 313.15,           # FIX: renamed from temperature
    pressures=None,
) -> Dict[str, Any]:
    """Bridge to the actual parser in gctmmc/runner.py."""
    from .runner import parse_tmmc_output as _parse
    return _parse(job_dir, T=T, pressures=pressures)


# Re-export parser utilities so tests can do:
#   from src.simulation.gctmmc.parser import synthetic_collection_matrix
from .runner import (           # noqa: E402
    synthetic_collection_matrix,
    collection_matrix_to_ln_pi,
    ln_pi_to_isotherm,
)


# ═══════════════════════════════════════════════════════════════════════
# 5. GC-TMMC RUNNER
# ═══════════════════════════════════════════════════════════════════════

class GCTMMCRunner:
    """Run GC-TMMC simulations for single-component isotherms."""

    def __init__(self, config: Optional[Union[GCTMMCConfig, Dict[str, Any]]] = None):
        if config is None:
            config = GCTMMCConfig()
        elif isinstance(config, dict):
            valid_keys = GCTMMCConfig.__dataclass_fields__.keys()
            config = GCTMMCConfig(**{k: v for k, v in config.items() if k in valid_keys})
        self.config   = config
        self.work_dir = Path(self.config.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def check_raspa(self) -> bool:
        exe = _resolve_raspa_executable(self.config.raspa_path)
        if exe is None:
            return False
        try:
            proc = subprocess.run([exe, "-h"], capture_output=True, text=True, timeout=10)
            return proc.returncode is not None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def run_single(
        self,
        mof_cif: Union[str, Path],
        molecule: str = "CO2",
        temperature: Optional[float] = None,
        clean_after: bool = True,
    ) -> Dict[str, Any]:
        """Run one GC-TMMC simulation."""
        cif_path = Path(mof_cif)
        T        = float(temperature if temperature is not None else self.config.temperature)
        molecule = str(molecule).strip()

        result: Dict[str, Any] = {
            "mof_id":      cif_path.stem,
            "molecule":    molecule,
            "temperature": T,
            "success":     False,
            "pressures":   np.array([], dtype=float),
            "loadings":    np.array([], dtype=float),
            "ln_pi":       np.array([], dtype=float),
            "warnings":    [],
            "error":       None,
            "returncode":  None,
            "job_dir":     None,
        }

        if not cif_path.exists():
            result["error"] = f"CIF not found: {cif_path}"
            return result

        exe = _resolve_raspa_executable(self.config.raspa_path)
        if exe is None:
            result["error"] = f"RASPA executable not found: {self.config.raspa_path}"
            logger.error(result["error"])
            return result

        raspa_data_dir = _resolve_raspa_data_dir(self.config.raspa_data_dir, exe)
        if raspa_data_dir is None:
            logger.warning("RASPA data directory could not be resolved.")

        job_id  = self._job_id(cif_path.stem, molecule, T)
        job_dir = self.work_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        result["job_dir"] = str(job_dir)

        try:
            input_str = generate_tmmc_input(
                mof_cif=cif_path, molecule=molecule,
                temperature=T, config=self.config, job_dir=job_dir,
            )
            (job_dir / "simulation.input").write_text(input_str, encoding="utf-8")

            cif_dest = job_dir / cif_path.name
            if not cif_dest.exists():
                shutil.copy2(cif_path, cif_dest)

            cmd = [exe, "-i", "simulation.input"]
            if raspa_data_dir:
                cmd.extend(["-d", str(raspa_data_dir)])

            proc = subprocess.run(
                cmd, cwd=str(job_dir), capture_output=True, text=True,
                timeout=int(self.config.timeout),
            )
            result["returncode"] = int(proc.returncode)
            (job_dir / "stdout.log").write_text(proc.stdout or "", encoding="utf-8")
            (job_dir / "stderr.log").write_text(proc.stderr or "", encoding="utf-8")

            if proc.returncode != 0:
                result["error"] = (proc.stderr or proc.stdout or "Non-zero exit")[:2000]
                return result

            parsed = parse_tmmc_output(job_dir, T=T)
            result.update(parsed)
            result["success"] = bool(parsed.get("success", False))
            if not result["success"] and not result.get("error"):
                result["error"] = "GC-TMMC finished, but no valid isotherm could be parsed."

        except subprocess.TimeoutExpired:
            result["error"] = f"Timeout after {self.config.timeout}s"

        except Exception as e:
            result["error"] = str(e)
            logger.exception("Unexpected GC-TMMC error for %s/%s", cif_path.stem, molecule)

        finally:
            if clean_after and job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)

        return result

    def run_multi_species(
        self,
        mof_cif: Union[str, Path],
        molecules: Sequence[str] = ("CO2", "N2", "H2O"),
        temperature: Optional[float] = None,
        clean_after: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        return {
            str(mol): self.run_single(mof_cif, str(mol), temperature, clean_after)
            for mol in molecules
        }

    def run_batch_mofs(
        self,
        mof_cifs: Sequence[Union[str, Path]],
        molecule: str = "CO2",
        temperature: Optional[float] = None,
        clean_after: bool = True,
    ) -> List[Dict[str, Any]]:
        return [
            self.run_single(cif, molecule, temperature, clean_after)
            for cif in mof_cifs
        ]

    def _job_id(self, mof_name: str, molecule: str, temperature: float) -> str:
        raw = f"{mof_name}_{molecule}_{float(temperature):.3f}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "GCTMMCConfig",
    "GCTMMCRunner",
    "generate_tmmc_input",
    "parse_tmmc_output",
    # Re-exported from runner.py
    "synthetic_collection_matrix",
    "collection_matrix_to_ln_pi",
    "ln_pi_to_isotherm",
]