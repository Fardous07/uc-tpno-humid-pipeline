#!/usr/bin/env python3
"""
src/simulation/gcmc/parser.py
──────────────────────────────
GCMC runner (RASPA2) for the UC-TPNO pipeline.

NOTE on project layout
──────────────────────
In this project the conventional names are swapped:
    parser.py  → GCMCConfig / generate_input / GCMCRunner  (the RUNNER)
    runner.py  → RASPAWriter / RASPAParser / parse_raspa_output  (the PARSER)

Fixes applied
─────────────
1. parse_raspa_output bridge — was calling from .runner import parse_raspa_output
   which now exists as a standalone function in runner.py. Works correctly.
2. results_to_arrays — re-exported from runner.py so tests that do
   ``from src.simulation.gcmc.parser import results_to_arrays`` work.
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
class GCMCConfig:
    """GCMC simulation parameters."""
    raspa_path:     str            = "simulate"
    raspa_data_dir: Optional[str]  = None
    work_dir:       str            = "simulation_workspace"
    forcefield:     str            = "ExampleMOFsForceField"
    cutoff_vdw:     float          = 12.0
    cutoff_coul:    float          = 12.0
    n_cycles:       int            = 100_000
    n_init:         int            = 50_000
    print_every:    int            = 10_000
    use_charges:    bool           = True
    timeout:        int            = 7200
    unit_cells_min: int            = 1


# ═══════════════════════════════════════════════════════════════════════
# 2. HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _resolve_raspa_executable(raspa_path: str) -> Optional[str]:
    if not raspa_path:
        return None
    p = Path(raspa_path).expanduser()
    if p.exists():
        return str(p.resolve())
    found = shutil.which(raspa_path)
    return found or None


def _resolve_raspa_data_dir(
    raspa_data_dir: Optional[str],
    exe_path: Optional[str],
) -> Optional[str]:
    """
    Resolve the RASPA prefix/data root.

    We pass the PREFIX ROOT (not prefix/share/raspa) because RASPA internally
    appends share/raspa when using -d.  Passing the full path would produce
    doubled paths like .../share/raspa/share/raspa/molecules/...
    """
    if raspa_data_dir:
        p = Path(raspa_data_dir).expanduser().resolve()
        if p.exists():
            return str(p)
        return None
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
    """Estimate unit-cell replication using the 2×cutoff rule."""
    try:
        text = cif_path.read_text(errors="ignore")
        lengths: List[float] = []
        for tag in ("_cell_length_a", "_cell_length_b", "_cell_length_c"):
            m = re.search(rf"{tag}\s+([+-]?\d+(?:\.\d+)?)", text)
            if m:
                lengths.append(float(m.group(1)))
        if len(lengths) == 3 and all(L > 0 for L in lengths):
            return tuple(
                max(int(np.ceil((2.0 * cutoff) / L)), min_cells) for L in lengths
            )
    except Exception:
        pass
    return (2, 2, 2)


def _normalize_composition(composition: Dict[str, float]) -> Dict[str, float]:
    if not composition:
        raise ValueError("Composition dictionary is empty.")
    cleaned: Dict[str, float] = {}
    for k, v in composition.items():
        val = float(v)
        if val < 0:
            raise ValueError(f"Negative mole fraction for {k}: {val}")
        cleaned[str(k)] = val
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("Composition sum must be > 0.")
    return {k: v / total for k, v in cleaned.items()}


# ═══════════════════════════════════════════════════════════════════════
# 3. INPUT GENERATION
# ═══════════════════════════════════════════════════════════════════════

_TEMPLATE_GCMC = """\
SimulationType                MonteCarlo
NumberOfCycles                {n_cycles}
NumberOfInitializationCycles  {n_init}
PrintEvery                    {print_every}
PrintPropertiesEvery          {print_every}

Forcefield                    {forcefield}
CutOffVDW                     {cutoff_vdw}
CutOffCoulomb                 {cutoff_coul}
ChargeMethod                  Ewald
EwaldPrecision                1e-6

Framework 0
FrameworkName                 {mof_name}
FrameworkFolders              {job_dir}
UnitCells                     {uc_a} {uc_b} {uc_c}
ExternalTemperature           {temperature}
ExternalPressure              {pressure_pa}
{charges_line}

Component 0 MoleculeName             CO2
            MoleculeDefinition       TraPPE
            MolFraction              {y_co2}
            TranslationProbability   0.5
            RotationProbability      0.5
            ReinsertionProbability   0.5
            SwapProbability          1.0
            CreateNumberOfMolecules  0

Component 1 MoleculeName             N2
            MoleculeDefinition       TraPPE
            MolFraction              {y_n2}
            TranslationProbability   0.5
            RotationProbability      0.5
            ReinsertionProbability   0.5
            SwapProbability          1.0
            CreateNumberOfMolecules  0
{h2o_block}
"""

_H2O_BLOCK = """
Component 2 MoleculeName             H2O
            MoleculeDefinition       TIP4P/2005
            MolFraction              {y_h2o}
            TranslationProbability   0.5
            RotationProbability      0.5
            ReinsertionProbability   0.5
            SwapProbability          1.0
            CreateNumberOfMolecules  0
"""


def generate_input(
    mof_cif: Union[str, Path],
    temperature: float,
    pressure: float,
    composition: Dict[str, float],
    config: Optional[GCMCConfig] = None,
    job_dir: Union[str, Path, None] = None,
) -> str:
    """Generate a RASPA2 GCMC input file string."""
    if config is None:
        config = GCMCConfig()

    cif_path = Path(mof_cif)
    mof_name = cif_path.stem

    comp   = _normalize_composition(composition)
    y_co2  = comp.get("CO2", 0.0)
    y_n2   = comp.get("N2",  0.0)
    y_h2o  = comp.get("H2O", 0.0)

    uc_a, uc_b, uc_c = _estimate_unit_cells(
        cif_path, cutoff=config.cutoff_vdw, min_cells=config.unit_cells_min,
    )

    h2o_block    = _H2O_BLOCK.format(y_h2o=y_h2o) if y_h2o > 1e-12 else ""
    charges_line = "UseChargesFromCIFFile         yes" if config.use_charges else ""
    pressure_pa  = float(pressure) * 1e5   # bar → Pa

    return _TEMPLATE_GCMC.format(
        n_cycles=int(config.n_cycles),
        n_init=int(config.n_init),
        print_every=int(config.print_every),
        forcefield=config.forcefield,
        cutoff_vdw=float(config.cutoff_vdw),
        cutoff_coul=float(config.cutoff_coul),
        mof_name=mof_name,
        job_dir=str(job_dir) if job_dir is not None else ".",
        uc_a=uc_a, uc_b=uc_b, uc_c=uc_c,
        temperature=float(temperature),
        pressure_pa=pressure_pa,
        charges_line=charges_line,
        y_co2=y_co2, y_n2=y_n2,
        h2o_block=h2o_block,
    )


# ═══════════════════════════════════════════════════════════════════════
# 4. PARSER BRIDGE + results_to_arrays re-export
# ═══════════════════════════════════════════════════════════════════════

def parse_raspa_output(job_dir: Union[str, Path]) -> Dict[str, Any]:
    """Bridge to the standalone parser in gcmc/runner.py."""
    from .runner import parse_raspa_output as _parse
    return _parse(job_dir)


# FIX: re-export results_to_arrays so tests can import it from gcmc.parser
from .runner import results_to_arrays   # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# 5. GCMC RUNNER
# ═══════════════════════════════════════════════════════════════════════

class GCMCRunner:
    """Manage RASPA2 GCMC simulations (composition-based interface)."""

    def __init__(self, config: Optional[Union[GCMCConfig, Dict[str, Any]]] = None):
        if config is None:
            config = GCMCConfig()
        elif isinstance(config, dict):
            valid_keys = GCMCConfig.__dataclass_fields__.keys()
            config = GCMCConfig(**{k: v for k, v in config.items() if k in valid_keys})
        self.config   = config
        self.work_dir = Path(self.config.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    # ── Public methods ────────────────────────────────────────────

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
        temperature: float,
        pressure: float,
        composition: Dict[str, float],
        clean_after: bool = True,
    ) -> Dict[str, Any]:
        """Run one GCMC simulation."""
        cif_path = Path(mof_cif)

        result: Dict[str, Any] = {
            "mof_id":      cif_path.stem,
            "temperature": float(temperature),
            "pressure":    float(pressure),
            "composition": composition,
            "success":     False,
            "loadings":    {},
            "energies":    {},
            "converged":   None,
            "warnings":    [],
            "enthalpy":    None,
            "error":       None,
            "returncode":  None,
            "job_dir":     None,
        }

        if not cif_path.exists():
            result["error"] = f"CIF not found: {cif_path}"
            return result

        try:
            comp = _normalize_composition(composition)
            result["composition"] = comp
        except Exception as e:
            result["error"] = f"Invalid composition: {e}"
            return result

        exe = _resolve_raspa_executable(self.config.raspa_path)
        if exe is None:
            result["error"] = f"RASPA executable not found: {self.config.raspa_path}"
            logger.error(result["error"])
            return result

        raspa_data_dir = _resolve_raspa_data_dir(self.config.raspa_data_dir, exe)
        if raspa_data_dir is None:
            logger.warning("RASPA data directory could not be resolved.")

        job_id  = self._job_id(cif_path.stem, float(temperature), float(pressure), comp)
        job_dir = self.work_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        result["job_dir"] = str(job_dir)

        try:
            input_str = generate_input(
                mof_cif=cif_path, temperature=float(temperature),
                pressure=float(pressure), composition=comp,
                config=self.config, job_dir=job_dir,
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

            parsed = parse_raspa_output(job_dir)
            result["loadings"]  = parsed.get("loadings", {})
            result["converged"] = parsed.get("converged", None)
            result["warnings"]  = parsed.get("warnings", [])

            if result["loadings"]:
                result["success"] = True
            else:
                result["error"] = "RASPA finished, but no loadings were parsed."

        except subprocess.TimeoutExpired:
            result["error"] = f"Timeout after {self.config.timeout}s"

        except Exception as e:
            result["error"] = str(e)
            logger.exception("Unexpected GCMC error for %s", cif_path.stem)

        finally:
            if clean_after and job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)

        return result

    def run_batch(
        self,
        mof_cifs: Sequence[Union[str, Path]],
        temperatures: Sequence[float],
        pressures: Sequence[float],
        compositions: Sequence[Dict[str, float]],
        clean_after: bool = True,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        total = len(mof_cifs) * len(temperatures) * len(pressures) * len(compositions)
        idx = 0
        for cif in mof_cifs:
            for T in temperatures:
                for P in pressures:
                    for comp in compositions:
                        idx += 1
                        logger.info("GCMC batch [%d/%d]", idx, total)
                        results.append(self.run_single(
                            mof_cif=cif, temperature=float(T), pressure=float(P),
                            composition=comp, clean_after=clean_after,
                        ))
        n_ok = sum(bool(r.get("success")) for r in results)
        logger.info("Batch complete: %d/%d succeeded", n_ok, total)
        return results

    def run_isotherm(
        self,
        mof_cif: Union[str, Path],
        temperature: float,
        pressures: Sequence[float],
        composition: Dict[str, float],
        clean_after: bool = True,
    ) -> Dict[str, Any]:
        results = [
            self.run_single(
                mof_cif=mof_cif, temperature=float(temperature), pressure=float(P),
                composition=composition, clean_after=clean_after,
            )
            for P in pressures
        ]
        pressure_arr = np.array([r["pressure"] for r in results], dtype=float)
        species: set = set()
        for r in results:
            species.update(r.get("loadings", {}).keys())
        loadings: Dict[str, np.ndarray] = {
            sp: np.array([r.get("loadings", {}).get(sp, np.nan) for r in results])
            for sp in sorted(species)
        }
        return {
            "mof_id":      Path(mof_cif).stem,
            "temperature": float(temperature),
            "composition": _normalize_composition(composition),
            "pressures":   pressure_arr,
            "loadings":    loadings,
            "n_success":   sum(bool(r.get("success")) for r in results),
            "n_total":     len(results),
            "results":     results,
        }

    # ── Internal ──────────────────────────────────────────────────

    def _job_id(self, mof_name: str, T: float, P: float, comp: Dict[str, float]) -> str:
        comp_str = "_".join(f"{k}{comp[k]:.6f}" for k in sorted(comp))
        raw = f"{mof_name}_T{T:.3f}_P{P:.6f}_{comp_str}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "GCMCConfig",
    "GCMCRunner",
    "generate_input",
    "parse_raspa_output",
    "results_to_arrays",
]