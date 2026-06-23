# /home/fardo/uc_tpno_humid_pipeline/uc_tpno_humid_pipeline/src/simulation/gcmc/parser.py
#!/usr/bin/env python3
"""
RASPA2 GCMC simulation runner.

IMPORTANT
---------
In this project layout:
    - src/simulation/gcmc/parser.py  -> RUNNER
    - src/simulation/gcmc/runner.py  -> OUTPUT PARSER

This file provides:
    - GCMCConfig
    - generate_input(...)
    - GCMCRunner
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
    """
    GCMC simulation parameters.
    """
    raspa_path: str = "simulate"
    raspa_data_dir: Optional[str] = None
    work_dir: str = "simulation_workspace"
    forcefield: str = "ExampleMOFsForceField"
    cutoff_vdw: float = 12.0
    cutoff_coul: float = 12.0
    n_cycles: int = 100_000
    n_init: int = 50_000
    print_every: int = 10_000
    use_charges: bool = True
    timeout: int = 7200
    unit_cells_min: int = 1


# ═══════════════════════════════════════════════════════════════════════
# 2. HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _resolve_raspa_executable(raspa_path: str) -> Optional[str]:
    """
    Resolve a RASPA executable from:
      - an explicit filesystem path
      - something available on PATH
    """
    if not raspa_path:
        return None

    p = Path(raspa_path).expanduser()
    if p.exists():
        return str(p.resolve())

    found = shutil.which(raspa_path)
    if found:
        return found

    return None


def _resolve_raspa_data_dir(
    raspa_data_dir: Optional[str],
    exe_path: Optional[str],
) -> Optional[str]:
    """
    Resolve the RASPA prefix/data root.

    Priority:
      1. explicit raspa_data_dir if provided
      2. infer from executable path:
            /.../envs/raspa-env/bin/simulate -> /.../envs/raspa-env

    IMPORTANT
    ---------
    For this project and your conda RASPA install, we pass the PREFIX ROOT,
    not prefix/share/raspa.

    Why:
      RASPA internally resolves share/raspa beneath the prefix when using -d.
      If we pass prefix/share/raspa, it can produce broken doubled paths like:
          .../share/raspa/share/raspa/molecules/TraPPE/CO2.def
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
    """
    Estimate unit-cell replication using the 2×cutoff rule.

    Falls back to (2, 2, 2) if parsing fails.
    """
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
    """
    Validate and normalize gas composition.
    """
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
    """
    Generate a RASPA2 GCMC input file string.
    """
    if config is None:
        config = GCMCConfig()

    cif_path = Path(mof_cif)
    mof_name = cif_path.stem

    comp = _normalize_composition(composition)
    y_co2 = comp.get("CO2", 0.0)
    y_n2 = comp.get("N2", 0.0)
    y_h2o = comp.get("H2O", 0.0)

    uc_a, uc_b, uc_c = _estimate_unit_cells(
        cif_path,
        cutoff=config.cutoff_vdw,
        min_cells=config.unit_cells_min,
    )

    h2o_block = _H2O_BLOCK.format(y_h2o=y_h2o) if y_h2o > 1e-12 else ""
    charges_line = "UseChargesFromCIFFile         yes" if config.use_charges else ""
    pressure_pa = float(pressure) * 1e5  # bar -> Pa

    return _TEMPLATE_GCMC.format(
        n_cycles=int(config.n_cycles),
        n_init=int(config.n_init),
        print_every=int(config.print_every),
        forcefield=config.forcefield,
        cutoff_vdw=float(config.cutoff_vdw),
        cutoff_coul=float(config.cutoff_coul),
        mof_name=mof_name,
        job_dir=str(job_dir) if job_dir is not None else ".",
        uc_a=uc_a,
        uc_b=uc_b,
        uc_c=uc_c,
        temperature=float(temperature),
        pressure_pa=pressure_pa,
        charges_line=charges_line,
        y_co2=y_co2,
        y_n2=y_n2,
        h2o_block=h2o_block,
    )


# ═══════════════════════════════════════════════════════════════════════
# 4. PARSER BRIDGE
# ═══════════════════════════════════════════════════════════════════════

def parse_raspa_output(job_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Compatibility bridge to the actual output parser in gcmc/runner.py
    """
    from .runner import parse_raspa_output as _parse_raspa_output
    return _parse_raspa_output(job_dir)


# ═══════════════════════════════════════════════════════════════════════
# 5. GCMC RUNNER
# ═══════════════════════════════════════════════════════════════════════

class GCMCRunner:
    """
    Manage RASPA2 GCMC simulations.
    """

    def __init__(self, config: Optional[Union[GCMCConfig, Dict[str, Any]]] = None):
        if config is None:
            config = GCMCConfig()
        elif isinstance(config, dict):
            valid_keys = GCMCConfig.__dataclass_fields__.keys()
            config = GCMCConfig(**{k: v for k, v in config.items() if k in valid_keys})

        self.config = config
        self.work_dir = Path(self.config.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _job_id(self, mof_name: str, T: float, P: float, comp: Dict[str, float]) -> str:
        comp_norm = _normalize_composition(comp)
        comp_str = "_".join(f"{k}{comp_norm[k]:.6f}" for k in sorted(comp_norm))
        raw = f"{mof_name}_T{T:.3f}_P{P:.6f}_{comp_str}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    def run_single(
        self,
        mof_cif: Union[str, Path],
        temperature: float,
        pressure: float,
        composition: Dict[str, float],
        clean_after: bool = True,
    ) -> Dict[str, Any]:
        """
        Run one GCMC simulation.
        """
        cif_path = Path(mof_cif)

        result: Dict[str, Any] = {
            "mof_id": cif_path.stem,
            "temperature": float(temperature),
            "pressure": float(pressure),
            "composition": composition,
            "success": False,
            "loadings": {},
            "energies": {},
            "converged": None,
            "warnings": [],
            "enthalpy": None,
            "error": None,
            "returncode": None,
            "job_dir": None,
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
            logger.warning(
                "RASPA data directory could not be resolved. "
                "RASPA may fail to find force-field files."
            )

        mof_name = cif_path.stem
        job_id = self._job_id(mof_name, float(temperature), float(pressure), comp)
        job_dir = self.work_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        result["job_dir"] = str(job_dir)

        try:
            input_str = generate_input(
                mof_cif=cif_path,
                temperature=float(temperature),
                pressure=float(pressure),
                composition=comp,
                config=self.config,
                job_dir=job_dir,
            )

            (job_dir / "simulation.input").write_text(input_str, encoding="utf-8")

            cif_dest = job_dir / cif_path.name
            if not cif_dest.exists():
                shutil.copy2(cif_path, cif_dest)

            logger.info(
                "Running GCMC | MOF=%s | T=%.2f K | P=%.5g bar | comp=%s",
                mof_name,
                float(temperature),
                float(pressure),
                comp,
            )

            cmd = [exe, "-i", "simulation.input"]
            if raspa_data_dir:
                cmd.extend(["-d", str(raspa_data_dir)])

            proc = subprocess.run(
                cmd,
                cwd=str(job_dir),
                capture_output=True,
                text=True,
                timeout=int(self.config.timeout),
            )

            result["returncode"] = int(proc.returncode)

            (job_dir / "stdout.log").write_text(proc.stdout or "", encoding="utf-8")
            (job_dir / "stderr.log").write_text(proc.stderr or "", encoding="utf-8")

            if proc.returncode != 0:
                result["error"] = (proc.stderr or proc.stdout or "Non-zero exit")[:2000]
                logger.warning(
                    "GCMC failed for %s (returncode=%s): %s",
                    mof_name,
                    proc.returncode,
                    result["error"][:300],
                )
                return result

            parsed = parse_raspa_output(job_dir)
            result["loadings"] = parsed.get("loadings", {})
            result["energies"] = parsed.get("energies", {})
            result["converged"] = parsed.get("converged", None)
            result["warnings"] = parsed.get("warnings", [])
            result["enthalpy"] = parsed.get("enthalpy", None)

            if result["loadings"]:
                result["success"] = True
            else:
                result["error"] = "RASPA finished, but no loadings were parsed from output files."

        except subprocess.TimeoutExpired:
            result["error"] = f"Timeout after {self.config.timeout}s"
            logger.warning("GCMC timeout for %s", mof_name)

        except Exception as e:
            result["error"] = str(e)
            logger.exception("Unexpected GCMC error for %s", mof_name)

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
        """
        Run a serial batch over the Cartesian product:
            mof_cifs × temperatures × pressures × compositions
        """
        results: List[Dict[str, Any]] = []
        total = len(mof_cifs) * len(temperatures) * len(pressures) * len(compositions)
        idx = 0

        for cif in mof_cifs:
            for T in temperatures:
                for P in pressures:
                    for comp in compositions:
                        idx += 1
                        logger.info("GCMC batch [%d/%d]", idx, total)
                        results.append(
                            self.run_single(
                                mof_cif=cif,
                                temperature=float(T),
                                pressure=float(P),
                                composition=comp,
                                clean_after=clean_after,
                            )
                        )

        n_ok = sum(bool(r.get("success", False)) for r in results)
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
        """
        Run a pressure sweep at fixed temperature and composition.
        """
        results = [
            self.run_single(
                mof_cif=mof_cif,
                temperature=float(temperature),
                pressure=float(P),
                composition=composition,
                clean_after=clean_after,
            )
            for P in pressures
        ]

        pressure_arr = np.array([r["pressure"] for r in results], dtype=float)
        species = set()
        for r in results:
            species.update(r.get("loadings", {}).keys())

        loadings: Dict[str, np.ndarray] = {}
        for sp in sorted(species):
            loadings[sp] = np.array(
                [r.get("loadings", {}).get(sp, np.nan) for r in results],
                dtype=float,
            )

        return {
            "mof_id": Path(mof_cif).stem,
            "temperature": float(temperature),
            "composition": _normalize_composition(composition),
            "pressures": pressure_arr,
            "loadings": loadings,
            "n_success": sum(bool(r.get("success", False)) for r in results),
            "n_total": len(results),
            "results": results,
        }

    def check_raspa(self) -> bool:
        """
        Check whether the configured RASPA executable is available.
        """
        exe = _resolve_raspa_executable(self.config.raspa_path)
        if exe is None:
            return False

        try:
            proc = subprocess.run(
                [exe, "-h"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return proc.returncode is not None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


__all__ = [
    "GCMCConfig",
    "GCMCRunner",
    "generate_input",
    "parse_raspa_output",
]
