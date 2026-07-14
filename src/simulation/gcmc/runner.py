"""
src/simulation/gcmc/runner.py
──────────────────────────────
GCMC output parser (RASPA2) for the UC-TPNO pipeline.

NOTE on project layout
──────────────────────
In this project the conventional names are swapped:
    parser.py  → GCMCConfig / generate_input / GCMCRunner  (the RUNNER)
    runner.py  → RASPAWriter / RASPAParser / GCMCInput      (the PARSER)

To satisfy test imports that do
    ``from src.simulation.gcmc.runner import GCMCConfig``
this file re-exports the runner symbols from parser.py.

Fixes applied
─────────────
1. Re-export GCMCConfig, generate_input, _estimate_unit_cells, GCMCRunner
   from parser.py so test imports work.
2. Rename the local advanced GCMCRunner → ChemPotGCMCRunner (no name clash).
3. Add standalone parse_raspa_output() using Component-header regexes that
   handle the single-file RASPA format used by the test fixture.
4. Add results_to_arrays() so it can be imported here or via parser.py.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Physical constants ────────────────────────────────────────────────
_R_KJ_MOL_K = 8.314462618e-3
_R_J_MOL_K  = 8.314462618
_COMPONENT_NAMES = ("CO2", "N2", "H2O")

# ── Module-level compiled regexes for parse_raspa_output ─────────────
_RE_COMPONENT = re.compile(r"^Component\s+\d+\s+\[(\w+)\]", re.IGNORECASE)
_RE_MOL_KG    = re.compile(
    r"Average\s+loading\s+absolute\s+\[mol/kg\s+framework\]\s+"
    r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
_RE_MOLEC_UC  = re.compile(
    r"Average\s+loading\s+absolute\s+\[molecules/unit\s+cell\]\s+"
    r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════
# RE-EXPORTS FROM parser.py
# (test imports GCMCConfig / generate_input / _estimate_unit_cells /
#  GCMCRunner from this module)
# ═══════════════════════════════════════════════════════════════════════

from .parser import (           # noqa: E402  (after constant defs is fine)
    GCMCConfig,
    generate_input,
    _estimate_unit_cells,
    GCMCRunner,
)


# ═══════════════════════════════════════════════════════════════════════
# STANDALONE OUTPUT PARSER
# ═══════════════════════════════════════════════════════════════════════

def parse_raspa_output(
    work_dir: Any,
    *,
    require_convergence: bool = False,
) -> Dict[str, Any]:
    """
    Parse all RASPA output files under ``work_dir/Output/System_0/``.

    Handles the standard RASPA single-file format where one ``.data`` file
    contains all components, identified by ``Component N [SPECIES]`` headers:

        Component 0 [CO2]    (total 200000 trial moves)

          Average loading absolute [mol/kg framework] 3.456000 +/- 0.123000

    When multiple ``.data`` files exist their loadings are averaged.

    Returns
    -------
    dict with:
        ``loadings``  : Dict[str, float] — mol/kg per species
        ``converged`` : bool
    """
    work_dir = Path(work_dir)
    out_dir  = work_dir / "Output" / "System_0"

    data_files = sorted(out_dir.glob("*.data")) if out_dir.is_dir() else []

    if not data_files:
        if require_convergence:
            raise FileNotFoundError(f"No RASPA .data files found in {out_dir}")
        logger.warning("No RASPA output files in %s — returning empty result.", out_dir)
        return {"loadings": {}, "converged": False}

    accumulated: Dict[str, List[float]] = {}

    for fp in data_files:
        current_species: Optional[str] = None
        for line in fp.read_text(encoding="utf-8", errors="replace").splitlines():
            comp_match = _RE_COMPONENT.match(line.strip())
            if comp_match:
                current_species = comp_match.group(1)
                continue
            if current_species is not None:
                m = _RE_MOL_KG.search(line)
                if m:
                    accumulated.setdefault(current_species, []).append(float(m.group(1)))
                    continue
                # Fallback: molecules/unit cell (stored as proxy if mol/kg absent)
                m2 = _RE_MOLEC_UC.search(line)
                if m2 and current_species not in accumulated:
                    logger.warning(
                        "Species %s: mol/kg not found; using molecules/uc as proxy.",
                        current_species,
                    )
                    accumulated.setdefault(current_species, []).append(float(m2.group(1)))

    averaged: Dict[str, float] = {
        sp: float(np.mean(vals)) for sp, vals in accumulated.items()
    }
    return {"loadings": averaged, "converged": True}


# ═══════════════════════════════════════════════════════════════════════
# RESULTS → ARRAY HELPER
# ═══════════════════════════════════════════════════════════════════════

def results_to_arrays(
    results: Sequence[Dict[str, Any]],
    species: Sequence[str],
) -> np.ndarray:
    """
    Convert a list of ``parse_raspa_output`` result dicts to a 2-D array.

    Parameters
    ----------
    results : Sequence of result dicts (each must have ``"loadings"`` key).
    species : Ordered species names — defines column order.

    Returns
    -------
    np.ndarray of shape ``(len(results), len(species))``, dtype float64.
    Missing species default to 0.0.
    """
    arr = np.zeros((len(results), len(species)), dtype=np.float64)
    for i, res in enumerate(results):
        loadings = res.get("loadings", {})
        for j, sp in enumerate(species):
            arr[i, j] = loadings.get(sp, 0.0)
    return arr


# ═══════════════════════════════════════════════════════════════════════
# DATA CLASSES (chemical-potential interface)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GCMCInput:
    """
    All parameters needed to define one GCMC simulation via chemical
    potentials (used by the active-learning loop).
    """
    cif_path:      Union[str, Path]
    mu_co2:        float
    mu_n2:         float
    mu_h2o:        float
    temperature:   float = 313.0
    n_cycles:      int   = 10_000
    n_init_cycles: int   = 2_000
    force_field:   str   = "UFF"
    unit_cells:    Tuple[int, int, int] = (1, 1, 1)
    mof_id:        str   = "unknown"

    @property
    def fugacities_pa(self) -> Dict[str, float]:
        """Convert μ (kJ/mol) → fugacity f = exp(μ / RT) in Pascal."""
        RT = _R_KJ_MOL_K * self.temperature
        fu = {}
        for name, mu in zip(_COMPONENT_NAMES, (self.mu_co2, self.mu_n2, self.mu_h2o)):
            exponent = float(np.clip(mu / RT, -500.0, 500.0))
            fu[name] = math.exp(exponent)
        return fu


@dataclass
class GCMCResult:
    """Output of one GCMC simulation run. Loadings in mol/kg."""
    mof_id:        str
    loading_co2:   float = float("nan")
    loading_n2:    float = float("nan")
    loading_h2o:   float = float("nan")
    loading_total: float = float("nan")
    std_co2:       float = float("nan")
    std_n2:        float = float("nan")
    std_h2o:       float = float("nan")
    n_cycles_run:  int   = 0
    wall_time_s:   float = 0.0
    success:       bool  = False
    error_msg:     str   = ""

    @property
    def loadings_array(self) -> np.ndarray:
        return np.array([self.loading_co2, self.loading_n2, self.loading_h2o],
                        dtype=np.float64)

    @property
    def stds_array(self) -> np.ndarray:
        return np.array([self.std_co2, self.std_n2, self.std_h2o], dtype=np.float64)

    def to_dict(self) -> Dict[str, float]:
        return {
            "loading_co2":   self.loading_co2,
            "loading_n2":    self.loading_n2,
            "loading_h2o":   self.loading_h2o,
            "loading_total": self.loading_total,
            "std_co2":       self.std_co2,
            "std_n2":        self.std_n2,
            "std_h2o":       self.std_h2o,
        }


# ═══════════════════════════════════════════════════════════════════════
# RASPA INPUT WRITER
# ═══════════════════════════════════════════════════════════════════════

class RASPAWriter:
    """Writes RASPA2 simulation.input for multicomponent GCMC."""

    def __init__(self, raspa_dir: Optional[Union[str, Path]] = None):
        if raspa_dir is None:
            raspa_dir = os.environ.get("RASPA_DIR", "/opt/RASPA2")
        self.raspa_dir = Path(raspa_dir)

    def write(self, gcmc_input: GCMCInput, work_dir: Path) -> Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        cif_src  = Path(gcmc_input.cif_path)
        cif_dest = work_dir / cif_src.name
        if not cif_dest.exists():
            shutil.copy2(cif_src, cif_dest)
        sim_path = work_dir / "simulation.input"
        sim_path.write_text(self._simulation_input(gcmc_input, cif_src.stem))
        return sim_path

    def _simulation_input(self, gi: GCMCInput, framework_name: str) -> str:
        fu    = gi.fugacities_pa
        uc    = gi.unit_cells
        n_prod = gi.n_cycles - gi.n_init_cycles
        lines = [
            "SimulationType                GCMC",
            f"NumberOfCycles                {n_prod}",
            f"NumberOfInitializationCycles  {gi.n_init_cycles}",
            "PrintEvery                    1000",
            "RestartFile                   no",
            "",
            f"Forcefield                    {gi.force_field}",
            "ChargeMethod                  Ewald",
            "CutOff                        12.0",
            "EwaldPrecision                1e-6",
            "UseChargesFromCIFFile         yes",
            "",
            "Framework                     0",
            f"FrameworkName                 {framework_name}",
            f"UnitCells                     {uc[0]} {uc[1]} {uc[2]}",
            "HeliumVoidFraction            0.29",
            f"ExternalTemperature           {gi.temperature:.2f}",
            "",
            f"NumberOfComponents            {len(_COMPONENT_NAMES)}",
            "",
        ]
        for idx, (name, fugacity) in enumerate(fu.items()):
            lines += self._component_block(idx, name, fugacity)
        return "\n".join(lines) + "\n"

    def _component_block(self, idx: int, name: str, fugacity_pa: float) -> List[str]:
        pressure_pa = max(fugacity_pa, 1e-30)
        return [
            f"Component {idx} MoleculeName                {name}",
            f"            MoleculeDefinition            TraPPE",
            f"            ExternalPressure              {pressure_pa:.6e}",
            f"            FugacityCoefficient           1.0",
            f"            TranslationProbability        0.5",
            f"            RotationProbability           0.5",
            f"            ReinsertionProbability        0.5",
            f"            SwapProbability               1.0",
            f"            CreateNumberOfMolecules       0",
            "",
        ]


# ═══════════════════════════════════════════════════════════════════════
# RASPA OUTPUT PARSER (class-based, for ChemPotGCMCRunner)
# ═══════════════════════════════════════════════════════════════════════

class RASPAParser:
    """
    Parse RASPA2 output files to extract per-component average loadings.
    Uses the Component-header regex so it handles both single-file and
    multi-file RASPA output layouts.
    """

    def parse(self, work_dir: Path, framework_name: str) -> Dict[str, Dict[str, float]]:
        parsed = parse_raspa_output(work_dir)
        result: Dict[str, Dict[str, float]] = {}
        for name in _COMPONENT_NAMES:
            loading = parsed["loadings"].get(name, float("nan"))
            result[name] = {"mean": loading, "std": float("nan")}
        return result


# ═══════════════════════════════════════════════════════════════════════
# CHEMICAL-POTENTIAL GCMC RUNNER (active-learning interface)
# Renamed from GCMCRunner to avoid collision with the re-exported one.
# ═══════════════════════════════════════════════════════════════════════

class ChemPotGCMCRunner:
    """
    Run GCMC simulations via RASPA2 using chemical-potential inputs.
    Used by the active-learning loop (``08_active_learning_loop.py``).
    """

    def __init__(
        self,
        raspa_exe:     Optional[Union[str, Path]] = None,
        raspa_dir:     Optional[Union[str, Path]] = None,
        n_retries:     int   = 3,
        retry_delay_s: float = 5.0,
        keep_work_dir: bool  = False,
        work_base_dir: Optional[Union[str, Path]] = None,
    ):
        self.raspa_exe     = self._resolve_exe(raspa_exe, raspa_dir)
        self.raspa_dir     = Path(raspa_dir or os.environ.get("RASPA_DIR", "/opt/RASPA2"))
        self.n_retries     = n_retries
        self.retry_delay_s = retry_delay_s
        self.keep_work_dir = keep_work_dir
        self.work_base_dir = Path(work_base_dir) if work_base_dir else None
        self._writer = RASPAWriter(self.raspa_dir)
        self._parser = RASPAParser()

    def run(self, gcmc_input: GCMCInput) -> GCMCResult:
        work_dir = Path(tempfile.mkdtemp(
            prefix=f"gcmc_{gcmc_input.mof_id}_",
            dir=self.work_base_dir,
        ))
        try:
            return self._run_with_retries(gcmc_input, work_dir)
        finally:
            if not self.keep_work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)

    def run_batch(self, inputs: List[GCMCInput], n_workers: int = 1) -> List[GCMCResult]:
        if n_workers <= 1:
            return [self.run(gi) for gi in inputs]
        from concurrent.futures import ProcessPoolExecutor, as_completed
        results: List[Optional[GCMCResult]] = [None] * len(inputs)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(self.run, gi): i for i, gi in enumerate(inputs)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    logger.error("Job %d failed: %s", idx, exc)
                    results[idx] = GCMCResult(
                        mof_id=inputs[idx].mof_id, success=False, error_msg=str(exc)
                    )
        return results  # type: ignore[return-value]

    def __call__(
        self,
        x: np.ndarray,
        cif_path: Union[str, Path],
        mof_id: str = "unknown",
        temperature: float = 313.0,
        **kwargs,
    ) -> Tuple[np.ndarray, float]:
        if len(x) >= 4:
            temperature = float(x[3])
        gi = GCMCInput(
            cif_path=cif_path,
            mu_co2=float(x[0]),
            mu_n2=float(x[1]),
            mu_h2o=float(x[2]),
            temperature=temperature,
            mof_id=mof_id,
            **kwargs,
        )
        result = self.run(gi)
        return result.loadings_array, result.wall_time_s

    def _run_with_retries(self, gcmc_input: GCMCInput, work_dir: Path) -> GCMCResult:
        framework_name = Path(gcmc_input.cif_path).stem
        self._writer.write(gcmc_input, work_dir)
        last_error = ""
        delay = self.retry_delay_s
        for attempt in range(self.n_retries + 1):
            if attempt > 0:
                logger.warning("GCMC retry %d/%d: %s", attempt, self.n_retries, last_error)
                time.sleep(delay)
                delay *= 2.0
            t0 = time.perf_counter()
            try:
                ok, last_error = self._execute_raspa(work_dir)
            except Exception as exc:
                last_error = str(exc)
                ok = False
            wall_time = time.perf_counter() - t0
            if ok:
                return self._parse_result(gcmc_input, work_dir, framework_name, wall_time)
            if self._is_fatal_error(last_error):
                break
        return GCMCResult(
            mof_id=gcmc_input.mof_id,
            success=False,
            error_msg=f"Failed after {self.n_retries + 1} attempt(s): {last_error}",
        )

    def _execute_raspa(self, work_dir: Path) -> Tuple[bool, str]:
        env = os.environ.copy()
        env["RASPA_DIR"] = str(self.raspa_dir)
        env.setdefault("DYLD_LIBRARY_PATH", str(self.raspa_dir / "lib"))
        env.setdefault("LD_LIBRARY_PATH",   str(self.raspa_dir / "lib"))
        try:
            proc = subprocess.run(
                [str(self.raspa_exe)],
                cwd=str(work_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=3600,
            )
        except subprocess.TimeoutExpired:
            return False, "RASPA timed out after 3600s"
        except FileNotFoundError:
            return False, f"RASPA executable not found: {self.raspa_exe}"
        if proc.returncode != 0:
            return False, f"RASPA exit code {proc.returncode}: {proc.stderr[-500:]}"
        if not (work_dir / "Output" / "System_0").is_dir():
            return False, "RASPA exited 0 but Output/System_0 not created"
        return True, ""

    def _parse_result(
        self, gcmc_input: GCMCInput, work_dir: Path,
        framework_name: str, wall_time: float,
    ) -> GCMCResult:
        parsed = self._parser.parse(work_dir, framework_name)
        co2 = parsed.get("CO2", {})
        n2  = parsed.get("N2",  {})
        h2o = parsed.get("H2O", {})
        q_co2 = co2.get("mean", float("nan"))
        q_n2  = n2.get("mean",  float("nan"))
        q_h2o = h2o.get("mean", float("nan"))
        finite = [q for q in (q_co2, q_n2, q_h2o) if math.isfinite(q)]
        q_total = sum(finite) if finite else float("nan")
        success = any(math.isfinite(q) for q in (q_co2, q_n2, q_h2o))
        return GCMCResult(
            mof_id=gcmc_input.mof_id,
            loading_co2=q_co2, loading_n2=q_n2, loading_h2o=q_h2o,
            loading_total=q_total,
            std_co2=co2.get("std", float("nan")),
            std_n2=n2.get("std",  float("nan")),
            std_h2o=h2o.get("std", float("nan")),
            n_cycles_run=gcmc_input.n_cycles,
            wall_time_s=wall_time,
            success=success,
            error_msg="" if success else "All component loadings are NaN",
        )

    @staticmethod
    def _resolve_exe(raspa_exe, raspa_dir) -> Path:
        if raspa_exe is not None:
            return Path(raspa_exe)
        rdir = raspa_dir or os.environ.get("RASPA_DIR", "")
        if rdir:
            candidate = Path(rdir) / "bin" / "simulate"
            if candidate.is_file():
                return candidate
        found = shutil.which("simulate") or shutil.which("raspa")
        if found:
            return Path(found)
        return Path("simulate")

    @staticmethod
    def _is_fatal_error(msg: str) -> bool:
        fatal = ("cannot open file", "cif file not found", "force field not found",
                 "unknown molecule", "no such file", "segmentation fault")
        msg_lower = msg.lower()
        return any(kw in msg_lower for kw in fatal)


# ═══════════════════════════════════════════════════════════════════════
# UNIT-CELL HELPER
# ═══════════════════════════════════════════════════════════════════════

def suggest_unit_cells(
    cif_path: Union[str, Path],
    cutoff_ang: float = 12.0,
) -> Tuple[int, int, int]:
    """Suggest supercell replication so every dimension ≥ 2 × cutoff."""
    try:
        text = Path(cif_path).read_text(errors="replace")
        a = _parse_cif_length(text, "_cell_length_a")
        b = _parse_cif_length(text, "_cell_length_b")
        c = _parse_cif_length(text, "_cell_length_c")
        return (
            max(1, math.ceil(2 * cutoff_ang / a)),
            max(1, math.ceil(2 * cutoff_ang / b)),
            max(1, math.ceil(2 * cutoff_ang / c)),
        )
    except Exception as exc:
        logger.debug("Could not determine unit cells from CIF: %s", exc)
        return (1, 1, 1)


def _parse_cif_length(text: str, tag: str) -> float:
    m = re.compile(rf"^\s*{re.escape(tag)}\s+([\d.]+)", re.MULTILINE | re.IGNORECASE).search(text)
    if not m:
        raise ValueError(f"Tag {tag} not found in CIF")
    return float(m.group(1))


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    # Re-exported from parser.py (for test imports)
    "GCMCConfig",
    "generate_input",
    "_estimate_unit_cells",
    "GCMCRunner",
    # Defined here
    "parse_raspa_output",
    "results_to_arrays",
    "GCMCInput",
    "GCMCResult",
    "RASPAWriter",
    "RASPAParser",
    "ChemPotGCMCRunner",
    "suggest_unit_cells",
]