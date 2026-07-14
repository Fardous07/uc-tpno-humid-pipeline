"""
Tests for src/simulation/ (GCMC runner/parser, GC-TMMC runner/parser).

These test configuration, input generation, and output parsing
without requiring RASPA2 to be installed.

Fixes vs previous version
──────────────────────────
1. fake_raspa_output fixture wrote a made-up format that no realistic
   parser regex would match.  Fixed: now writes authentic RASPA output
   with "Component N [X]" headers and "Average loading absolute [mol/kg
   framework]" lines that match the actual parser regexes.

2. _estimate_unit_cells is now explicitly exported from runner.py so
   the direct import in TestGCMCInput works without accessing private API.

3. All four src.simulation.{gcmc,gctmmc}.{runner,parser} modules were
   missing entirely — every import would have raised ModuleNotFoundError.
   The companion implementation files (gcmc_runner.py, gcmc_parser.py,
   gctmmc_runner.py, gctmmc_parser.py) in this outputs directory provide
   those modules.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════
# GCMC CONFIG + INPUT GENERATION
# ═══════════════════════════════════════════════════════════

class TestGCMCConfig:
    def test_defaults(self):
        from src.simulation.gcmc.runner import GCMCConfig
        cfg = GCMCConfig()
        assert cfg.forcefield in ("UFF", "DREIDING")
        assert cfg.cutoff_vdw > 0
        assert cfg.n_cycles > 0

    def test_custom(self):
        from src.simulation.gcmc.runner import GCMCConfig
        cfg = GCMCConfig(forcefield="DREIDING", n_cycles=50000, timeout=3600)
        assert cfg.forcefield == "DREIDING"
        assert cfg.n_cycles == 50000


class TestGCMCInput:
    def test_dry_input(self):
        from src.simulation.gcmc.runner import generate_input, GCMCConfig
        cfg = GCMCConfig()
        text = generate_input(
            mof_name="HKUST-1",
            temperature=313.15,
            pressure=1.0,
            composition={"CO2": 0.15, "N2": 0.85},
            config=cfg,
        )
        assert "HKUST-1" in text
        assert "CO2" in text or "CO_2" in text
        assert "H2O" not in text  # dry — no water component

    def test_humid_input(self):
        from src.simulation.gcmc.runner import generate_input, GCMCConfig
        cfg = GCMCConfig()
        text = generate_input(
            mof_name="HKUST-1",
            temperature=313.15,
            pressure=1.0,
            composition={"CO2": 0.15, "N2": 0.75, "H2O": 0.10},
            config=cfg,
        )
        assert "H2O" in text  # humid — includes water

    def test_unit_cell_estimation(self):
        # FIX: _estimate_unit_cells must be exported from runner.py,
        # not just a private function.
        from src.simulation.gcmc.runner import estimate_unit_cells
        # Large cell — one repetition is sufficient (2*12=24 < 25)
        uc = estimate_unit_cells(25.0, 25.0, 25.0, cutoff=12.0)
        assert uc == (1, 1, 1)
        # Small cell — must replicate (ceil(24/8)=3)
        uc2 = estimate_unit_cells(8.0, 8.0, 8.0, cutoff=12.0)
        assert all(u >= 2 for u in uc2)


class TestGCMCRunner:
    def test_graceful_no_raspa(self, tmp_path):
        from src.simulation.gcmc.runner import GCMCRunner, GCMCConfig
        cfg = GCMCConfig(raspa_path="nonexistent_binary", work_dir=str(tmp_path))
        runner = GCMCRunner(cfg)
        assert not runner.check_raspa()

    def test_run_single_no_raspa(self, tmp_path):
        from src.simulation.gcmc.runner import GCMCRunner, GCMCConfig
        cfg = GCMCConfig(raspa_path="nonexistent", work_dir=str(tmp_path))
        runner = GCMCRunner(cfg)
        cif = tmp_path / "TEST.cif"
        cif.write_text(
            "data_TEST\n"
            "_cell_length_a 25\n_cell_length_b 25\n_cell_length_c 25\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        )
        result = runner.run_single(
            cif, temperature=313.15, pressure=1.0,
            composition={"CO2": 0.15, "N2": 0.85},
        )
        assert result["success"] is False
        assert "error" in result


# ═══════════════════════════════════════════════════════════
# GCMC PARSER
# ═══════════════════════════════════════════════════════════

class TestGCMCParser:
    @pytest.fixture
    def fake_raspa_output(self, tmp_path):
        """
        Create a minimal RASPA output directory.

        FIX: previous fixture wrote a made-up tabular format that no
        realistic parser regex would match.  Now writes authentic RASPA
        output with Component headers and 'Average loading absolute'
        lines that match the parser's _RE_MOL_KG / _RE_MOLEC_UC patterns.
        """
        out_dir = tmp_path / "Output" / "System_0"
        out_dir.mkdir(parents=True)

        # Authentic RASPA output format
        data = (
            "Component 0 [CO2]    (total 200000 trial moves)\n"
            "\n"
            "  Average loading absolute [molecules/unit cell] 12.340000 +/- 0.500000\n"
            "  Average loading absolute [mol/kg framework] 3.456000 +/- 0.123000\n"
            "\n"
            "  Average Host-Adsorbate energy [K]:   -12345.600000 +/- 100.000000\n"
            "  Enthalpy of adsorption [kJ/mol]:         -32.500000 +/- 1.000000\n"
            "\n"
            "Component 1 [N2]    (total 200000 trial moves)\n"
            "\n"
            "  Average loading absolute [molecules/unit cell] 2.810000 +/- 0.200000\n"
            "  Average loading absolute [mol/kg framework] 0.789000 +/- 0.045000\n"
            "\n"
            "  Average Host-Adsorbate energy [K]:    -2345.600000 +/- 50.000000\n"
            "  Enthalpy of adsorption [kJ/mol]:         -15.200000 +/- 0.500000\n"
        )
        (out_dir / "output.data").write_text(data)
        return tmp_path

    def test_parse_loadings(self, fake_raspa_output):
        from src.simulation.gcmc.parser import parse_raspa_output
        result = parse_raspa_output(fake_raspa_output)
        assert "CO2" in result["loadings"]
        assert result["loadings"]["CO2"] == pytest.approx(3.456, rel=0.01)
        assert result["loadings"]["N2"]  == pytest.approx(0.789, rel=0.01)

    def test_parse_converged(self, fake_raspa_output):
        from src.simulation.gcmc.parser import parse_raspa_output
        result = parse_raspa_output(fake_raspa_output)
        assert result["converged"] is True

    def test_results_to_arrays(self):
        from src.simulation.gcmc.parser import results_to_arrays
        results = [
            {"loadings": {"CO2": 1.0, "N2": 0.5, "H2O": 0.1}},
            {"loadings": {"CO2": 2.0, "N2": 0.3, "H2O": 0.2}},
        ]
        arr = results_to_arrays(results, species=["CO2", "N2", "H2O"])
        assert arr.shape == (2, 3)
        assert arr[0, 0] == pytest.approx(1.0)
        assert arr[1, 1] == pytest.approx(0.3)


# ═══════════════════════════════════════════════════════════
# GC-TMMC CONFIG + INPUT
# ═══════════════════════════════════════════════════════════

class TestGCTMMCConfig:
    def test_defaults(self):
        from src.simulation.gctmmc.runner import GCTMMCConfig
        cfg = GCTMMCConfig()
        assert cfg.N_max > 0
        assert cfg.n_cycles > 0

    def test_custom(self):
        from src.simulation.gctmmc.runner import GCTMMCConfig
        cfg = GCTMMCConfig(N_max=50, temperature=300.0)
        assert cfg.N_max == 50
        assert cfg.temperature == pytest.approx(300.0)


class TestGCTMMCInput:
    def test_tmmc_input_generation(self):
        from src.simulation.gctmmc.runner import generate_tmmc_input, GCTMMCConfig
        cfg = GCTMMCConfig(N_max=100)
        text = generate_tmmc_input(
            mof_name="HKUST-1",
            molecule="CO2",
            temperature=313.15,
            config=cfg,
        )
        assert "TMC" in text or "TMMC" in text or "Transition" in text
        assert "HKUST-1" in text


class TestGCTMMCRunner:
    def test_graceful_no_raspa(self, tmp_path):
        from src.simulation.gctmmc.runner import GCTMMCRunner, GCTMMCConfig
        cfg = GCTMMCConfig(raspa_path="nonexistent", work_dir=str(tmp_path))
        runner = GCTMMCRunner(cfg)
        cif = tmp_path / "TEST.cif"
        cif.write_text("data_T\n")
        result = runner.run_single(cif, "CO2", 313.15)
        assert result["success"] is False
        assert "error" in result


# ═══════════════════════════════════════════════════════════
# GC-TMMC PARSER
# ═══════════════════════════════════════════════════════════

class TestGCTMMCParser:
    def test_synthetic_collection_matrix(self):
        from src.simulation.gctmmc.parser import synthetic_collection_matrix
        C = synthetic_collection_matrix(N_max=80)
        assert C.shape == (81, 3)
        assert (C >= 0).all()

    def test_collection_to_ln_pi(self):
        from src.simulation.gctmmc.parser import (
            synthetic_collection_matrix,
            collection_matrix_to_ln_pi,
        )
        C   = synthetic_collection_matrix(N_max=50)
        ln_pi = collection_matrix_to_ln_pi(C)
        assert len(ln_pi) == 51
        assert ln_pi[0] == pytest.approx(0.0, abs=1e-12)  # normalised reference

    def test_ln_pi_to_isotherm_monotonic(self):
        from src.simulation.gctmmc.parser import (
            synthetic_collection_matrix,
            collection_matrix_to_ln_pi,
            ln_pi_to_isotherm,
        )
        C     = synthetic_collection_matrix(N_max=80)
        ln_pi = collection_matrix_to_ln_pi(C)
        pressures = np.logspace(-2, 1, 30)
        result    = ln_pi_to_isotherm(ln_pi, T=313.15, pressures=pressures)
        loadings  = result["loadings"]
        # Loadings should be monotonically non-decreasing with pressure
        assert all(
            loadings[i] <= loadings[i + 1] + 1e-8
            for i in range(len(loadings) - 1)
        )

    def test_parse_tmmc_output(self, tmp_path):
        from src.simulation.gctmmc.parser import (
            synthetic_collection_matrix,
            parse_tmmc_output,
        )
        out_dir = tmp_path / "Output" / "System_0"
        out_dir.mkdir(parents=True)

        C = synthetic_collection_matrix(N_max=50)
        lines = [
            f"{i}  {row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}"
            for i, row in enumerate(C)
        ]
        (out_dir / "CollectionMatrix.dat").write_text("\n".join(lines))

        result = parse_tmmc_output(tmp_path, T=313.15)
        assert result["success"] is True
        assert len(result["pressures"]) > 0
        assert len(result["loadings"]) == len(result["pressures"])