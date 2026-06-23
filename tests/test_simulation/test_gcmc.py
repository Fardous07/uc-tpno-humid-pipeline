"""
Tests for src/simulation/ (GCMC runner/parser, GC-TMMC runner/parser).

These test configuration, input generation, and output parsing
without requiring RASPA2 to be installed.
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
        assert "H2O" not in text  # dry — no water

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
        from src.simulation.gcmc.runner import _estimate_unit_cells
        # Large cell — should be (1,1,1)
        uc = _estimate_unit_cells(25.0, 25.0, 25.0, cutoff=12.0)
        assert uc == (1, 1, 1)
        # Small cell — should replicate
        uc2 = _estimate_unit_cells(8.0, 8.0, 8.0, cutoff=12.0)
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
        # Create a dummy CIF
        cif = tmp_path / "TEST.cif"
        cif.write_text("data_TEST\n_cell_length_a 25\n_cell_length_b 25\n"
                       "_cell_length_c 25\n_cell_angle_alpha 90\n"
                       "_cell_angle_beta 90\n_cell_angle_gamma 90\n")
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
        """Create a minimal RASPA output directory."""
        out_dir = tmp_path / "Output" / "System_0"
        out_dir.mkdir(parents=True)
        data = """\
Average loading absolute [mol/kg framework]
    Component 0 [CO2]   3.456 +/- 0.123
    Component 1 [N2]    0.789 +/- 0.045

Average loading absolute [molecules/unit cell]
    Component 0 [CO2]   12.34 +/- 0.5
    Component 1 [N2]    2.81 +/- 0.2

Host/Adsorbate energy:
    Component 0 [CO2]  -12345.6 +/- 100.0
    Component 1 [N2]   -2345.6 +/- 50.0

Enthalpy of adsorption:
    Component 0 [CO2]  -32.5 +/- 1.0
    Component 1 [N2]   -15.2 +/- 0.5
"""
        (out_dir / "output.data").write_text(data)
        return tmp_path

    def test_parse_loadings(self, fake_raspa_output):
        from src.simulation.gcmc.parser import parse_raspa_output
        result = parse_raspa_output(fake_raspa_output)
        assert "CO2" in result["loadings"]
        assert result["loadings"]["CO2"] == pytest.approx(3.456, rel=0.01)
        assert result["loadings"]["N2"] == pytest.approx(0.789, rel=0.01)

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
            synthetic_collection_matrix, collection_matrix_to_ln_pi,
        )
        C = synthetic_collection_matrix(N_max=50)
        ln_pi = collection_matrix_to_ln_pi(C)
        assert len(ln_pi) == 51
        assert ln_pi[0] == 0.0  # normalised

    def test_ln_pi_to_isotherm_monotonic(self):
        from src.simulation.gctmmc.parser import (
            synthetic_collection_matrix,
            collection_matrix_to_ln_pi,
            ln_pi_to_isotherm,
        )
        C = synthetic_collection_matrix(N_max=80)
        ln_pi = collection_matrix_to_ln_pi(C)
        pressures = np.logspace(-2, 1, 30)
        result = ln_pi_to_isotherm(ln_pi, T=313.15, pressures=pressures)
        loadings = result["loadings"]
        # Loadings should be monotonically non-decreasing
        assert all(loadings[i] <= loadings[i + 1] + 1e-8
                    for i in range(len(loadings) - 1))

    def test_parse_tmmc_output(self, tmp_path):
        from src.simulation.gctmmc.parser import (
            synthetic_collection_matrix, parse_tmmc_output,
        )
        # Create fake output directory
        out_dir = tmp_path / "Output" / "System_0"
        out_dir.mkdir(parents=True)
        C = synthetic_collection_matrix(N_max=50)
        lines = []
        for i, row in enumerate(C):
            lines.append(f"{i}  {row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}")
        (out_dir / "CollectionMatrix.dat").write_text("\n".join(lines))

        result = parse_tmmc_output(tmp_path, T=313.15)
        assert result["success"]
        assert len(result["pressures"]) > 0
        assert len(result["loadings"]) == len(result["pressures"])