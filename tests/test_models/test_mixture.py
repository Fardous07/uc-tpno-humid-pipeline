"""
Tests for src/models/mixture/ (IAST, NeuralMixture, SPLINT).

IAST tests are pure-Python; NeuralMixture and SPLINT require torch.
"""

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════
# IAST (pure Python / numpy)
# ═══════════════════════════════════════════════════════════

class TestLangmuir:
    def test_langmuir_basic(self):
        from src.models.mixture.iast import Langmuir
        iso = Langmuir(q_sat=5.0, K=1.0)
        # At P=0, q=0
        assert iso.loading(0.0) == pytest.approx(0.0, abs=1e-10)
        # At P→∞, q→q_sat
        assert iso.loading(1e6) == pytest.approx(5.0, rel=0.01)
        # At P=1, q=5*1/(1+1)=2.5
        assert iso.loading(1.0) == pytest.approx(2.5, rel=0.01)

    def test_langmuir_spreading_pressure(self):
        from src.models.mixture.iast import Langmuir
        iso = Langmuir(q_sat=5.0, K=1.0)
        sp = iso.spreading_pressure(1.0)
        # ∫₀¹ q/P dP = q_sat*K*ln(1+K*P)/K = 5*ln(2) ≈ 3.466
        assert sp == pytest.approx(5.0 * np.log(2), rel=0.05)


class TestDualSiteLangmuir:
    def test_two_sites(self):
        from src.models.mixture.iast import DualSiteLangmuir
        iso = DualSiteLangmuir(q_sat1=3.0, K1=2.0, q_sat2=2.0, K2=0.1)
        q = iso.loading(1.0)
        expected = 3.0 * 2.0 / (1 + 2.0) + 2.0 * 0.1 / (1 + 0.1)
        assert q == pytest.approx(expected, rel=0.01)


class TestFreundlich:
    def test_freundlich(self):
        from src.models.mixture.iast import Freundlich
        iso = Freundlich(K=2.0, n=2.0)
        q = iso.loading(4.0)
        assert q == pytest.approx(2.0 * 4.0 ** (1 / 2.0), rel=0.01)


class TestIASTCalculator:
    def test_binary_mixture(self):
        from src.models.mixture.iast import IASTCalculator, Langmuir
        iso_co2 = Langmuir(q_sat=5.0, K=2.0)
        iso_n2 = Langmuir(q_sat=4.0, K=0.3)
        calc = IASTCalculator()
        result = calc.solve(
            isotherms=[iso_co2, iso_n2],
            y=[0.15, 0.85],
            P_total=1.0,
        )
        assert result is not None
        assert len(result["loadings"]) == 2
        assert all(q >= 0 for q in result["loadings"])
        # CO2 should have higher loading due to higher K
        assert result["selectivity"] > 1.0

    def test_pure_component_limit(self):
        from src.models.mixture.iast import IASTCalculator, Langmuir
        iso = Langmuir(q_sat=5.0, K=1.0)
        calc = IASTCalculator()
        result = calc.solve(
            isotherms=[iso],
            y=[1.0],
            P_total=1.0,
        )
        assert result["loadings"][0] == pytest.approx(
            iso.loading(1.0), rel=0.05
        )


# ═══════════════════════════════════════════════════════════
# SPLINT
# ═══════════════════════════════════════════════════════════

class TestSPLINTConfig:
    def test_defaults(self):
        from src.models.mixture.splint import SPLINTConfig
        cfg = SPLINTConfig()
        assert cfg is not None


class TestMargulesSPLINT:
    def test_activity_coefficients(self):
        from src.models.mixture.splint import MargulesSPLINT
        model = MargulesSPLINT(n_components=2)
        # For x = [0.5, 0.5], activity coefficients should be symmetric
        x = np.array([0.5, 0.5])
        gamma = model.activity_coefficients(x)
        assert len(gamma) == 2
        assert gamma[0] == pytest.approx(gamma[1], rel=0.01)


# ═══════════════════════════════════════════════════════════
# NEURAL MIXTURE (requires torch)
# ═══════════════════════════════════════════════════════════

class TestNeuralMixtureModel:
    def test_instantiation(self):
        torch = pytest.importorskip("torch")
        from src.models.mixture.neural_mixture import NeuralMixtureModel, NeuralMixtureConfig
        cfg = NeuralMixtureConfig()
        model = NeuralMixtureModel(cfg)
        assert model is not None
        assert sum(p.numel() for p in model.parameters()) > 0