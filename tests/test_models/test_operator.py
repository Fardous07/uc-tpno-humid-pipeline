"""
Tests for src/models/operator/ (TPNO + losses).

Covers: TPNOConfig, ThermodynamicPotentialNO, ICNN,
        ThermodynamicLoss, LossConfig, ThermodynamicValidator.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


class TestTPNOConfig:
    def test_defaults(self):
        from src.models.operator.tpno import TPNOConfig
        cfg = TPNOConfig()
        assert cfg.emb_dim > 0
        assert cfg.n_conditions > 0
        assert cfg.n_components > 0

    def test_custom(self):
        from src.models.operator.tpno import TPNOConfig
        cfg = TPNOConfig(emb_dim=64, hidden_dim=128, n_layers=3)
        assert cfg.emb_dim == 64
        assert cfg.hidden_dim == 128


class TestICNN:
    def test_forward_shape(self):
        from src.models.operator.tpno import ICNN
        icnn = ICNN(in_dim=4, hidden_dim=32, out_dim=1, n_layers=3)
        x = torch.randn(8, 4)
        y = icnn(x)
        assert y.shape == (8, 1)

    def test_convexity(self):
        """ICNN output should be convex in input (non-negative weights)."""
        from src.models.operator.tpno import ICNN
        icnn = ICNN(in_dim=2, hidden_dim=16, out_dim=1, n_layers=3)
        # Check along a line: f(tx + (1-t)y) <= t*f(x) + (1-t)*f(y)
        x = torch.randn(1, 2)
        y = torch.randn(1, 2)
        t_vals = torch.linspace(0, 1, 20)
        violations = 0
        with torch.no_grad():
            fx = icnn(x).item()
            fy = icnn(y).item()
            for t in t_vals:
                z = t * x + (1 - t) * y
                fz = icnn(z).item()
                bound = t.item() * fx + (1 - t.item()) * fy
                if fz > bound + 1e-5:
                    violations += 1
        assert violations == 0, f"ICNN convexity violated {violations} times"


class TestThermodynamicPotentialNO:
    def test_instantiation(self):
        from src.models.operator.tpno import ThermodynamicPotentialNO, TPNOConfig
        encoder = nn.Linear(32, 64)  # dummy encoder
        cfg = TPNOConfig(emb_dim=64, n_conditions=4, n_components=3)
        model = ThermodynamicPotentialNO(encoder, cfg)
        assert model is not None

    def test_parameter_count(self):
        from src.models.operator.tpno import ThermodynamicPotentialNO, TPNOConfig
        encoder = nn.Linear(32, 64)
        cfg = TPNOConfig(emb_dim=64, hidden_dim=64, n_layers=2)
        model = ThermodynamicPotentialNO(encoder, cfg)
        n = sum(p.numel() for p in model.parameters())
        assert n > 0


class TestThermodynamicLoss:
    def test_instantiation(self):
        from src.models.operator.losses import ThermodynamicLoss, LossConfig
        cfg = LossConfig()
        loss_fn = ThermodynamicLoss(cfg)
        assert loss_fn is not None


class TestThermodynamicValidator:
    def test_instantiation(self):
        from src.models.operator.losses import ThermodynamicValidator
        tv = ThermodynamicValidator()
        assert tv is not None