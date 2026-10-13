"""
Tests for src/models/operator/ (TPNO + losses).

Covers: TPNOConfig, ThermodynamicPotentialNO, ICNN,
        ThermodynamicLoss, LossConfig, ThermodynamicValidator.

The physics guarantees (Maxwell symmetry, PSD Jacobian, monotonicity,
convexity of Phi) are tested in depth in test_tpno_physics.py.

Run with pytest, or directly:  python tests/test_models/test_operator.py
"""
import sys
from pathlib import Path

import torch

nn = torch.nn
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class TestTPNOConfig:
    def test_defaults(self):
        from src.models.operator.tpno import TPNOConfig
        cfg = TPNOConfig()
        assert cfg.emb_dim > 0
        assert cfg.n_conditions > cfg.n_components > 0
        # defaults must give a convex potential network
        assert cfg.activation == "softplus"
        assert cfg.use_layer_norm is False
        assert cfg.film_conditioning is True

    def test_custom(self):
        from src.models.operator.tpno import TPNOConfig
        cfg = TPNOConfig(emb_dim=64, hidden_dim=128, n_layers=3)
        assert cfg.emb_dim == 64
        assert cfg.hidden_dim == 128


class TestICNN:
    def test_forward_shape_and_positive(self):
        from src.models.operator.tpno import ICNN
        icnn = ICNN(input_dim=3, hidden_dim=32, n_layers=3, dropout=0.0)
        y = icnn(torch.randn(8, 3))
        assert y.shape == (8, 1)
        assert (y > 0).all(), "Phi = pV must be positive"

    def test_convexity_random_weights(self):
        """Jensen's inequality for many random weights, points and directions."""
        from src.models.operator.tpno import ICNN
        for seed in range(5):
            torch.manual_seed(seed)
            icnn = ICNN(input_dim=3, hidden_dim=16, n_layers=3, dropout=0.0)
            with torch.no_grad():
                for p in icnn.parameters():
                    p.add_(torch.randn_like(p) * 0.5)
            x, y = torch.randn(200, 3), torch.randn(200, 3)
            t = torch.rand(200, 1)
            with torch.no_grad():
                gap = icnn(t * x + (1 - t) * y) - (t * icnn(x) + (1 - t) * icnn(y))
            assert gap.max().item() < 1e-5, f"ICNN convexity violated (seed {seed}): {gap.max().item():.2e}"

    def test_rejects_non_convex_settings(self):
        from src.models.operator.tpno import ICNN
        for kwargs in ({"activation": "swish"}, {"use_layer_norm": True}):
            try:
                ICNN(input_dim=3, hidden_dim=8, **kwargs)
            except ValueError:
                continue
            raise AssertionError(f"ICNN accepted {kwargs}")


class TestThermodynamicPotentialNO:
    def test_instantiation_and_forward(self):
        from src.models.operator.tpno import ThermodynamicPotentialNO, TPNOConfig
        encoder = nn.Embedding(2, 64)          # stand-in encoder: graph index -> embedding
        cfg = TPNOConfig(emb_dim=64, n_conditions=4, n_components=3, dropout=0.0)
        model = ThermodynamicPotentialNO(encoder, cfg).eval()
        cond = torch.randn(2, 5, 4)
        out = model(torch.arange(2), cond, return_potential=True)
        assert out["q_pred"].shape == (2, 5, 3)
        assert out["sigma"].shape == (2, 5, 3)
        assert torch.allclose(out["omega"], -out["phi"])

    def test_parameter_count(self):
        from src.models.operator.tpno import ThermodynamicPotentialNO, TPNOConfig
        model = ThermodynamicPotentialNO(nn.Embedding(2, 64), TPNOConfig(emb_dim=64, hidden_dim=64, n_layers=2))
        assert sum(p.numel() for p in model.parameters()) > 0


class TestThermodynamicLoss:
    def test_instantiation(self):
        from src.models.operator.losses import ThermodynamicLoss, LossConfig
        assert ThermodynamicLoss(LossConfig()) is not None


class TestThermodynamicValidator:
    def test_instantiation(self):
        # ThermodynamicValidator lives in tpno.py (not losses.py)
        from src.models.operator.tpno import ThermodynamicValidator
        assert ThermodynamicValidator() is not None


if __name__ == "__main__":
    failed = total = 0
    for cls_name, cls in sorted(globals().items()):
        if not (cls_name.startswith("Test") and isinstance(cls, type)):
            continue
        for name in sorted(dir(cls)):
            if not name.startswith("test_"):
                continue
            total += 1
            try:
                getattr(cls(), name)()
                print(f"PASS  {cls_name}.{name}")
            except Exception as e:
                failed += 1
                print(f"FAIL  {cls_name}.{name}: {e}")
    print(f"\n{total - failed}/{total} passed")
    sys.exit(1 if failed else 0)
