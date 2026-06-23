"""
Tests for src/models/uq/ (conformal, SWAG).

Conformal tests are pure numpy; SWAG requires torch.
"""

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════
# CONFORMAL (pure numpy)
# ═══════════════════════════════════════════════════════════

class TestConformalConfig:
    def test_defaults(self):
        from src.models.uq.conformal import ConformalConfig
        cfg = ConformalConfig()
        assert 0 < cfg.alpha < 1
        assert cfg.method in ("split", "weighted", "mondrian", "cv")

    def test_custom(self):
        from src.models.uq.conformal import ConformalConfig
        cfg = ConformalConfig(alpha=0.05, method="weighted")
        assert cfg.alpha == 0.05
        assert cfg.method == "weighted"


class TestConformalCalibrator:
    def _make_calibration_data(self, n=500, seed=42):
        rng = np.random.RandomState(seed)
        y_true = rng.randn(n)
        y_pred = y_true + rng.randn(n) * 0.3
        y_std = np.abs(rng.randn(n)) * 0.3 + 0.1
        return {"y_true": y_true, "y_pred": y_pred, "y_std": y_std}

    def test_calibrate_split(self):
        from src.models.uq.conformal import ConformalCalibrator, ConformalConfig
        cfg = ConformalConfig(alpha=0.1, method="split")
        cal = ConformalCalibrator(cfg)
        data = self._make_calibration_data()
        cal.calibrate(data)
        assert cal.is_fitted

    def test_predict_intervals(self):
        from src.models.uq.conformal import ConformalCalibrator, ConformalConfig
        cfg = ConformalConfig(alpha=0.1, method="split")
        cal = ConformalCalibrator(cfg)
        cal.calibrate(self._make_calibration_data(n=300))

        rng = np.random.RandomState(99)
        test_data = {
            "y_pred": rng.randn(100),
            "y_std": np.abs(rng.randn(100)) * 0.3 + 0.1,
        }
        intervals = cal.predict_intervals(test_data)
        assert "lower" in intervals
        assert "upper" in intervals
        assert (intervals["upper"] >= intervals["lower"]).all()

    def test_coverage_90pct(self):
        from src.models.uq.conformal import ConformalCalibrator, ConformalConfig
        cfg = ConformalConfig(alpha=0.1, method="split")
        cal = ConformalCalibrator(cfg)

        rng = np.random.RandomState(42)
        n_cal, n_test = 500, 500
        y_true_all = rng.randn(n_cal + n_test)
        y_pred_all = y_true_all + rng.randn(n_cal + n_test) * 0.3
        y_std_all = np.ones(n_cal + n_test) * 0.4

        cal.calibrate({
            "y_true": y_true_all[:n_cal],
            "y_pred": y_pred_all[:n_cal],
            "y_std": y_std_all[:n_cal],
        })
        intervals = cal.predict_intervals({
            "y_pred": y_pred_all[n_cal:],
            "y_std": y_std_all[n_cal:],
        })
        y_test = y_true_all[n_cal:]
        covered = (y_test >= intervals["lower"]) & (y_test <= intervals["upper"])
        coverage = covered.mean()
        # Should be close to 90% (alpha=0.1)
        assert coverage >= 0.80, f"Coverage {coverage:.2%} too low"


class TestEvaluateCoverage:
    def test_perfect_coverage(self):
        from src.models.uq.conformal import evaluate_coverage
        y_true = np.array([1.0, 2.0, 3.0])
        intervals = {
            "lower": np.array([0.0, 1.0, 2.0]),
            "upper": np.array([2.0, 3.0, 4.0]),
        }
        result = evaluate_coverage(intervals, y_true)
        assert result["overall_coverage"] == pytest.approx(1.0)

    def test_zero_coverage(self):
        from src.models.uq.conformal import evaluate_coverage
        y_true = np.array([10.0, 20.0, 30.0])
        intervals = {
            "lower": np.array([0.0, 0.0, 0.0]),
            "upper": np.array([1.0, 1.0, 1.0]),
        }
        result = evaluate_coverage(intervals, y_true)
        assert result["overall_coverage"] == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════
# SWAG (requires torch)
# ═══════════════════════════════════════════════════════════

class TestSWAGConfig:
    def test_defaults(self):
        pytest.importorskip("torch")
        from src.models.uq.swag import SWAGConfig
        cfg = SWAGConfig()
        assert cfg.rank > 0
        assert cfg.lr > 0


class TestSWAGWrapper:
    def test_instantiation(self):
        torch = pytest.importorskip("torch")
        from src.models.uq.swag import SWAGWrapper, SWAGConfig
        base = torch.nn.Linear(10, 3)
        cfg = SWAGConfig()
        swag = SWAGWrapper(base, cfg)
        assert swag is not None
        assert hasattr(swag, "collect")
        assert hasattr(swag, "sample")

    def test_collect_and_sample(self):
        torch = pytest.importorskip("torch")
        from src.models.uq.swag import SWAGWrapper, SWAGConfig
        base = torch.nn.Linear(10, 3)
        cfg = SWAGConfig(rank=5)
        swag = SWAGWrapper(base, cfg)
        # Collect a few snapshots
        for _ in range(10):
            # Simulate different params
            with torch.no_grad():
                for p in base.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            swag.collect()
        # Sample should not raise
        swag.sample()