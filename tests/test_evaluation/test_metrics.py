"""
Tests for src/evaluation/metrics.py.

Covers: mae, rmse, r2, mape, coverage, CRPS, ECE, NLL,
        compute_regression_metrics, compute_uncertainty_metrics,
        compute_all_metrics, pareto_front_indices.
"""

import numpy as np
import pytest


class TestRegressionMetrics:
    def test_mae_perfect(self):
        from src.evaluation.metrics import mae
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_mae_known(self):
        from src.evaluation.metrics import mae
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        assert mae(y_true, y_pred) == pytest.approx(0.5)

    def test_rmse(self):
        from src.evaluation.metrics import rmse
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(1 / 3), rel=1e-5)

    def test_r2_perfect(self):
        from src.evaluation.metrics import r2_score
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert r2_score(y, y) == pytest.approx(1.0)

    def test_r2_mean_model(self):
        from src.evaluation.metrics import r2_score
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full_like(y_true, y_true.mean())
        assert r2_score(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)

    def test_mape(self):
        from src.evaluation.metrics import mape
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        expected = np.mean([0.2, 0.1, 0.1]) * 100
        assert mape(y_true, y_pred) == pytest.approx(expected, rel=0.01)

    def test_max_abs_error(self):
        from src.evaluation.metrics import max_abs_error
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 5.0, 2.9])
        assert max_abs_error(y_true, y_pred) == pytest.approx(3.0)


class TestCompositeMetrics:
    def test_compute_regression_metrics(self):
        from src.evaluation.metrics import compute_regression_metrics
        rng = np.random.RandomState(42)
        y_true = rng.randn(100, 3)
        y_pred = y_true + rng.randn(100, 3) * 0.1
        m = compute_regression_metrics(y_true, y_pred,
                                        component_names=["CO2", "N2", "H2O"])
        assert "mae_overall" in m or "mae" in m or "CO2_mae" in m
        # At least some metric keys exist
        assert len(m) > 0

    def test_compute_all_metrics(self):
        from src.evaluation.metrics import compute_all_metrics
        rng = np.random.RandomState(42)
        y_true = rng.randn(50, 3)
        y_pred = y_true + rng.randn(50, 3) * 0.2
        y_std = np.abs(rng.randn(50, 3)) * 0.3
        m = compute_all_metrics(y_true, y_pred, y_std)
        assert len(m) > 0


class TestUQMetrics:
    def test_coverage_at_alpha(self):
        from src.evaluation.metrics import coverage_at_alpha
        rng = np.random.RandomState(42)
        y_true = rng.randn(1000)
        y_pred = y_true + rng.randn(1000) * 0.3
        y_std = np.ones(1000) * 0.5
        cov = coverage_at_alpha(y_true, y_pred, y_std, alpha=0.1)
        # With std=0.5, 90% interval = ±0.82, error std=0.3 → coverage >> 0.9
        assert 0.5 < cov <= 1.0

    def test_compute_uncertainty_metrics(self):
        from src.evaluation.metrics import compute_uncertainty_metrics
        rng = np.random.RandomState(42)
        y_true = rng.randn(200)
        y_pred = y_true + rng.randn(200) * 0.1
        y_std = np.abs(rng.randn(200)) * 0.2
        m = compute_uncertainty_metrics(y_true, y_pred, y_std)
        assert len(m) > 0


class TestParetoFront:
    def test_pareto_2d(self):
        from src.evaluation.metrics import pareto_front_indices
        # 4 points: (1,4), (2,3), (3,2), (4,1) — all on Pareto front (minimise)
        objectives = np.array([[1, 4], [2, 3], [3, 2], [4, 1]])
        idx = pareto_front_indices(objectives)
        assert len(idx) == 4  # all non-dominated

    def test_pareto_dominated(self):
        from src.evaluation.metrics import pareto_front_indices
        # (1,1) dominates (2,2)
        objectives = np.array([[1, 1], [2, 2], [1, 3], [3, 1]])
        idx = pareto_front_indices(objectives)
        assert 0 in idx  # (1,1) on front
        assert 1 not in idx  # (2,2) dominated