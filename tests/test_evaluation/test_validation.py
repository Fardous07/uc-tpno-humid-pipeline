"""
Tests for src/evaluation/validator.py.

Covers: DataQualityChecker, StatisticalValidator, ModelValidator.
"""

import numpy as np
import pytest


class TestDataQualityChecker:
    def test_no_issues_clean_data(self):
        from src.evaluation.validator import DataQualityChecker
        dqc = DataQualityChecker()
        rng = np.random.RandomState(42)
        y = rng.randn(100, 3)
        report = dqc.check(y)
        assert isinstance(report, dict)
        assert report.get("n_samples", 0) == 100

    def test_detects_nan(self):
        from src.evaluation.validator import DataQualityChecker
        dqc = DataQualityChecker()
        y = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
        report = dqc.check(y)
        assert report.get("n_nan", 0) > 0 or report.get("has_nan", False)

    def test_detects_inf(self):
        from src.evaluation.validator import DataQualityChecker
        dqc = DataQualityChecker()
        y = np.array([[1.0], [np.inf], [3.0]])
        report = dqc.check(y)
        assert report.get("n_inf", 0) > 0 or report.get("has_inf", False)


class TestStatisticalValidator:
    def test_residual_normality(self):
        from src.evaluation.validator import StatisticalValidator
        sv = StatisticalValidator()
        rng = np.random.RandomState(42)
        residuals = rng.randn(200)  # truly normal
        report = sv.check_residuals(residuals)
        assert isinstance(report, dict)

    def test_distribution_shift(self):
        from src.evaluation.validator import StatisticalValidator
        sv = StatisticalValidator()
        rng = np.random.RandomState(42)
        X_train = rng.randn(500, 4)
        X_test = rng.randn(100, 4) + 2.0  # shifted
        report = sv.check_distribution_shift(X_train, X_test)
        assert isinstance(report, dict)


class TestModelValidator:
    def test_full_report(self):
        from src.evaluation.validator import ModelValidator
        mv = ModelValidator()
        rng = np.random.RandomState(42)
        y_true = rng.randn(100, 3)
        y_pred = y_true + rng.randn(100, 3) * 0.1
        y_std = np.abs(rng.randn(100, 3)) * 0.2
        X_train = rng.randn(200, 4)
        X_test = rng.randn(100, 4)

        report = mv.full_report(
            y_true=y_true, y_pred=y_pred, y_std=y_std,
            X_train=X_train, X_test=X_test,
        )
        assert isinstance(report, dict)
        assert len(report) > 0

    def test_save_report(self, tmp_path):
        from src.evaluation.validator import ModelValidator
        mv = ModelValidator()
        report = {"test": {"mae": 0.1, "r2": 0.95}}
        path = tmp_path / "report.json"
        mv.save_report(report, path)
        assert path.exists()