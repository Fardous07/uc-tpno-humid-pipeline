"""
Model validation for UC-TPNO.

Orchestrates three validation tiers:

1.  **Thermodynamic consistency** — delegates to
    ``models.operator.losses.ThermodynamicValidator`` for convexity,
    monotonicity, and Henry-region checks.
2.  **Statistical validation** — holdout R², residual normality,
    heteroscedasticity, and out-of-distribution (OOD) detection.
3.  **Data quality** — missing-value audit, outlier detection,
    feature-target correlation, and train/test distribution shift.

The top-level ``ModelValidator.full_report()`` runs everything and
returns a structured dict suitable for logging or JSON export.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1.  DATA QUALITY CHECKS
# ═══════════════════════════════════════════════════════════════════════

class DataQualityChecker:
    """
    Audit datasets before training or evaluation.

    Checks
    ──────
    *  Missing / NaN / Inf values.
    *  Outliers via IQR and z-score.
    *  Feature variance (constant features).
    *  Train/test distribution shift (KS test per feature).
    """

    def __init__(self, z_threshold: float = 4.0, iqr_factor: float = 3.0):
        self.z_threshold = z_threshold
        self.iqr_factor = iqr_factor

    def check_missing(self, X: np.ndarray, name: str = "X") -> Dict[str, Any]:
        """Count NaN and Inf values."""
        n_nan = int(np.isnan(X).sum())
        n_inf = int(np.isinf(X).sum())
        total = X.size
        return {
            f"{name}_n_nan": n_nan,
            f"{name}_n_inf": n_inf,
            f"{name}_missing_frac": (n_nan + n_inf) / max(total, 1),
            f"{name}_clean": n_nan == 0 and n_inf == 0,
        }

    def check_outliers(self, X: np.ndarray, name: str = "X") -> Dict[str, Any]:
        """Detect outliers via z-score and IQR methods."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Z-score outliers
        means = np.nanmean(X, axis=0)
        stds = np.nanstd(X, axis=0).clip(min=1e-8)
        z_scores = np.abs((X - means) / stds)
        n_z_outliers = int((z_scores > self.z_threshold).sum())

        # IQR outliers
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        lower = q1 - self.iqr_factor * iqr
        upper = q3 + self.iqr_factor * iqr
        n_iqr_outliers = int(((X < lower) | (X > upper)).sum())

        return {
            f"{name}_z_outliers": n_z_outliers,
            f"{name}_iqr_outliers": n_iqr_outliers,
            f"{name}_outlier_frac": n_z_outliers / max(X.size, 1),
        }

    def check_variance(self, X: np.ndarray, name: str = "X", threshold: float = 1e-10) -> Dict[str, Any]:
        """Find near-constant features."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        variances = np.nanvar(X, axis=0)
        n_low_var = int((variances < threshold).sum())
        return {
            f"{name}_n_low_variance": n_low_var,
            f"{name}_low_var_frac": n_low_var / max(X.shape[1], 1),
        }

    def check_distribution_shift(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        max_features: int = 50,
    ) -> Dict[str, Any]:
        """
        Kolmogorov-Smirnov test per feature for train/test shift.

        Returns fraction of features with significant shift (p < 0.05).
        """
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1)

        d = min(X_train.shape[1], max_features)
        p_values = []
        for j in range(d):
            _, p = sp_stats.ks_2samp(X_train[:, j], X_test[:, j])
            p_values.append(p)

        p_arr = np.array(p_values)
        return {
            "ks_shift_fraction": float(np.mean(p_arr < 0.05)),
            "ks_min_p": float(p_arr.min()),
            "ks_mean_p": float(p_arr.mean()),
        }

    def full_check(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_test: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run all data quality checks."""
        report: Dict[str, Any] = {}
        report.update(self.check_missing(X, "X"))
        report.update(self.check_missing(y, "y"))
        report.update(self.check_outliers(X, "X"))
        report.update(self.check_outliers(y, "y"))
        report.update(self.check_variance(X, "X"))

        if X_test is not None:
            report.update(self.check_distribution_shift(X, X_test))

        return report


# ═══════════════════════════════════════════════════════════════════════
# 2.  STATISTICAL VALIDATION
# ═══════════════════════════════════════════════════════════════════════

class StatisticalValidator:
    """
    Post-training statistical checks on model predictions.

    Tests
    ─────
    *  **Residual normality** — Shapiro-Wilk (N < 5000) or
       D'Agostino-Pearson.
    *  **Heteroscedasticity** — Breusch-Pagan (residual² vs ŷ).
    *  **Residual autocorrelation** — Durbin-Watson statistic.
    *  **OOD detection** — Mahalanobis distance of test from train.
    """

    def residual_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """Analyse residual distribution."""
        residuals = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()

        results: Dict[str, Any] = {
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
            "residual_skewness": float(sp_stats.skew(residuals)),
            "residual_kurtosis": float(sp_stats.kurtosis(residuals)),
        }

        # Normality test
        n = len(residuals)
        if n < 5000:
            stat, p = sp_stats.shapiro(residuals[:min(n, 5000)])
            results["normality_test"] = "shapiro"
        else:
            stat, p = sp_stats.normaltest(residuals)
            results["normality_test"] = "dagostino"
        results["normality_stat"] = float(stat)
        results["normality_p"] = float(p)
        results["residuals_normal"] = p > 0.05

        # Durbin-Watson
        diffs = np.diff(residuals)
        dw = float(np.sum(diffs ** 2) / max(np.sum(residuals ** 2), 1e-15))
        results["durbin_watson"] = dw
        results["autocorrelation_flag"] = dw < 1.5 or dw > 2.5

        return results

    def heteroscedasticity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Breusch-Pagan-like test: regress residual² on ŷ.

        A significant slope indicates heteroscedasticity.
        """
        residuals = (np.asarray(y_true) - np.asarray(y_pred)).ravel()
        yp = np.asarray(y_pred).ravel()
        res_sq = residuals ** 2

        slope, intercept, r, p, se = sp_stats.linregress(yp, res_sq)
        return {
            "heterosc_slope": float(slope),
            "heterosc_p": float(p),
            "heterosc_flag": p < 0.05,
        }

    def ood_detection(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        percentile: float = 95.0,
    ) -> Dict[str, Any]:
        """
        Mahalanobis distance-based OOD detection.

        Points with distance above the train-set percentile are
        flagged as potential OOD.
        """
        X_tr = np.asarray(X_train, dtype=np.float64)
        X_te = np.asarray(X_test, dtype=np.float64)

        mu = X_tr.mean(axis=0)
        cov = np.cov(X_tr.T) + 1e-6 * np.eye(X_tr.shape[1])
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        def mahal(X):
            diff = X - mu
            return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

        d_train = mahal(X_tr)
        d_test = mahal(X_te)
        threshold = float(np.percentile(d_train, percentile))

        n_ood = int((d_test > threshold).sum())
        return {
            "ood_threshold": threshold,
            "ood_count": n_ood,
            "ood_fraction": n_ood / max(len(X_te), 1),
            "mahal_test_mean": float(d_test.mean()),
            "mahal_test_max": float(d_test.max()),
        }


# ═══════════════════════════════════════════════════════════════════════
# 3.  THERMODYNAMIC VALIDATION WRAPPER
# ═══════════════════════════════════════════════════════════════════════

class ThermodynamicValidationWrapper:
    """
    Wraps ``models.operator.losses.ThermodynamicValidator`` for
    use in the evaluation pipeline.

    If the low-level validator is not available (e.g. no torch),
    falls back gracefully.
    """

    def __init__(self, n_test_points: int = 100):
        self.n_test_points = n_test_points
        self._validator = None

    def _get_validator(self):
        if self._validator is None:
            try:
                from src.models.operator.losses import ThermodynamicValidator
                self._validator = ThermodynamicValidator(self.n_test_points)
            except ImportError:
                logger.warning("ThermodynamicValidator not available (torch missing?).")
        return self._validator

    def check_all(
        self,
        model: Any,
        graphs: Any,
        conditions: Any,
    ) -> Dict[str, Any]:
        """
        Run all thermodynamic checks: convexity, monotonicity, Henry.

        Parameters
        ----------
        model      : TPNO model (nn.Module).
        graphs     : Graph batch from encoder.
        conditions : ``[B, P, D]`` condition tensor.

        Returns
        -------
        Combined metrics dict.
        """
        val = self._get_validator()
        if val is None:
            return {"thermo_checks": "unavailable"}

        results: Dict[str, Any] = {}

        try:
            results.update(val.check_convexity(model, graphs, conditions))
        except Exception as e:
            results["convexity_error"] = str(e)

        try:
            results.update(val.check_monotonicity(model, graphs, conditions))
        except Exception as e:
            results["monotonicity_error"] = str(e)

        try:
            results.update(val.check_henry_region(model, graphs, conditions))
        except Exception as e:
            results["henry_error"] = str(e)

        return results


# ═══════════════════════════════════════════════════════════════════════
# 4.  UNIFIED MODEL VALIDATOR
# ═══════════════════════════════════════════════════════════════════════

class ModelValidator:
    """
    Top-level validator combining all tiers.

    Example
    ───────
    >>> val = ModelValidator()
    >>> report = val.full_report(
    ...     y_true=y_test, y_pred=pred_mean, y_std=pred_std,
    ...     X_train=X_train, X_test=X_test,
    ... )
    >>> val.save_report(report, "validation_report.json")
    """

    def __init__(self):
        self.data_checker = DataQualityChecker()
        self.stat_validator = StatisticalValidator()
        self.thermo_wrapper = ThermodynamicValidationWrapper()

    def full_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: Optional[np.ndarray] = None,
        X_train: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        model: Any = None,
        graphs: Any = None,
        conditions: Any = None,
    ) -> Dict[str, Any]:
        """
        Run all validation checks and return structured report.

        Parameters
        ----------
        y_true, y_pred : Ground truth and predictions.
        y_std          : Predictive std (optional, for UQ checks).
        X_train, X_test: Feature matrices (optional, for shift/OOD).
        y_train        : Training targets (optional, for data checks).
        model, graphs, conditions : For thermodynamic checks.

        Returns
        -------
        Nested dict: ``{"data_quality": {...}, "statistical": {...},
        "thermodynamic": {...}, "summary": {...}}``.
        """
        report: Dict[str, Any] = {}

        # ── Data quality ─────────────────────────────────────────
        dq: Dict[str, Any] = {}
        if X_test is not None:
            dq.update(self.data_checker.check_missing(X_test, "X_test"))
        dq.update(self.data_checker.check_missing(y_true, "y_true"))
        dq.update(self.data_checker.check_missing(y_pred, "y_pred"))
        if X_train is not None and X_test is not None:
            dq.update(self.data_checker.check_distribution_shift(X_train, X_test))
        if y_train is not None:
            dq.update(self.data_checker.check_outliers(y_train, "y_train"))
        report["data_quality"] = dq

        # ── Statistical validation ───────────────────────────────
        stat: Dict[str, Any] = {}
        stat.update(self.stat_validator.residual_analysis(y_true, y_pred))
        stat.update(self.stat_validator.heteroscedasticity(y_true, y_pred))
        if X_train is not None and X_test is not None:
            stat.update(self.stat_validator.ood_detection(X_train, X_test))
        report["statistical"] = stat

        # ── Thermodynamic consistency ────────────────────────────
        if model is not None and graphs is not None and conditions is not None:
            report["thermodynamic"] = self.thermo_wrapper.check_all(
                model, graphs, conditions,
            )

        # ── Summary flags ────────────────────────────────────────
        from .metrics import compute_regression_metrics, compute_uncertainty_metrics

        reg = compute_regression_metrics(y_true, y_pred)
        summary: Dict[str, Any] = {
            "r2": reg.get("r2", 0.0),
            "rmse": reg.get("rmse", 0.0),
            "residuals_normal": stat.get("residuals_normal", False),
            "heteroscedastic": stat.get("heterosc_flag", False),
        }

        if y_std is not None:
            uq = compute_uncertainty_metrics(y_true, y_pred, y_std)
            summary["ece"] = uq.get("ece", None)
            summary["coverage_90"] = uq.get("coverage_90", None)

        if X_train is not None and X_test is not None:
            summary["ood_fraction"] = stat.get("ood_fraction", 0.0)
            summary["distribution_shift"] = dq.get("ks_shift_fraction", 0.0)

        report["summary"] = summary
        return report

    @staticmethod
    def save_report(report: Dict[str, Any], path: Union[str, Path]) -> None:
        """Save report to JSON (with numpy serialisation)."""
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            raise TypeError(f"Cannot serialise {type(obj)}")

        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=convert)
        logger.info(f"Validation report saved to {path}")

    @staticmethod
    def load_report(path: Union[str, Path]) -> Dict[str, Any]:
        """Load report from JSON."""
        with open(path) as f:
            return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "DataQualityChecker",
    "StatisticalValidator",
    "ThermodynamicValidationWrapper",
    "ModelValidator",
]