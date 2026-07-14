"""
Distribution-free conformal prediction for uncertainty calibration.

This module provides finite-sample coverage-guaranteed prediction intervals
for the UC-TPNO pipeline.  Four conformal prediction strategies are
implemented behind a single ``ConformalCalibrator`` façade:

1.  **Split conformal** — the simplest method; uses a held-out
    calibration set to compute a single quantile of nonconformity
    scores.  Guarantees marginal coverage ≥ 1−α.
2.  **Weighted conformal** — accounts for covariate shift between
    calibration and test domains by re-weighting calibration scores
    with a classifier-estimated density ratio (Tibshirani et al., 2019).
3.  **Group / Mondrian conformal** — fits separate quantiles per group
    (e.g. MOF topology, temperature bin), providing *conditional*
    coverage within each group (Vovk et al., 2005).
4.  **CV+ conformal** — uses cross-validation out-of-fold residuals so
    every data point contributes to both training and calibration
    (Barber et al., 2021).

Additionally:

*   ``NonconformityScores`` — factory for absolute, normalised,
    studentised, and quantile-based score functions.
*   ``DensityRatioEstimator`` — classifier-based p_target/p_source
    estimation for the weighted method.
*   ``evaluate_coverage`` — post-hoc evaluation of empirical coverage,
    interval width, sharpness, and conditional-coverage statistics.

All methods operate on NumPy arrays (CPU) and are model-agnostic:
they only require predictions, ground truth, and (optionally)
predicted uncertainties.

References
──────────
[1] Angelopoulos & Bates (2021). A Gentle Introduction to Conformal
    Prediction and Distribution-Free Uncertainty Quantification.
[2] Tibshirani et al. (2019). Conformal Prediction Under Covariate
    Shift. NeurIPS.
[3] Vovk et al. (2005). Algorithmic Learning in a Random World.
[4] Barber et al. (2021). Predictive Inference with the Jackknife+.
    Annals of Statistics.
[5] Romano et al. (2019). Conformalized Quantile Regression. NeurIPS.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ConformalConfig:
    """
    Configuration for conformal prediction.

    Attributes
    ──────────
    alpha          : Miscoverage rate (e.g. 0.1 → 90 % intervals).
    method         : ``'split'``, ``'weighted'``, ``'mondrian'``, or
                     ``'cv'``.
    score_method   : Nonconformity score: ``'absolute'``,
                     ``'normalized'``, ``'studentized'``.
    n_bins         : Number of groups when ``method='mondrian'`` and
                     groups are derived from a continuous feature via
                     equal-frequency binning.
    cal_size       : Fraction of data reserved for calibration when
                     the caller performs its own train/cal split.
    density_ratio  : Classifier for density-ratio estimation in the
                     weighted method: ``'gbm'``, ``'rf'``, ``'logistic'``.
    weight_clip    : ``(low, high)`` clip range for density-ratio
                     weights to prevent extreme values.
    """

    alpha: float = 0.1
    method: str = "split"
    score_method: str = "absolute"
    n_bins: int = 5
    cal_size: float = 0.2
    density_ratio: str = "gbm"
    weight_clip: Tuple[float, float] = (0.1, 10.0)


# ═══════════════════════════════════════════════════════════════════════
# 2.  NONCONFORMITY SCORES
# ═══════════════════════════════════════════════════════════════════════

class NonconformityScores:
    """
    Factory for nonconformity score functions.

    Every method is a static function with signature
    ``(y_true, y_pred, [y_std]) → scores`` returning a 1-D array of
    non-negative scalars.  Larger values = more nonconformal.
    """

    @staticmethod
    def absolute_error(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """|y − ŷ|"""
        return np.abs(y_true - y_pred)

    @staticmethod
    def squared_error(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """(y − ŷ)²"""
        return (y_true - y_pred) ** 2

    @staticmethod
    def normalized_error(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """|y − ŷ| / σ  (locally adaptive)."""
        return np.abs(y_true - y_pred) / (np.asarray(y_std) + 1e-8)

    @staticmethod
    def studentized_error(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        leverage: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """|y − ŷ| / (σ √(1−h))"""
        denom = np.asarray(y_std) + 1e-8
        if leverage is not None:
            denom = denom * np.sqrt(np.maximum(1.0 - leverage, 1e-8))
        return np.abs(y_true - y_pred) / denom

    @staticmethod
    def quantile_error(
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """max(lower − y, 0) + max(y − upper, 0)."""
        return np.maximum(lower - y_true, 0) + np.maximum(y_true - upper, 0)


def _get_score_fn(name: str) -> Callable:
    """Look up a score function by name."""
    mapping = {
        "absolute": NonconformityScores.absolute_error,
        "squared": NonconformityScores.squared_error,
        "normalized": NonconformityScores.normalized_error,
        "studentized": NonconformityScores.studentized_error,
    }
    if name not in mapping:
        raise ValueError(
            f"Unknown score method '{name}'. Choose from {list(mapping)}."
        )
    return mapping[name]


# ═══════════════════════════════════════════════════════════════════════
# 3.  DENSITY-RATIO ESTIMATION  (for weighted conformal)
# ═══════════════════════════════════════════════════════════════════════

class DensityRatioEstimator:
    """
    Classifier-based density-ratio estimator:

        w(x) = p_target(x) / p_source(x)

    A binary classifier is trained to distinguish source (label 0) from
    target (label 1) covariates.  The ratio is then
    ``P(target | x) / P(source | x)``, clipped for stability.

    Supported classifiers: ``'gbm'`` (default), ``'rf'``, ``'logistic'``.

    FIX: original code normalised weights BEFORE clipping, so clipping
    destroyed the normalisation.  Correct order: clip first, then
    normalise to mean 1 so that the weighted quantile is well-scaled.
    """

    def __init__(
        self,
        method: str = "gbm",
        clip: Tuple[float, float] = (0.1, 10.0),
    ):
        self.method = method
        self.clip = clip
        self._clf = None

    def fit(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
    ) -> "DensityRatioEstimator":
        """Train the classifier on source ∪ target."""
        X = np.vstack([X_source, X_target])
        y = np.concatenate([np.zeros(len(X_source)), np.ones(len(X_target))])

        if self.method == "gbm":
            from sklearn.ensemble import GradientBoostingClassifier

            self._clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )
        elif self.method == "rf":
            from sklearn.ensemble import RandomForestClassifier

            self._clf = RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42,
            )
        elif self.method == "logistic":
            from sklearn.linear_model import LogisticRegression

            self._clf = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown density-ratio method: {self.method}")

        self._clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return density-ratio weights w(x) for each row of *X*."""
        if self._clf is None:
            raise RuntimeError("Call .fit() before .predict().")
        proba = self._clf.predict_proba(X)
        p_src, p_tgt = proba[:, 0], proba[:, 1]
        w = p_tgt / (p_src + 1e-8)
        # FIX: clip BEFORE normalising so the clipped weights still have
        # mean 1 and the weighted quantile is correctly scaled.
        # Original: normalise then clip → clipping broke normalisation.
        w = np.clip(w, self.clip[0], self.clip[1])
        w = w / w.mean()    # normalise to mean 1 after clipping
        return w


# ═══════════════════════════════════════════════════════════════════════
# 4.  CONFORMAL PREDICTORS
# ═══════════════════════════════════════════════════════════════════════

# ── Helpers ──────────────────────────────────────────────────────────

def _finite_sample_quantile(
    scores: np.ndarray,
    alpha: float,
) -> float:
    """
    Compute the conformal quantile with the finite-sample correction:

        q = ⌈(n+1)(1−α)⌉ / n – th quantile of *scores*.

    This ensures the guaranteed marginal coverage ≥ 1−α.
    """
    n = len(scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def _weighted_quantile(
    scores: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> float:
    """
    Weighted conformal quantile: the smallest score s such that the
    cumulative normalised weight of all calibration scores ≤ s
    exceeds ``(1−α)(1 + 1/n)``.

    FIX: original used ``np.interp`` (linear interpolation between
    scores), which can return a value strictly between two calibration
    scores.  The conformal guarantee requires the *step-function*
    quantile — the actual smallest calibration score that pushes
    cumulative weight over the target level.  Fixed to use
    ``np.searchsorted`` on the sorted cumulative weights.
    """
    n = len(scores)
    target = (1 - alpha) * (1 + 1.0 / n)

    idx = np.argsort(scores)
    sorted_scores = scores[idx]
    sorted_weights = weights[idx]
    cum_w = np.cumsum(sorted_weights / sorted_weights.sum())

    # Find the first index where cumulative weight reaches target
    pos = np.searchsorted(cum_w, target, side="left")
    if pos >= len(sorted_scores):
        # All scores needed; return the largest
        return float(sorted_scores[-1])
    return float(sorted_scores[pos])


# ── Split conformal ─────────────────────────────────────────────────

class SplitConformalPredictor:
    """
    Basic split conformal prediction.

    Uses a single calibration set to compute one global quantile.
    """

    def __init__(self, config: ConformalConfig):
        self.config = config
        self.quantile: Optional[float] = None
        self.calibration_scores: Optional[np.ndarray] = None

    def fit(self, scores: np.ndarray, **kwargs) -> "SplitConformalPredictor":
        self.calibration_scores = scores
        self.quantile = _finite_sample_quantile(scores, self.config.alpha)
        return self

    def predict(
        self,
        test_scores: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(lower_offset, upper_offset)``."""
        if self.quantile is None:
            raise RuntimeError("Call .fit() first.")
        return (
            np.full_like(test_scores, -self.quantile),
            np.full_like(test_scores, self.quantile),
        )


# ── Weighted conformal ──────────────────────────────────────────────

class WeightedConformalPredictor:
    """
    Weighted conformal prediction for covariate shift
    (Tibshirani et al., 2019).
    """

    def __init__(self, config: ConformalConfig):
        self.config = config
        self.quantile: Optional[float] = None
        self.calibration_scores: Optional[np.ndarray] = None
        self.calibration_weights: Optional[np.ndarray] = None
        self._dre = DensityRatioEstimator(
            method=config.density_ratio,
            clip=config.weight_clip,
        )

    def fit(
        self,
        scores: np.ndarray,
        calibration_covariates: np.ndarray,
        target_covariates: np.ndarray,
        **kwargs,
    ) -> "WeightedConformalPredictor":
        self.calibration_scores = scores
        self._dre.fit(calibration_covariates, target_covariates)
        self.calibration_weights = self._dre.predict(calibration_covariates)
        self.quantile = _weighted_quantile(
            scores, self.calibration_weights, self.config.alpha,
        )
        return self

    def predict(
        self,
        test_scores: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.quantile is None:
            raise RuntimeError("Call .fit() first.")
        return (
            np.full_like(test_scores, -self.quantile),
            np.full_like(test_scores, self.quantile),
        )


# ── Group / Mondrian conformal ──────────────────────────────────────

class GroupConformalPredictor:
    """
    Group (Mondrian) conformal prediction (Vovk et al., 2005).

    Separate quantiles per group → conditional coverage within groups.
    """

    def __init__(self, config: ConformalConfig):
        self.config = config
        self.group_quantiles: Dict[Any, float] = {}
        self.group_scores: Dict[Any, np.ndarray] = {}
        self.group_counts: Dict[Any, int] = {}
        self._fallback_quantile: Optional[float] = None

    def fit(
        self,
        scores: np.ndarray,
        calibration_groups: np.ndarray,
        **kwargs,
    ) -> "GroupConformalPredictor":
        for g in np.unique(calibration_groups):
            mask = calibration_groups == g
            gs = scores[mask]
            if len(gs) == 0:
                continue
            self.group_scores[g] = gs
            self.group_counts[g] = len(gs)
            self.group_quantiles[g] = _finite_sample_quantile(gs, self.config.alpha)

        # Global fallback
        self._fallback_quantile = _finite_sample_quantile(scores, self.config.alpha)
        return self

    def predict(
        self,
        test_scores: np.ndarray,
        test_groups: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.zeros_like(test_scores)
        upper = np.zeros_like(test_scores)

        for g in np.unique(test_groups):
            mask = test_groups == g
            q = self.group_quantiles.get(g, self._fallback_quantile)
            if q is None:
                q = 0.0
            lower[mask] = -q
            upper[mask] = q

        return lower, upper


# ── CV+ conformal ───────────────────────────────────────────────────

class CVPlusConformalPredictor:
    """
    Cross-validation+ conformal prediction (Barber et al., 2021).

    Every data point is used for both training and calibration via
    K-fold out-of-fold residuals.
    """

    def __init__(self, config: ConformalConfig, n_folds: int = 5):
        self.config = config
        self.n_folds = n_folds
        self.quantile: Optional[float] = None
        self.calibration_scores: Optional[np.ndarray] = None

    def fit(
        self,
        scores_per_fold: List[np.ndarray],
        **kwargs,
    ) -> "CVPlusConformalPredictor":
        """
        Parameters
        ----------
        scores_per_fold : List of K arrays, one per fold, each
            containing the out-of-fold nonconformity scores.
        """
        self.calibration_scores = np.concatenate(scores_per_fold)
        self.quantile = _finite_sample_quantile(
            self.calibration_scores, self.config.alpha,
        )
        return self

    def predict(
        self,
        test_scores: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.quantile is None:
            raise RuntimeError("Call .fit() first.")
        return (
            np.full_like(test_scores, -self.quantile),
            np.full_like(test_scores, self.quantile),
        )


# ═══════════════════════════════════════════════════════════════════════
# 5.  CONFORMAL CALIBRATOR  (unified façade)
# ═══════════════════════════════════════════════════════════════════════

class ConformalCalibrator:
    """
    Unified interface for conformal calibration of TPNO predictions.

    Usage
    ─────
    >>> cfg = ConformalConfig(alpha=0.1, method='weighted',
    ...                       score_method='normalized')
    >>> cal = ConformalCalibrator(cfg)
    >>> cal.calibrate({
    ...     'y_true': y_cal, 'y_pred': mu_cal, 'y_std': sigma_cal,
    ...     'covariates': X_cal, 'target_covariates': X_test,
    ... })
    >>> intervals = cal.predict_intervals({'y_pred': mu_test, 'y_std': sigma_test})
    >>> # intervals = {'mean', 'lower', 'upper', 'coverage_probability'}

    Parameters
    ----------
    config : ``ConformalConfig``
    """

    def __init__(self, config: Optional[ConformalConfig] = None):
        self.config = config or ConformalConfig()
        self._predictor = self._make_predictor()
        self._score_fn = _get_score_fn(self.config.score_method)
        self.is_fitted = False

    def _make_predictor(self):
        m = self.config.method
        if m == "split":
            return SplitConformalPredictor(self.config)
        if m == "weighted":
            return WeightedConformalPredictor(self.config)
        if m == "mondrian":
            return GroupConformalPredictor(self.config)
        if m == "cv":
            return CVPlusConformalPredictor(self.config)
        warnings.warn(f"Unknown method '{m}'; falling back to split.")
        return SplitConformalPredictor(self.config)

    # ── score computation ────────────────────────────────────────

    def compute_scores(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute nonconformity scores using the configured method."""
        kwargs: Dict[str, Any] = {}
        if y_std is not None:
            kwargs["y_std"] = y_std
        return self._score_fn(y_true, y_pred, **kwargs)

    # ── calibrate ────────────────────────────────────────────────

    def calibrate(
        self,
        calibration_data: Dict[str, np.ndarray],
    ) -> "ConformalCalibrator":
        """
        Calibrate the conformal predictor on a calibration set.

        Parameters
        ----------
        calibration_data : Dict with keys:
            * ``y_true``  — ground truth.
            * ``y_pred``  — point predictions.
            * ``y_std``   — predicted std-dev (opt., needed for
              ``'normalized'`` / ``'studentized'`` scores).
            * ``covariates``        — calibration covariates (weighted).
            * ``target_covariates`` — target covariates (weighted).
            * ``groups``            — group labels (Mondrian).
            * ``scores_per_fold``   — list of fold-score arrays (CV+).
        """
        method = self.config.method

        if method == "cv":
            if "scores_per_fold" not in calibration_data:
                raise ValueError("CV+ requires 'scores_per_fold'.")
            self._predictor.fit(
                scores_per_fold=calibration_data["scores_per_fold"],
            )
            self.is_fitted = True
            return self

        # Compute scores
        scores = self.compute_scores(
            calibration_data["y_true"],
            calibration_data["y_pred"],
            calibration_data.get("y_std"),
        )

        if method == "weighted":
            if "covariates" not in calibration_data:
                raise ValueError("Weighted conformal requires 'covariates'.")
            if "target_covariates" not in calibration_data:
                raise ValueError("Weighted conformal requires 'target_covariates'.")
            self._predictor.fit(
                scores,
                calibration_covariates=calibration_data["covariates"],
                target_covariates=calibration_data["target_covariates"],
            )
        elif method == "mondrian":
            if "groups" not in calibration_data:
                raise ValueError("Mondrian conformal requires 'groups'.")
            self._predictor.fit(
                scores,
                calibration_groups=calibration_data["groups"],
            )
        else:
            self._predictor.fit(scores)

        self.is_fitted = True
        return self

    # ── predict ──────────────────────────────────────────────────

    def predict_intervals(
        self,
        test_data: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Generate prediction intervals for test data.

        Parameters
        ----------
        test_data : Dict with keys:
            * ``y_pred`` — point predictions.
            * ``y_std``  — predicted std-dev (opt.).
            * ``groups`` — group labels (Mondrian only).

        Returns
        -------
        Dict with ``mean``, ``lower``, ``upper``, ``coverage_probability``.
        """
        if not self.is_fitted:
            raise RuntimeError("Call .calibrate() first.")

        y_pred = test_data["y_pred"]
        dummy_scores = np.zeros_like(y_pred)

        kwargs: Dict[str, Any] = {}
        if self.config.method == "mondrian":
            if "groups" not in test_data:
                raise ValueError("Mondrian requires 'groups' in test_data.")
            kwargs["test_groups"] = test_data["groups"]

        lower_off, upper_off = self._predictor.predict(dummy_scores, **kwargs)

        # If using normalized scores, scale the offsets by local sigma
        if self.config.score_method in ("normalized", "studentized"):
            y_std = test_data.get("y_std")
            if y_std is not None:
                lower_off = lower_off * y_std
                upper_off = upper_off * y_std

        return {
            "mean": y_pred,
            "lower": y_pred + lower_off,
            "upper": y_pred + upper_off,
            "coverage_probability": 1.0 - self.config.alpha,
        }

    # ── adaptive intervals ───────────────────────────────────────

    def adaptive_intervals(
        self,
        test_data: Dict[str, np.ndarray],
        calibration_data: Dict[str, np.ndarray],
        lambda_reg: float = 0.1,
    ) -> Dict[str, np.ndarray]:
        """
        Locally-adaptive intervals scaled by a learned difficulty model
        (Romano et al., 2019 — Conformalized Quantile Regression idea).

        Fits a Random Forest on calibration absolute errors to estimate
        local difficulty, then rescales the base interval widths.
        """
        from sklearn.ensemble import RandomForestRegressor

        # Fit difficulty model
        X_cal = calibration_data.get(
            "features",
            np.ones((len(calibration_data["y_true"]), 1)),
        )
        abs_err = np.abs(calibration_data["y_true"] - calibration_data["y_pred"])

        rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
        rf.fit(X_cal, abs_err)

        X_test = test_data.get(
            "features",
            np.ones((len(test_data["y_pred"]), 1)),
        )
        local_scale = rf.predict(X_test) + lambda_reg

        # Base intervals
        base = self.predict_intervals(test_data)
        half_width = (base["upper"] - base["lower"]) / 2.0
        scaled_hw = half_width * local_scale / np.mean(local_scale)

        y_pred = test_data["y_pred"]
        return {
            "mean": y_pred,
            "lower": y_pred - scaled_hw,
            "upper": y_pred + scaled_hw,
            "coverage_probability": 1.0 - self.config.alpha,
        }


# ═══════════════════════════════════════════════════════════════════════
# 6.  EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate_coverage(
    intervals: Dict[str, np.ndarray],
    y_true: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Evaluate empirical coverage and interval quality.

    Parameters
    ----------
    intervals : Dict with ``lower``, ``upper``, and optionally
                ``coverage_probability``.
    y_true    : Ground-truth values.
    groups    : Optional group labels for conditional-coverage analysis.

    Returns
    -------
    Dict of scalar metrics:

    * ``overall_coverage``    — fraction of y_true inside [lower, upper].
    * ``target_coverage``     — the nominal 1−α.
    * ``coverage_error``      — overall − target.
    * ``mean_width``          — average interval width.
    * ``median_width``        — median interval width.
    * ``std_width``           — std-dev of widths.
    * ``sharpness``           — −mean_width (higher = narrower = better).
    * ``length_deviation``    — relative std of widths.
    * ``n_samples``           — number of test points.
    * ``conditional_coverage`` — per-group coverage (if *groups* given).
    """
    lower = intervals["lower"]
    upper = intervals["upper"]
    target = intervals.get("coverage_probability", 0.9)

    covered = (y_true >= lower) & (y_true <= upper)
    overall = float(np.mean(covered))

    width = upper - lower
    mw = float(np.mean(width))

    results: Dict[str, Any] = {
        "overall_coverage": overall,
        "target_coverage": float(target),
        "coverage_error": overall - float(target),
        "mean_width": mw,
        "median_width": float(np.median(width)),
        "std_width": float(np.std(width)),
        "sharpness": -mw,
        "length_deviation": float(np.std(width) / mw) if mw > 0 else 0.0,
        "n_samples": len(y_true),
    }

    if groups is not None:
        cond: Dict[str, float] = {}
        for g in np.unique(groups):
            mask = groups == g
            cond[f"group_{g}"] = float(np.mean(covered[mask]))
        results["conditional_coverage"] = cond
        results["min_conditional_coverage"] = float(min(cond.values()))
        results["max_conditional_coverage"] = float(max(cond.values()))

    return results


def multi_alpha_coverage(
    calibrator: ConformalCalibrator,
    calibration_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    y_true_test: np.ndarray,
    alphas: Optional[Sequence[float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Sweep over multiple α values and report observed coverage vs.
    nominal coverage for calibration-plot construction.

    Returns dict with ``alphas``, ``nominal``, ``observed``, ``widths``.
    """
    if alphas is None:
        alphas = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    nominal, observed, widths = [], [], []

    for a in alphas:
        cfg = ConformalConfig(
            alpha=a,
            method=calibrator.config.method,
            score_method=calibrator.config.score_method,
            density_ratio=calibrator.config.density_ratio,
            weight_clip=calibrator.config.weight_clip,
        )
        cal = ConformalCalibrator(cfg)
        cal.calibrate(calibration_data)
        iv = cal.predict_intervals(test_data)
        ev = evaluate_coverage(iv, y_true_test)

        nominal.append(1.0 - a)
        observed.append(ev["overall_coverage"])
        widths.append(ev["mean_width"])

    return {
        "alphas": np.array(alphas),
        "nominal": np.array(nominal),
        "observed": np.array(observed),
        "widths": np.array(widths),
    }


# ═══════════════════════════════════════════════════════════════════════
# 7.  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    # Config
    "ConformalConfig",
    # Scores
    "NonconformityScores",
    # Density ratio
    "DensityRatioEstimator",
    # Predictors
    "SplitConformalPredictor",
    "WeightedConformalPredictor",
    "GroupConformalPredictor",
    "CVPlusConformalPredictor",
    # Unified interface
    "ConformalCalibrator",
    # Evaluation
    "evaluate_coverage",
    "multi_alpha_coverage",
]