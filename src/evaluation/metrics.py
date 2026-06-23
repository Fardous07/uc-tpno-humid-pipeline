"""
Evaluation metrics for adsorption prediction.

Three families of metrics, mirroring the three pillars of UC-TPNO:

1.  **Regression** — MAE, RMSE, R², MAPE, per-component errors.
2.  **Uncertainty quantification** — calibration curves, coverage at
    multiple confidence levels, sharpness, CRPS, NLL.
3.  **Multi-objective / Pareto** — hypervolume, Pareto fraction,
    spread, spacing.

All functions accept plain ``numpy`` arrays.  A convenience
``compute_all_metrics`` aggregates everything.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1.  REGRESSION METRICS
# ═══════════════════════════════════════════════════════════════════════

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%)."""
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))))


def max_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Maximum absolute error."""
    return float(np.max(np.abs(y_true - y_pred)))


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
    component_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute all regression metrics.

    Parameters
    ----------
    y_true : ``[N]`` or ``[N, C]`` ground-truth values.
    y_pred : ``[N]`` or ``[N, C]`` predicted values.
    prefix : String prefix for metric keys.
    component_names : Per-component labels (e.g. ``['CO2','N2','H2O']``).

    Returns
    -------
    Dict of metric name → value.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Flatten for overall metrics
    yt_flat = y_true.ravel()
    yp_flat = y_pred.ravel()

    metrics = {
        f"{prefix}mae": mae(yt_flat, yp_flat),
        f"{prefix}rmse": rmse(yt_flat, yp_flat),
        f"{prefix}r2": r2_score(yt_flat, yp_flat),
        f"{prefix}mape": mape(yt_flat, yp_flat),
        f"{prefix}max_abs_error": max_abs_error(yt_flat, yp_flat),
    }

    # Per-component metrics
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        C = y_true.shape[1]
        if component_names is None:
            component_names = [f"comp{i}" for i in range(C)]
        for i in range(C):
            name = component_names[i]
            metrics[f"{prefix}mae_{name}"] = mae(y_true[:, i], y_pred[:, i])
            metrics[f"{prefix}rmse_{name}"] = rmse(y_true[:, i], y_pred[:, i])
            metrics[f"{prefix}r2_{name}"] = r2_score(y_true[:, i], y_pred[:, i])

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# 2.  UNCERTAINTY QUANTIFICATION METRICS
# ═══════════════════════════════════════════════════════════════════════

def coverage_at_alpha(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    Empirical coverage of the ``(1−α)`` Gaussian prediction interval.

    E.g. ``alpha=0.1`` checks 90% coverage.
    """
    z = sp_stats.norm.ppf(1.0 - alpha / 2.0)
    lower = y_pred - z * y_std
    upper = y_pred + z * y_std
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Quantile calibration curve and Expected Calibration Error (ECE).

    Returns
    -------
    expected : ``[n_bins]`` expected coverage levels.
    observed : ``[n_bins]`` observed coverage.
    ece      : scalar ECE = mean |expected − observed|.
    """
    alphas = np.linspace(0.05, 0.95, n_bins)
    expected = 1.0 - alphas
    observed = np.array([coverage_at_alpha(y_true, y_pred, y_std, a) for a in alphas])
    ece = float(np.mean(np.abs(expected - observed)))
    return expected, observed, ece


def sharpness(y_std: np.ndarray) -> float:
    """Mean predictive standard deviation (lower = sharper)."""
    return float(np.mean(y_std))


def interval_width(
    y_std: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Mean width of the (1−α) prediction interval."""
    z = sp_stats.norm.ppf(1.0 - alpha / 2.0)
    return float(np.mean(2.0 * z * y_std))


def gaussian_nll(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
) -> float:
    """Mean Gaussian negative log-likelihood."""
    var = y_std ** 2 + 1e-8
    nll = 0.5 * (np.log(2.0 * np.pi * var) + (y_true - y_pred) ** 2 / var)
    return float(np.mean(nll))


def crps_gaussian(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
) -> float:
    """
    Continuous Ranked Probability Score for Gaussian predictive
    distributions (Gneiting & Raftery, 2007).
    """
    z = (y_true - y_pred) / (y_std + 1e-8)
    crps = y_std * (z * (2.0 * sp_stats.norm.cdf(z) - 1.0)
                    + 2.0 * sp_stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(np.abs(crps)))


def compute_uncertainty_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute all UQ metrics.

    Parameters
    ----------
    y_true : ``[N]`` or ``[N, C]`` ground truth.
    y_pred : Mean predictions.
    y_std  : Predicted standard deviations.
    prefix : Key prefix.
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    ys = np.asarray(y_std).ravel()

    metrics: Dict[str, float] = {}

    # Standardised residuals
    std_res = (yt - yp) / (ys + 1e-8)
    metrics[f"{prefix}std_residual_mean"] = float(np.mean(std_res))
    metrics[f"{prefix}std_residual_std"] = float(np.std(std_res))

    # Coverage at standard levels
    for alpha in [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]:
        level = int((1 - alpha) * 100)
        cov = coverage_at_alpha(yt, yp, ys, alpha)
        metrics[f"{prefix}coverage_{level}"] = cov
        metrics[f"{prefix}cal_error_{level}"] = cov - (1 - alpha)

    # ECE
    _, _, ece = calibration_error(yt, yp, ys)
    metrics[f"{prefix}ece"] = ece

    # Sharpness & interval width
    metrics[f"{prefix}sharpness"] = sharpness(ys)
    metrics[f"{prefix}mean_interval_width_90"] = interval_width(ys, 0.1)

    # NLL & CRPS
    metrics[f"{prefix}nll"] = gaussian_nll(yt, yp, ys)
    metrics[f"{prefix}crps"] = crps_gaussian(yt, yp, ys)

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# 3.  MULTI-OBJECTIVE / PARETO METRICS
# ═══════════════════════════════════════════════════════════════════════

def pareto_front_indices(
    objectives: np.ndarray,
    minimize: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    """
    Find indices of Pareto-optimal points.

    Parameters
    ----------
    objectives : ``[N, M]`` objective values.
    minimize   : Per-objective flag; ``True`` = minimise.
                 Default: all maximised.

    Returns
    -------
    1-D array of Pareto-optimal indices.
    """
    N = len(objectives)
    if minimize is not None:
        obj = objectives.copy()
        for j, m in enumerate(minimize):
            if m:
                obj[:, j] = -obj[:, j]
    else:
        obj = objectives

    is_pareto = np.ones(N, dtype=bool)
    for i in range(N):
        if not is_pareto[i]:
            continue
        for j in range(N):
            if i == j or not is_pareto[j]:
                continue
            if np.all(obj[j] >= obj[i]) and np.any(obj[j] > obj[i]):
                is_pareto[i] = False
                break

    return np.where(is_pareto)[0]


def hypervolume_2d(
    pareto_points: np.ndarray,
    ref_point: np.ndarray,
) -> float:
    """
    2-D hypervolume indicator (exact, O(N log N)).

    Both objectives are assumed to be **maximised**.  Points must
    dominate the reference point.

    Parameters
    ----------
    pareto_points : ``[K, 2]`` Pareto-optimal objective values.
    ref_point     : ``[2]`` reference point (dominated by all).
    """
    pts = pareto_points[pareto_points[:, 0] > ref_point[0]]
    pts = pts[pts[:, 1] > ref_point[1]]

    if len(pts) == 0:
        return 0.0

    # Sort by first objective descending
    pts = pts[np.argsort(-pts[:, 0])]

    hv = 0.0
    prev_y = ref_point[1]
    for x, y in pts:
        if y > prev_y:
            hv += (x - ref_point[0]) * (y - prev_y)
            prev_y = y

    return float(hv)


def pareto_spacing(pareto_points: np.ndarray) -> float:
    """
    Mean nearest-neighbour distance on the Pareto front (uniformity).
    """
    if len(pareto_points) < 2:
        return 0.0

    from scipy.spatial.distance import cdist
    D = cdist(pareto_points, pareto_points)
    np.fill_diagonal(D, np.inf)
    nn_dists = D.min(axis=1)
    return float(np.mean(nn_dists))


def compute_pareto_metrics(
    objectives: np.ndarray,
    ref_point: Optional[np.ndarray] = None,
    minimize: Optional[Sequence[bool]] = None,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute multi-objective Pareto metrics.

    Parameters
    ----------
    objectives : ``[N, M]`` objective values.
    ref_point  : Reference for hypervolume (auto-chosen if None).
    minimize   : Per-objective minimisation flags.
    prefix     : Key prefix.
    """
    objectives = np.asarray(objectives, dtype=np.float64)
    N, M = objectives.shape

    pf_idx = pareto_front_indices(objectives, minimize)
    pf = objectives[pf_idx]

    metrics = {
        f"{prefix}n_pareto": len(pf_idx),
        f"{prefix}pareto_fraction": len(pf_idx) / max(N, 1),
    }

    # Spread (range of each objective on front)
    if len(pf) > 1:
        ranges = pf.max(axis=0) - pf.min(axis=0)
        metrics[f"{prefix}pareto_spread"] = float(np.mean(ranges))
        metrics[f"{prefix}pareto_spacing"] = pareto_spacing(pf)

    # 2-D hypervolume
    if M == 2 and len(pf) > 0:
        # Ensure maximisation form
        pf_max = pf.copy()
        if minimize is not None:
            for j, m in enumerate(minimize):
                if m:
                    pf_max[:, j] = -pf_max[:, j]

        if ref_point is None:
            ref_point = pf_max.min(axis=0) - 0.1 * (pf_max.max(axis=0) - pf_max.min(axis=0) + 1e-8)

        metrics[f"{prefix}hypervolume"] = hypervolume_2d(pf_max, np.asarray(ref_point))

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# 4.  AGGREGATE
# ═══════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    objectives: Optional[np.ndarray] = None,
    component_names: Optional[List[str]] = None,
    pareto_ref: Optional[np.ndarray] = None,
    pareto_minimize: Optional[Sequence[bool]] = None,
) -> Dict[str, float]:
    """
    One-call convenience: regression + UQ + Pareto.

    Parameters
    ----------
    y_true, y_pred : Ground truth and predictions.
    y_std          : Predictive uncertainty (optional).
    objectives     : Multi-objective values for Pareto (optional).
    component_names: Labels for per-component metrics.
    pareto_ref     : Reference point for hypervolume.
    pareto_minimize: Which objectives to minimise.

    Returns
    -------
    Flat dict of all metrics.
    """
    m: Dict[str, float] = {}

    m.update(compute_regression_metrics(y_true, y_pred,
                                         component_names=component_names))

    if y_std is not None:
        m.update(compute_uncertainty_metrics(y_true, y_pred, y_std))

    if objectives is not None:
        m.update(compute_pareto_metrics(objectives, pareto_ref, pareto_minimize))

    return m


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    # Regression
    "mae", "mse", "rmse", "r2_score", "mape", "max_abs_error",
    "compute_regression_metrics",
    # UQ
    "coverage_at_alpha", "calibration_error", "sharpness",
    "interval_width", "gaussian_nll", "crps_gaussian",
    "compute_uncertainty_metrics",
    # Pareto
    "pareto_front_indices", "hypervolume_2d", "pareto_spacing",
    "compute_pareto_metrics",
    # Aggregate
    "compute_all_metrics",
]