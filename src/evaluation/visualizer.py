"""
Publication-quality visualisation for UC-TPNO results.

Plot types
──────────
1.  **Parity plots** — predicted vs true (with UQ bands).
2.  **Isotherms** — loading vs pressure per component.
3.  **Calibration curves** — observed vs expected coverage.
4.  **Residual diagnostics** — histogram, Q-Q, heteroscedasticity.
5.  **Pareto fronts** — 2-D multi-objective with shading.
6.  **Benchmark comparison** — bar chart of metrics across models.
7.  **Training curves** — loss/metric vs epoch.

All functions follow the pattern:

    ``plot_xxx(data, ..., save_path=None, ax=None) → fig``

If ``ax`` is provided, the plot is drawn on that axis (for subplots);
otherwise a new figure is created.  ``save_path`` triggers a high-DPI
save.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Lazy matplotlib import (avoid backend issues on headless servers)
_MPL_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    _MPL_AVAILABLE = True
except ImportError:
    logger.warning("matplotlib not available; visualiser disabled.")


# ═══════════════════════════════════════════════════════════════════════
# 0.  STYLE
# ═══════════════════════════════════════════════════════════════════════

# Colour palette (CB-friendly, 8 colours)
PALETTE = [
    "#0077BB",  # blue
    "#CC3311",  # red
    "#009988",  # teal
    "#EE7733",  # orange
    "#33BBEE",  # cyan
    "#EE3377",  # magenta
    "#BBBBBB",  # grey
    "#000000",  # black
]


def _setup_style() -> None:
    """Apply consistent publication style."""
    if not _MPL_AVAILABLE:
        return
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig, path: Optional[Union[str, Path]]) -> None:
    if path is not None and _MPL_AVAILABLE:
        fig.savefig(str(path), dpi=300, bbox_inches="tight")
        logger.info(f"Figure saved to {path}")


def _get_ax(ax, figsize=(6, 5)):
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════
# 1.  PARITY PLOT
# ═══════════════════════════════════════════════════════════════════════

def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    component_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Predicted vs true scatter with optional uncertainty shading.

    If ``y_true`` is ``[N, C]`` with C > 1, produces one subplot
    per component.
    """
    if not _MPL_AVAILABLE:
        return None
    _setup_style()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        if y_std is not None:
            y_std = np.asarray(y_std).reshape(-1, 1)

    C = y_true.shape[1]
    if component_names is None:
        component_names = [f"Component {i}" for i in range(C)]

    fig, axes = plt.subplots(1, C, figsize=(5 * C, 5), squeeze=False)

    for i in range(C):
        ax = axes[0, i]
        yt, yp = y_true[:, i], y_pred[:, i]

        ax.scatter(yt, yp, s=12, alpha=0.5, color=PALETTE[0], edgecolors="none")

        # Perfect line
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        margin = 0.05 * (hi - lo + 1e-8)
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "k--", alpha=0.5, linewidth=1, label="y = x")

        # UQ band
        if y_std is not None:
            ys = y_std[:, i] if y_std.ndim > 1 else y_std
            order = np.argsort(yt)
            ax.fill_between(
                yt[order],
                (yp - 1.96 * ys)[order],
                (yp + 1.96 * ys)[order],
                alpha=0.15, color=PALETTE[0], label="95% CI",
            )

        # R² annotation
        from .metrics import r2_score as _r2
        r2 = _r2(yt, yp)
        ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes,
                va="top", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

        ax.set_xlabel(f"True {component_names[i]} (mol/kg)")
        ax.set_ylabel(f"Predicted {component_names[i]} (mol/kg)")
        ax.set_title(component_names[i])
        ax.legend(loc="lower right", framealpha=0.8)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 2.  ISOTHERM PLOT
# ═══════════════════════════════════════════════════════════════════════

def plot_isotherms(
    pressures: np.ndarray,
    loadings_true: np.ndarray,
    loadings_pred: Optional[np.ndarray] = None,
    loadings_std: Optional[np.ndarray] = None,
    component_names: Optional[List[str]] = None,
    mof_id: Optional[str] = None,
    log_scale: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Adsorption isotherms: loading vs pressure.
    """
    if not _MPL_AVAILABLE:
        return None
    _setup_style()

    loadings_true = np.asarray(loadings_true)
    if loadings_true.ndim == 1:
        loadings_true = loadings_true.reshape(-1, 1)
    C = loadings_true.shape[1]

    if component_names is None:
        component_names = [f"Component {i}" for i in range(C)]

    fig, axes = plt.subplots(1, C, figsize=(5 * C, 4), squeeze=False)

    for i in range(C):
        ax = axes[0, i]
        P = np.asarray(pressures)
        qt = loadings_true[:, i]

        ax.plot(P, qt, "o-", color=PALETTE[0], markersize=4, label="True")

        if loadings_pred is not None:
            qp = np.asarray(loadings_pred)
            if qp.ndim > 1:
                qp = qp[:, i]
            ax.plot(P, qp, "s--", color=PALETTE[1], markersize=4, label="Predicted")

            if loadings_std is not None:
                qs = np.asarray(loadings_std)
                if qs.ndim > 1:
                    qs = qs[:, i]
                ax.fill_between(P, qp - 1.96 * qs, qp + 1.96 * qs,
                                alpha=0.15, color=PALETTE[1])

        if log_scale and P.min() > 0:
            ax.set_xscale("log")

        ax.set_xlabel("Pressure (bar)")
        ax.set_ylabel(f"{component_names[i]} Loading (mol/kg)")
        ttl = component_names[i]
        if mof_id:
            ttl += f" — {mof_id}"
        ax.set_title(ttl)
        ax.legend(framealpha=0.8)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 3.  CALIBRATION CURVE
# ═══════════════════════════════════════════════════════════════════════

def plot_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 15,
    title: str = "UQ Calibration",
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Calibration curve: observed coverage vs expected confidence level.
    """
    if not _MPL_AVAILABLE:
        return None
    _setup_style()

    from .metrics import calibration_error
    expected, observed, ece = calibration_error(
        np.asarray(y_true).ravel(),
        np.asarray(y_pred).ravel(),
        np.asarray(y_std).ravel(),
        n_bins,
    )

    fig, ax = _get_ax(ax, figsize=(5, 5))

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.plot(expected, observed, "o-", color=PALETTE[0], markersize=5,
            label=f"Model (ECE = {ece:.3f})")
    ax.fill_between(expected, expected - 0.05, expected + 0.05,
                    alpha=0.1, color="grey", label="± 0.05 band")

    ax.set_xlabel("Expected coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", framealpha=0.8)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 4.  RESIDUAL DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Three-panel residual diagnostics: histogram, Q-Q, residual vs ŷ.
    """
    if not _MPL_AVAILABLE:
        return None
    _setup_style()

    residuals = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
    yp = np.asarray(y_pred).ravel()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram
    ax1.hist(residuals, bins=50, density=True, alpha=0.7, color=PALETTE[0])
    x_g = np.linspace(residuals.min(), residuals.max(), 200)
    from scipy.stats import norm
    ax1.plot(x_g, norm.pdf(x_g, residuals.mean(), residuals.std()),
             "k-", linewidth=1.5, label="Gaussian fit")
    ax1.set_xlabel("Residual")
    ax1.set_ylabel("Density")
    ax1.set_title("Residual Distribution")
    ax1.legend()

    # Q-Q plot
    from scipy.stats import probplot
    probplot(residuals, plot=ax2)
    ax2.set_title("Q-Q Plot")

    # Residual vs predicted
    ax3.scatter(yp, residuals, s=8, alpha=0.4, color=PALETTE[0], edgecolors="none")
    ax3.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Residual")
    ax3.set_title("Residual vs Predicted")

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 5.  PARETO FRONT
# ═══════════════════════════════════════════════════════════════════════

def plot_pareto(
    objectives: np.ndarray,
    pareto_idx: Optional[np.ndarray] = None,
    labels: Tuple[str, str] = ("Objective 1", "Objective 2"),
    title: str = "Pareto Front",
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    2-D scatter with Pareto front highlighted.
    """
    if not _MPL_AVAILABLE:
        return None
    _setup_style()

    objectives = np.asarray(objectives)
    fig, ax = _get_ax(ax, figsize=(6, 5))

    if pareto_idx is None:
        from .metrics import pareto_front_indices
        pareto_idx = pareto_front_indices(objectives)

    non_pareto = np.setdiff1d(np.arange(len(objectives)), pareto_idx)

    ax.scatter(objectives[non_pareto, 0], objectives[non_pareto, 1],
               s=15, alpha=0.3, color=PALETTE[6], label="Dominated")

    pf = objectives[pareto_idx]
    order = np.argsort(pf[:, 0])
    ax.plot(pf[order, 0], pf[order, 1], "o-", color=PALETTE[1],
            markersize=6, linewidth=1.5, label="Pareto front")

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    ax.legend(framealpha=0.8)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 6.  BENCHMARK BAR CHART
# ═══════════════════════════════════════════════════════════════════════

def plot_benchmark(
    comparison: Dict[str, Dict[str, float]],
    metric: str = "rmse",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Grouped bar chart comparing models on a single metric.

    Parameters
    ----------
    comparison : ``{model_name: {metric: value}}`` from
                 ``Benchmarker.comparison_table()``.
    metric     : Which metric to plot.
    """
    if not _MPL_AVAILABLE:
        return None
    _setup_style()

    names = list(comparison.keys())
    values = [comparison[n].get(metric, 0.0) for n in names]

    fig, ax = _get_ax(ax, figsize=(max(4, len(names) * 1.2), 5))

    bars = ax.bar(names, values, color=[PALETTE[i % len(PALETTE)] for i in range(len(names))],
                  alpha=0.85, edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"Model Comparison — {metric.upper()}")
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 7.  TRAINING CURVES
# ═══════════════════════════════════════════════════════════════════════

def plot_training_curves(
    history: Sequence[Dict[str, float]],
    keys: Optional[List[str]] = None,
    title: str = "Training History",
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Plot loss/metric curves over epochs.

    Parameters
    ----------
    history : List of per-epoch metric dicts (from Trainer.history).
    keys    : Metric keys to plot (default: auto-detect loss keys).
    """
    if not _MPL_AVAILABLE or len(history) == 0:
        return None
    _setup_style()

    if keys is None:
        keys = [k for k in history[0] if "loss" in k.lower() or "lr" in k.lower()]
        if not keys:
            keys = list(history[0].keys())[:4]

    n_keys = len(keys)
    fig, axes = plt.subplots(1, n_keys, figsize=(5 * n_keys, 4), squeeze=False)

    epochs = np.arange(len(history))

    for j, key in enumerate(keys):
        ax = axes[0, j]
        values = [h.get(key, float("nan")) for h in history]
        ax.plot(epochs, values, color=PALETTE[j % len(PALETTE)], linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.set_title(key)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 8.  COMPOSITE RESULT VISUALISER
# ═══════════════════════════════════════════════════════════════════════

class ResultVisualizer:
    """
    High-level interface that generates a standard set of figures.

    Example
    ───────
    >>> vis = ResultVisualizer(save_dir="results/figures")
    >>> vis.full_report(y_true, y_pred, y_std, history=trainer.history)
    """

    def __init__(
        self,
        save_dir: Optional[Union[str, Path]] = None,
        component_names: Optional[List[str]] = None,
    ):
        self.save_dir = Path(save_dir) if save_dir else None
        self.component_names = component_names or ["CO₂", "N₂", "H₂O"]

        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Optional[Path]:
        return self.save_dir / name if self.save_dir else None

    def full_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: Optional[np.ndarray] = None,
        history: Optional[Sequence[Dict]] = None,
        benchmark: Optional[Dict[str, Dict[str, float]]] = None,
        objectives: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Generate all standard plots.

        Returns dict of figure name → matplotlib Figure.
        """
        figs: Dict[str, Any] = {}

        figs["parity"] = plot_parity(
            y_true, y_pred, y_std,
            component_names=self.component_names,
            save_path=self._path("parity.png"),
        )

        figs["residuals"] = plot_residuals(
            y_true, y_pred,
            save_path=self._path("residuals.png"),
        )

        if y_std is not None:
            figs["calibration"] = plot_calibration(
                y_true, y_pred, y_std,
                save_path=self._path("calibration.png"),
            )

        if history is not None:
            figs["training"] = plot_training_curves(
                history,
                save_path=self._path("training_curves.png"),
            )

        if benchmark is not None:
            figs["benchmark"] = plot_benchmark(
                benchmark,
                save_path=self._path("benchmark.png"),
            )

        if objectives is not None and objectives.shape[1] == 2:
            figs["pareto"] = plot_pareto(
                objectives,
                save_path=self._path("pareto.png"),
            )

        if _MPL_AVAILABLE:
            plt.close("all")

        return figs


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "plot_parity",
    "plot_isotherms",
    "plot_calibration",
    "plot_residuals",
    "plot_pareto",
    "plot_benchmark",
    "plot_training_curves",
    "ResultVisualizer",
]