"""
Benchmarking: compare UC-TPNO against baseline models.

Baselines
─────────
1.  **IAST** — Ideal Adsorbed Solution Theory (``mixture/iast.py``).
2.  **Linear / MLP** — simple regression on MOF descriptors.
3.  **GCMC** — ground-truth simulation data (treated as reference).
4.  **GP** — Gaussian Process surrogate (``process/surrogate.py``).

The ``Benchmarker`` class runs each model on the same held-out test
set, computes standardised metrics (from ``metrics.py``), and
produces a comparison table + statistical significance tests.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats as sp_stats

from .metrics import (
    compute_regression_metrics,
    compute_uncertainty_metrics,
    mae,
    rmse,
    r2_score,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1.  BASELINE WRAPPERS
# ═══════════════════════════════════════════════════════════════════════

class BaselineModel:
    """
    Uniform interface for any baseline model.

    Parameters
    ----------
    name       : Human-readable name (e.g. ``'IAST'``).
    predict_fn : ``(X_test) → y_pred [N, C]``  or
                 ``(X_test) → (y_pred, y_std)`` for models with UQ.
    has_uq     : Whether ``predict_fn`` returns uncertainty.
    """

    def __init__(
        self,
        name: str,
        predict_fn: Callable,
        has_uq: bool = False,
    ):
        self.name = name
        self.predict_fn = predict_fn
        self.has_uq = has_uq

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Returns
        -------
        (y_pred, y_std) — y_std is None if ``has_uq=False``.
        """
        start = time.perf_counter()
        result = self.predict_fn(X)
        elapsed = time.perf_counter() - start

        if self.has_uq and isinstance(result, tuple) and len(result) == 2:
            y_pred, y_std = result
        else:
            y_pred = result if not isinstance(result, tuple) else result[0]
            y_std = None

        return np.asarray(y_pred), (np.asarray(y_std) if y_std is not None else None)


class LinearBaseline:
    """
    Ordinary least-squares linear regression baseline.
    """

    def __init__(self):
        self._W: Optional[np.ndarray] = None
        self._b: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Solve via normal equations with regularisation
        lam = 1e-4
        XtX = X.T @ X + lam * np.eye(X.shape[1])
        Xty = X.T @ y
        self._W = np.linalg.solve(XtX, Xty)
        self._b = y.mean(0) - X.mean(0) @ self._W

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self._W + self._b


class MLPBaseline:
    """
    Simple 2-layer MLP baseline (uses NeuralSurrogate internally).
    """

    def __init__(self, hidden_dims: List[int] = None):
        self.hidden_dims = hidden_dims or [64, 32]
        self._surr = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from src.models.process.surrogate import NeuralSurrogate, NeuralSurrogateConfig

        target_names = [f"target_{i}" for i in range(y.shape[1] if y.ndim > 1 else 1)]
        self._surr = NeuralSurrogate(
            NeuralSurrogateConfig(hidden_dims=self.hidden_dims, epochs=300),
            target_names=target_names,
        )
        self._surr.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = self._surr.predict(X)
        cols = [pred[name]["mean"] for name in self._surr.target_names]
        return np.column_stack(cols) if len(cols) > 1 else cols[0]


# ═══════════════════════════════════════════════════════════════════════
# 2.  BENCHMARKER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """Results for one model on one test set."""

    name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    y_pred: Optional[np.ndarray] = None
    y_std: Optional[np.ndarray] = None
    elapsed_s: float = 0.0

    def __repr__(self) -> str:
        r2 = self.metrics.get("r2", float("nan"))
        rmse_v = self.metrics.get("rmse", float("nan"))
        return f"BenchmarkResult({self.name}: R²={r2:.4f}, RMSE={rmse_v:.4f}, t={self.elapsed_s:.2f}s)"


class Benchmarker:
    """
    Run standardised benchmarks across multiple models.

    Parameters
    ----------
    y_true          : ``[N, C]`` ground-truth test targets.
    component_names : Per-component labels.

    Example
    ───────
    >>> bench = Benchmarker(y_true=y_test, component_names=['CO2','N2','H2O'])
    >>> bench.add_model("TPNO", tpno_predict_fn, has_uq=True)
    >>> bench.add_model("IAST", iast_predict_fn)
    >>> bench.add_baseline_linear(X_train, y_train)
    >>> results = bench.run(X_test)
    >>> table = bench.comparison_table(results)
    """

    def __init__(
        self,
        y_true: np.ndarray,
        component_names: Optional[List[str]] = None,
    ):
        self.y_true = np.asarray(y_true, dtype=np.float64)
        self.component_names = component_names
        self.models: List[BaselineModel] = []

    def add_model(
        self,
        name: str,
        predict_fn: Callable,
        has_uq: bool = False,
    ) -> None:
        """Register a model for benchmarking."""
        self.models.append(BaselineModel(name, predict_fn, has_uq))

    def add_baseline_linear(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit and register a linear regression baseline."""
        lin = LinearBaseline()
        lin.fit(X_train, y_train)
        self.add_model("Linear", lin.predict)

    def add_baseline_mlp(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hidden_dims: Optional[List[int]] = None,
    ) -> None:
        """Fit and register an MLP baseline."""
        mlp = MLPBaseline(hidden_dims)
        mlp.fit(X_train, y_train)
        self.add_model("MLP", mlp.predict)

    def run(self, X_test: np.ndarray) -> List[BenchmarkResult]:
        """
        Evaluate all registered models on X_test.

        Parameters
        ----------
        X_test : ``[N, d]`` test features.

        Returns
        -------
        List of ``BenchmarkResult``, one per model.
        """
        results = []
        for model in self.models:
            start = time.perf_counter()
            y_pred, y_std = model.predict(X_test)
            elapsed = time.perf_counter() - start

            # Regression metrics
            metrics = compute_regression_metrics(
                self.y_true, y_pred,
                prefix="",
                component_names=self.component_names,
            )

            # UQ metrics
            if y_std is not None:
                metrics.update(compute_uncertainty_metrics(
                    self.y_true, y_pred, y_std,
                ))

            result = BenchmarkResult(
                name=model.name,
                metrics=metrics,
                y_pred=y_pred,
                y_std=y_std,
                elapsed_s=elapsed,
            )
            results.append(result)
            logger.info(f"Benchmark {model.name}: R²={metrics.get('r2',0):.4f}, "
                        f"RMSE={metrics.get('rmse',0):.4f}, t={elapsed:.2f}s")

        return results

    # ── Comparison table ─────────────────────────────────────────

    def comparison_table(
        self,
        results: List[BenchmarkResult],
        metrics_to_show: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Build a model → metrics comparison dict.

        Parameters
        ----------
        results         : From ``run()``.
        metrics_to_show : Subset of metric keys (default: core set).

        Returns
        -------
        ``{model_name: {metric: value}}``.
        """
        if metrics_to_show is None:
            metrics_to_show = ["mae", "rmse", "r2", "mape", "max_abs_error"]

        table = {}
        for r in results:
            row = {k: r.metrics.get(k, float("nan")) for k in metrics_to_show}
            row["time_s"] = r.elapsed_s
            table[r.name] = row

        return table

    # ── Statistical significance ─────────────────────────────────

    def paired_test(
        self,
        result_a: BenchmarkResult,
        result_b: BenchmarkResult,
        test: str = "wilcoxon",
    ) -> Dict[str, Any]:
        """
        Paired significance test between two models.

        Uses per-sample absolute errors as paired observations.

        Parameters
        ----------
        result_a, result_b : Results to compare.
        test : ``'wilcoxon'`` or ``'ttest'``.

        Returns
        -------
        Dict with ``"statistic"``, ``"p_value"``, ``"significant"``
        (at α = 0.05), and ``"better_model"``.
        """
        err_a = np.abs(self.y_true.ravel() - result_a.y_pred.ravel())
        err_b = np.abs(self.y_true.ravel() - result_b.y_pred.ravel())

        if test == "wilcoxon":
            diff = err_a - err_b
            diff = diff[diff != 0]
            if len(diff) < 10:
                return {"statistic": 0, "p_value": 1.0, "significant": False, "better_model": "tie"}
            stat, p = sp_stats.wilcoxon(diff)
        elif test == "ttest":
            stat, p = sp_stats.ttest_rel(err_a, err_b)
        else:
            raise ValueError(f"Unknown test '{test}'.")

        better = result_a.name if np.mean(err_a) < np.mean(err_b) else result_b.name

        return {
            "test": test,
            "statistic": float(stat),
            "p_value": float(p),
            "significant": p < 0.05,
            "better_model": better,
            "mean_err_a": float(np.mean(err_a)),
            "mean_err_b": float(np.mean(err_b)),
        }

    def pairwise_tests(
        self,
        results: List[BenchmarkResult],
    ) -> Dict[str, Dict[str, Any]]:
        """Run pairwise significance tests between all models."""
        tests = {}
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                key = f"{results[i].name}_vs_{results[j].name}"
                tests[key] = self.paired_test(results[i], results[j])
        return tests

    # ── Summary ──────────────────────────────────────────────────

    def summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Return a structured summary of the benchmark."""
        table = self.comparison_table(results)
        best_r2 = max(results, key=lambda r: r.metrics.get("r2", -1))
        best_rmse = min(results, key=lambda r: r.metrics.get("rmse", 1e10))

        return {
            "table": table,
            "best_r2": best_r2.name,
            "best_rmse": best_rmse.name,
            "n_models": len(results),
            "n_test": len(self.y_true),
        }


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "BaselineModel",
    "LinearBaseline",
    "MLPBaseline",
    "BenchmarkResult",
    "Benchmarker",
]