"""
Process surrogate models for fast PVSA emulation.

Running a full PVSA cycle simulation for every MOF candidate is
expensive.  This module builds **surrogate models** that learn the
mapping from MOF descriptors + operating conditions → process KPIs,
enabling rapid screening of large MOF databases.

Two surrogate approaches
────────────────────────
1.  **GPSurrogate** — Gaussian Process regression with an RBF kernel.
    Provides calibrated uncertainty and works well with small
    datasets (< 1 k training points).  Uses ``scikit-learn`` or
    ``scipy`` — no deep-learning dependencies.

2.  **NeuralSurrogate** — small feedforward MLP for larger datasets.
    Faster inference than GP at scale (> 1 k points), but needs more
    training data and doesn't naturally provide uncertainty (combine
    with SWAG or ensemble for UQ).

Both surrogates implement the same ``predict()`` interface:

    ``(X_descriptors, conditions) → Dict["kpi_name": predictions]``

Integration
───────────
*  ``pvsa.py`` generates training data by running PVSA cycles.
*  ``kpi.py`` defines the target KPIs.
*  ``active/acquisition.py`` uses the surrogate as the objective
   model in the Bayesian optimisation loop.

References
──────────
[1] Rasmussen & Williams (2006). Gaussian Processes for Machine
    Learning. MIT Press.
[2] Farmahini et al. (2018). Performance-Based Screening of Porous
    Materials for Carbon Capture. Chemical Reviews.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import linalg, optimize

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GPSurrogateConfig:
    """
    Gaussian Process surrogate hyperparameters.

    Attributes
    ──────────
    kernel         : ``'rbf'``, ``'matern32'``, or ``'matern52'``.
    length_scale   : Initial RBF length scale (or ``'auto'``).
    noise          : Observation noise variance σ²_n.
    optimise_hyper : Optimise kernel hyperparameters via MLL.
    n_restarts     : Random restarts for hyperparameter optimisation.
    alpha          : Tikhonov regularisation added to diagonal.
    normalise_X    : Standardise input features.
    normalise_Y    : Standardise targets.
    """

    kernel: str = "rbf"
    length_scale: Union[str, float] = "auto"
    noise: float = 1e-4
    optimise_hyper: bool = True
    n_restarts: int = 5
    alpha: float = 1e-8
    normalise_X: bool = True
    normalise_Y: bool = True


@dataclass
class NeuralSurrogateConfig:
    """
    Neural surrogate hyperparameters.

    Attributes
    ──────────
    hidden_dims   : List of hidden layer widths.
    activation    : ``'relu'``, ``'silu'``, or ``'gelu'``.
    dropout       : Dropout rate.
    lr            : Learning rate.
    epochs        : Training epochs (= L-BFGS maxiter).
    batch_size    : Mini-batch size.
    weight_decay  : L2 regularisation.
    early_stop    : Patience for early stopping (0 = disabled).
    """

    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    activation: str = "silu"
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 500
    batch_size: int = 64
    weight_decay: float = 1e-4
    early_stop: int = 30


# ═══════════════════════════════════════════════════════════════════════
# 2.  KERNEL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def _sq_dist(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix [N1, N2]."""
    X1sq = (X1 ** 2).sum(axis=1, keepdims=True)
    X2sq = (X2 ** 2).sum(axis=1, keepdims=True)
    return (X1sq + X2sq.T - 2.0 * X1 @ X2.T).clip(min=0.0)
    # clip(min=0): floating-point cancellation can produce tiny negatives
    # on the diagonal; clipping keeps kernels valid without affecting results.


def rbf_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    length_scale: float,
    signal_var: float = 1.0,
) -> np.ndarray:
    """RBF (squared exponential) kernel: k(x,x') = σ² exp(-‖x-x'‖²/(2l²))."""
    sq_dist = _sq_dist(X1, X2)
    return signal_var * np.exp(-0.5 * sq_dist / length_scale ** 2)


def matern32_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    length_scale: float,
    signal_var: float = 1.0,
) -> np.ndarray:
    """Matérn 3/2 kernel."""
    dist = np.sqrt(_sq_dist(X1, X2).clip(min=1e-30))
    r = np.sqrt(3.0) * dist / length_scale
    return signal_var * (1.0 + r) * np.exp(-r)


def matern52_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    length_scale: float,
    signal_var: float = 1.0,
) -> np.ndarray:
    """Matérn 5/2 kernel."""
    dist = np.sqrt(_sq_dist(X1, X2).clip(min=1e-30))
    r = np.sqrt(5.0) * dist / length_scale
    return signal_var * (1.0 + r + r ** 2 / 3.0) * np.exp(-r)


_KERNEL_FNS = {
    "rbf": rbf_kernel,
    "matern32": matern32_kernel,
    "matern52": matern52_kernel,
}


# ═══════════════════════════════════════════════════════════════════════
# 3.  GP SURROGATE
# ═══════════════════════════════════════════════════════════════════════

class GPSurrogate:
    """
    Gaussian Process surrogate for process KPI prediction.

    Fits independent GPs for each target KPI (multi-output via
    independent single-output GPs).

    Parameters
    ----------
    config     : ``GPSurrogateConfig``.
    target_names: List of KPI names to predict.

    Example
    ───────
    >>> gp = GPSurrogate(target_names=["purity", "recovery", "energy_MJ_ton"])
    >>> gp.fit(X_train, Y_train)  # Y: [N, 3]
    >>> pred = gp.predict(X_test)
    >>> pred["purity"]["mean"], pred["purity"]["std"]
    """

    def __init__(
        self,
        config: Optional[Union[GPSurrogateConfig, Dict]] = None,
        target_names: Optional[List[str]] = None,
    ):
        if config is None:
            config = GPSurrogateConfig()
        elif isinstance(config, dict):
            config = GPSurrogateConfig(**{
                k: v for k, v in config.items()
                if k in GPSurrogateConfig.__dataclass_fields__
            })
        self.config = config
        self.target_names = target_names or ["purity", "recovery", "energy_MJ_ton"]

        self._fitted = False
        self._X_train: Optional[np.ndarray] = None
        self._Y_trains: Dict[str, np.ndarray] = {}
        self._L_caches: Dict[str, np.ndarray] = {}    # Cholesky of K + σ²I
        self._alpha_caches: Dict[str, np.ndarray] = {} # L⁻ᵀ L⁻¹ y

        # Normalisation
        self._X_mean: Optional[np.ndarray] = None
        self._X_std: Optional[np.ndarray] = None
        self._Y_means: Dict[str, float] = {}
        self._Y_stds: Dict[str, float] = {}

        # Kernel params per target
        self._length_scales: Dict[str, float] = {}
        self._signal_vars: Dict[str, float] = {}
        self._noise_vars: Dict[str, float] = {}

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _kernel(self, X1: np.ndarray, X2: np.ndarray, target: str) -> np.ndarray:
        fn = _KERNEL_FNS.get(self.config.kernel, rbf_kernel)
        return fn(
            X1, X2,
            length_scale=self._length_scales.get(target, 1.0),
            signal_var=self._signal_vars.get(target, 1.0),
        )

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> Dict[str, float]:
        """
        Fit independent GPs for each target.

        Parameters
        ----------
        X : ``[N, d]`` input features.
        Y : ``[N, n_targets]`` target KPI values (columns aligned
            with ``target_names``).

        Returns
        -------
        Dict of target → negative log marginal likelihood.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        assert Y.shape[1] == len(self.target_names), \
            f"Y has {Y.shape[1]} cols but {len(self.target_names)} targets"

        # Normalise X
        if self.config.normalise_X:
            self._X_mean = X.mean(axis=0)
            self._X_std = X.std(axis=0).clip(min=1e-8)
            X = (X - self._X_mean) / self._X_std
        self._X_train = X

        nll_results = {}

        for i, name in enumerate(self.target_names):
            y = Y[:, i].copy()

            # Normalise Y
            if self.config.normalise_Y:
                self._Y_means[name] = float(y.mean())
                self._Y_stds[name] = max(float(y.std()), 1e-8)
                y = (y - self._Y_means[name]) / self._Y_stds[name]
            self._Y_trains[name] = y

            # Initial length scale
            if self.config.length_scale == "auto":
                ls0 = float(np.median(np.sqrt(_sq_dist(X[:50], X[:50]).ravel() + 1e-12)))
                ls0 = max(ls0, 0.1)
            else:
                ls0 = float(self.config.length_scale)

            # Optimise hyperparameters
            if self.config.optimise_hyper:
                ls, sv, nv, nll = self._optimise_hyperparams(X, y, ls0, name)
            else:
                ls, sv, nv = ls0, 1.0, self.config.noise
                nll = 0.0

            self._length_scales[name] = ls
            self._signal_vars[name] = sv
            self._noise_vars[name] = nv

            # Cache Cholesky decomposition
            K = self._kernel(X, X, name)
            K += (nv + self.config.alpha) * np.eye(len(X))
            L = linalg.cholesky(K, lower=True)
            alpha = linalg.cho_solve((L, True), y)

            self._L_caches[name] = L
            self._alpha_caches[name] = alpha
            nll_results[name] = nll

        self._fitted = True
        logger.info(f"GPSurrogate fitted on {X.shape[0]} points, {len(self.target_names)} targets.")
        return nll_results

    def _optimise_hyperparams(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ls0: float,
        target: str,
    ) -> Tuple[float, float, float, float]:
        """Optimise (length_scale, signal_var, noise_var) via MLL."""
        N = len(X)
        kernel_fn = _KERNEL_FNS.get(self.config.kernel, rbf_kernel)

        def neg_log_marginal(log_params):
            ls = np.exp(log_params[0])
            sv = np.exp(log_params[1])
            nv = np.exp(log_params[2])

            K = kernel_fn(X, X, ls, sv) + (nv + self.config.alpha) * np.eye(N)
            try:
                L = linalg.cholesky(K, lower=True)
            except linalg.LinAlgError:
                return 1e10

            alpha = linalg.cho_solve((L, True), y)
            nll = (0.5 * y @ alpha
                   + np.sum(np.log(np.diag(L)))
                   + 0.5 * N * np.log(2 * np.pi))
            return float(nll)

        best = None
        best_nll = float("inf")
        rng = np.random.RandomState(42)

        for _ in range(self.config.n_restarts):
            x0 = np.array([
                np.log(ls0) + rng.randn() * 0.5,
                rng.randn() * 0.5,
                np.log(self.config.noise) + rng.randn() * 0.5,
            ])
            try:
                res = optimize.minimize(neg_log_marginal, x0, method="L-BFGS-B")
                if res.fun < best_nll:
                    best_nll = res.fun
                    best = res.x
            except Exception:
                continue

        if best is None:
            return ls0, 1.0, self.config.noise, 0.0

        return np.exp(best[0]), np.exp(best[1]), np.exp(best[2]), best_nll

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Predict KPIs with uncertainty.

        Parameters
        ----------
        X          : ``[M, d]`` test features.
        return_std : Whether to compute predictive std.

        Returns
        -------
        ``{target_name: {"mean": [M], "std": [M]}}``.
        """
        if not self._fitted:
            raise RuntimeError("GPSurrogate not fitted.")

        X = np.asarray(X, dtype=np.float64)
        if self.config.normalise_X:
            X = (X - self._X_mean) / self._X_std

        results = {}
        for name in self.target_names:
            K_star = self._kernel(X, self._X_train, name)
            mean = K_star @ self._alpha_caches[name]

            out: Dict[str, np.ndarray] = {}

            # De-normalise
            if self.config.normalise_Y:
                mean = mean * self._Y_stds[name] + self._Y_means[name]

            out["mean"] = mean

            if return_std:
                L = self._L_caches[name]
                V = linalg.solve_triangular(L, K_star.T, lower=True)
                K_ss = self._kernel(X, X, name)
                var = np.diag(K_ss) - (V ** 2).sum(axis=0)
                var = var.clip(min=0.0)

                if self.config.normalise_Y:
                    var *= self._Y_stds[name] ** 2

                out["std"] = np.sqrt(var)

            results[name] = out

        return results

    def predict_flat(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Return just the mean predictions as a flat dict."""
        pred = self.predict(X, return_std=False)
        return {name: pred[name]["mean"] for name in self.target_names}

    @property
    def n_train(self) -> int:
        return self._X_train.shape[0] if self._X_train is not None else 0

    def summary(self) -> Dict[str, Any]:
        return {
            "fitted": self._fitted,
            "n_train": self.n_train,
            "targets": self.target_names,
            "kernel": self.config.kernel,
            "length_scales": dict(self._length_scales),
        }


# ═══════════════════════════════════════════════════════════════════════
# 4.  NEURAL SURROGATE
# ═══════════════════════════════════════════════════════════════════════

class NeuralSurrogate:
    """
    MLP surrogate for process KPI prediction (pure numpy/scipy
    implementation for portability; torch version in training/).

    This is a lightweight 2-layer MLP with ReLU activations trained
    via L-BFGS, suitable for < 10 k training points and fast
    inference.  For larger scale, wrap a PyTorch ``nn.Module`` with
    the same ``predict()`` interface.

    Parameters
    ----------
    config       : ``NeuralSurrogateConfig``.
    target_names : KPI names.
    """

    def __init__(
        self,
        config: Optional[Union[NeuralSurrogateConfig, Dict]] = None,
        target_names: Optional[List[str]] = None,
    ):
        if config is None:
            config = NeuralSurrogateConfig()
        elif isinstance(config, dict):
            config = NeuralSurrogateConfig(**{
                k: v for k, v in config.items()
                if k in NeuralSurrogateConfig.__dataclass_fields__
            })
        self.config = config
        self.target_names = target_names or ["purity", "recovery", "energy_MJ_ton"]

        self._fitted = False
        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []

        # Normalisation stats
        self._X_mean: Optional[np.ndarray] = None
        self._X_std: Optional[np.ndarray] = None
        self._Y_mean: Optional[np.ndarray] = None
        self._Y_std: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """MLP forward pass."""
        h = X
        for i, (W, b) in enumerate(zip(self._weights, self._biases)):
            h = h @ W + b
            if i < len(self._weights) - 1:
                h = self._relu(h)
        return h

    def _pack_params(self) -> np.ndarray:
        parts = []
        for W, b in zip(self._weights, self._biases):
            parts.append(W.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts)

    def _unpack_params(self, flat: np.ndarray, dims: List[Tuple[int, int]]) -> None:
        self._weights = []
        self._biases = []
        offset = 0
        for in_d, out_d in dims:
            n_w = in_d * out_d
            W = flat[offset:offset + n_w].reshape(in_d, out_d)
            offset += n_w
            b = flat[offset:offset + out_d]
            offset += out_d
            self._weights.append(W)
            self._biases.append(b)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train the MLP via L-BFGS.

        Parameters
        ----------
        X : ``[N, d]`` inputs.
        Y : ``[N, n_targets]`` targets.

        Returns
        -------
        Dict with ``"final_loss"`` and ``"n_train"``.

        FIX: removed dead ``loss_and_grad`` function that was defined
        but never passed to ``optimize.minimize`` (which used
        ``_loss_only`` with ``jac=False`` instead).  The dead code
        also called ``optimize.approx_fprime`` internally, which would
        have made gradient evaluation O(n_params) × slower than
        needed.  The optimizer now explicitly uses ``jac=False`` and
        scipy's built-in finite-difference gradient.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # Normalise
        self._X_mean = X.mean(0)
        self._X_std = X.std(0).clip(min=1e-8)
        self._Y_mean = Y.mean(0)
        self._Y_std = Y.std(0).clip(min=1e-8)

        Xn = (X - self._X_mean) / self._X_std
        Yn = (Y - self._Y_mean) / self._Y_std

        # Architecture
        d_in = Xn.shape[1]
        d_out = Yn.shape[1]
        hidden = self.config.hidden_dims
        layer_dims = [(d_in, hidden[0])]
        for i in range(len(hidden) - 1):
            layer_dims.append((hidden[i], hidden[i + 1]))
        layer_dims.append((hidden[-1], d_out))

        # Xavier init
        rng = np.random.RandomState(42)
        self._weights = []
        self._biases = []
        for in_d, out_d in layer_dims:
            scale = np.sqrt(2.0 / (in_d + out_d))
            self._weights.append(rng.randn(in_d, out_d) * scale)
            self._biases.append(np.zeros(out_d))

        wd = self.config.weight_decay
        x0 = self._pack_params()

        # FIX: removed the dead loss_and_grad() function that was never
        # called.  Using _loss_only with jac=False (scipy finite-diff gradient).
        result = optimize.minimize(
            lambda p: self._loss_only(p, Xn, Yn, layer_dims, wd),
            x0,
            method="L-BFGS-B",
            jac=False,
            options={"maxiter": self.config.epochs, "ftol": 1e-10},
        )

        self._unpack_params(result.x, layer_dims)
        self._fitted = True

        logger.info(f"NeuralSurrogate fitted: loss={result.fun:.6f}, N={X.shape[0]}")
        return {"final_loss": float(result.fun), "n_train": X.shape[0]}

    def _loss_only(
        self,
        flat: np.ndarray,
        Xn: np.ndarray,
        Yn: np.ndarray,
        dims: List[Tuple[int, int]],
        wd: float,
    ) -> float:
        self._unpack_params(flat, dims)
        pred = self._forward(Xn)
        loss = 0.5 * np.mean((pred - Yn) ** 2)
        if wd > 0:
            for W in self._weights:
                loss += 0.5 * wd * np.sum(W ** 2)
        return float(loss)

    def predict(
        self,
        X: np.ndarray,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Predict KPIs.

        Parameters
        ----------
        X : ``[M, d]`` features.

        Returns
        -------
        ``{target_name: {"mean": [M]}}``.
        """
        if not self._fitted:
            raise RuntimeError("NeuralSurrogate not fitted.")

        X = np.asarray(X, dtype=np.float64)
        Xn = (X - self._X_mean) / self._X_std
        Yn = self._forward(Xn)

        # De-normalise
        Y = Yn * self._Y_std + self._Y_mean

        results = {}
        for i, name in enumerate(self.target_names):
            results[name] = {"mean": Y[:, i]}

        return results

    def predict_flat(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        pred = self.predict(X)
        return {name: pred[name]["mean"] for name in self.target_names}

    def summary(self) -> Dict[str, Any]:
        n_params = sum(W.size + b.size for W, b in zip(self._weights, self._biases)) if self._fitted else 0
        return {
            "fitted": self._fitted,
            "targets": self.target_names,
            "n_params": n_params,
            "hidden_dims": self.config.hidden_dims,
        }


# ═══════════════════════════════════════════════════════════════════════
# 5.  FACTORY
# ═══════════════════════════════════════════════════════════════════════

def build_surrogate(
    method: str = "gp",
    target_names: Optional[List[str]] = None,
    **kwargs: Any,
) -> Union[GPSurrogate, NeuralSurrogate]:
    """
    Factory function for building surrogates.

    Parameters
    ----------
    method       : ``'gp'`` or ``'neural'``.
    target_names : KPI names.
    **kwargs     : Forwarded to the config dataclass.
    """
    if method == "gp":
        config = GPSurrogateConfig(**{
            k: v for k, v in kwargs.items()
            if k in GPSurrogateConfig.__dataclass_fields__
        })
        return GPSurrogate(config, target_names)
    elif method == "neural":
        config = NeuralSurrogateConfig(**{
            k: v for k, v in kwargs.items()
            if k in NeuralSurrogateConfig.__dataclass_fields__
        })
        return NeuralSurrogate(config, target_names)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'gp' or 'neural'.")


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "GPSurrogateConfig",
    "NeuralSurrogateConfig",
    "GPSurrogate",
    "NeuralSurrogate",
    "build_surrogate",
    "rbf_kernel",
    "matern32_kernel",
    "matern52_kernel",
]