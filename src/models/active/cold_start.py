"""
Cold start selection for active learning.

Farthest Point Sampling (FPS) selects an initial diverse set of MOFs
when no labels are available yet.  This prevents the active learning
loop from being initialized with a biased/clustered sample.

The name 'cold_start' refers to the state where we have no simulation
data for any MOFs — we're starting 'cold' with only structural features.

Why we need this:
- Random initial MOFs might all come from the same topology family
- Active learning performance is sensitive to initial training set
- FPS ensures chemical space diversity from the first iteration

Methods implemented:
1. Farthest Point Sampling (FPS) — select diverse points
2. k-Means seeding             — select cluster centers
3. Max-min diversity           — maximize minimum distance
4. DPP sampling                — determinantal point process
5. Stratified sampling         — ensure coverage across feature space
6. Hybrid                      — mix of random + diverse

Fix vs. original
----------------
BUG FIXED: ColdStartSelector.select had

    return np.array([] if return_indices else [])[0]

Both ternary branches produce [] so this always evaluates to
np.array([])[0], raising IndexError (index 0 is out of bounds for an
empty array) whenever k <= 0.  Removed the erroneous [0] subscript.

References:
[1] Gonzalez (1985). Clustering to Minimize the Maximum Intercluster Distance.
[2] Kulesza & Taskar (2012). Determinantal Point Processes for Machine Learning.
[3] Settles (2009). Active Learning Literature Survey.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO
License : MIT
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: F401 (public re-export)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1.  FARTHEST POINT SAMPLING
# ═══════════════════════════════════════════════════════════════════════

def farthest_point_sampling(
    X:           np.ndarray,
    k:           int,
    seed:        int = 42,
    metric:      str = "euclidean",
    initial_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) for diverse subset selection.

    Algorithm:
    1. Pick the first point (random or specified).
    2. For each subsequent point, pick the point farthest from all
       already-selected points.
    3. Repeat until k points are selected.

    Parameters
    ----------
    X           : [N, d] feature matrix.
    k           : Number of points to select.
    seed        : Random seed.
    metric      : Distance metric (passed to scipy.spatial.distance.cdist).
    initial_idx : Index of the first point.  If None, chosen randomly.

    Returns
    -------
    np.ndarray [k] : Indices of selected points.
    """
    if k <= 0:
        return np.array([], dtype=int)
    if k >= X.shape[0]:
        return np.arange(X.shape[0], dtype=int)

    rng = np.random.RandomState(seed)
    n   = X.shape[0]

    if initial_idx is None:
        initial_idx = rng.randint(0, n)

    selected    = np.empty(k, dtype=int)
    selected[0] = initial_idx

    # Running minimum distance from each point to the nearest selected point
    min_dist             = np.full(n, np.inf)
    min_dist[initial_idx] = 0.0

    for i in range(1, k):
        last_idx     = selected[i - 1]
        dist_to_last = cdist(X[last_idx: last_idx + 1], X, metric=metric).flatten()
        min_dist     = np.minimum(min_dist, dist_to_last)

        # Use a copy to exclude already-selected points without corrupting min_dist
        tmp                  = min_dist.copy()
        tmp[selected[:i]]    = -np.inf
        next_idx             = int(np.argmax(tmp))
        selected[i]          = next_idx

    return selected


# ═══════════════════════════════════════════════════════════════════════
# 2.  K-MEANS SEEDING
# ═══════════════════════════════════════════════════════════════════════

def kmeans_seeding(
    X:        np.ndarray,
    k:        int,
    seed:     int = 42,
    n_init:   int = 10,
    max_iter: int = 300,
) -> np.ndarray:
    """
    Select initial points using k-means++ cluster centres.

    For each cluster the data point closest to the centroid is returned,
    giving a well-distributed, reproducible subset.

    Parameters
    ----------
    X        : [N, d] feature matrix.
    k        : Number of points to select.
    seed     : Random seed.
    n_init   : Number of k-means initialisations.
    max_iter : Maximum k-means iterations.

    Returns
    -------
    np.ndarray [k] : Indices of selected points (one per cluster).
    """
    from sklearn.cluster import KMeans

    if k <= 0:
        return np.array([], dtype=int)
    if k >= X.shape[0]:
        return np.arange(X.shape[0], dtype=int)

    kmeans = KMeans(
        n_clusters=min(k, X.shape[0]),
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
        algorithm="lloyd",
    )
    kmeans.fit(X)

    centers = kmeans.cluster_centers_
    labels  = kmeans.labels_
    selected: List[int] = []

    for i in range(centers.shape[0]):
        cluster_mask = labels == i
        if not cluster_mask.any():
            continue
        cluster_points = X[cluster_mask]
        distances      = cdist(centers[i: i + 1], cluster_points).flatten()
        closest_idx    = np.argmin(distances)
        original_idx   = int(np.where(cluster_mask)[0][closest_idx])
        selected.append(original_idx)

    return np.array(selected[:k], dtype=int)


# ═══════════════════════════════════════════════════════════════════════
# 3.  MAX-MIN DIVERSITY
# ═══════════════════════════════════════════════════════════════════════

def max_min_diversity(
    X:    np.ndarray,
    k:    int,
    seed: int = 42,
) -> np.ndarray:
    """
    Max-min diversity selection.

    Greedily selects points that maximise the minimum distance to the
    already-selected set.  Functionally identical to FPS but always
    starts from a random point.

    Parameters
    ----------
    X    : [N, d] feature matrix.
    k    : Number of points to select.
    seed : Random seed.

    Returns
    -------
    np.ndarray [k] : Indices of selected points.
    """
    rng = np.random.RandomState(seed)

    if k <= 0:
        return np.array([], dtype=int)
    if k >= X.shape[0]:
        return np.arange(X.shape[0], dtype=int)

    n           = X.shape[0]
    selected    = np.empty(k, dtype=int)
    selected[0] = rng.randint(0, n)
    min_dist    = np.full(n, np.inf)

    for i in range(1, k):
        last_idx     = selected[i - 1]
        dist_to_last = cdist(X[last_idx: last_idx + 1], X).flatten()
        min_dist     = np.minimum(min_dist, dist_to_last)

        # Copy before excluding so min_dist stays valid for future iters
        tmp               = min_dist.copy()
        tmp[selected[:i]] = -np.inf
        next_idx          = int(np.argmax(tmp))
        selected[i]       = next_idx

    return selected


# ═══════════════════════════════════════════════════════════════════════
# 4.  STRATIFIED SAMPLING
# ═══════════════════════════════════════════════════════════════════════

def stratified_sampling(
    X:      np.ndarray,
    k:      int,
    n_bins: int = 5,
    seed:   int = 42,
) -> np.ndarray:
    """
    Stratified sampling across feature space.

    Uses k-means to cluster, then samples from each cluster
    proportionally to its size.

    Parameters
    ----------
    X      : [N, d] feature matrix.
    k      : Number of points to select.
    n_bins : Number of strata (clusters).
    seed   : Random seed.

    Returns
    -------
    np.ndarray [k] : Indices of selected points.
    """
    from sklearn.cluster import KMeans

    if k <= 0:
        return np.array([], dtype=int)
    if k >= X.shape[0]:
        return np.arange(X.shape[0], dtype=int)

    rng        = np.random.RandomState(seed)
    n          = X.shape[0]
    n_clusters = min(n_bins, n)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X)

    unique, counts       = np.unique(labels, return_counts=True)
    probs                = counts / n
    samples_per_cluster  = np.floor(probs * k).astype(int)

    # Distribute remaining quota
    while samples_per_cluster.sum() < k:
        idx                      = rng.choice(len(unique))
        samples_per_cluster[idx] += 1

    selected: List[int] = []
    for cluster_id, n_samples in zip(unique, samples_per_cluster):
        if n_samples <= 0:
            continue
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) <= n_samples:
            selected.extend(cluster_indices.tolist())
        else:
            sampled = rng.choice(cluster_indices, size=n_samples, replace=False)
            selected.extend(sampled.tolist())

    return np.array(selected[:k], dtype=int)


# ═══════════════════════════════════════════════════════════════════════
# 5.  DETERMINANTAL POINT PROCESS (DPP) SAMPLING
# ═══════════════════════════════════════════════════════════════════════

def dpp_sampling(
    X:     np.ndarray,
    k:     int,
    sigma: float = 1.0,
    seed:  int   = 42,
) -> np.ndarray:
    """
    Determinantal Point Process (DPP) sampling for diversity.

    DPPs favour diverse subsets via an RBF kernel matrix where
    similarity between points reduces the probability of co-selection.

    Parameters
    ----------
    X     : [N, d] feature matrix.
    k     : Number of points to select.
    sigma : RBF kernel bandwidth.
    seed  : Random seed.

    Returns
    -------
    np.ndarray [k] : Indices of selected points.

    References
    ----------
    Kulesza & Taskar (2012). Determinantal Point Processes for ML.
    """
    if k <= 0:
        return np.array([], dtype=int)
    if k >= X.shape[0]:
        return np.arange(X.shape[0], dtype=int)

    rng = np.random.RandomState(seed)
    n   = X.shape[0]

    D = cdist(X, X, metric="euclidean")
    K = np.exp(-(D ** 2) / (2 * sigma ** 2))

    selected:  List[int] = []
    remaining: List[int] = list(range(n))

    for _ in range(k):
        if len(selected) == 0:
            probs = np.diag(K)[remaining]
        else:
            K_ss = K[np.ix_(selected, selected)]
            K_rs = K[np.ix_(remaining, selected)]
            K_rr = K[np.ix_(remaining, remaining)]
            try:
                K_ss_inv = np.linalg.inv(K_ss + 1e-6 * np.eye(len(selected)))
            except np.linalg.LinAlgError:
                K_ss_inv = np.linalg.pinv(K_ss)
            conditional_cov = K_rr - K_rs @ K_ss_inv @ K_rs.T
            probs = np.diag(conditional_cov)

        probs = np.maximum(probs, 0.0)
        if probs.sum() == 0.0:
            idx = int(rng.choice(remaining))
        else:
            probs = probs / probs.sum()
            idx   = remaining[int(rng.choice(len(remaining), p=probs))]

        selected.append(idx)
        remaining.remove(idx)

    return np.array(selected, dtype=int)


# ═══════════════════════════════════════════════════════════════════════
# 6.  HYBRID SELECTION STRATEGIES
# ═══════════════════════════════════════════════════════════════════════

def hybrid_selection(
    X:               np.ndarray,
    k:               int,
    method:          str   = "fps",
    random_fraction: float = 0.2,
    seed:            int   = 42,
    **kwargs,
) -> np.ndarray:
    """
    Hybrid selection combining random and diverse sampling.

    Parameters
    ----------
    X               : [N, d] feature matrix.
    k               : Number of points to select.
    method          : Diversity method: 'fps'|'kmeans'|'maxmin'|'stratified'|'dpp'.
    random_fraction : Fraction of points selected randomly.
    seed            : Random seed.

    Returns
    -------
    np.ndarray [k] : Indices of selected points.
    """
    if k <= 0:
        return np.array([], dtype=int)
    if k >= X.shape[0]:
        return np.arange(X.shape[0], dtype=int)

    rng       = np.random.RandomState(seed)
    n_random  = int(np.ceil(k * random_fraction))
    n_diverse = k - n_random

    all_indices    = np.arange(X.shape[0])
    random_indices = rng.choice(all_indices, size=n_random, replace=False)

    remaining_mask              = np.ones(X.shape[0], dtype=bool)
    remaining_mask[random_indices] = False
    X_remaining                 = X[remaining_mask]
    remaining_indices           = np.where(remaining_mask)[0]

    diverse_indices = np.array([], dtype=int)
    if n_diverse > 0 and len(X_remaining) > 0:
        if method == "fps":
            rel = farthest_point_sampling(X_remaining, n_diverse, seed=seed + 1, **kwargs)
        elif method == "kmeans":
            rel = kmeans_seeding(X_remaining, n_diverse, seed=seed + 1, **kwargs)
        elif method == "maxmin":
            rel = max_min_diversity(X_remaining, n_diverse, seed=seed + 1)
        elif method == "stratified":
            rel = stratified_sampling(
                X_remaining, n_diverse,
                n_bins=kwargs.get("n_bins", 5), seed=seed + 1,
            )
        elif method == "dpp":
            rel = dpp_sampling(
                X_remaining, n_diverse,
                sigma=kwargs.get("sigma", 1.0), seed=seed + 1,
            )
        else:
            raise ValueError(
                f"Unknown method: '{method}'. "
                "Choose from: fps, kmeans, maxmin, stratified, dpp."
            )
        diverse_indices = remaining_indices[rel]

    selected = np.concatenate([random_indices, diverse_indices])
    rng.shuffle(selected)
    return selected[:k]


# ═══════════════════════════════════════════════════════════════════════
# 7.  COLD START SELECTOR CLASS
# ═══════════════════════════════════════════════════════════════════════

class ColdStartSelector:
    """
    Unified interface for cold-start selection strategies.

    Parameters
    ----------
    method : str
        Selection method: 'fps'|'kmeans'|'maxmin'|'stratified'|'dpp'|'hybrid'.
    k      : Default number of points to select.
    seed   : Random seed.
    **kwargs : Additional arguments forwarded to the selection function.

    Example
    -------
    >>> selector = ColdStartSelector(method='fps', k=20, seed=42)
    >>> indices  = selector.select(X)
    """

    _METHODS = {
        "fps":        farthest_point_sampling,
        "kmeans":     kmeans_seeding,
        "maxmin":     max_min_diversity,
        "stratified": stratified_sampling,
        "dpp":        dpp_sampling,
        "hybrid":     hybrid_selection,
    }

    def __init__(
        self,
        method: str = "fps",
        k:      int = 20,
        seed:   int = 42,
        **kwargs,
    ):
        if method not in self._METHODS:
            raise ValueError(
                f"Unknown method: '{method}'. "
                f"Available: {list(self._METHODS.keys())}"
            )
        self.method = method
        self.k      = k
        self.seed   = seed
        self.kwargs = kwargs

    def select(
        self,
        X:              np.ndarray,
        k:              Optional[int] = None,
        return_indices: bool          = True,
    ) -> Union[np.ndarray, np.ndarray]:
        """
        Select diverse points from X.

        Parameters
        ----------
        X              : [N, d] feature matrix.
        k              : Points to select (overrides self.k when given).
        return_indices : If True, return integer indices; otherwise return
                         the selected rows of X.

        Returns
        -------
        np.ndarray : Selected indices [k] or selected points [k, d].
        """
        k = k if k is not None else self.k

        # FIX: original had `np.array([] if return_indices else [])[0]`
        # Both ternary branches produce [] so this always created
        # np.array([])[0] → IndexError.  Removed the erroneous [0].
        if k <= 0:
            return np.array([], dtype=int) if return_indices else X[np.array([], dtype=int)]

        if k >= X.shape[0]:
            indices = np.arange(X.shape[0], dtype=int)
            return indices if return_indices else X[indices]

        indices = self._METHODS[self.method](X, k, seed=self.seed, **self.kwargs)
        return indices if return_indices else X[indices]

    def select_with_scores(
        self,
        X: np.ndarray,
        k: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select points and return their diversity scores.

        For FPS, scores are the minimum distance to the selected set at
        the moment each point was chosen.  For other methods, scores are
        computed post-hoc as the minimum distance to the selected set.

        Parameters
        ----------
        X : [N, d] feature matrix.
        k : Points to select (overrides self.k).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray] : (indices [k], scores [k])
        """
        k = k if k is not None else self.k

        if k <= 0:
            return np.array([], dtype=int), np.array([])
        if k >= X.shape[0]:
            return np.arange(X.shape[0], dtype=int), np.ones(X.shape[0])

        if self.method == "fps":
            return self._fps_with_scores(X, k)

        indices    = self.select(X, k, return_indices=True)
        selected_X = X[indices]
        min_dist   = np.full(X.shape[0], np.inf)
        for point in selected_X:
            dist     = cdist(point.reshape(1, -1), X).flatten()
            min_dist = np.minimum(min_dist, dist)

        return indices, min_dist[indices]

    def _fps_with_scores(
        self,
        X: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """FPS variant that also returns the diversity score per point."""
        if k <= 0:
            return np.array([], dtype=int), np.array([])
        if k >= X.shape[0]:
            return np.arange(X.shape[0], dtype=int), np.ones(X.shape[0])

        rng         = np.random.RandomState(self.seed)
        n           = X.shape[0]
        selected    = np.empty(k, dtype=int)
        scores      = np.zeros(k, dtype=float)
        initial_idx = rng.randint(0, n)
        selected[0] = initial_idx
        scores[0]   = 0.0

        min_dist              = np.full(n, np.inf)
        min_dist[initial_idx] = 0.0

        for i in range(1, k):
            last_idx     = selected[i - 1]
            dist_to_last = cdist(X[last_idx: last_idx + 1], X).flatten()
            min_dist     = np.minimum(min_dist, dist_to_last)

            tmp               = min_dist.copy()
            tmp[selected[:i]] = -np.inf
            next_idx          = int(np.argmax(tmp))
            selected[i]       = next_idx
            scores[i]         = float(min_dist[next_idx])

        return selected, scores

    def summary(self) -> Dict[str, Any]:
        """Return a dict summarising the selector configuration."""
        return {
            "method": self.method,
            "k":      self.k,
            "seed":   self.seed,
            "kwargs": self.kwargs,
        }


# ═══════════════════════════════════════════════════════════════════════
# 8.  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "farthest_point_sampling",
    "kmeans_seeding",
    "max_min_diversity",
    "stratified_sampling",
    "dpp_sampling",
    "hybrid_selection",
    "ColdStartSelector",
]