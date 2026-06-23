from __future__ import annotations
import numpy as np

def farthest_point_sampling(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Simple FPS for diversity."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.empty(k, dtype=int)
    idx[0] = rng.integers(0, n)
    dist = np.full(n, np.inf)
    for i in range(1, k):
        d = np.linalg.norm(X - X[idx[i-1]], axis=1)
        dist = np.minimum(dist, d)
        idx[i] = int(np.argmax(dist))
    return idx
