"""
Dataset splitting strategies for MOF adsorption data.

Strategies
──────────
1.  **Random** — standard i.i.d. split.
2.  **Scaffold** — split by MOF topology so test MOFs have unseen
    topologies (measures generalisation to new structures).
3.  **Humidity** — train on dry, test on humid (measures domain
    shift under water presence).

All methods return ``(train_idx, val_idx, test_idx)`` as lists of
integer indices into the MOF ID list.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Split MOF dataset into train / val / test.

    Parameters
    ----------
    method       : ``'random'``, ``'scaffold'``, or ``'humidity'``.
    test_size    : Fraction for test set.
    val_size     : Fraction for validation set.
    random_state : Seed for reproducibility.

    Example
    ───────
    >>> sp = DataSplitter("random", test_size=0.1, val_size=0.1)
    >>> train, val, test = sp.split(mof_ids)
    >>> sp.save_splits(mof_ids, train, val, test, "splits.json")
    """

    def __init__(
        self,
        method: str = "random",
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42,
    ):
        if method not in ("random", "scaffold", "humidity"):
            raise ValueError(f"Unknown method '{method}'.")
        self.method = method
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split(
        self,
        mof_ids: List[str],
        metadata: Optional[Any] = None,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Split MOF IDs into train / val / test index lists.

        Parameters
        ----------
        mof_ids  : List of MOF identifiers.
        metadata : ``DataFrame`` with ``mof_id`` column and:
                   * ``topology`` column (for scaffold split).
                   * ``humidity`` column (for humidity split).

        Returns
        -------
        ``(train_idx, val_idx, test_idx)`` — sorted integer lists.
        """
        n = len(mof_ids)
        rng = np.random.RandomState(self.random_state)

        if self.method == "random":
            return self._random_split(n, rng)
        elif self.method == "scaffold":
            return self._scaffold_split(mof_ids, metadata, rng)
        elif self.method == "humidity":
            return self._humidity_split(mof_ids, metadata, rng)
        else:
            raise ValueError(f"Unknown method '{self.method}'.")

    # ── Random ───────────────────────────────────────────────────

    def _random_split(
        self, n: int, rng: np.random.RandomState,
    ) -> Tuple[List[int], List[int], List[int]]:
        indices = rng.permutation(n)
        n_test = max(1, int(n * self.test_size))
        n_val = max(1, int(n * self.val_size))

        test_idx = indices[:n_test].tolist()
        val_idx = indices[n_test:n_test + n_val].tolist()
        train_idx = indices[n_test + n_val:].tolist()

        return sorted(train_idx), sorted(val_idx), sorted(test_idx)

    # ── Scaffold ─────────────────────────────────────────────────

    def _scaffold_split(
        self,
        mof_ids: List[str],
        metadata: Any,
        rng: np.random.RandomState,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Split by MOF topology so test set has unseen topologies."""
        import pandas as pd

        if metadata is None or "topology" not in metadata.columns:
            raise ValueError("Scaffold split requires metadata with 'topology' column.")

        # Map mof_id → topology
        meta_idx = metadata.set_index("mof_id")
        topos = []
        for mid in mof_ids:
            if mid in meta_idx.index:
                topos.append(meta_idx.loc[mid, "topology"])
            else:
                topos.append("__unknown__")

        unique_topos = list(set(topos))
        rng.shuffle(unique_topos)

        n_top = len(unique_topos)
        n_test_top = max(1, int(n_top * self.test_size))
        n_val_top = max(1, int(n_top * self.val_size))

        test_topos = set(unique_topos[:n_test_top])
        val_topos = set(unique_topos[n_test_top:n_test_top + n_val_top])

        train_idx, val_idx, test_idx = [], [], []
        for i, t in enumerate(topos):
            if t in test_topos:
                test_idx.append(i)
            elif t in val_topos:
                val_idx.append(i)
            else:
                train_idx.append(i)

        return sorted(train_idx), sorted(val_idx), sorted(test_idx)

    # ── Humidity domain shift ────────────────────────────────────

    def _humidity_split(
        self,
        mof_ids: List[str],
        metadata: Any,
        rng: np.random.RandomState,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Train on dry conditions, test on humid.

        Humidity threshold: samples with ``humidity >= 0.05`` go to
        test set.  Training is further split into train / val.
        """
        import pandas as pd

        if metadata is None or "humidity" not in metadata.columns:
            raise ValueError("Humidity split requires metadata with 'humidity' column.")

        meta_idx = metadata.set_index("mof_id")
        dry_idx, humid_idx = [], []

        for i, mid in enumerate(mof_ids):
            if mid in meta_idx.index:
                h = meta_idx.loc[mid, "humidity"]
            else:
                h = 0.0
            if h >= 0.05:
                humid_idx.append(i)
            else:
                dry_idx.append(i)

        test_idx = humid_idx

        # Split dry into train / val
        dry_arr = np.array(dry_idx)
        rng.shuffle(dry_arr)
        n_val = max(1, int(len(dry_arr) * self.val_size / (1 - self.test_size)))
        val_idx = dry_arr[:n_val].tolist()
        train_idx = dry_arr[n_val:].tolist()

        return sorted(train_idx), sorted(val_idx), sorted(test_idx)

    # ── Persistence ──────────────────────────────────────────────

    def save_splits(
        self,
        mof_ids: List[str],
        train_idx: List[int],
        val_idx: List[int],
        test_idx: List[int],
        path: Union[str, Path],
    ) -> None:
        """Save split MOF IDs to JSON."""
        splits = {
            "train": [mof_ids[i] for i in train_idx],
            "val": [mof_ids[i] for i in val_idx],
            "test": [mof_ids[i] for i in test_idx],
            "method": self.method,
            "test_size": self.test_size,
            "val_size": self.val_size,
            "random_state": self.random_state,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(splits, f, indent=2)
        logger.info(f"Saved splits to {path}: train={len(train_idx)}, "
                     f"val={len(val_idx)}, test={len(test_idx)}")

    @staticmethod
    def load_splits(
        path: Union[str, Path],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Load split MOF IDs from JSON."""
        with open(path) as f:
            data = json.load(f)
        return data["train"], data["val"], data["test"]


__all__ = ["DataSplitter"]