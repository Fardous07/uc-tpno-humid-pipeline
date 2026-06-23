"""
Base MOF graph dataset.

Loads pre-built PyG graph files (``.pt``) from disk, with optional
registry metadata (cell params, topology, source database, etc.).

This is the foundation for ``AdsorptionDataset`` which adds
operating conditions and adsorption targets.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MOFDataset(Dataset):
    """
    PyTorch Dataset that loads MOF graph files.

    Parameters
    ----------
    graph_dir     : Directory containing ``{mof_id}.pt`` files.
    registry_path : Optional path to MOF registry (parquet or CSV)
                    with ``mof_id`` column and metadata.
    mof_ids       : Explicit list of MOF IDs to include.
                    If None, discovers from ``graph_dir``.
    transform     : Optional transform applied to each Data object.

    Example
    ───────
    >>> ds = MOFDataset("data/graphs")
    >>> ds[0]  # {'mof_id': 'HKUST-1', 'graph': Data(...), 'metadata': {...}}
    """

    def __init__(
        self,
        graph_dir: Union[str, Path],
        registry_path: Optional[Union[str, Path]] = None,
        mof_ids: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
    ):
        self.graph_dir = Path(graph_dir)
        self.transform = transform

        # Discover MOF IDs
        if mof_ids is not None:
            self.mof_ids = list(mof_ids)
        else:
            self.mof_ids = sorted(
                f.stem for f in self.graph_dir.glob("*.pt")
            )

        # Load registry
        self.registry = None
        if registry_path is not None:
            rp = Path(registry_path)
            try:
                import pandas as pd
                if rp.suffix == ".parquet":
                    self.registry = pd.read_parquet(rp).set_index("mof_id")
                else:
                    self.registry = pd.read_csv(rp).set_index("mof_id")
            except Exception as e:
                logger.warning(f"Could not load registry: {e}")

        logger.info(f"MOFDataset: {len(self.mof_ids)} MOFs from {self.graph_dir}")

    def __len__(self) -> int:
        return len(self.mof_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        mof_id = self.mof_ids[idx]
        graph_path = self.graph_dir / f"{mof_id}.pt"
        graph = torch.load(str(graph_path), weights_only=False)

        if self.transform is not None:
            graph = self.transform(graph)

        metadata = {}
        if self.registry is not None and mof_id in self.registry.index:
            metadata = self.registry.loc[mof_id].to_dict()

        return {"mof_id": mof_id, "graph": graph, "metadata": metadata}

    def get_dataloader(
        self,
        indices: Optional[List[int]] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        """Create a DataLoader with graph-aware collation."""
        from torch_geometric.loader import DataLoader as PyGLoader

        if indices is not None:
            subset = torch.utils.data.Subset(self, indices)
        else:
            subset = self

        return PyGLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """Collate: batch graphs via PyG, keep metadata as lists."""
        from torch_geometric.data import Batch

        mof_ids = [item["mof_id"] for item in batch]
        graphs = [item["graph"] for item in batch]
        metadata = [item["metadata"] for item in batch]

        batched_graph = Batch.from_data_list(graphs)

        return {
            "mof_ids": mof_ids,
            "graphs": batched_graph,
            "metadata": metadata,
        }


__all__ = ["MOFDataset"]