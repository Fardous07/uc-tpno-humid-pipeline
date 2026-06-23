"""
Adsorption isotherm dataset for UC-TPNO training.

Each sample corresponds to *one MOF* and contains:
*  ``graphs``     — PyG Data object (structure graph).
*  ``conditions`` — ``[P, D]`` tensor of operating conditions
                    (μ_CO₂, μ_N₂, μ_H₂O, T) at P pressure points.
*  ``loadings``   — ``[P, C]`` tensor of adsorption targets
                    (q_CO₂, q_N₂, q_H₂O).

The collate function pads variable-length condition/loading tensors
to the maximum P in the batch, producing:
*  ``graphs``     — PyG ``Batch`` object
*  ``conditions`` — ``[B, P_max, D]``
*  ``loadings``   — ``[B, P_max, C]``
*  ``mask``       — ``[B, P_max]`` bool — True for real points

This matches the format ``TPNOTrainer._to_device()`` expects.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class AdsorptionDataset(Dataset):
    """
    Dataset for multi-component adsorption data.

    Parameters
    ----------
    registry_path     : Path to MOF registry (parquet/csv) with ``mof_id``.
    adsorption_path   : Path to adsorption data (parquet/csv) with
                        ``mof_id``, condition columns, and target columns.
    graph_dir         : Directory containing ``{mof_id}.pt`` graphs.
    condition_columns : Column names for conditions.
    target_columns    : Column names for targets.
    transform         : Optional transform on the graph.

    Example
    ───────
    >>> ds = AdsorptionDataset(
    ...     "data/mof_registry.parquet",
    ...     "data/adsorption.parquet",
    ...     "data/graphs",
    ... )
    >>> loader = ds.get_dataloader(batch_size=8)
    >>> batch = next(iter(loader))
    >>> batch["graphs"]      # PyG Batch
    >>> batch["conditions"]  # [8, P_max, 4]
    >>> batch["loadings"]    # [8, P_max, 3]
    """

    def __init__(
        self,
        registry_path: Union[str, Path],
        adsorption_path: Union[str, Path],
        graph_dir: Union[str, Path],
        condition_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
    ):
        import pandas as pd

        self.graph_dir = Path(graph_dir)
        self.transform = transform

        # Default columns
        if condition_columns is None:
            condition_columns = ["mu_CO2", "mu_N2", "mu_H2O", "T"]
        if target_columns is None:
            target_columns = ["co2_loading_molkg", "n2_loading_molkg", "h2o_loading_molkg"]

        self.condition_columns = condition_columns
        self.target_columns = target_columns

        # Load registry
        rp = Path(registry_path)
        self.registry = (pd.read_parquet(rp) if rp.suffix == ".parquet"
                         else pd.read_csv(rp))

        # Load adsorption data
        ap = Path(adsorption_path)
        self.adsorption = (pd.read_parquet(ap) if ap.suffix == ".parquet"
                           else pd.read_csv(ap))

        # Unique MOFs that have both graph files and adsorption data
        ads_mofs = set(self.adsorption["mof_id"].unique())
        graph_mofs = {f.stem for f in self.graph_dir.glob("*.pt")}
        valid_mofs = ads_mofs & graph_mofs
        self.mof_ids = sorted(valid_mofs)

        # Precompute per-MOF row indices
        self._mof_to_rows: Dict[str, np.ndarray] = {}
        grouped = self.adsorption.groupby("mof_id")
        for mof_id in self.mof_ids:
            if mof_id in grouped.groups:
                self._mof_to_rows[mof_id] = grouped.groups[mof_id].values

        logger.info(f"AdsorptionDataset: {len(self.mof_ids)} MOFs, "
                     f"{len(self.adsorption)} data points.")

    def __len__(self) -> int:
        return len(self.mof_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        mof_id = self.mof_ids[idx]

        # Graph
        graph_path = self.graph_dir / f"{mof_id}.pt"
        graph = torch.load(str(graph_path), weights_only=False)
        if self.transform is not None:
            graph = self.transform(graph)

        # Conditions and targets for this MOF
        rows = self._mof_to_rows[mof_id]
        df = self.adsorption.iloc[rows]

        conditions = torch.tensor(
            df[self.condition_columns].values, dtype=torch.float32,
        )
        loadings = torch.tensor(
            df[self.target_columns].values, dtype=torch.float32,
        )

        return {
            "mof_id": mof_id,
            "graphs": graph,
            "conditions": conditions,   # [P, D]
            "loadings": loadings,       # [P, C]
            "n_points": len(conditions),
        }

    # ── DataLoader ───────────────────────────────────────────────

    def get_dataloader(
        self,
        indices: Optional[List[int]] = None,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        """
        Create a DataLoader with padded collation.

        The returned batches have:
        *  ``graphs``     : PyG Batch
        *  ``conditions`` : ``[B, P_max, D]``
        *  ``loadings``   : ``[B, P_max, C]``
        *  ``mask``       : ``[B, P_max]``
        *  ``mof_ids``    : list of str
        """
        if indices is not None:
            subset = torch.utils.data.Subset(self, indices)
        else:
            subset = self

        return torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory,
        )

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """
        Pad variable-length conditions/loadings to max P in batch.

        Returns ``graphs`` (PyG Batch), ``conditions`` ``[B, P_max, D]``,
        ``loadings`` ``[B, P_max, C]``, ``mask`` ``[B, P_max]``.
        """
        from torch_geometric.data import Batch

        mof_ids = [item["mof_id"] for item in batch]
        graphs = Batch.from_data_list([item["graphs"] for item in batch])

        # Determine max points
        n_pts = [item["n_points"] for item in batch]
        P_max = max(n_pts)
        B = len(batch)
        D = batch[0]["conditions"].shape[-1]
        C = batch[0]["loadings"].shape[-1]

        conditions = torch.zeros(B, P_max, D)
        loadings = torch.zeros(B, P_max, C)
        mask = torch.zeros(B, P_max, dtype=torch.bool)

        for i, item in enumerate(batch):
            p = item["n_points"]
            conditions[i, :p] = item["conditions"]
            loadings[i, :p] = item["loadings"]
            mask[i, :p] = True

        return {
            "mof_ids": mof_ids,
            "graphs": graphs,
            "conditions": conditions,
            "loadings": loadings,
            "mask": mask,
            "n_points": n_pts,
        }


# ═══════════════════════════════════════════════════════════════════════
# SYNTHETIC DATASET (for testing without real data)
# ═══════════════════════════════════════════════════════════════════════

class SyntheticAdsorptionDataset(Dataset):
    """
    Generates fake adsorption data for unit testing.

    Each sample has a random graph, random conditions, and Langmuir-
    based loadings.  Useful for testing the full pipeline without
    real CIF files.

    Parameters
    ----------
    n_mofs       : Number of fake MOFs.
    n_points     : Condition points per MOF.
    n_atoms_range: (min, max) atoms per graph.
    n_conditions : Condition dimensions.
    n_components : Loading components.
    seed         : Random seed.
    """

    def __init__(
        self,
        n_mofs: int = 50,
        n_points: int = 20,
        n_atoms_range: tuple = (10, 50),
        n_conditions: int = 4,
        n_components: int = 3,
        seed: int = 42,
    ):
        from torch_geometric.data import Data

        rng = np.random.RandomState(seed)
        self.samples = []

        for i in range(n_mofs):
            n_atoms = rng.randint(*n_atoms_range)

            # Random graph
            z = torch.tensor(rng.randint(1, 80, n_atoms), dtype=torch.long)
            pos = torch.tensor(rng.randn(n_atoms, 3) * 5, dtype=torch.float32)
            cell = torch.eye(3, dtype=torch.float32).unsqueeze(0) * 25.0

            # Random edges (radius graph approximation)
            n_edges = n_atoms * 8
            src = torch.tensor(rng.randint(0, n_atoms, n_edges), dtype=torch.long)
            dst = torch.tensor(rng.randint(0, n_atoms, n_edges), dtype=torch.long)
            edge_index = torch.stack([src, dst])

            graph = Data(z=z, pos=pos, cell=cell, edge_index=edge_index,
                         num_nodes=n_atoms)

            # Random conditions (μ_CO2, μ_N2, μ_H2O, T)
            conditions = torch.tensor(
                rng.randn(n_points, n_conditions).astype(np.float32),
            )

            # Langmuir-ish loadings
            K = rng.uniform(0.1, 2.0, n_components)
            q_sat = rng.uniform(1.0, 8.0, n_components)
            mu = conditions[:, :n_components].numpy()
            P_eff = np.exp(mu)
            loadings_np = q_sat * K * P_eff / (1.0 + K * P_eff)
            loadings_np += rng.randn(*loadings_np.shape) * 0.05
            loadings = torch.tensor(loadings_np.clip(0), dtype=torch.float32)

            self.samples.append({
                "mof_id": f"SYNTH_{i:04d}",
                "graphs": graph,
                "conditions": conditions,
                "loadings": loadings,
                "n_points": n_points,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def get_dataloader(
        self,
        indices: Optional[List[int]] = None,
        batch_size: int = 8,
        shuffle: bool = True,
        **kwargs,
    ):
        if indices is not None:
            subset = torch.utils.data.Subset(self, indices)
        else:
            subset = self
        return torch.utils.data.DataLoader(
            subset, batch_size=batch_size, shuffle=shuffle,
            collate_fn=AdsorptionDataset.collate_fn,
        )


__all__ = ["AdsorptionDataset", "SyntheticAdsorptionDataset"]