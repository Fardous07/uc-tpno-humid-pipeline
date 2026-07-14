from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class AdsorptionDataset(Dataset):
    """
    Dataset for multicomponent humid adsorption in MOFs.

    Each sample is one MOF; __getitem__ returns all its (condition, loading)
    pairs padded to a common length by the collate function.

    Fixes vs. original
    ------------------
    1. BUG FIXED: grouped.groups returns index LABELS, not positional integers.
       iloc[labels] silently returns wrong rows when the DataFrame index is
       non-contiguous (e.g. after dropna).  Now uses grouped.indices which
       always gives positional integers.
    2. BUG FIXED: NaN loadings from failed/partial GCMC runs are silently
       propagated to tensors and corrupt training.  Rows with NaN in any
       target column are dropped and the index is reset before building the
       lookup table (which also guarantees contiguous integer positions).
    3. Removed lazy_loading flag — both old branches did identical work at
       init time (pre-computed all row indices), so the flag saved no memory.
       One clean path is simpler and correct.
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

        if condition_columns is None:
            condition_columns = ["mu_CO2", "mu_N2", "mu_H2O", "T"]
        if target_columns is None:
            target_columns = ["co2_loading_molkg", "n2_loading_molkg", "h2o_loading_molkg"]

        self.condition_columns = condition_columns
        self.target_columns = target_columns

        # --- registry (MOF metadata) ---
        rp = Path(registry_path)
        self.registry = (
            pd.read_parquet(rp) if rp.suffix == ".parquet" else pd.read_csv(rp)
        )

        # --- adsorption data ---
        ap = Path(adsorption_path)
        adsorption_raw = (
            pd.read_parquet(ap) if ap.suffix == ".parquet" else pd.read_csv(ap)
        )

        # FIX 2: drop rows where any loading is NaN (failed GCMC points)
        # then reset_index so iloc positions == index labels (fixes FIX 1).
        n_before = len(adsorption_raw)
        self.adsorption = (
            adsorption_raw
            .dropna(subset=self.target_columns)
            .reset_index(drop=True)
        )
        n_dropped = n_before - len(self.adsorption)
        if n_dropped:
            logger.warning(
                f"Dropped {n_dropped}/{n_before} adsorption rows with NaN loadings."
            )

        # --- valid MOFs: must have both adsorption data and a graph file ---
        ads_mofs   = set(self.adsorption["mof_id"].unique())
        graph_mofs = {f.stem for f in self.graph_dir.glob("*.pt")}
        valid_mofs = ads_mofs & graph_mofs
        self.mof_ids = sorted(valid_mofs)

        if not self.mof_ids:
            raise RuntimeError(
                "No MOFs found with both adsorption data and a graph file. "
                f"Adsorption MOFs: {len(ads_mofs)}, Graph MOFs: {len(graph_mofs)}"
            )

        # FIX 1: grouped.indices gives positional integers — safe with iloc.
        grouped = self.adsorption.groupby("mof_id")
        self._mof_to_rows: Dict[str, np.ndarray] = {
            mof_id: grouped.indices[mof_id]
            for mof_id in self.mof_ids
            if mof_id in grouped.groups
        }

        logger.info(
            f"AdsorptionDataset: {len(self.mof_ids)} MOFs, "
            f"{len(self.adsorption)} clean data points "
            f"({n_dropped} NaN rows removed)"
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.mof_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        mof_id = self.mof_ids[idx]

        # Load graph
        graph_path = self.graph_dir / f"{mof_id}.pt"
        graph = torch.load(str(graph_path), weights_only=False)
        if self.transform is not None:
            graph = self.transform(graph)

        # Slice adsorption rows (positional — safe after reset_index)
        rows = self._mof_to_rows[mof_id]
        df   = self.adsorption.iloc[rows]

        conditions = torch.tensor(
            df[self.condition_columns].values, dtype=torch.float32
        )
        loadings = torch.tensor(
            df[self.target_columns].values, dtype=torch.float32
        )

        return {
            "mof_id":     mof_id,
            "graphs":     graph,
            "conditions": conditions,   # [P, D]
            "loadings":   loadings,     # [P, C]
            "n_points":   len(conditions),
        }

    # ------------------------------------------------------------------
    # DataLoader factory
    # ------------------------------------------------------------------

    def get_dataloader(
        self,
        indices: Optional[List[int]] = None,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> torch.utils.data.DataLoader:
        dataset = torch.utils.data.Subset(self, indices) if indices is not None else self
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=AdsorptionDataset.collate_fn,
            pin_memory=pin_memory,
        )

    # ------------------------------------------------------------------
    # Collate: pad variable-length pressure grids to a common length
    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        from torch_geometric.data import Batch

        mof_ids = [item["mof_id"] for item in batch]
        graphs  = Batch.from_data_list([item["graphs"] for item in batch])
        n_pts   = [item["n_points"] for item in batch]

        B     = len(batch)
        P_max = max(n_pts)
        D     = batch[0]["conditions"].shape[-1]
        C     = batch[0]["loadings"].shape[-1]

        conditions = torch.zeros(B, P_max, D)
        loadings   = torch.zeros(B, P_max, C)
        mask       = torch.zeros(B, P_max, dtype=torch.bool)

        for i, item in enumerate(batch):
            p = item["n_points"]
            conditions[i, :p] = item["conditions"]
            loadings[i,   :p] = item["loadings"]
            mask[i,       :p] = True

        return {
            "mof_ids":    mof_ids,
            "graphs":     graphs,
            "conditions": conditions,   # [B, P_max, D]
            "loadings":   loadings,     # [B, P_max, C]
            "mask":       mask,         # [B, P_max]  True where real data
            "n_points":   n_pts,
        }

    # ------------------------------------------------------------------
    # Convenience: compute normalization statistics from a loader
    # ------------------------------------------------------------------

    def compute_normalization_stats(
        self, indices: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute per-column mean and std for conditions and loadings.

        Pass the result directly to model.set_normalization():
            stats = dataset.compute_normalization_stats(train_indices)
            model.set_normalization(**stats)
        """
        all_cond: List[torch.Tensor] = []
        all_load: List[torch.Tensor] = []

        idx_iter = indices if indices is not None else range(len(self))
        for i in idx_iter:
            item = self[i]
            all_cond.append(item["conditions"])   # [P, D]
            all_load.append(item["loadings"])     # [P, C]

        cond_cat = torch.cat(all_cond, dim=0)   # [N_total, D]
        load_cat = torch.cat(all_load, dim=0)   # [N_total, C]

        return {
            "mu_mean": cond_cat.mean(0),
            "mu_std":  cond_cat.std(0).clamp(min=1e-6),
            "q_mean":  load_cat.mean(0),
            "q_std":   load_cat.std(0).clamp(min=1e-6),
        }


# ---------------------------------------------------------------------------
# Synthetic dataset (for unit tests and shape checks)
# ---------------------------------------------------------------------------

class SyntheticAdsorptionDataset(Dataset):
    """
    Langmuir-ish synthetic data for fast testing without real CIFs.

    Fix: self-loops removed from edge_index so graph structure better
    matches real NequIP inputs (which exclude i==j by construction).
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
        self.samples: List[Dict[str, Any]] = []

        for i in range(n_mofs):
            n_atoms = rng.randint(*n_atoms_range)

            z   = torch.tensor(rng.randint(1, 80, n_atoms), dtype=torch.long)
            pos = torch.tensor(rng.randn(n_atoms, 3) * 5.0, dtype=torch.float32)
            cell = torch.eye(3, dtype=torch.float32).unsqueeze(0) * 25.0

            # Build edges without self-loops
            n_edges = n_atoms * 8
            src = rng.randint(0, n_atoms, n_edges)
            dst = rng.randint(0, n_atoms, n_edges)
            no_self = src != dst
            src, dst = src[no_self], dst[no_self]
            edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

            graph = Data(
                z=z, pos=pos, cell=cell,
                edge_index=edge_index,
                num_nodes=n_atoms,
            )

            # Conditions: [n_points, n_conditions]
            conditions = torch.tensor(
                rng.randn(n_points, n_conditions).astype(np.float32)
            )

            # Langmuir loadings: q = q_sat * K*P / (1 + K*P)
            # μ = RT ln(P) → P_eff = exp(μ)  (RT = 1 for synthetic)
            K     = rng.uniform(0.1, 2.0, n_components)
            q_sat = rng.uniform(1.0, 8.0, n_components)
            mu    = conditions[:, :n_components].numpy()
            P_eff = np.exp(mu)
            loadings_np = q_sat * K * P_eff / (1.0 + K * P_eff)
            loadings_np += rng.randn(*loadings_np.shape) * 0.05
            loadings = torch.tensor(loadings_np.clip(0), dtype=torch.float32)

            self.samples.append({
                "mof_id":     f"SYNTH_{i:04d}",
                "graphs":     graph,
                "conditions": conditions,
                "loadings":   loadings,
                "n_points":   n_points,
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
    ) -> torch.utils.data.DataLoader:
        dataset = torch.utils.data.Subset(self, indices) if indices is not None else self
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=AdsorptionDataset.collate_fn,
        )


__all__ = ["AdsorptionDataset", "SyntheticAdsorptionDataset"]