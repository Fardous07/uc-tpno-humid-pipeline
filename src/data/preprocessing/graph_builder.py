"""
CIF → PyTorch Geometric graph builder.

Converts sanitised CIF files into ``torch_geometric.data.Data``
objects containing the fields that our E(3)-equivariant encoders
(NequIP, Equiformer, GemNet, SE(3)-Transformer) expect:

    z          : ``[N]``    int   — atomic numbers
    pos        : ``[N, 3]`` float — Cartesian positions
    cell       : ``[3, 3]`` float — unit-cell matrix
    pbc        : ``[3]``    bool  — periodic boundary conditions
    edge_index : ``[2, E]`` long  — neighbour list
    edge_attr  : ``[E, D]`` float — edge features (length, unit vector)

Supports:
*  Radius-based neighbour list (with PBC).
*  k-nearest-neighbours (kNN) graph.
*  Pre-computed edge features (distance + unit displacement vector).
*  Batch processing of entire directories.
*  Loading back from saved ``.pt`` files.

Dependencies
────────────
*  ``ase`` — CIF I/O and periodic neighbour lists.
*  ``torch_geometric`` — Data container.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GraphConfig:
    """
    Parameters for graph construction.

    Attributes
    ──────────
    cutoff     : Radius cutoff for neighbour list [Å].
    max_neighbours : Cap edges per node (0 = unlimited).
    self_loops : Include self-edges.
    edge_features : Include distance + unit vector as ``edge_attr``.
    """
    cutoff: float = 6.0
    max_neighbours: int = 0
    self_loops: bool = False
    edge_features: bool = True


# ═══════════════════════════════════════════════════════════════════════
# 2.  CORE CONVERSION
# ═══════════════════════════════════════════════════════════════════════

def cif_to_graph(
    cif_path: Union[str, Path],
    config: Optional[GraphConfig] = None,
) -> "torch_geometric.data.Data":
    """
    Convert a CIF file to a PyG ``Data`` object.

    Parameters
    ----------
    cif_path : Path to sanitised CIF.
    config   : ``GraphConfig`` (uses defaults if None).

    Returns
    -------
    ``Data(z, pos, cell, pbc, edge_index, edge_attr, num_nodes)``
    """
    from ase.io import read as ase_read
    from ase.neighborlist import neighbor_list
    from torch_geometric.data import Data

    if config is None:
        config = GraphConfig()

    # Read CIF
    atoms = ase_read(str(cif_path))

    # Atomic numbers
    z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)

    # Positions
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)

    # Cell
    cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float32)

    # Periodic boundary conditions
    pbc = torch.tensor(atoms.get_pbc(), dtype=torch.bool)

    # Build neighbour list with PBC
    i_idx, j_idx, dist, D_vec = neighbor_list("ijdD", atoms, cutoff=config.cutoff)

    # Remove self-loops unless requested
    if not config.self_loops:
        mask = i_idx != j_idx
        i_idx, j_idx = i_idx[mask], j_idx[mask]
        dist, D_vec = dist[mask], D_vec[mask]

    # Cap neighbours per atom
    if config.max_neighbours > 0:
        keep = _cap_neighbours(i_idx, dist, config.max_neighbours)
        i_idx, j_idx = i_idx[keep], j_idx[keep]
        dist, D_vec = dist[keep], D_vec[keep]

    edge_index = torch.tensor(np.stack([i_idx, j_idx]), dtype=torch.long)

    # Edge features
    edge_attr = None
    if config.edge_features and len(dist) > 0:
        distances = torch.tensor(dist, dtype=torch.float32).unsqueeze(-1)
        unit_vec = torch.tensor(D_vec, dtype=torch.float32)
        norms = distances.clamp(min=1e-8)
        unit_vec = unit_vec / norms
        edge_attr = torch.cat([distances, unit_vec], dim=-1)  # [E, 4]

    data = Data(
        z=z,
        pos=pos,
        cell=cell.unsqueeze(0),    # [1, 3, 3] for batching
        pbc=pbc,
        edge_index=edge_index,
        num_nodes=len(z),
    )

    if edge_attr is not None:
        data.edge_attr = edge_attr

    return data


def _cap_neighbours(
    i_idx: np.ndarray,
    dist: np.ndarray,
    max_k: int,
) -> np.ndarray:
    """Keep only the ``max_k`` closest neighbours per node."""
    keep = []
    order = np.argsort(dist)
    counts = {}
    for idx in order:
        node = i_idx[idx]
        counts.setdefault(node, 0)
        if counts[node] < max_k:
            keep.append(idx)
            counts[node] += 1
    return np.array(keep)


# ═══════════════════════════════════════════════════════════════════════
# 3.  BATCH BUILDER
# ═══════════════════════════════════════════════════════════════════════

class GraphBuilder:
    """
    Batch-convert CIF files to PyG graphs.

    Parameters
    ----------
    config : ``GraphConfig``.

    Example
    ───────
    >>> builder = GraphBuilder()
    >>> builder.build_all("data/cifs_sanitized", "data/graphs")
    """

    def __init__(self, config: Optional[Union[GraphConfig, Dict]] = None):
        if config is None:
            config = GraphConfig()
        elif isinstance(config, dict):
            config = GraphConfig(**{
                k: v for k, v in config.items()
                if k in GraphConfig.__dataclass_fields__
            })
        self.config = config

    def build_single(
        self,
        cif_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> "torch_geometric.data.Data":
        """
        Convert one CIF and optionally save as ``.pt``.

        Returns the Data object.
        """
        data = cif_to_graph(cif_path, self.config)

        # Store MOF ID as metadata
        data.mof_id = Path(cif_path).stem

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, str(output_path))

        return data

    def build_all(
        self,
        cif_dir: Union[str, Path],
        output_dir: Union[str, Path],
        pattern: str = "*.cif",
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert all CIFs in a directory to ``.pt`` graph files.

        Parameters
        ----------
        cif_dir       : Input directory of sanitised CIFs.
        output_dir    : Output directory for ``.pt`` files.
        pattern       : Glob pattern.
        skip_existing : Skip if ``.pt`` already exists.

        Returns
        -------
        Summary dict: ``n_total``, ``n_success``, ``n_skipped``,
        ``n_failed``, ``failures``.
        """
        cif_dir = Path(cif_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cif_files = sorted(cif_dir.glob(pattern))
        n_success = n_skipped = n_failed = 0
        failures = []

        for cif in cif_files:
            pt_path = output_dir / f"{cif.stem}.pt"

            if skip_existing and pt_path.exists():
                n_skipped += 1
                continue

            try:
                self.build_single(cif, pt_path)
                n_success += 1
            except Exception as e:
                n_failed += 1
                failures.append({"mof_id": cif.stem, "error": str(e)})
                logger.warning(f"Graph build failed for {cif.stem}: {e}")

        logger.info(f"Built {n_success} graphs, skipped {n_skipped}, "
                     f"failed {n_failed} out of {len(cif_files)}.")

        return {
            "n_total": len(cif_files),
            "n_success": n_success,
            "n_skipped": n_skipped,
            "n_failed": n_failed,
            "failures": failures,
        }

    @staticmethod
    def load_graph(pt_path: Union[str, Path]) -> "torch_geometric.data.Data":
        """Load a saved ``.pt`` graph file."""
        return torch.load(str(pt_path), weights_only=False)


__all__ = ["GraphConfig", "GraphBuilder", "cif_to_graph"]