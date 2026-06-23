"""
GemNet-style directional message-passing encoder for MOF structures.

This module implements a GemNet-OC–inspired graph neural network
(Gasteiger et al., 2022) that uses **directional information** from
both interatomic distances and **angles** between edge triplets,
combined with efficient radial and spherical basis expansions.

Compared to the pure tensor-product encoders (NequIP, Equiformer),
GemNet achieves competitive accuracy with a more straightforward
message-passing design that avoids explicit irreducible representations,
making it easier to train and faster on non-GPU-specialised hardware.

Architecture overview
─────────────────────
1.  **Atom embedding** — learnable table (atomic number → vector).
2.  **Radial basis** — Gaussian RBFs with polynomial envelope.
3.  **Spherical basis** — 2-D Fourier–Bessel expansion of angles for
    directional resolution (simplified from full spherical harmonics).
4.  **Interaction blocks** (``GemNetBlock``):
        a.  Edge-update MLP combines source/destination node features,
            radial basis, and angular information.
        b.  Efficient bilinear layer mixes radial and angular channels.
        c.  Residual message aggregation via ``scatter_add``.
5.  **Readout** — invariant scalar projection → global mean pool → MLP
    → ``[B, emb_dim]`` MOF embedding.

The interface is identical to ``NequIPEncoder`` /
``EquiformerEncoder``, so encoders can be swapped via the
``EncoderAdapter``.

References
──────────
[1] Gasteiger et al. (2022). GemNet-OC: Developing Graph Neural
    Networks for Large and Diverse Molecular Simulation Datasets.
    Trans. Machine Learning Research.
[2] Gasteiger et al. (2021). GemNet: Universal Directional Graph
    Neural Networks for Molecules. NeurIPS.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# Lazy dependency gate
# ═══════════════════════════════════════════════════════════════════════

def _require_scatter():
    try:
        from torch_scatter import scatter
        return scatter
    except ImportError as e:
        raise ImportError(
            "gemnet.py requires `torch_scatter`. "
            "Install via: pip install torch-scatter"
        ) from e


# ═══════════════════════════════════════════════════════════════════════
# 1.  RADIAL & ANGULAR BASES
# ═══════════════════════════════════════════════════════════════════════

class GaussianRBF(nn.Module):
    """Gaussian radial basis with trainable centres and widths."""

    def __init__(self, n_rbf: int = 64, cutoff: float = 6.0, trainable: bool = False):
        super().__init__()
        self.cutoff = cutoff
        centres = torch.linspace(0.0, cutoff, n_rbf)
        widths = torch.full((n_rbf,), (cutoff / max(n_rbf - 1, 1)))
        if trainable:
            self.centres = nn.Parameter(centres)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer("centres", centres)
            self.register_buffer("widths", widths)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * ((d.unsqueeze(-1) - self.centres) / self.widths).pow(2))


class PolynomialEnvelope(nn.Module):
    """Smooth polynomial cutoff envelope (p = 6)."""

    def __init__(self, cutoff: float = 6.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        x = d / self.cutoff
        env = 1.0 - 6.0 * x.pow(5) + 15.0 * x.pow(4) - 10.0 * x.pow(3)
        return env.clamp(min=0.0) * (d < self.cutoff).float()


class SphericalBasis(nn.Module):
    """
    Simplified spherical basis for angular resolution.

    Instead of full spherical harmonics, we use a 2-D Fourier–Bessel
    expansion of the angle θ between each edge triplet (i→j, j→k).
    This is cheaper than tensor products while still encoding
    directional information.
    """

    def __init__(self, n_sph: int = 7, n_radial: int = 6, cutoff: float = 6.0):
        super().__init__()
        self.n_sph = n_sph
        self.n_radial = n_radial
        # Bessel frequencies
        freqs = math.pi * torch.arange(1, n_sph + 1).float()
        self.register_buffer("freqs", freqs)
        self.envelope = PolynomialEnvelope(cutoff)

    def forward(
        self,
        cos_angle: torch.Tensor,
        dist_kj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        cos_angle : ``[n_triplets]`` cosine of angle at central atom j.
        dist_kj   : ``[n_triplets]`` distance of the second edge in triplet.

        Returns
        -------
        ``[n_triplets, n_sph]`` angular features.
        """
        # Chebyshev-like expansion of cos(θ)
        # T_0 = 1, T_1 = cos θ, T_n = 2 cos θ T_{n-1} - T_{n-2}
        theta = torch.acos(cos_angle.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        basis = torch.sin(self.freqs.unsqueeze(0) * theta.unsqueeze(-1))  # [T, n_sph]
        env = self.envelope(dist_kj).unsqueeze(-1)  # [T, 1]
        return basis * env


# ═══════════════════════════════════════════════════════════════════════
# 2.  EDGE-UPDATE / INTERACTION BLOCK
# ═══════════════════════════════════════════════════════════════════════

class BilinearLayer(nn.Module):
    """Efficient bilinear mixing of radial and angular channels."""

    def __init__(self, radial_dim: int, angular_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(radial_dim, angular_dim, out_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, r: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """``r [*, R]``, ``a [*, A]`` → ``[*, O]``."""
        # Einsum: ...r, ...a, rao -> ...o
        return torch.einsum("...r,...a,rao->...o", r, a, self.W) + self.bias


class GemNetBlock(nn.Module):
    """
    Single GemNet interaction block.

    Updates edge messages using:

    1.  Source/dest node features + radial basis (pair interaction).
    2.  Triplet angular features via ``BilinearLayer`` (directional).
    3.  Aggregation to destination node (scatter-add).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_rbf: int = 64,
        n_sph: int = 7,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.scatter = _require_scatter()

        # Edge-level MLP: node_src ⊕ node_dst ⊕ rbf → message
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Angular refinement (bilinear radial × angular → hidden)
        self.bilinear = BilinearLayer(n_rbf, n_sph, hidden_dim)

        # Combine pair + angular
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        rbf: torch.Tensor,
        triplet_index: torch.Tensor,
        triplet_rbf_kj: torch.Tensor,
        triplet_sph: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x             : ``[N, H]`` node features.
        edge_index    : ``[2, E]`` edges.
        rbf           : ``[E, n_rbf]`` radial basis of edge distances.
        triplet_index : ``[3, T]`` triplet indices ``(i, j, k)`` where
                        j is the central atom.  Row 0 = edge (i→j) idx
                        in ``edge_index``, row 1 = edge (j→k) idx.
        triplet_rbf_kj: ``[T, n_rbf]`` radial basis for the k→j edge.
        triplet_sph   : ``[T, n_sph]`` spherical basis for the angle.

        Returns
        -------
        ``[N, H]`` updated node features.
        """
        src, dst = edge_index

        # ── Pair messages ────────────────────────────────────────
        edge_input = torch.cat([x[src], x[dst], rbf], dim=-1)
        msg_pair = self.edge_mlp(edge_input)  # [E, H]

        # ── Angular refinement (per triplet → scatter to edge) ───
        if triplet_index is not None and triplet_index.shape[1] > 0:
            edge_ij_idx = triplet_index[0]  # which edge i→j
            ang_feat = self.bilinear(triplet_rbf_kj, triplet_sph)  # [T, H]
            # Scatter angular info onto the i→j edges
            ang_per_edge = self.scatter(
                ang_feat, edge_ij_idx, dim=0,
                dim_size=edge_index.shape[1], reduce="mean",
            )
        else:
            ang_per_edge = torch.zeros_like(msg_pair)

        # ── Combine pair + angular ───────────────────────────────
        msg = self.combine(torch.cat([msg_pair, ang_per_edge], dim=-1))

        # ── Aggregate to destination ─────────────────────────────
        agg = self.scatter(msg, dst, dim=0, dim_size=x.shape[0], reduce="add")

        return self.norm(x + self.dropout(agg))


# ═══════════════════════════════════════════════════════════════════════
# 3.  TRIPLET BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_triplets(
    edge_index: torch.Tensor,
    pos: torch.Tensor,
    cell: Optional[torch.Tensor] = None,
    use_pbc: bool = True,
    max_neighbours: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build angle triplets ``(edge_ij_idx, edge_jk_idx)`` from the
    edge list and compute cos(angle) at the central atom.

    Returns ``(triplet_index [2, T], cos_angles [T], dist_kj [T])``.
    """
    src, dst = edge_index  # i→j means src=i, dst=j

    # For every edge i→j, find all edges j→k (k ≠ i) to form triplets
    # Build adjacency mapping: for each node j, list of outgoing edge indices
    n_nodes = int(max(src.max(), dst.max())) + 1
    n_edges = edge_index.shape[1]

    # Outgoing edges per node  (node j → list of edge indices where src == j)
    # We use dst as the central atom, so edges arriving at j: dst == j
    # Then find edges leaving j: src == j
    # This is O(E²/N) in the worst case; cap with max_neighbours.

    edge_ij_list = []
    edge_jk_list = []

    # Group edges by destination node
    # edge indices where dst == j gives us edges arriving at j (i→j)
    device = edge_index.device

    # For efficiency, build via sorting
    sort_idx_dst = torch.argsort(dst)
    sorted_dst = dst[sort_idx_dst]
    # Find boundaries
    unique_dst, counts = torch.unique_consecutive(sorted_dst, return_counts=True)

    # Also group edges by source (outgoing from j): src == j
    sort_idx_src = torch.argsort(src)
    sorted_src = src[sort_idx_src]
    unique_src, counts_src = torch.unique_consecutive(sorted_src, return_counts=True)

    # Build lookup: node → outgoing edge indices
    out_edges: Dict[int, torch.Tensor] = {}
    offset = 0
    for node, cnt in zip(unique_src.tolist(), counts_src.tolist()):
        out_edges[node] = sort_idx_src[offset : offset + min(cnt, max_neighbours)]
        offset += cnt

    # For each incoming edge i→j, pair with outgoing edges j→k
    offset = 0
    for j_node, cnt in zip(unique_dst.tolist(), counts.tolist()):
        in_idxs = sort_idx_dst[offset : offset + cnt]  # edges arriving at j
        offset += cnt

        if j_node not in out_edges:
            continue
        out_idxs = out_edges[j_node]  # edges leaving j

        for ij_idx in in_idxs:
            i_node = src[ij_idx].item()
            for jk_idx in out_idxs:
                k_node = dst[jk_idx].item()
                if k_node == i_node:
                    continue  # skip reverse edge
                edge_ij_list.append(ij_idx)
                edge_jk_list.append(jk_idx)

    if len(edge_ij_list) == 0:
        triplet_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        cos_angles = torch.zeros(0, device=device)
        dist_kj = torch.zeros(0, device=device)
        return triplet_index, cos_angles, dist_kj

    triplet_index = torch.stack([
        torch.tensor(edge_ij_list, device=device),
        torch.tensor(edge_jk_list, device=device),
    ])  # [2, T]

    # Compute vectors and angles
    vec_ij = pos[dst[triplet_index[0]]] - pos[src[triplet_index[0]]]
    vec_jk = pos[dst[triplet_index[1]]] - pos[src[triplet_index[1]]]

    # PBC correction
    if use_pbc and cell is not None:
        cell_inv = torch.linalg.inv(cell.float())
        for v in [vec_ij, vec_jk]:
            frac = v @ cell_inv.T
            v -= torch.round(frac) @ cell.float()

    d_ij = vec_ij.norm(dim=-1).clamp(min=1e-8)
    d_jk = vec_jk.norm(dim=-1).clamp(min=1e-8)

    cos_angles = (vec_ij * vec_jk).sum(dim=-1) / (d_ij * d_jk)
    cos_angles = cos_angles.clamp(-1.0, 1.0)

    return triplet_index, cos_angles, d_jk


# ═══════════════════════════════════════════════════════════════════════
# 4.  GEMNET ENCODER
# ═══════════════════════════════════════════════════════════════════════

class GemNetEncoder(nn.Module):
    """
    GemNet-OC–style directional message-passing encoder.

    Parameters
    ----------
    n_species  : Embedding table size.
    emb_dim    : Hidden dimension.
    n_layers   : Number of interaction blocks.
    n_rbf      : Radial basis functions.
    n_sph      : Spherical/angular basis dimension.
    cutoff     : Cutoff radius in Å.
    drop_rate  : Dropout probability.
    use_pbc    : Apply minimum-image convention.
    max_neighbours : Cap for triplet construction.
    """

    def __init__(
        self,
        n_species: int = 100,
        emb_dim: int = 128,
        n_layers: int = 4,
        n_rbf: int = 64,
        n_sph: int = 7,
        cutoff: float = 6.0,
        drop_rate: float = 0.1,
        use_pbc: bool = True,
        max_neighbours: int = 50,
    ):
        super().__init__()
        self.scatter = _require_scatter()

        self.emb_dim = emb_dim
        self.cutoff = cutoff
        self.use_pbc = use_pbc
        self.max_neighbours = max_neighbours

        # Node embedding
        self.node_emb = nn.Embedding(n_species, emb_dim)

        # Radial + angular bases
        self.rbf = GaussianRBF(n_rbf, cutoff)
        self.envelope = PolynomialEnvelope(cutoff)
        self.sph = SphericalBasis(n_sph=n_sph, n_radial=n_rbf, cutoff=cutoff)

        # Interaction blocks
        self.blocks = nn.ModuleList([
            GemNetBlock(
                hidden_dim=emb_dim, n_rbf=n_rbf,
                n_sph=n_sph, drop_rate=drop_rate,
            )
            for _ in range(n_layers)
        ])

        # Readout MLP
        self.readout = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Dropout(drop_rate),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    # ── Data unpacking ───────────────────────────────────────────

    @staticmethod
    def _unpack(data: Any) -> Tuple[torch.Tensor, ...]:
        if hasattr(data, "z"):
            z = data.z
            pos = data.pos
            ei = data.edge_index
            cell = getattr(data, "cell", None)
            batch = getattr(data, "batch", None)
        else:
            z = data["atom_types"]
            pos = data["pos"]
            ei = data["edge_index"]
            cell = data.get("cell")
            batch = data.get("batch")
        if batch is None:
            batch = torch.zeros(len(z), dtype=torch.long, device=z.device)
        return z, pos, ei, cell, batch

    # ── Forward ──────────────────────────────────────────────────

    def forward(self, data: Any) -> torch.Tensor:
        """
        Parameters
        ----------
        data : dict or PyG ``Data`` with ``atom_types/z``, ``pos``,
               ``edge_index``, optionally ``cell``, ``batch``.

        Returns
        -------
        ``[B, emb_dim]`` MOF-level embeddings.
        """
        z, pos, edge_index, cell, batch = self._unpack(data)
        n_mofs = int(batch.max().item()) + 1

        # Node features
        x = self.node_emb(z)  # [N, H]

        # Radial basis for all edges
        src, dst = edge_index
        vec = pos[dst] - pos[src]
        if self.use_pbc and cell is not None:
            cell_inv = torch.linalg.inv(cell.float())
            frac = vec @ cell_inv.T
            vec = vec - torch.round(frac) @ cell.float()
        dist = vec.norm(dim=-1).clamp(min=1e-8)

        rbf = self.rbf(dist) * self.envelope(dist).unsqueeze(-1)  # [E, n_rbf]

        # Build triplets + angular basis
        triplet_idx, cos_ang, dist_kj = build_triplets(
            edge_index, pos, cell, self.use_pbc, self.max_neighbours,
        )
        if triplet_idx.shape[1] > 0:
            triplet_rbf_kj = self.rbf(dist_kj) * self.envelope(dist_kj).unsqueeze(-1)
            triplet_sph = self.sph(cos_ang, dist_kj)
        else:
            triplet_rbf_kj = torch.zeros(0, self.rbf.centres.shape[0], device=x.device)
            triplet_sph = torch.zeros(0, self.sph.n_sph, device=x.device)

        # Interaction blocks
        for block in self.blocks:
            x = block(x, edge_index, rbf, triplet_idx, triplet_rbf_kj, triplet_sph)

        # Readout
        x_pooled = self.scatter(x, batch, dim=0, dim_size=n_mofs, reduce="mean")
        return self.readout(x_pooled)  # [B, emb_dim]

    @property
    def num_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "GemNetEncoder",
    "GemNetBlock",
    "GaussianRBF",
    "PolynomialEnvelope",
    "SphericalBasis",
    "BilinearLayer",
    "build_triplets",
]