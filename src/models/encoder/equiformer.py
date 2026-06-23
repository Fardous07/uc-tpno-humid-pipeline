"""
Equiformer: E(3)-equivariant transformer for MOF structure encoding.

This module implements the Equiformer architecture
(Liao et al., 2023), which combines SE(3)-equivariant tensor
products with a multi-head attention mechanism over atomic graphs.
It is an alternative to the NequIP encoder in our pipeline, offering
higher expressiveness for complex MOF geometries at the cost of
additional compute.

Architecture overview
─────────────────────
1.  **Atom embedding** — learnable table maps atomic numbers to
    scalar features.
2.  **Radial basis** — Gaussian RBFs with smooth polynomial envelope.
3.  **Spherical harmonics** — unit edge vectors are expanded into
    irreducible representations up to order ``lmax``.
4.  **Equivariant attention blocks** — each ``EquiformerBlock`` performs:
        a.  Pre-norm via equivariant layer normalisation.
        b.  Multi-head equivariant dot-product attention with radial
            edge bias.
        c.  Scatter-aggregate messages to destination nodes.
        d.  Equivariant FFN (``o3.Linear`` → norm → activation →
            ``o3.Linear``).
        e.  Residual connections and equivariant dropout.
5.  **Invariant readout** — project to scalars via ``o3.Linear``,
    then global mean-scatter + MLP → ``[B, emb_dim]``.

Dependencies (lazy)
───────────────────
``e3nn``, ``torch_scatter``, ``torch_geometric`` — imported at
class instantiation time.  If absent, the module is importable
but construction will raise ``ImportError``.

References
──────────
[1] Liao et al. (2023). Equiformer: Equivariant Graph Attention
    Transformer for 3D Atomistic Graphs. ICLR.
[2] Geiger & Smidt (2022). e3nn: Euclidean Neural Networks.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# Lazy dependency gate
# ═══════════════════════════════════════════════════════════════════════

def _require_e3nn():
    try:
        from e3nn import o3
        from e3nn.nn import BatchNorm
        return o3, BatchNorm
    except ImportError as e:
        raise ImportError(
            "equiformer.py requires `e3nn`. Install with: pip install e3nn"
        ) from e


def _require_scatter():
    try:
        from torch_scatter import scatter
        return scatter
    except ImportError as e:
        raise ImportError(
            "equiformer.py requires `torch_scatter`. "
            "Install with: pip install torch-scatter"
        ) from e


# ═══════════════════════════════════════════════════════════════════════
# 1.  BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════════

class EquivariantLayerNorm(nn.Module):
    """
    Equivariant layer normalisation.

    * **Scalars (l = 0)**: standard mean/std normalisation with
      optional affine parameters.
    * **Higher-order (l > 0)**: normalise by RMS only (no mean
      subtraction, which would break equivariance).
    """

    def __init__(self, irreps, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        o3, _ = _require_e3nn()
        self.irreps = o3.Irreps(irreps)
        self.eps = eps

        if affine:
            # One scale parameter per (mul, ir) block
            self.weight = nn.ParameterList()
            for mul, ir in self.irreps:
                self.weight.append(nn.Parameter(torch.ones(mul)))
        else:
            self.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fields = []
        idx = 0
        w_idx = 0
        for mul, ir in self.irreps:
            dim = mul * ir.dim
            field = x[..., idx : idx + dim].reshape(*x.shape[:-1], mul, ir.dim)

            if ir.l == 0:
                # Scalar: full normalisation
                mean = field.mean(dim=-2, keepdim=True)
                var = field.var(dim=-2, keepdim=True, unbiased=False)
                field = (field - mean) / (var + self.eps).sqrt()
            else:
                # Higher-order: RMS normalisation (preserve direction)
                rms = field.pow(2).mean(dim=-1, keepdim=True).sqrt()
                field = field / (rms + self.eps)

            if self.weight is not None:
                w = self.weight[w_idx].view(1, mul, 1) if x.dim() == 2 else self.weight[w_idx].view(*([1]*(x.dim()-1)), mul, 1)
                field = field * w

            fields.append(field.reshape(*x.shape[:-1], dim))
            idx += dim
            w_idx += 1

        return torch.cat(fields, dim=-1)


class EquivariantDropout(nn.Module):
    """
    Equivariant dropout — same mask across all components of each irrep
    so equivariance is preserved.
    """

    def __init__(self, irreps, p: float = 0.1):
        super().__init__()
        o3, _ = _require_e3nn()
        self.irreps = o3.Irreps(irreps)
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x

        fields = []
        idx = 0
        for mul, ir in self.irreps:
            dim = mul * ir.dim
            field = x[..., idx : idx + dim]
            # One mask element per multiplicity (broadcast over ir.dim)
            mask_shape = list(field.shape[:-1]) + [mul, 1]
            mask = torch.ones(mask_shape, device=field.device, dtype=field.dtype)
            mask = F.dropout(mask, p=self.p, training=self.training)
            mask = mask.expand(*field.shape[:-1], mul, ir.dim).reshape(*field.shape[:-1], dim)
            fields.append(field * mask)
            idx += dim

        return torch.cat(fields, dim=-1)


class RadialBasis(nn.Module):
    """Gaussian RBFs with smooth polynomial envelope."""

    def __init__(self, n_rbf: int = 32, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.register_buffer("widths", torch.full((n_rbf,), cutoff / max(n_rbf - 1, 1)))

    def _envelope(self, r: torch.Tensor) -> torch.Tensor:
        """p = 5 polynomial envelope → 0 at cutoff."""
        x = r / self.cutoff
        return (1.0 - 6.0 * x.pow(5) + 15.0 * x.pow(4) - 10.0 * x.pow(3)).clamp(min=0.0)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        rbf = torch.exp(-0.5 * ((distances.unsqueeze(-1) - self.centers) / self.widths).pow(2))
        return rbf * self._envelope(distances).unsqueeze(-1)


# ═══════════════════════════════════════════════════════════════════════
# 2.  EQUIVARIANT GRAPH ATTENTION
# ═══════════════════════════════════════════════════════════════════════

class EquivariantGraphAttention(nn.Module):
    """
    Multi-head equivariant dot-product attention.

    Q, K live in a low-dimensional scalar irrep space for efficiency;
    V is projected into the full output irreps.  Edge distances
    contribute a learned radial bias per head.
    """

    def __init__(
        self,
        irreps_node: str,
        n_heads: int = 8,
        radial_dim: int = 64,
        cutoff: float = 5.0,
    ):
        super().__init__()
        o3, _ = _require_e3nn()
        self.scatter = _require_scatter()
        self.irreps = o3.Irreps(irreps_node)
        self.n_heads = n_heads
        self.cutoff = cutoff

        # Scalar dimension per head for Q/K
        d_qk = max(self.irreps.dim // (4 * n_heads), 1)
        self.d_qk = d_qk

        # Linear projections (equivariant → scalar for Q/K)
        qk_irreps = o3.Irreps(f"{d_qk * n_heads}x0e")
        self.q_proj = o3.Linear(self.irreps, qk_irreps)
        self.k_proj = o3.Linear(self.irreps, qk_irreps)

        # Value projection (equivariant → equivariant)
        v_irreps = o3.Irreps(f"{n_heads}x{self.irreps}")
        self.v_proj = o3.Linear(self.irreps, v_irreps)

        # Output projection (combine heads)
        self.out_proj = o3.Linear(v_irreps, self.irreps)

        # Edge bias from radial features
        self.edge_bias_mlp = nn.Sequential(
            nn.Linear(radial_dim, radial_dim),
            nn.SiLU(),
            nn.Linear(radial_dim, n_heads),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_radial: torch.Tensor,
        edge_lengths: torch.Tensor,
    ) -> torch.Tensor:
        n_nodes = x.shape[0]
        src, dst = edge_index

        q = self.q_proj(x).reshape(n_nodes, self.n_heads, self.d_qk)
        k = self.k_proj(x).reshape(n_nodes, self.n_heads, self.d_qk)
        v = self.v_proj(x)  # [N, n_heads * irreps_dim]

        # Attention scores: dot(q_i, k_j) + edge_bias
        scores = (q[dst] * k[src]).sum(dim=-1) / math.sqrt(self.d_qk)
        scores = scores + self.edge_bias_mlp(edge_radial)

        # Cutoff mask
        mask = (edge_lengths < self.cutoff).float().unsqueeze(-1)
        scores = scores * mask + (1.0 - mask) * (-1e9)

        # Softmax per destination node
        scores_exp = torch.exp(scores - scores.max())
        denom = self.scatter(scores_exp, dst, dim=0, dim_size=n_nodes, reduce="sum")
        attn = scores_exp / (denom[dst] + 1e-8)  # [E, H]

        # Weighted messages
        v_dim = v.shape[-1] // self.n_heads
        v_per_head = v[src].reshape(-1, self.n_heads, v_dim)
        msg = (attn.unsqueeze(-1) * v_per_head).reshape(-1, self.n_heads * v_dim)
        agg = self.scatter(msg, dst, dim=0, dim_size=n_nodes, reduce="sum")

        return self.out_proj(agg)


# ═══════════════════════════════════════════════════════════════════════
# 3.  EQUIFORMER BLOCK
# ═══════════════════════════════════════════════════════════════════════

class EquiformerBlock(nn.Module):
    """
    Pre-norm equivariant transformer block:
    ``x → x + Dropout(Attn(Norm(x)))``
    ``x → x + Dropout(FFN(Norm(x)))``
    """

    def __init__(
        self,
        irreps_node: str,
        n_heads: int = 8,
        radial_dim: int = 64,
        cutoff: float = 5.0,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        o3, _ = _require_e3nn()
        self.irreps = o3.Irreps(irreps_node)

        # Attention sub-layer
        self.norm1 = EquivariantLayerNorm(irreps_node)
        self.attn = EquivariantGraphAttention(
            irreps_node, n_heads=n_heads,
            radial_dim=radial_dim, cutoff=cutoff,
        )
        self.drop1 = EquivariantDropout(irreps_node, p=drop_rate)

        # FFN sub-layer  (Linear → activation → Linear)
        self.norm2 = EquivariantLayerNorm(irreps_node)
        # Use scalar gate for non-linearity: project → scalars → SiLU → back
        scalar_irreps = o3.Irreps(f"{self.irreps.dim}x0e")
        self.ffn_up = o3.Linear(self.irreps, scalar_irreps)
        self.ffn_act = nn.SiLU()
        self.ffn_down = o3.Linear(scalar_irreps, self.irreps)
        self.drop2 = EquivariantDropout(irreps_node, p=drop_rate)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_radial: torch.Tensor,
        edge_lengths: torch.Tensor,
    ) -> torch.Tensor:
        # Attention + residual
        x = x + self.drop1(self.attn(self.norm1(x), edge_index, edge_radial, edge_lengths))
        # FFN + residual
        h = self.norm2(x)
        h = self.ffn_down(self.ffn_act(self.ffn_up(h)))
        x = x + self.drop2(h)
        return x


# ═══════════════════════════════════════════════════════════════════════
# 4.  EQUIFORMER ENCODER
# ═══════════════════════════════════════════════════════════════════════

class EquiformerEncoder(nn.Module):
    """
    Full Equiformer encoder: graph → MOF embedding ``h ∈ ℝ^{emb_dim}``.

    Interface is identical to ``NequIPEncoder`` so the two can be
    swapped via the ``EncoderAdapter``.

    Parameters
    ----------
    n_species  : Max atomic number (embedding table size).
    emb_dim    : Scalar embedding dimension.
    n_layers   : Number of transformer blocks.
    lmax       : Maximum spherical-harmonic order.
    n_heads    : Attention heads per block.
    n_rbf      : Number of radial basis functions.
    cutoff     : Cutoff radius in Å.
    drop_rate  : Dropout probability.
    use_pbc    : Apply minimum-image convention for periodic crystals.
    """

    def __init__(
        self,
        n_species: int = 100,
        emb_dim: int = 128,
        n_layers: int = 4,
        lmax: int = 2,
        n_heads: int = 8,
        n_rbf: int = 32,
        cutoff: float = 5.0,
        drop_rate: float = 0.1,
        use_pbc: bool = True,
    ):
        super().__init__()
        o3, _ = _require_e3nn()
        self.scatter = _require_scatter()

        self.emb_dim = emb_dim
        self.cutoff = cutoff
        self.use_pbc = use_pbc

        # ── Spherical harmonics ──────────────────────────────────
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.sh = o3.SphericalHarmonics(
            self.irreps_sh, normalize=True, normalization="component",
        )

        # ── Node embedding ───────────────────────────────────────
        self.node_emb = nn.Embedding(n_species, emb_dim)

        # ── Node irreps (scalars + vectors + rank-2) ─────────────
        vec_dim = emb_dim // 2
        if lmax >= 2:
            irreps_str = f"{emb_dim}x0e + {vec_dim}x1o + {vec_dim}x2e"
        elif lmax >= 1:
            irreps_str = f"{emb_dim}x0e + {vec_dim}x1o"
        else:
            irreps_str = f"{emb_dim}x0e"

        self.irreps_node = o3.Irreps(irreps_str)
        node_dim = self.irreps_node.dim

        # Lift scalar embedding to node irreps
        self.lift = o3.Linear(o3.Irreps(f"{emb_dim}x0e"), self.irreps_node)

        # ── Radial basis + MLP ───────────────────────────────────
        self.rbf = RadialBasis(n_rbf, cutoff)
        radial_dim = emb_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(n_rbf, radial_dim),
            nn.SiLU(),
            nn.Linear(radial_dim, radial_dim),
        )

        # ── Transformer blocks ───────────────────────────────────
        self.blocks = nn.ModuleList([
            EquiformerBlock(
                irreps_node=irreps_str,
                n_heads=n_heads,
                radial_dim=radial_dim,
                cutoff=cutoff,
                drop_rate=drop_rate,
            )
            for _ in range(n_layers)
        ])

        # ── Readout: project to scalars → pool → MLP ────────────
        self.readout_proj = o3.Linear(self.irreps_node, o3.Irreps(f"{emb_dim}x0e"))
        self.readout_mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim * 2),
            nn.LayerNorm(emb_dim * 2),
            nn.SiLU(),
            nn.Dropout(drop_rate),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    # ── PBC edge vectors ─────────────────────────────────────────

    def _edge_vectors(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        shifts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(vectors [E,3], lengths [E])`` with PBC."""
        src, dst = edge_index
        vec = pos[dst] - pos[src]

        if shifts is not None:
            vec = vec + shifts
        elif self.use_pbc and cell is not None:
            cell_inv = torch.linalg.inv(cell)
            frac = vec @ cell_inv.T
            vec = vec - torch.round(frac) @ cell.float()

        lengths = vec.norm(dim=-1).clamp(min=1e-8)
        return vec, lengths

    # ── Data unpacking (dict or PyG) ─────────────────────────────

    @staticmethod
    def _unpack(data: Any) -> Tuple[torch.Tensor, ...]:
        if hasattr(data, "z"):
            z = data.z
            pos = data.pos
            ei = data.edge_index
            cell = getattr(data, "cell", None)
            shifts = getattr(data, "edge_shifts", None)
            batch = getattr(data, "batch", None)
        else:
            z = data["atom_types"]
            pos = data["pos"]
            ei = data["edge_index"]
            cell = data.get("cell")
            shifts = data.get("edge_shifts")
            batch = data.get("batch")

        if batch is None:
            batch = torch.zeros(len(z), dtype=torch.long, device=z.device)
        return z, pos, ei, cell, shifts, batch

    # ── Forward ──────────────────────────────────────────────────

    def forward(self, data: Any) -> torch.Tensor:
        """
        Parameters
        ----------
        data : dict or ``torch_geometric.data.Data`` with keys
               ``atom_types / z``, ``pos``, ``edge_index``, and
               optionally ``cell``, ``edge_shifts``, ``batch``.

        Returns
        -------
        ``[B, emb_dim]`` MOF-level embeddings.
        """
        z, pos, edge_index, cell, shifts, batch = self._unpack(data)
        n_mofs = int(batch.max().item()) + 1

        # Node features: scalar embedding → lift to full irreps
        x = self.node_emb(z)  # [N, emb_dim]
        x = self.lift(x)      # [N, irreps_node.dim]

        # Edge geometry
        vec, lengths = self._edge_vectors(pos, edge_index, cell, shifts)
        edge_sh = self.sh(vec / lengths.unsqueeze(-1))       # [E, sh_dim]
        edge_radial = self.edge_mlp(self.rbf(lengths))       # [E, radial_dim]

        # Mask beyond cutoff
        mask = (lengths < self.cutoff).float().unsqueeze(-1)
        edge_radial = edge_radial * mask

        # Transformer blocks
        for block in self.blocks:
            x = block(x, edge_index, edge_radial, lengths)

        # Readout
        x_scalar = self.readout_proj(x)  # [N, emb_dim]
        x_pooled = self.scatter(x_scalar, batch, dim=0, dim_size=n_mofs, reduce="mean")
        return self.readout_mlp(x_pooled)  # [B, emb_dim]

    # ── Introspection ────────────────────────────────────────────

    @property
    def num_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "EquiformerEncoder",
    "EquiformerBlock",
    "EquivariantGraphAttention",
    "EquivariantLayerNorm",
    "EquivariantDropout",
    "RadialBasis",
]
