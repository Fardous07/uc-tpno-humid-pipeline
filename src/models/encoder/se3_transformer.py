"""
SE(3)-equivariant transformer for MOF structure encoding.

This module implements a lightweight SE(3)-equivariant graph
transformer that combines **tensor-product convolutions** with
**attention-weighted message passing**.  It occupies a design
point between the full Equiformer (heavier, more expressive) and
the NequIP encoder (purely convolutional, no attention).

Architecture overview
─────────────────────
1.  **Atom embedding** — scalar embedding of atomic numbers.
2.  **Radial basis** — Gaussian RBFs with smooth cutoff envelope.
3.  **Spherical harmonics** — edge-direction encoding up to ``lmax``.
4.  **SE3TransformerLayer** (repeated ``n_layers`` times):
        a.  *Equivariant tensor-product message* — neighbour features
            ⊗ edge spherical harmonics, weighted by a learned radial
            MLP (same as NequIP).
        b.  *Scalar attention gate* — dot-product attention on the
            scalar (l = 0) channels modulates message strength.
        c.  *Scatter aggregation* + residual + equivariant layer norm.
5.  **Invariant readout** — ``o3.Linear`` to scalars, mean-pool per
    MOF, then output MLP → ``[B, emb_dim]``.

The interface is identical to ``NequIPEncoder`` and
``EquiformerEncoder``, so encoders can be swapped via the
``EncoderAdapter``.

Dependencies (lazy)
───────────────────
``e3nn``, ``torch_scatter`` — imported at construction time.

References
──────────
[1] Fuchs et al. (2020). SE(3)-Transformers: 3D Roto-Translation
    Equivariant Attention Networks. NeurIPS.
[2] Geiger & Smidt (2022). e3nn: Euclidean Neural Networks.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO2 Capture in MOFs
License : MIT
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =======================================================================
# Lazy dependency gate
# =======================================================================

def _require_e3nn():
    try:
        from e3nn import o3
        from e3nn.nn import BatchNorm
        return o3, BatchNorm
    except ImportError as e:
        raise ImportError(
            "se3_transformer.py requires `e3nn`. "
            "Install with: pip install e3nn"
        ) from e


def _require_scatter():
    try:
        from torch_scatter import scatter
        return scatter
    except ImportError as e:
        raise ImportError(
            "se3_transformer.py requires `torch_scatter`. "
            "Install with: pip install torch-scatter"
        ) from e


# =======================================================================
# 1.  BUILDING BLOCKS
# =======================================================================

class RadialBasisSE3(nn.Module):
    """Gaussian RBFs with polynomial cutoff envelope."""

    def __init__(self, n_rbf: int = 32, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff
        self.register_buffer("centres", torch.linspace(0.0, cutoff, n_rbf))
        self.register_buffer("widths", torch.full((n_rbf,), cutoff / max(n_rbf - 1, 1)))

    def _envelope(self, r: torch.Tensor) -> torch.Tensor:
        x = r / self.cutoff
        return (1.0 - 6.0 * x.pow(5) + 15.0 * x.pow(4) - 10.0 * x.pow(3)).clamp(min=0.0)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        rbf = torch.exp(-0.5 * ((d.unsqueeze(-1) - self.centres) / self.widths).pow(2))
        return rbf * self._envelope(d).unsqueeze(-1)


class RadialMLP(nn.Module):
    """Two-layer MLP mapping radial basis to per-path weights."""

    def __init__(self, n_rbf: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_rbf, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, rbf: torch.Tensor) -> torch.Tensor:
        return self.net(rbf)


# =======================================================================
# 2.  SE(3)-TRANSFORMER LAYER
# =======================================================================

class SE3TransformerLayer(nn.Module):
    """
    One SE(3)-transformer message-passing layer.

    Each layer:
    1.  Computes equivariant tensor-product messages (neighbour
        features ⊗ edge SH), weighted by a radial MLP.
    2.  Applies dot-product attention on scalar channels to gate
        message contributions.
    3.  Aggregates messages, adds a residual connection, and applies
        equivariant BatchNorm.

    Parameters
    ----------
    irreps_in   : Input node irreps string.
    irreps_out  : Output node irreps string.
    irreps_sh   : Spherical-harmonic irreps for edge attributes.
    n_rbf       : Radial basis dimension.
    n_heads     : Number of scalar attention heads.
    drop_rate   : Dropout on aggregated messages.
    """

    def __init__(
        self,
        irreps_in: str,
        irreps_out: str,
        irreps_sh: str,
        n_rbf: int = 32,
        n_heads: int = 4,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        o3, BatchNorm = _require_e3nn()
        self.scatter = _require_scatter()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.n_heads = n_heads

        # Tensor product: node ⊗ edge_sh → message
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_sh, self.irreps_out, shared_weights=False,
        )
        self.radial_mlp = RadialMLP(n_rbf, self.tp.weight_numel)

        # Self-interaction (linear skip)
        self.self_interaction = o3.Linear(self.irreps_in, self.irreps_out)

        # Scalar attention gate — Q/K operate on l=0 channels only
        scalar_dim = sum(mul for mul, ir in self.irreps_in if ir.l == 0)
        d_head = max(scalar_dim // n_heads, 1)
        self.q_proj = nn.Linear(scalar_dim, d_head * n_heads, bias=False)
        self.k_proj = nn.Linear(scalar_dim, d_head * n_heads, bias=False)
        self.d_head = d_head

        # Radial attention bias
        self.attn_radial = nn.Sequential(
            nn.Linear(n_rbf, n_rbf),
            nn.SiLU(),
            nn.Linear(n_rbf, n_heads),
        )

        # Norm + dropout
        self.bn = BatchNorm(self.irreps_out)
        self.dropout = nn.Dropout(drop_rate)

        # Scalar slice end index (l=0 channels are laid out first)
        self._scalar_end = scalar_dim

    def _extract_scalars(self, x: torch.Tensor) -> torch.Tensor:
        """Extract l=0 channels from an irreps tensor."""
        return x[..., : self._scalar_end]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_rbf: torch.Tensor,
        edge_lengths: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        n_nodes = x.shape[0]

        # Tensor-product messages
        tp_weights = self.radial_mlp(edge_rbf)          # [E, weight_numel]
        msg = self.tp(x[src], edge_sh, tp_weights)       # [E, irreps_out.dim]

        # Scalar attention
        x_scalar = self._extract_scalars(x)
        q = self.q_proj(x_scalar).reshape(n_nodes, self.n_heads, self.d_head)
        k = self.k_proj(x_scalar).reshape(n_nodes, self.n_heads, self.d_head)

        scores = (q[dst] * k[src]).sum(dim=-1) / math.sqrt(self.d_head)
        scores = scores + self.attn_radial(edge_rbf)     # [E, H]

        # Softmax per destination node (subtract global max for numerical stability;
        # the constant cancels within each destination group's ratio)
        scores_exp = torch.exp(scores - scores.max())
        denom = self.scatter(scores_exp, dst, dim=0, dim_size=n_nodes, reduce="sum")
        attn = scores_exp / (denom[dst] + 1e-8)          # [E, H]

        # Average heads → scalar gate, then gate equivariant messages
        gate = attn.mean(dim=-1, keepdim=True)            # [E, 1]
        msg = msg * gate                                   # [E, irreps_out.dim]

        # Aggregate + residual + norm
        agg = self.scatter(msg, dst, dim=0, dim_size=n_nodes, reduce="sum")
        agg = self.dropout(agg)
        skip = self.self_interaction(x)
        return self.bn(agg + skip)


# =======================================================================
# 3.  SE(3)-TRANSFORMER ENCODER
# =======================================================================

class SE3TransformerEncoder(nn.Module):
    """
    Full SE(3)-transformer encoder: graph → ``[B, emb_dim]``.

    Parameters
    ----------
    n_species  : Embedding table size (max atomic number).
    emb_dim    : Scalar embedding / output dimension.
    n_layers   : Number of SE3TransformerLayers.
    lmax       : Maximum spherical-harmonic order.
    n_heads    : Attention heads per layer.
    n_rbf      : Radial basis functions.
    cutoff     : Cutoff radius in Å.
    drop_rate  : Dropout probability.
    use_pbc    : Minimum-image convention for periodic crystals.
    """

    def __init__(
        self,
        n_species: int = 100,
        emb_dim: int = 128,
        n_layers: int = 4,
        lmax: int = 2,
        n_heads: int = 4,
        n_rbf: int = 32,
        cutoff: float = 5.0,
        drop_rate: float = 0.1,
        use_pbc: bool = True,
    ):
        super().__init__()
        o3, BatchNorm = _require_e3nn()
        self.scatter = _require_scatter()

        self.emb_dim = emb_dim
        self.cutoff = cutoff
        self.use_pbc = use_pbc

        # Spherical harmonics
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.sh = o3.SphericalHarmonics(
            self.irreps_sh, normalize=True, normalization="component",
        )

        # Node embedding
        self.node_emb = nn.Embedding(n_species, emb_dim)

        # Build hidden irreps string
        vec_dim = emb_dim // 2
        if lmax >= 2:
            irreps_str = f"{emb_dim}x0e + {vec_dim}x1o + {vec_dim}x2e"
        elif lmax >= 1:
            irreps_str = f"{emb_dim}x0e + {vec_dim}x1o"
        else:
            irreps_str = f"{emb_dim}x0e"

        self.irreps_hidden = o3.Irreps(irreps_str)

        # Lift scalars to hidden irreps
        self.lift = o3.Linear(o3.Irreps(f"{emb_dim}x0e"), self.irreps_hidden)

        # Radial basis
        self.rbf = RadialBasisSE3(n_rbf, cutoff)

        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                SE3TransformerLayer(
                    irreps_in=str(self.irreps_hidden),
                    irreps_out=str(self.irreps_hidden),
                    irreps_sh=str(self.irreps_sh),
                    n_rbf=n_rbf,
                    n_heads=n_heads,
                    drop_rate=drop_rate,
                )
            )

        # Readout
        self.readout_proj = o3.Linear(self.irreps_hidden, o3.Irreps(f"{emb_dim}x0e"))
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

    def _edge_vectors(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        shifts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        vec = pos[dst] - pos[src]
        if shifts is not None:
            vec = vec + shifts
        elif self.use_pbc and cell is not None:
            cell_inv = torch.linalg.inv(cell.float())
            frac = vec @ cell_inv.T
            vec = vec - torch.round(frac) @ cell.float()
        lengths = vec.norm(dim=-1).clamp(min=1e-8)
        return vec, lengths

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

    def forward(self, data: Any) -> torch.Tensor:
        """
        Parameters
        ----------
        data : dict or PyG ``Data`` with ``atom_types/z``, ``pos``,
               ``edge_index``, optionally ``cell``, ``edge_shifts``,
               ``batch``.

        Returns
        -------
        ``[B, emb_dim]`` MOF-level embeddings.
        """
        z, pos, edge_index, cell, shifts, batch = self._unpack(data)
        n_mofs = int(batch.max().item()) + 1

        # Node features
        x = self.node_emb(z)
        x = self.lift(x)

        # Edge geometry
        vec, lengths = self._edge_vectors(pos, edge_index, cell, shifts)
        edge_sh = self.sh(vec / lengths.unsqueeze(-1))
        edge_rbf = self.rbf(lengths)

        # Zero out features beyond cutoff
        mask = (lengths < self.cutoff).float().unsqueeze(-1)
        edge_sh = edge_sh * mask
        edge_rbf = edge_rbf * mask

        # SE(3)-transformer layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_sh, edge_rbf, lengths)

        # Readout
        x_scalar = self.readout_proj(x)
        x_pooled = self.scatter(x_scalar, batch, dim=0, dim_size=n_mofs, reduce="mean")
        return self.readout_mlp(x_pooled)

    @property
    def num_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# =======================================================================
# PUBLIC API
# =======================================================================

__all__ = [
    "SE3TransformerEncoder",
    "SE3TransformerLayer",
    "RadialBasisSE3",
    "RadialMLP",
]