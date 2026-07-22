"""
NequIP-style E(3)-equivariant graph neural network for MOF encoding.

This module implements an SE(3)-equivariant message-passing neural network
inspired by NequIP (Batzner et al., 2022, Nature Communications) and
adapted for periodic crystalline MOF structures.

Architecture overview
─────────────────────
1.  **Atom embedding** — learnable embedding table maps atomic numbers
    to scalar features (``emb_dim × 0e`` irreps).  (run_008: optionally
    mixes a per-atom partial charge into this scalar embedding.)
2.  **Radial basis** — Gaussian RBFs with a smooth polynomial cutoff
    envelope encode interatomic distances.
3.  **Spherical harmonics** — unit edge vectors are expanded into SO(3)
    irreducible representations up to order ``lmax``.
4.  **Equivariant message-passing layers** — each layer performs:
        a.  Fully-connected tensor product of neighbour features ⊗ edge
            spherical harmonics, weighted by a learned radial MLP.
        b.  Scatter-add aggregation of messages onto destination nodes.
        c.  Equivariant self-interaction (linear skip connection).
        d.  Equivariant batch normalisation.
5.  **Invariant pooling** — an ``o3.Linear`` projects to scalars, then
    a global mean-scatter over nodes yields a single ``emb_dim``-vector
    per MOF in the batch.
6.  **Output MLP** — a two-layer MLP with LayerNorm + SiLU produces the
    final MOF embedding ``h ∈ ℝ^{emb_dim}``, which is passed to the
    TPNO operator.

Key properties
──────────────
* **E(3)-invariant output** — the final embedding is a scalar (``0e``),
  so rotating, translating, or reflecting the crystal leaves the
  embedding unchanged.  (Partial charge is a scalar, so mixing it in
  preserves invariance.)
* **Periodic boundary conditions** — minimum-image convention is applied
  when a unit cell is provided, and ASE-computed edge shifts are
  supported for exact PBC handling.
* **Configurable equivariance** — ``lmax`` controls the maximum
  spherical-harmonic order (0 = invariant only, 1 = vectors,
  2 = rank-2 tensors).  Higher ``lmax`` is more expressive but slower.

References
──────────
[1] Batzner et al. (2022). E(3)-equivariant graph neural networks for
    data-efficient and accurate interatomic potentials. Nature Comms.
[2] Geiger & Smidt (2022). e3nn: Euclidean Neural Networks.
    arXiv:2207.09453.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from e3nn import o3
    from e3nn.nn import BatchNorm as E3BatchNorm
except ImportError:
    raise ImportError(
        "e3nn is required for the NequIP encoder.  "
        "Install with:  pip install e3nn"
    )

try:
    from torch_scatter import scatter
except ImportError:
    # Fallback: use torch_geometric's scatter or manual implementation
    try:
        from torch_geometric.utils import scatter as _pyg_scatter

        def scatter(src, index, dim=0, dim_size=None, reduce="sum"):
            return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)
    except ImportError:
        def scatter(src, index, dim=0, dim_size=None, reduce="sum"):
            """Minimal fallback scatter for sum reduction."""
            if dim_size is None:
                dim_size = int(index.max()) + 1
            out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
            if reduce == "sum":
                return out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src), src)
            elif reduce == "mean":
                counts = torch.zeros(dim_size, device=src.device)
                counts.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
                out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src), src)
                counts = counts.clamp(min=1).unsqueeze(-1)
                return out / counts
            raise ValueError(f"Unsupported reduce: {reduce}")


# ═══════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class NequIPConfig:
    """
    Hyperparameters for the NequIP encoder.

    Attributes
    ──────────
    n_species      : Maximum atomic number supported (+1 for padding).
    emb_dim        : Width of the scalar (``0e``) channel.
    n_layers       : Number of equivariant message-passing layers.
    lmax           : Maximum spherical-harmonic order for edge features.
    n_rbf          : Number of Gaussian radial basis functions.
    cutoff         : Neighbour-list cutoff radius [Å].
    use_pbc        : Apply minimum-image convention for PBC.
    radial_hidden  : Hidden width of the per-edge radial MLP.
    radial_layers  : Depth of the radial MLP (number of hidden layers).
    invariant_layers : Depth of the output MLP after pooling.
    invariant_neurons : Width of the output MLP hidden layers.
    avg_num_neighbours : Expected average number of neighbours, used
                         for message normalisation (``1/sqrt(N)``).
    envelope_exponent  : Exponent *p* of the polynomial envelope.
    use_sc         : Use self-connection (equivariant skip) in each layer.
    use_charges    : run_008 — mix a per-atom partial charge into the
                     scalar node embedding (requires graphs to carry ``q``).
    """

    n_species: int = 100
    emb_dim: int = 128
    n_layers: int = 4
    lmax: int = 2
    n_rbf: int = 32
    cutoff: float = 5.0
    use_pbc: bool = True
    radial_hidden: int = 64
    radial_layers: int = 2
    invariant_layers: int = 2
    invariant_neurons: int = 128
    avg_num_neighbours: float = 50.0
    envelope_exponent: int = 5
    use_sc: bool = True
    use_charges: bool = False   # run_008: mix per-atom charge into node embedding


# ═══════════════════════════════════════════════════════════════════════
# 2.  CUTOFF & RADIAL BASIS
# ═══════════════════════════════════════════════════════════════════════

class PolynomialEnvelope(nn.Module):
    r"""
    Polynomial envelope that smoothly goes to zero at the cutoff.

    .. math::
        u(r) = 1 - \frac{(p+1)(p+2)}{2} x^p
             + p(p+2) x^{p+1}
             - \frac{p(p+1)}{2} x^{p+2}

    where :math:`x = r / r_{\mathrm{cut}}` and *p* is the exponent.
    Satisfies :math:`u(0)=1`, :math:`u(1)=0`, :math:`u'(1)=0`,
    :math:`u''(1)=0`.
    """

    def __init__(self, cutoff: float, exponent: int = 5):
        super().__init__()
        self.cutoff = cutoff
        self.p = exponent

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        x = r / self.cutoff
        p = self.p
        env = (
            1.0
            - 0.5 * (p + 1) * (p + 2) * x.pow(p)
            + p * (p + 2) * x.pow(p + 1)
            - 0.5 * p * (p + 1) * x.pow(p + 2)
        )
        # Hard zero beyond cutoff
        return env * (r < self.cutoff).float()


class CosineCutoff(nn.Module):
    r"""
    Smooth cosine cutoff:

    .. math::
        f_c(r) = \frac{1}{2}\left[1 + \cos\!\left(\pi \frac{r}{r_c}\right)\right]

    for :math:`r < r_c`, and zero otherwise.
    """

    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        mask = (r < self.cutoff).float()
        return 0.5 * (1.0 + torch.cos(math.pi * r / self.cutoff)) * mask


class GaussianRadialBasis(nn.Module):
    """
    Gaussian radial basis functions (fixed centres and widths).

    The centres are spaced linearly from 0 to ``cutoff`` and the widths
    are set so neighbouring Gaussians overlap.
    """

    def __init__(self, n_rbf: int, cutoff: float):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff

        centres = torch.linspace(0.0, cutoff, n_rbf)
        widths = torch.full((n_rbf,), (cutoff / (n_rbf - 1)) ** 2)

        self.register_buffer("centres", centres)
        self.register_buffer("widths", widths)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        r : ``[n_edges]`` interatomic distances.

        Returns
        -------
        ``[n_edges, n_rbf]`` RBF features.
        """
        diff = r.unsqueeze(-1) - self.centres  # [n_edges, n_rbf]
        return torch.exp(-0.5 * diff.pow(2) / self.widths)


class RadialEmbedding(nn.Module):
    """
    Compose Gaussian RBFs → envelope → MLP to produce per-edge scalar
    features of dimension ``out_dim``.
    """

    def __init__(
        self,
        n_rbf: int,
        cutoff: float,
        out_dim: int,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        envelope_exponent: int = 5,
    ):
        super().__init__()
        self.rbf = GaussianRadialBasis(n_rbf, cutoff)
        self.envelope = PolynomialEnvelope(cutoff, exponent=envelope_exponent)

        layers: list[nn.Module] = [nn.Linear(n_rbf, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """``[n_edges]`` → ``[n_edges, out_dim]``."""
        rbf = self.rbf(r)            # [n_edges, n_rbf]
        env = self.envelope(r)       # [n_edges]
        return self.mlp(rbf) * env.unsqueeze(-1)


# ═══════════════════════════════════════════════════════════════════════
# 3.  EQUIVARIANT MESSAGE-PASSING LAYER
# ═══════════════════════════════════════════════════════════════════════

class EquivariantInteraction(nn.Module):
    """
    Single E(3)-equivariant message-passing layer.

    For each edge ``(j → i)`` the layer:

    1.  Computes an edge weight vector from the radial embedding via a
        learned MLP (output dimension = ``tp.weight_numel``).
    2.  Performs a **fully-connected tensor product** of the source node
        features with the edge spherical harmonics, weighted by the
        radial MLP output.
    3.  Scatter-sums messages onto destination nodes.
    4.  Adds a **self-connection** (equivariant skip) of the input.
    5.  Applies **equivariant batch normalisation**.

    The optional ``avg_num_neighbours`` normalises the aggregated
    messages by ``1/√N̄`` for variance stabilisation.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        irreps_sh: o3.Irreps,
        n_rbf: int,
        cutoff: float,
        radial_hidden: int = 64,
        radial_layers: int = 2,
        envelope_exponent: int = 5,
        avg_num_neighbours: float = 1.0,
        use_sc: bool = True,
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.avg_num_neighbours = avg_num_neighbours
        self.use_sc = use_sc

        # ── Tensor product (neighbour feat ⊗ edge SH) ───────────
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            internal_weights=False,
            shared_weights=False,
        )

        # ── Radial weight network ────────────────────────────────
        # Maps RBF features → tensor-product weights
        self.radial_embed = RadialEmbedding(
            n_rbf=n_rbf,
            cutoff=cutoff,
            out_dim=self.tp.weight_numel,
            hidden_dim=radial_hidden,
            n_hidden=radial_layers,
            envelope_exponent=envelope_exponent,
        )

        # ── Self-connection (equivariant skip) ───────────────────
        if use_sc:
            self.self_connection = o3.Linear(self.irreps_in, self.irreps_out)
        else:
            self.self_connection = None

        # ── Batch normalisation ──────────────────────────────────
        self.bn = E3BatchNorm(self.irreps_out)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        node_features : ``[N, irreps_in.dim]``
        edge_index    : ``[2, E]``
        edge_sh       : ``[E, irreps_sh.dim]``  (spherical harmonics)
        edge_lengths  : ``[E]``

        Returns
        -------
        ``[N, irreps_out.dim]``
        """
        src, dst = edge_index
        n_nodes = node_features.shape[0]

        # Edge weights from radial embedding
        weights = self.radial_embed(edge_lengths)  # [E, tp.weight_numel]

        # Tensor product: source features ⊗ edge SH, weighted
        messages = self.tp(node_features[src], edge_sh, weights)  # [E, irreps_out.dim]

        # Aggregate onto destination nodes
        agg = scatter(messages, dst, dim=0, dim_size=n_nodes, reduce="sum")

        # Normalise by expected neighbour count
        agg = agg / math.sqrt(self.avg_num_neighbours)

        # Self-connection
        if self.self_connection is not None:
            agg = agg + self.self_connection(node_features)

        # Batch normalisation
        out = self.bn(agg)

        return out


# ═══════════════════════════════════════════════════════════════════════
# 4.  INVARIANT POOLING
# ═══════════════════════════════════════════════════════════════════════

class InvariantPooling(nn.Module):
    """
    Project equivariant node features → scalars, then scatter-mean
    over nodes to obtain one vector per graph (MOF).
    """

    def __init__(self, irreps_in: o3.Irreps, output_dim: int):
        super().__init__()
        self.linear = o3.Linear(irreps_in, o3.Irreps(f"{output_dim}x0e"))
        self.bn = E3BatchNorm(o3.Irreps(f"{output_dim}x0e"))

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : ``[N, irreps_in.dim]`` node features.
        batch    : ``[N]`` graph membership indices.
        n_graphs : Number of graphs in the batch.

        Returns
        -------
        ``[n_graphs, output_dim]`` graph-level invariant embeddings.
        """
        x = self.linear(x)   # [N, output_dim]
        x = self.bn(x)
        return scatter(x, batch, dim=0, dim_size=n_graphs, reduce="mean")


# ═══════════════════════════════════════════════════════════════════════
# 5.  NEQUIP ENCODER  (top-level module)
# ═══════════════════════════════════════════════════════════════════════

class NequIPEncoder(nn.Module):
    """
    NequIP-style E(3)-equivariant encoder for periodic MOF structures.

    Produces a **rotation-invariant** MOF embedding vector
    ``h ∈ ℝ^{emb_dim}`` suitable as input to the TPNO operator.

    Parameters
    ----------
    n_species      : see :class:`NequIPConfig`
    emb_dim        : see :class:`NequIPConfig`
    n_layers       : see :class:`NequIPConfig`
    lmax           : see :class:`NequIPConfig`
    n_rbf          : see :class:`NequIPConfig`
    cutoff         : see :class:`NequIPConfig`
    use_pbc        : see :class:`NequIPConfig`
    use_charges    : see :class:`NequIPConfig` (run_008)
    config         : Alternatively, pass a :class:`NequIPConfig` directly.

    Accepted input formats
    ──────────────────────
    The :meth:`forward` method accepts *either* a ``dict`` or a
    ``torch_geometric.data.Data`` object.  The expected fields are:

    * ``z`` or ``atom_types`` — ``[N]`` atomic numbers (``torch.long``).
    * ``pos`` — ``[N, 3]`` Cartesian positions.
    * ``edge_index`` — ``[2, E]`` edge list.
    * ``edge_attr`` — ``[E, 1]`` or ``[E]`` edge lengths (optional; if
      absent they are computed from ``pos``).
    * ``cell`` — ``[1, 3, 3]`` or ``[3, 3]`` lattice vectors (optional).
    * ``q`` — ``[N]`` per-atom partial charges (optional; used only when
      ``use_charges`` is True, defaults to zeros if absent).
    * ``batch`` — ``[N]`` graph-membership indices (default: single graph).
    """

    def __init__(
        self,
        n_species: int = 100,
        emb_dim: int = 128,
        n_layers: int = 4,
        lmax: int = 2,
        n_rbf: int = 32,
        cutoff: float = 5.0,
        use_pbc: bool = True,
        avg_num_neighbours: float = 15.0,
        use_charges: bool = False,
        *,
        config: Optional[NequIPConfig] = None,
    ):
        super().__init__()

        # Resolve config
        if config is not None:
            c = config
        else:
            c = NequIPConfig(
                n_species=n_species,
                emb_dim=emb_dim,
                n_layers=n_layers,
                lmax=lmax,
                n_rbf=n_rbf,
                cutoff=cutoff,
                use_pbc=use_pbc,
                avg_num_neighbours=avg_num_neighbours,
                use_charges=use_charges,
            )

        self.config = c
        self.emb_dim = c.emb_dim
        self.n_layers = c.n_layers
        self.cutoff = c.cutoff
        self.use_pbc = c.use_pbc

        # ── Spherical harmonics irreps for edges ─────────────────
        self.irreps_sh = o3.Irreps.spherical_harmonics(c.lmax)

        # ── Node embedding ───────────────────────────────────────
        self.node_embedding = nn.Embedding(c.n_species, c.emb_dim)
        nn.init.uniform_(self.node_embedding.weight, -math.sqrt(3), math.sqrt(3))

        # run_008: optional per-atom charge → embedding contribution.
        # Zero-init so the encoder starts identical to the no-charge model
        # and learns to use charge from there (safe, near-identity start).
        self.use_charges = c.use_charges
        if c.use_charges:
            self.charge_proj = nn.Linear(1, c.emb_dim)
            nn.init.zeros_(self.charge_proj.weight)
            nn.init.zeros_(self.charge_proj.bias)

        # ── Build irreps schedule ────────────────────────────────
        # Input: scalars only
        irreps_node_input = o3.Irreps(f"{c.emb_dim}x0e")

        # Intermediate: scalars + vectors (+ optional l=2)
        irreps_hidden_list: list[str] = []
        for i in range(c.lmax + 1):
            parity = "e" if i % 2 == 0 else "o"
            mul = c.emb_dim if i == 0 else c.emb_dim // 2
            irreps_hidden_list.append(f"{mul}x{i}{parity}")
        irreps_hidden = o3.Irreps("+".join(irreps_hidden_list))

        # ── Message-passing layers ───────────────────────────────
        self.layers = nn.ModuleList()

        irreps_in = irreps_node_input
        for _ in range(c.n_layers):
            layer = EquivariantInteraction(
                irreps_in=irreps_in,
                irreps_out=irreps_hidden,
                irreps_sh=self.irreps_sh,
                n_rbf=c.n_rbf,
                cutoff=c.cutoff,
                radial_hidden=c.radial_hidden,
                radial_layers=c.radial_layers,
                envelope_exponent=c.envelope_exponent,
                avg_num_neighbours=c.avg_num_neighbours,
                use_sc=c.use_sc,
            )
            self.layers.append(layer)
            irreps_in = irreps_hidden  # subsequent layers: hidden → hidden

        self._irreps_node_final = irreps_hidden

        # ── Invariant pooling → graph embedding ──────────────────
        self.pooling = InvariantPooling(irreps_hidden, c.emb_dim)

        # ── Output MLP ───────────────────────────────────────────
        mlp_layers: list[nn.Module] = []
        in_dim = c.emb_dim
        for _ in range(c.invariant_layers - 1):
            mlp_layers += [
                nn.Linear(in_dim, c.invariant_neurons),
                nn.LayerNorm(c.invariant_neurons),
                nn.SiLU(),
            ]
            in_dim = c.invariant_neurons
        mlp_layers.append(nn.Linear(in_dim, c.emb_dim))
        self.output_mlp = nn.Sequential(*mlp_layers)

        # ── Cosine cutoff (for masking SH) ───────────────────────
        self.cosine_cutoff = CosineCutoff(c.cutoff)

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _unpack_input(
        self,
        data: Union[Dict[str, Any], Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Extract ``(z, pos, edge_index, cell, batch)`` from either a dict
        or a PyG ``Data`` object.
        """
        if hasattr(data, "z"):
            # PyG Data
            z = data.z
            pos = data.pos
            edge_index = data.edge_index
            cell = getattr(data, "cell", None)
            batch = getattr(data, "batch", None)
        else:
            z = data.get("z", data.get("atom_types"))
            pos = data["pos"]
            edge_index = data["edge_index"]
            cell = data.get("cell", None)
            batch = data.get("batch", None)

        if batch is None:
            batch = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)

        return z, pos, edge_index, cell, batch

    def _get_charges(
        self,
        data: Union[Dict[str, Any], Any],
        n_atoms: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        run_008: return per-atom charge as ``[N, 1]``; zeros if the graph
        carries none (keeps the encoder well-defined for charge-less MOFs).
        """
        q = None
        if hasattr(data, "q"):
            q = data.q
        elif isinstance(data, dict):
            q = data.get("q")
        if q is None:
            return torch.zeros(n_atoms, 1, device=device)
        return q.to(device).float().view(-1, 1)

    def _compute_edge_vectors(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        cell: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute edge vectors and lengths, applying minimum-image
        convention when ``use_pbc`` is True and a cell is provided.

        Returns ``(vec, lengths)`` each of shape ``[E, 3]`` and ``[E]``.
        """
        src, dst = edge_index
        vec = pos[dst] - pos[src]  # [E, 3]

        if self.use_pbc and cell is not None:
            # Reshape cell to [3, 3] if batched as [1, 3, 3]
            c = cell.squeeze()
            if c.dim() == 2:
                c = c.float()
                # Fractional coords
                c_inv = torch.linalg.inv(c)
                frac = vec @ c_inv.T
                frac = frac - torch.round(frac)
                vec = frac @ c

        lengths = torch.linalg.norm(vec, dim=-1)
        return vec, lengths

    # ──────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────

    def forward(self, data: Union[Dict[str, Any], Any]) -> torch.Tensor:
        """
        Forward pass through the NequIP encoder.

        Parameters
        ----------
        data : dict or ``torch_geometric.data.Data``
            See class docstring for expected fields.

        Returns
        -------
        ``torch.Tensor`` of shape ``[n_graphs, emb_dim]`` — one
        rotation-invariant embedding vector per MOF.
        """
        z, pos, edge_index, cell, batch = self._unpack_input(data)
        n_graphs = int(batch.max().item()) + 1

        # ── Node features ────────────────────────────────────────
        x = self.node_embedding(z)  # [N, emb_dim]
        x = x * math.sqrt(self.emb_dim)
        if self.use_charges:
            x = x + self.charge_proj(self._get_charges(data, x.shape[0], x.device))

        # ── Edge attributes ──────────────────────────────────────
        vec, lengths = self._compute_edge_vectors(pos, edge_index, cell)

        # Spherical harmonics of unit edge vectors
        unit_vec = vec / (lengths.unsqueeze(-1) + 1e-8)
        edge_sh = o3.spherical_harmonics(
            list(range(self.config.lmax + 1)),
            unit_vec,
            normalize=True,
            normalization="component",
        )  # [E, irreps_sh.dim]

        # Apply smooth cutoff to SH
        cutoff_vals = self.cosine_cutoff(lengths)  # [E]
        edge_sh = edge_sh * cutoff_vals.unsqueeze(-1)

        # ── Message-passing layers ───────────────────────────────
        for layer in self.layers:
            x = layer(x, edge_index, edge_sh, lengths)

        # ── Invariant pooling ────────────────────────────────────
        h = self.pooling(x, batch, n_graphs)  # [n_graphs, emb_dim]

        # ── Output MLP ───────────────────────────────────────────
        h = self.output_mlp(h)  # [n_graphs, emb_dim]

        return h

    # ──────────────────────────────────────────────────────────────
    # Auxiliary methods
    # ──────────────────────────────────────────────────────────────

    def get_node_features(
        self,
        data: Union[Dict[str, Any], Any],
    ) -> torch.Tensor:
        """
        Return per-node equivariant features *before* pooling.

        Useful for per-atom analysis, attention visualisation, or
        node-level auxiliary tasks.

        Returns ``[N, irreps_node_final.dim]``.
        """
        z, pos, edge_index, cell, batch = self._unpack_input(data)

        x = self.node_embedding(z) * math.sqrt(self.emb_dim)
        if self.use_charges:
            x = x + self.charge_proj(self._get_charges(data, x.shape[0], x.device))
        vec, lengths = self._compute_edge_vectors(pos, edge_index, cell)
        unit_vec = vec / (lengths.unsqueeze(-1) + 1e-8)
        edge_sh = o3.spherical_harmonics(
            list(range(self.config.lmax + 1)),
            unit_vec, normalize=True, normalization="component",
        )
        cutoff_vals = self.cosine_cutoff(lengths)
        edge_sh = edge_sh * cutoff_vals.unsqueeze(-1)

        for layer in self.layers:
            x = layer(x, edge_index, edge_sh, lengths)

        return x

    def freeze(self) -> None:
        """Freeze all parameters (for transfer learning)."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True

    def freeze_embedding(self) -> None:
        """Freeze only the atom embedding table."""
        for p in self.node_embedding.parameters():
            p.requires_grad = False

    @property
    def output_dim(self) -> int:
        """Dimension of the output MOF embedding."""
        return self.emb_dim

    @property
    def num_parameters(self) -> Dict[str, int]:
        """Return total, trainable, and frozen parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def extra_repr(self) -> str:
        c = self.config
        return (
            f"n_species={c.n_species}, emb_dim={c.emb_dim}, "
            f"n_layers={c.n_layers}, lmax={c.lmax}, "
            f"n_rbf={c.n_rbf}, cutoff={c.cutoff}, "
            f"use_pbc={c.use_pbc}, use_charges={c.use_charges}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 6.  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "NequIPConfig",
    "NequIPEncoder",
    "EquivariantInteraction",
    "GaussianRadialBasis",
    "RadialEmbedding",
    "PolynomialEnvelope",
    "CosineCutoff",
    "InvariantPooling",
]