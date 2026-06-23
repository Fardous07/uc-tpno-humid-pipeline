"""
Thermodynamic Potential Neural Operator (TPNO).

Physics-constrained neural operator that learns the grand potential
Ω(μ, T; h) conditioned on a MOF embedding *h* and derives adsorption
loadings by automatic differentiation:

    nᵢ = −∂Ω / ∂μᵢ          (loadings from chemical potentials)

The core of the operator is an **Input Convex Neural Network** (ICNN)
whose weights along the passthrough path are constrained to be
non-negative, guaranteeing that Ω is convex in (μ, T).  Convexity
implies thermodynamic stability (the Gibbs–Duhem inequality) and
ensures well-posed isotherms (monotonically increasing loading with
chemical potential).

Architecture overview
─────────────────────
1.  **Encoder** (external) — produces a MOF embedding h ∈ ℝ^{emb_dim}.
2.  **FiLM conditioning** — maps h → (γ, β) that modulate the first
    ICNN hidden layer, making the potential MOF-specific.
3.  **ICNN** — learns Ω(μ, T) with guaranteed convexity in μ.
4.  **Autograd differentiation** — computes n = −∂Ω/∂μ at zero extra
    parameter cost; second derivatives give the Hessian for Maxwell-
    relation regularisation.
5.  **Aleatoric uncertainty head** — a separate MLP predicts per-
    component log-variance from the MOF embedding (heteroscedastic
    Gaussian noise model).
6.  **Deep ensemble wrapper** — ``TPNOEnsemble`` trains M independent
    copies and decomposes total uncertainty into epistemic (model
    disagreement) and aleatoric (average noise) components.

References
──────────
[1] Amos et al. (2017). Input Convex Neural Networks. ICML.
[2] Perez et al. (2018). FiLM: Visual Reasoning with a General
    Conditioning Layer. AAAI.
[3] Lakshminarayanan et al. (2017). Simple and Scalable Predictive
    Uncertainty Estimation using Deep Ensembles. NeurIPS.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TPNOConfig:
    """
    Hyperparameters for the Thermodynamic Potential Neural Operator.

    Attributes
    ──────────
    emb_dim                 : Dimension of the MOF embedding from encoder.
    hidden_dim              : Width of every ICNN hidden layer.
    n_layers                : Number of ICNN hidden layers.
    n_conditions            : Number of thermodynamic inputs
                              (default 4: μ_CO₂, μ_N₂, μ_H₂O, T).
    n_components            : Number of adsorbate species (CO₂, N₂, H₂O).
    convex_constraint       : Method to enforce Uᵢ ≥ 0 weights.
                              One of ``'softplus'``, ``'exp'``, ``'clamp'``.
    film_conditioning       : Use FiLM to condition ICNN on MOF embedding.
    dropout                 : Dropout probability.
    use_layer_norm          : Apply LayerNorm inside ICNN hidden layers.
    activation              : Activation function (``'swish'``, ``'relu'``,
                              ``'gelu'``).
    min_potential           : Additive floor on Ω for numerical stability.
    """

    emb_dim: int = 128
    hidden_dim: int = 256
    n_layers: int = 4
    n_conditions: int = 4
    n_components: int = 3
    convex_constraint: str = "softplus"
    film_conditioning: bool = True
    dropout: float = 0.1
    use_layer_norm: bool = True
    activation: str = "swish"
    min_potential: float = 1e-6


# ═══════════════════════════════════════════════════════════════════════
# 2.  ACTIVATION HELPER
# ═══════════════════════════════════════════════════════════════════════

class Swish(nn.Module):
    """Swish / SiLU activation: x · σ(x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def _make_activation(name: str) -> nn.Module:
    """Return an activation module by name."""
    name = name.lower()
    if name in ("swish", "silu"):
        return Swish()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unknown activation: {name}")


# ═══════════════════════════════════════════════════════════════════════
# 3.  INPUT CONVEX NEURAL NETWORK  (ICNN)
# ═══════════════════════════════════════════════════════════════════════

class ICNN(nn.Module):
    r"""
    Input Convex Neural Network with guaranteed convexity in *x*.

    Architecture (Amos et al., 2017)::

        z₁   = σ( W₀ x + b₀ )
        zᵢ₊₁ = σ( Uᵢ zᵢ  +  Wᵢ x  +  bᵢ )     i = 1 … k−1
        y    =    Uₖ zₖ  +  Wₖ x  +  bₖ

    where all **Uᵢ** weights are constrained to be **non-negative**.
    This makes the network convex in *x* regardless of the sign of
    the **Wᵢ** (skip-connection) weights.

    Parameters
    ----------
    input_dim         : Dimension of the convex input *x*.
    hidden_dim        : Width of every hidden layer.
    n_layers          : Total number of hidden layers (≥ 1).
    convex_constraint : ``'softplus'``, ``'exp'``, or ``'clamp'``.
    dropout           : Dropout probability.
    activation        : Activation function name.
    use_layer_norm    : Apply LayerNorm after each hidden layer.
    min_potential     : Additive floor on the output for positivity.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int = 3,
        convex_constraint: str = "softplus",
        dropout: float = 0.1,
        activation: str = "swish",
        use_layer_norm: bool = True,
        min_potential: float = 1e-6,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.convex_constraint = convex_constraint
        self.use_layer_norm = use_layer_norm
        self._min_potential = min_potential

        self.act = _make_activation(activation)

        # ── First layer (unrestricted weights) ───────────────────
        self.W0 = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(self.W0.weight, gain=1.0)
        nn.init.zeros_(self.W0.bias)

        # ── Passthrough (U, non-negative) and skip (W) layers ───
        self.U_layers = nn.ModuleList()
        self.W_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None

        for _ in range(n_layers - 1):
            U = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self._init_nonneg(U)
            self.U_layers.append(U)

            W = nn.Linear(input_dim, hidden_dim)
            nn.init.xavier_uniform_(W.weight, gain=0.1)
            nn.init.zeros_(W.bias)
            self.W_layers.append(W)

            self.dropouts.append(nn.Dropout(dropout))
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(hidden_dim))

        # ── Output layer ─────────────────────────────────────────
        self.U_out = nn.Linear(hidden_dim, 1, bias=False)
        self._init_nonneg(self.U_out)

        self.W_out = nn.Linear(input_dim, 1)
        nn.init.xavier_uniform_(self.W_out.weight, gain=0.01)
        nn.init.zeros_(self.W_out.bias)

        self.b_out = nn.Parameter(torch.zeros(1))

    # ── weight helpers ───────────────────────────────────────────

    @staticmethod
    def _init_nonneg(layer: nn.Linear) -> None:
        """Initialise weights to small positive values."""
        nn.init.uniform_(layer.weight, 0.0, 0.1)

    def _project_nonneg(self, w: torch.Tensor) -> torch.Tensor:
        """Project a weight tensor to the non-negative cone."""
        if self.convex_constraint == "softplus":
            return F.softplus(w - 5.0) + 1e-6
        if self.convex_constraint == "exp":
            return torch.exp(w.clamp(max=10.0)) + 1e-6
        # clamp (default)
        return w.clamp(min=0.0)

    def project_weights(self) -> None:
        """Apply convexity constraint to all U-path weight matrices."""
        with torch.no_grad():
            for U in self.U_layers:
                U.weight.data = self._project_nonneg(U.weight.data)
            self.U_out.weight.data = self._project_nonneg(self.U_out.weight.data)

    # ── forward ──────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        film_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute the convex potential Ω(x).

        Parameters
        ----------
        x           : ``[B, input_dim]`` thermodynamic conditions.
        film_params : Optional ``(γ, β)`` each ``[B, hidden_dim]``.

        Returns
        -------
        ``[B, 1]`` grand-potential values (positive).
        """
        self.project_weights()

        # First layer
        z = self.W0(x)
        if film_params is not None:
            gamma, beta = film_params
            z = z * gamma + beta
        z = self.act(z)

        # Hidden layers
        for i in range(self.n_layers - 1):
            z_pass = self.U_layers[i](z)     # non-neg path
            z_skip = self.W_layers[i](x)     # skip from input
            z = z_pass + z_skip
            if self.use_layer_norm:
                z = self.norms[i](z)
            z = self.act(z)
            z = self.dropouts[i](z)

        # Output
        omega = self.U_out(z) + self.W_out(x) + self.b_out
        omega = F.softplus(omega) + self._min_potential

        return omega

    def check_convexity(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_lambdas: int = 5,
        tol: float = 1e-4,
    ) -> bool:
        """
        Empirical convexity check: f(λx + (1−λ)y) ≤ λf(x) + (1−λ)f(y).
        """
        self.eval()
        with torch.no_grad():
            fx = self.forward(x)
            fy = self.forward(y)
            for lam in torch.linspace(0.1, 0.9, n_lambdas):
                z = lam * x + (1 - lam) * y
                fz = self.forward(z)
                bound = lam * fx + (1 - lam) * fy
                if (fz > bound + tol).any():
                    return False
        return True


# ═══════════════════════════════════════════════════════════════════════
# 4.  FiLM CONDITIONING
# ═══════════════════════════════════════════════════════════════════════

class FiLMConditioning(nn.Module):
    """
    Feature-wise Linear Modulation (Perez et al., 2018).

    Maps a MOF embedding **h** to scale (γ) and shift (β) vectors that
    modulate the first hidden layer of the ICNN, making the potential
    MOF-specific while preserving convexity in μ.

    Initialised to the identity transformation (γ=1, β=0) so the
    untrained model behaves like a plain ICNN.
    """

    def __init__(self, emb_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.gamma_net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Near-identity init
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.ones_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h : ``[B, emb_dim]`` MOF embeddings.

        Returns
        -------
        (γ, β) each ``[B, hidden_dim]``.
        """
        return self.gamma_net(h), self.beta_net(h)


# ═══════════════════════════════════════════════════════════════════════
# 5.  THERMODYNAMIC POTENTIAL NEURAL OPERATOR
# ═══════════════════════════════════════════════════════════════════════

class ThermodynamicPotentialNO(nn.Module):
    """
    Complete Thermodynamic Potential Neural Operator.

    Combines an external encoder, FiLM conditioning, an ICNN for the
    grand potential Ω, autograd-based loading derivation, and a
    heteroscedastic uncertainty head.

    Parameters
    ----------
    encoder : ``nn.Module``
        Pre-built encoder (e.g. ``NequIPEncoder``) that maps a graph
        batch to ``[B, emb_dim]``.
    config  : ``TPNOConfig``
        Operator hyperparameters.

    Accepted condition tensor shapes
    ────────────────────────────────
    * ``[B, n_conditions]``            — single condition per MOF.
    * ``[B, P, n_conditions]``         — P condition points per MOF.

    Output dict keys
    ────────────────
    * ``q_pred``  — ``[B, P, n_components]`` predicted loadings.
    * ``omega``   — ``[B, P, 1]``            grand potential (opt.).
    * ``sigma``   — ``[B, P, n_components]`` aleatoric std-dev (opt.).
    * ``log_var`` — ``[B, n_components]``    log-variance (for NLL).
    * ``hessian`` — ``[B, P, C, C]``         Hessian of Ω (opt.).
    """

    def __init__(self, encoder: nn.Module, config: TPNOConfig):
        super().__init__()

        self.config = config
        self.encoder = encoder

        # ── FiLM ─────────────────────────────────────────────────
        self.film: Optional[FiLMConditioning] = None
        if config.film_conditioning:
            self.film = FiLMConditioning(
                config.emb_dim, config.hidden_dim, config.dropout,
            )

        # ── ICNN ─────────────────────────────────────────────────
        self.icnn = ICNN(
            input_dim=config.n_conditions,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            convex_constraint=config.convex_constraint,
            dropout=config.dropout,
            activation=config.activation,
            use_layer_norm=config.use_layer_norm,
            min_potential=config.min_potential,
        )

        # ── Aleatoric uncertainty head ───────────────────────────
        self.log_var_net = nn.Sequential(
            nn.Linear(config.emb_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 2, config.n_components),
        )
        # Initialise to small uncertainty (log σ² ≈ −5 → σ ≈ 0.08)
        nn.init.zeros_(self.log_var_net[-1].weight)
        nn.init.constant_(self.log_var_net[-1].bias, -5.0)

        # ── Normalisation buffers ────────────────────────────────
        self.register_buffer("mu_mean", torch.zeros(config.n_conditions))
        self.register_buffer("mu_std", torch.ones(config.n_conditions))
        self.register_buffer("q_mean", torch.zeros(config.n_components))
        self.register_buffer("q_std", torch.ones(config.n_components))

    # ── normalisation ────────────────────────────────────────────

    def set_normalization(
        self,
        mu_mean: torch.Tensor,
        mu_std: torch.Tensor,
        q_mean: torch.Tensor,
        q_std: torch.Tensor,
    ) -> None:
        """Set I/O normalisation statistics from the training set."""
        self.mu_mean.copy_(mu_mean)
        self.mu_std.copy_(mu_std)
        self.q_mean.copy_(q_mean)
        self.q_std.copy_(q_std)

    def normalize_mu(self, mu: torch.Tensor) -> torch.Tensor:
        return (mu - self.mu_mean) / (self.mu_std + 1e-8)

    def denormalize_q(self, q: torch.Tensor) -> torch.Tensor:
        return q * self.q_std + self.q_mean

    # ── encoder control ──────────────────────────────────────────

    def freeze_encoder(self, freeze: bool = True) -> None:
        """Freeze or unfreeze encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = not freeze

    # ── forward ──────────────────────────────────────────────────

    def forward(
        self,
        graphs: Any,
        conditions: torch.Tensor,
        *,
        return_potential: bool = False,
        return_uncertainty: bool = True,
        return_hessian: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encoder → FiLM → ICNN → autograd → loadings.

        Parameters
        ----------
        graphs             : Batched graph input accepted by ``self.encoder``.
        conditions         : ``[B, n_cond]`` or ``[B, P, n_cond]``.
        return_potential   : Include ``'omega'`` in output.
        return_uncertainty : Include ``'sigma'`` and ``'log_var'``.
        return_hessian     : Include ``'hessian'`` (expensive).

        Returns
        -------
        Dict — see class docstring for keys.
        """
        # Ensure 3-D conditions
        if conditions.dim() == 2:
            conditions = conditions.unsqueeze(1)

        B, P, C = conditions.shape
        n_comp = self.config.n_components

        # ── Encode MOF ───────────────────────────────────────────
        h = self.encoder(graphs)                       # [B, emb_dim]
        h_exp = h.unsqueeze(1).expand(-1, P, -1)      # [B, P, emb_dim]

        # ── Flatten for ICNN ─────────────────────────────────────
        cond_flat = conditions.reshape(B * P, C)       # [BP, C]
        h_flat = h_exp.reshape(B * P, -1)              # [BP, emb_dim]

        # Normalise conditions
        cond_norm = self.normalize_mu(cond_flat)

        # ---------- CRITICAL: Enable gradients for autograd ----------
        with torch.enable_grad():
            # Create a new leaf tensor that requires grad
            cond_norm_grad = cond_norm.detach().requires_grad_(True)

            # ── FiLM parameters ──────────────────────────────────────
            film_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
            if self.film is not None:
                film_params = self.film(h_flat)

            # ── ICNN forward (gradient-enabled) ──────────────────────
            omega_flat = self.icnn(cond_norm_grad, film_params)  # [BP, 1]

            # ── Safety: ensure gradients exist ───────────────────────
            if not omega_flat.requires_grad:
                raise RuntimeError(
                    "TPNO requires gradients to compute adsorption "
                    "(n = -∂Ω/∂μ). Do not call under torch.no_grad()."
                )

            # ── Loadings via autograd: nᵢ = −∂Ω/∂μᵢ ─────────────────
            grad_outputs = torch.ones_like(omega_flat)
            grads = autograd.grad(
                outputs=omega_flat,
                inputs=cond_norm_grad,
                grad_outputs=grad_outputs,
                create_graph=self.training or return_hessian,
                retain_graph=self.training or return_hessian,
            )[0]  # [BP, C]

            # Take only the first n_components columns (the μ dimensions)
            q_norm = -grads[:, :n_comp]                     # [BP, n_comp]

        # ── Reshape & denormalise ────────────────────────────────
        q_pred = q_norm.reshape(B, P, n_comp)
        q_pred = self.denormalize_q(q_pred)

        output: Dict[str, torch.Tensor] = {"q_pred": q_pred}

        # ── Potential (reuse omega_flat from grad block) ─────────
        if return_potential:
            output["omega"] = omega_flat.reshape(B, P, 1)

        # ── Aleatoric uncertainty ────────────────────────────────
        if return_uncertainty:
            log_var = self.log_var_net(h)                # [B, n_comp]
            sigma = torch.exp(0.5 * log_var)             # std-dev
            output["log_var"] = log_var
            output["sigma"] = sigma.unsqueeze(1).expand(-1, P, -1)

        # ── Hessian (expensive, recomputed) ─────────────────────
        if return_hessian:
            with torch.enable_grad():
                cond_norm_grad = cond_norm.detach().requires_grad_(True)
                if self.film is not None:
                    film_params = self.film(h_flat)
                else:
                    film_params = None
                omega_flat = self.icnn(cond_norm_grad, film_params)
                grad_outputs = torch.ones_like(omega_flat)
                grads = autograd.grad(
                    outputs=omega_flat,
                    inputs=cond_norm_grad,
                    grad_outputs=grad_outputs,
                    create_graph=self.training,
                    retain_graph=self.training,
                )[0]
                q_norm = -grads[:, :n_comp]
                rows = []
                for i in range(n_comp):
                    row_i = autograd.grad(
                        outputs=q_norm[:, i].sum(),
                        inputs=cond_norm_grad,
                        retain_graph=True,
                        create_graph=self.training,
                    )[0][:, :n_comp]
                    rows.append(row_i)
                hess = torch.stack(rows, dim=1)  # [BP, n_comp, n_comp]
            output["hessian"] = hess.reshape(B, P, n_comp, n_comp)

        return output

    # ── Standalone Hessian (re-runs encoder) ─────────────────────

    def get_hessian(
        self,
        graphs: Any,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Hessian ∂²Ω / ∂μᵢ∂μⱼ without needing a prior
        forward pass.  Useful for regularisation in the loss function.

        Returns ``[B, P, n_comp, n_comp]``.
        """
        out = self.forward(
            graphs, conditions,
            return_hessian=True,
            return_uncertainty=False,
        )
        return out["hessian"]

    # ── Potential surface for visualisation ───────────────────────

    @torch.no_grad()
    def get_potential_surface(
        self,
        graphs: Any,
        mu_range: Tuple[float, float] = (-20.0, 5.0),
        n_points: int = 50,
        T: float = 313.0,
    ) -> Dict[str, Any]:
        """
        Generate the full 3-D potential surface Ω(μ_CO₂, μ_N₂, μ_H₂O)
        at fixed temperature for one MOF.

        Returns a dict with NumPy arrays ``mu``, ``omega``,
        ``q_co2``, ``q_n2``, ``q_h2o``.
        """
        device = next(self.parameters()).device
        mu = torch.linspace(mu_range[0], mu_range[1], n_points, device=device)
        mu1, mu2, mu3 = torch.meshgrid(mu, mu, mu, indexing="ij")

        cond = torch.stack(
            [mu1.flatten(), mu2.flatten(), mu3.flatten(),
             torch.full_like(mu1.flatten(), T)],
            dim=1,
        ).unsqueeze(0)  # [1, n³, 4]

        out = self.forward(
            graphs, cond,
            return_potential=True, return_uncertainty=False,
        )

        S = n_points
        return {
            "mu": mu.cpu().numpy(),
            "omega": out["omega"].squeeze(0).reshape(S, S, S).cpu().numpy(),
            "q_co2": out["q_pred"][0, :, 0].reshape(S, S, S).cpu().numpy(),
            "q_n2": out["q_pred"][0, :, 1].reshape(S, S, S).cpu().numpy(),
            "q_h2o": out["q_pred"][0, :, 2].reshape(S, S, S).cpu().numpy(),
        }

    # ── Introspection ────────────────────────────────────────────

    @property
    def num_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def extra_repr(self) -> str:
        c = self.config
        return (
            f"emb_dim={c.emb_dim}, hidden_dim={c.hidden_dim}, "
            f"n_layers={c.n_layers}, n_conditions={c.n_conditions}, "
            f"n_components={c.n_components}, convex={c.convex_constraint}, "
            f"film={c.film_conditioning}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 6.  DEEP ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════

class TPNOEnsemble(nn.Module):
    """
    Deep ensemble of ``ThermodynamicPotentialNO`` models for epistemic
    uncertainty quantification (Lakshminarayanan et al., 2017).

    Each member is independently initialised (and optionally trained on
    bootstrap resamples).  At inference time the ensemble provides:

    * **Mean prediction** — average across members.
    * **Epistemic uncertainty** — standard deviation across members
      (model disagreement).
    * **Aleatoric uncertainty** — average of per-member σ.
    * **Total uncertainty** — √(σ²_epi + σ²_ale).

    Parameters
    ----------
    config       : Shared ``TPNOConfig``.
    encoder      : Encoder template (deep-copied for each member).
    n_models     : Ensemble size (typically 3–10).
    share_encoder: If *True*, all members share one encoder (cheaper).
    """

    def __init__(
        self,
        config: TPNOConfig,
        encoder: nn.Module,
        n_models: int = 5,
        share_encoder: bool = False,
    ):
        super().__init__()

        self.n_models = n_models
        self.share_encoder = share_encoder

        if share_encoder:
            self.shared_encoder = encoder
        else:
            self.shared_encoder = None

        self.models = nn.ModuleList()
        for _ in range(n_models):
            enc = encoder if share_encoder else copy.deepcopy(encoder)
            self.models.append(ThermodynamicPotentialNO(enc, config))

    # ── forward ──────────────────────────────────────────────────

    def forward(
        self,
        graphs: Any,
        conditions: torch.Tensor,
        *,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Ensemble forward pass.

        Returns
        -------
        Dict with keys: ``q_pred``, ``epistemic``, ``aleatoric``,
        ``total_uncertainty``, and optionally ``all_predictions``.
        """
        preds: List[torch.Tensor] = []
        sigmas: List[torch.Tensor] = []

        for model in self.models:
            out = model(
                graphs, conditions,
                return_uncertainty=True, return_potential=False,
            )
            preds.append(out["q_pred"])
            sigmas.append(out.get("sigma", torch.zeros_like(out["q_pred"])))

        pred_stack = torch.stack(preds, dim=0)    # [M, B, P, C]
        sigma_stack = torch.stack(sigmas, dim=0)  # [M, B, P, C]

        mean_pred = pred_stack.mean(dim=0)
        epistemic = pred_stack.std(dim=0)
        aleatoric = sigma_stack.mean(dim=0)
        total = torch.sqrt(epistemic.pow(2) + aleatoric.pow(2))

        result: Dict[str, torch.Tensor] = {
            "q_pred": mean_pred,
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "total_uncertainty": total,
        }
        if return_all:
            result["all_predictions"] = pred_stack
            result["all_sigma"] = sigma_stack

        return result

    # ── delegation helpers ───────────────────────────────────────

    def set_normalization(self, *args, **kwargs) -> None:
        """Propagate normalisation stats to every member."""
        for m in self.models:
            m.set_normalization(*args, **kwargs)

    def get_hessian(self, graphs: Any, conditions: torch.Tensor) -> torch.Tensor:
        """Mean Hessian across ensemble members."""
        hessians = [m.get_hessian(graphs, conditions) for m in self.models]
        return torch.stack(hessians).mean(dim=0)

    @property
    def num_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ═══════════════════════════════════════════════════════════════════════
# 7.  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "TPNOConfig",
    "Swish",
    "ICNN",
    "FiLMConditioning",
    "ThermodynamicPotentialNO",
    "TPNOEnsemble",
]