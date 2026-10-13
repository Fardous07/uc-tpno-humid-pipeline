from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TPNOConfig:
    emb_dim: int = 128
    hidden_dim: int = 256
    n_layers: int = 4
    n_conditions: int = 4       # μ_CO2, μ_N2, μ_H2O, T
    n_components: int = 3       # CO2, N2, H2O
    convex_constraint: str = "softplus"   # "softplus" | "exp" | "clamp"
    film_conditioning: bool = True        # required: T enters through FiLM
    dropout: float = 0.1
    # the potential network must be convex in mu. That needs a convex,
    # non-decreasing activation (softplus / relu) and NO LayerNorm on the
    # hidden-state path (Amos et al., ICML 2017).
    use_layer_norm: bool = False
    activation: str = "softplus"
    min_potential: float = 1e-6
    # --- target transform -------------------------------------------------
    # "log1p"    : train on log1p(loading); predictions are expm1'd back to
    #              real mol/kg. Compresses the heavy CO2 loading tail and
    #              rebalances the per-gas scale (helps CO2 + H2O). The
    #              normalization stats (q_mean/q_std) MUST be computed on
    #              log1p(loadings) — see AdsorptionDataset.compute_normalization_stats.
    # "identity" : linear standardization of raw loadings.
    target_transform: str = "log1p"


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name in ("swish", "silu"):
        return Swish()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "softplus":
        return nn.Softplus()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unknown activation: {name}")


# ---------------------------------------------------------------------------
# ICNN  (Input Convex Neural Network)
#
# Raw (unconstrained) parameters are stored; the non-negativity constraint is
# applied in the forward pass via _pos(raw_weight), so autograd differentiates
# through softplus.
# ---------------------------------------------------------------------------

class ICNN(nn.Module):
    """
    Input Convex Neural Network (Amos, Xu & Kolter, ICML 2017).

    The output is convex in x because
      * the weights on the hidden-state (z) path (U_raw, U_out_raw) are
        non-negative, via W_pos = softplus(W_raw - 5) + eps,
      * the activation is convex AND non-decreasing (softplus or relu),
      * x enters each layer only through affine terms (unconstrained
        weights, optionally scaled/shifted by FiLM -- still affine in x),
      * the output softplus is convex and non-decreasing, and dropout masks
        are non-negative scalings.
    A non-negative combination of convex functions composed with a convex
    non-decreasing function is convex, so convexity holds by induction over
    layers for ANY parameter values. LayerNorm or a non-monotone activation
    (e.g. swish) breaks this, so they are rejected below.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int = 3,
        convex_constraint: str = "softplus",
        dropout: float = 0.1,
        activation: str = "softplus",
        use_layer_norm: bool = False,
        min_potential: float = 1e-6,
    ):
        super().__init__()
        if activation.lower() not in ("softplus", "relu"):
            raise ValueError(
                f"ICNN activation '{activation}' is not convex and non-decreasing; "
                "use 'softplus' (or 'relu') to keep the potential convex."
            )
        if use_layer_norm:
            raise ValueError(
                "LayerNorm on the ICNN hidden path breaks convexity; set use_layer_norm=False."
            )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.convex_constraint = convex_constraint
        self.use_layer_norm = use_layer_norm
        self._min_potential = min_potential
        self.act = _make_activation(activation)

        # --- first layer (unconstrained — no z passthrough yet) ---
        self.W0 = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(self.W0.weight, gain=1.0)
        nn.init.zeros_(self.W0.bias)

        # --- hidden layers ---
        # U_raw : raw parameters for the z-passthrough (constrained non-neg
        #         via _pos() in forward — correct gradient flow)
        # W_layers : skip connections from x (unconstrained)
        self.U_raw    = nn.ParameterList()
        self.U_bias   = nn.ParameterList()
        self.W_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.norms    = nn.ModuleList() if use_layer_norm else None

        for _ in range(n_layers - 1):
            # Initialise small positive so softplus(w - 5) ≈ 0
            # → near-zero passthrough at the start of training
            U_raw = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
            nn.init.uniform_(U_raw, 0.0, 0.1)
            self.U_raw.append(U_raw)

            U_b = nn.Parameter(torch.zeros(hidden_dim))
            self.U_bias.append(U_b)

            W = nn.Linear(input_dim, hidden_dim)
            nn.init.xavier_uniform_(W.weight, gain=0.1)
            nn.init.zeros_(W.bias)
            self.W_layers.append(W)

            self.dropouts.append(nn.Dropout(dropout))
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(hidden_dim))

        # --- output layer ---
        self.U_out_raw  = nn.Parameter(torch.empty(1, hidden_dim))
        nn.init.uniform_(self.U_out_raw, 0.0, 0.1)
        self.U_out_bias = nn.Parameter(torch.zeros(1))

        self.W_out = nn.Linear(input_dim, 1)
        nn.init.xavier_uniform_(self.W_out.weight, gain=0.01)
        nn.init.zeros_(self.W_out.bias)

        self.b_out = nn.Parameter(torch.zeros(1))

    def _pos(self, w: torch.Tensor) -> torch.Tensor:
        """
        Non-negativity constraint applied IN the forward pass so that
        autograd differentiates through it (correct gradient flow).
        """
        if self.convex_constraint == "softplus":
            return F.softplus(w - 5.0) + 1e-6
        if self.convex_constraint == "exp":
            return torch.exp(w.clamp(max=10.0)) + 1e-6
        # clamp — gradient is 0 when w < 0 (suboptimal but still valid)
        return w.clamp(min=0.0)

    def forward(
        self,
        x: torch.Tensor,
        film_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        x           : [N, input_dim]
        film_params : optional (gamma, beta) each [N, hidden_dim]
        returns phi : [N, 1]  (strictly positive, convex in x)
        """
        # First layer
        z = self.W0(x)
        if film_params is not None:
            gamma, beta = film_params
            z = z * gamma + beta
        z = self.act(z)

        # Hidden layers — reparameterised U weights applied inline
        for i in range(self.n_layers - 1):
            W_pos  = self._pos(self.U_raw[i])              # [H, H] non-neg
            z_pass = F.linear(z, W_pos, self.U_bias[i])   # z passthrough
            z_skip = self.W_layers[i](x)                   # skip from x
            if film_params is not None:
                # Modulate the affine-in-x skip term: stays affine in x,
                # so ICNN convexity is preserved while every layer is
                # conditioned on the MOF embedding.
                z_skip = z_skip * gamma + beta
            z = z_pass + z_skip
            if self.use_layer_norm:
                z = self.norms[i](z)
            z = self.act(z)
            z = self.dropouts[i](z)

        # Output layer
        W_out_pos = self._pos(self.U_out_raw)              # [1, H] non-neg
        omega = (
            F.linear(z, W_out_pos, self.U_out_bias)
            + self.W_out(x)
            + self.b_out
        )
        # Phi = -Omega = pV >= 0; softplus keeps it positive and convex.
        omega = F.softplus(omega) + self._min_potential
        return omega


# ---------------------------------------------------------------------------
# FiLM conditioning  (MOF embedding → scale/shift for ICNN first layer)
# ---------------------------------------------------------------------------

class FiLMConditioning(nn.Module):
    """
    Feature-wise Linear Modulation.
    Near-identity initialisation: γ ≈ 1, β ≈ 0 at the start of training.
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
        # Near-identity init: γ→1, β→0
        nn.init.normal_(self.gamma_net[-1].weight, std=0.01)
        nn.init.ones_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.gamma_net(h), self.beta_net(h)


# ---------------------------------------------------------------------------
# ThermodynamicPotentialNO  (single model)
# ---------------------------------------------------------------------------

class ThermodynamicPotentialNO(nn.Module):
    """
    Thermodynamic Potential Neural Operator.

    Architecture
    ------------
    encoder(graph)          → h ∈ ℝ^{emb_dim}           (MOF embedding)
    FiLM([h, T̃])            → (γ, β)                     (MOF- and T-specific modulation)
    ICNN(μ̃; γ, β)           → Φ = −Ω ≥ 0, convex in μ̃   (pressure-like potential, pV)
    n_i = c/σ_i · ∂Φ/∂μ̃_i = c · ∂Φ/∂μ_i                 (loadings, physical mol/kg)

    Physics guaranteed for ANY parameter values (no penalty needed):
      * ∂n_i/∂μ_j = c · ∂²Φ/∂μ_i∂μ_j is symmetric (Maxwell relations) and
        positive semi-definite (thermodynamic stability; equals β·Cov(N_i, N_j)
        in the grand canonical ensemble), so in particular ∂n_i/∂μ_i ≥ 0.
    c is a constant (mean of the μ standard deviations), so these hold in
    physical units, not just in normalised ones.

    Ω is concave in μ, so Φ = −Ω is convex and ∂n_i/∂μ_i ≥ 0. Loadings are
    produced directly in physical units (a per-component transform would
    break Maxwell symmetry); the log/standardisation transform lives only in
    the loss (normalize_q).

    Normalization
    -------------
    The normalization composes an optional target transform
    (config.target_transform) with standardization:
        normalize_q(q)   :  real → (T(q) - q_mean) / q_std      (for the loss)
        denormalize_q(z) :  z → T^{-1}(z * q_std + q_mean)      (→ real mol/kg)
    with T = log1p (inverse expm1) by default. q_pred therefore always comes
    back in physical mol/kg, so eval / PVSA / KPI code is unaffected.
    """

    def __init__(self, encoder: nn.Module, config: TPNOConfig):
        super().__init__()
        self.config = config
        self.encoder = encoder

        if not config.film_conditioning:
            raise ValueError("film_conditioning must be True: temperature enters the potential through FiLM.")
        n_extra = config.n_conditions - config.n_components   # T (and any other non-μ inputs)
        # FiLM sees the MOF embedding AND the non-μ conditions (T), so Φ is
        # convex in μ for each fixed (MOF, T) -- exactly what thermodynamics
        # requires -- without forcing convexity in T.
        self.film: Optional[FiLMConditioning] = FiLMConditioning(
            config.emb_dim + n_extra, config.hidden_dim, config.dropout
        )

        self.icnn = ICNN(
            input_dim=config.n_components,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            convex_constraint=config.convex_constraint,
            dropout=config.dropout,
            activation=config.activation,
            use_layer_norm=config.use_layer_norm,
            min_potential=config.min_potential,
        )

        # Aleatoric uncertainty head (heteroscedastic), evaluated PER
        # CONDITION from [h, normalised conditions]. Outputs log σ² of the
        # loading in the loss's normalised log1p space.
        self.log_var_net = nn.Sequential(
            nn.Linear(config.emb_dim + config.n_conditions, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 2, config.n_components),
        )
        # Init: σ_z = 1 (log σ² = 0), the scale of early residuals in the
        # standardised target space.
        nn.init.zeros_(self.log_var_net[-1].weight)
        nn.init.zeros_(self.log_var_net[-1].bias)

        # Auxiliary direct head: [h, mu_norm] -> q_norm.
        # Provides a FIRST-ORDER gradient path to the encoder (the ICNN
        # path only reaches the encoder via a mixed second derivative,
        # which can make the encoder collapse).
        self.aux_head = nn.Sequential(
            nn.Linear(config.emb_dim + config.n_conditions, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.n_components),
        )

        # Normalization buffers (set by set_normalization before training).
        # q_mean / q_std are stats of the TRANSFORMED target
        # (e.g. log1p(loading)) when config.target_transform == "log1p".
        self.register_buffer("mu_mean", torch.zeros(config.n_conditions))
        self.register_buffer("mu_std",  torch.ones(config.n_conditions))
        self.register_buffer("q_mean",  torch.zeros(config.n_components))
        self.register_buffer("q_std",   torch.ones(config.n_components))
        # c in n = c · ∂Φ/∂μ: a fixed constant (set with the normalisation),
        # so Maxwell symmetry / PSD hold exactly in physical units.
        self.register_buffer("n_scale", torch.ones(()))

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def set_normalization(
        self,
        mu_mean: torch.Tensor,
        mu_std:  torch.Tensor,
        q_mean:  torch.Tensor,
        q_std:   torch.Tensor,
    ) -> None:
        """Call this BEFORE training with statistics from the training set."""
        self.mu_mean.copy_(mu_mean)
        self.mu_std.copy_(mu_std)
        self.q_mean.copy_(q_mean)
        self.q_std.copy_(q_std)
        # c = mean σ_μ, so n ≈ ∂Φ/∂μ̃ in magnitude at initialisation.
        self.n_scale.fill_(float(self.mu_std[: self.config.n_components].mean()))

    def normalize_mu(self, mu: torch.Tensor) -> torch.Tensor:
        return (mu - self.mu_mean) / (self.mu_std + 1e-8)

    # --- composable target transform T and its inverse ----------

    def _fwd_transform(self, q: torch.Tensor) -> torch.Tensor:
        """Real loading → transform space T(q)."""
        if getattr(self.config, "target_transform", "identity") == "log1p":
            # clamp keeps log1p in-domain (x > -1) for any stray negatives
            return torch.log1p(q.clamp(min=-1.0 + 1e-6))
        return q

    def _inv_transform(self, y: torch.Tensor) -> torch.Tensor:
        """Transform space → real loading T^{-1}(y)."""
        if getattr(self.config, "target_transform", "identity") == "log1p":
            # clamp the exponent to avoid expm1 overflow on pathological
            # early-training outputs (expm1(10) ≈ 2.2e4, still finite)
            return torch.expm1(y.clamp(max=10.0))
        return y

    def normalize_q(self, q: torch.Tensor) -> torch.Tensor:
        """
        Real loading (mol/kg) → normalized transform space.
        Used by the loss so it compares in the (tail-compressed) log space.
        """
        return (self._fwd_transform(q) - self.q_mean) / (self.q_std + 1e-8)

    def denormalize_q(self, q_norm: torch.Tensor) -> torch.Tensor:
        """
        Normalized transform space → real loading (mol/kg).
        Applied to every prediction, so q_pred is always physical.
        """
        return self._inv_transform(q_norm * self.q_std + self.q_mean)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def freeze_encoder(self, freeze: bool = True) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = not freeze

    @property
    def num_parameters(self) -> Dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    # ------------------------------------------------------------------
    # Core _icnn_forward helper  (shared by forward and forward_with_embedding)
    # ------------------------------------------------------------------

    def _icnn_forward(
        self,
        h_flat: torch.Tensor,        # [B*P, emb_dim]
        cond_flat: torch.Tensor,     # [B*P, n_conditions]  (UN-normalised)
        B: int,
        P: int,
        return_potential: bool,
        return_uncertainty: bool,
        return_hessian: bool,
        h_for_sigma: torch.Tensor,   # [B, emb_dim]  (for log_var_net)
    ) -> Dict[str, torch.Tensor]:
        """
        Shared computation kernel used by both forward() and
        forward_with_embedding().
        """
        n_comp = self.config.n_components
        cond_norm = self.normalize_mu(cond_flat)   # [B*P, D]  standardised (μ̃, T̃)
        need_graph = self.training or return_hessian
        mu_std = self.mu_std[:n_comp].float()       # σ_i of each μ_i

        # Potential + derivatives in fp32 even under AMP: autograd-derived
        # loadings are too precision-sensitive for fp16.
        with torch.enable_grad(), torch.amp.autocast("cuda", enabled=False):
            h_flat = h_flat.float()
            cond_norm = cond_norm.float()

            # Fresh leaf for the μ̃ part only; T̃ (and any other extra
            # condition) goes to FiLM, so Φ is convex in μ̃ alone.
            mu = cond_norm[:, :n_comp].detach().requires_grad_(True)
            extra = cond_norm[:, n_comp:]

            film_params = self.film(torch.cat([h_flat, extra], dim=-1))
            phi_flat = self.icnn(mu, film_params)     # [B*P, 1]  Φ = −Ω

            # n_i = c · ∂Φ/∂μ_i = (c/σ_i) · ∂Φ/∂μ̃_i   (physical mol/kg)
            grads = autograd.grad(
                outputs=phi_flat,
                inputs=mu,
                grad_outputs=torch.ones_like(phi_flat),
                create_graph=need_graph,
                retain_graph=need_graph,
            )[0]                                      # [B*P, n_comp]
            q_flat = grads * (self.n_scale / mu_std)  # [B*P, n_comp]

            # ∂n_i/∂μ_j = (∂n_i/∂μ̃_j)/σ_j = c ∂²Φ/∂μ_i∂μ_j : symmetric PSD.
            if return_hessian:
                rows: List[torch.Tensor] = []
                for i in range(n_comp):
                    row_i = autograd.grad(
                        outputs=q_flat[:, i].sum(),
                        inputs=mu,
                        retain_graph=True,
                        create_graph=self.training,
                    )[0] / mu_std                     # [B*P, n_comp]
                    rows.append(row_i)
                hess_flat = torch.stack(rows, dim=1)  # [B*P, n_comp, n_comp]

        q_pred = q_flat.reshape(B, P, n_comp)
        output: Dict[str, torch.Tensor] = {"q_pred": q_pred}

        # Auxiliary direct prediction (first-order encoder gradient path).
        # Training signal only -- never used as the model's prediction.
        q_aux_norm = self.aux_head(torch.cat([h_flat, cond_norm], dim=-1))
        output["q_aux"] = self.denormalize_q(q_aux_norm.reshape(B, P, n_comp))

        if return_potential:
            output["phi"]   = phi_flat.reshape(B, P, 1)
            output["omega"] = -output["phi"]          # grand potential Ω = −Φ

        if return_hessian:
            output["hessian"] = hess_flat.reshape(B, P, n_comp, n_comp)

        if return_uncertainty:
            # Per-condition aleatoric uncertainty (σ varies with P, T and RH).
            # h is detached: the variance head is fitted ON TOP of the
            # embedding and cannot pull the encoder (its NLL is large early
            # in training and would otherwise distort the representation).
            log_var = self.log_var_net(
                torch.cat([h_flat.detach(), cond_norm], dim=-1)
            ).reshape(B, P, n_comp)                    # log σ² in normalised log space
            # Smoothly bounded to [-10, 10] (σ_z in ~[0.007, 150]) so the NLL
            # cannot overflow or collapse; tanh keeps gradients everywhere.
            log_var = 10.0 * torch.tanh(log_var / 10.0)
            sigma_norm = torch.exp(0.5 * log_var)
            output["log_var"]    = log_var
            output["sigma_norm"] = sigma_norm
            # Physical σ (mol/kg) by the delta method through normalize_q:
            # z = (T(n) − m)/s  ⇒  dn/dz = s / T'(n), T = log1p ⇒ (1+n)·s.
            n_det = q_pred.detach().clamp(min=0.0)
            if getattr(self.config, "target_transform", "identity") == "log1p":
                dn_dz = (1.0 + n_det) * self.q_std
            else:
                dn_dz = self.q_std.expand_as(n_det)
            output["sigma"] = sigma_norm * dn_dz

        return output

    # ------------------------------------------------------------------
    # forward  (runs encoder internally)
    # ------------------------------------------------------------------

    def forward(
        self,
        graphs: Any,
        conditions: torch.Tensor,
        *,
        return_potential:   bool = False,
        return_uncertainty: bool = True,
        return_hessian:     bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        graphs     : torch_geometric Batch
        conditions : [B, P, D] or [B, D] (single condition)
        """
        if conditions.dim() == 2:
            conditions = conditions.unsqueeze(1)
        B, P, C = conditions.shape

        h = self.encoder(graphs)                          # [B, emb_dim]
        h_exp  = h.unsqueeze(1).expand(-1, P, -1)
        h_flat = h_exp.reshape(B * P, -1)                # [B*P, emb_dim]
        cond_flat = conditions.reshape(B * P, C)

        return self._icnn_forward(
            h_flat, cond_flat, B, P,
            return_potential, return_uncertainty, return_hessian,
            h_for_sigma=h,
        )

    # ------------------------------------------------------------------
    # forward_with_embedding  (encoder already run externally)
    # ------------------------------------------------------------------

    def forward_with_embedding(
        self,
        mof_embedding: torch.Tensor,
        conditions:    torch.Tensor,
        *,
        return_potential:   bool = False,
        return_uncertainty: bool = True,
        return_hessian:     bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Use when the encoder has already been run (e.g. in ensemble with
        shared encoder) to avoid re-running it for every ensemble member.

        mof_embedding : [B, emb_dim]
        conditions    : [B, P, D] or [B, D]
        """
        if conditions.dim() == 2:
            conditions = conditions.unsqueeze(1)
        B, P, C = conditions.shape

        h_exp  = mof_embedding.unsqueeze(1).expand(-1, P, -1)
        h_flat = h_exp.reshape(B * P, -1)
        cond_flat = conditions.reshape(B * P, C)

        return self._icnn_forward(
            h_flat, cond_flat, B, P,
            return_potential, return_uncertainty, return_hessian,
            h_for_sigma=mof_embedding,
        )

    # ------------------------------------------------------------------
    # Hessian convenience  (for physics loss — no caching)
    # ------------------------------------------------------------------

    def get_hessian(
        self,
        graphs: Any,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return ∂n_i/∂μ_j — shape [B, P, n_comp, n_comp].

        Note: hessian caching was removed because id() reuse after GC and
        non-unique sum-based keys caused stale cache hits.  This function
        is only called during physics loss computation (once per batch)
        so the overhead is negligible.
        """
        out = self.forward(
            graphs, conditions,
            return_hessian=True,
            return_uncertainty=False,
            return_potential=False,
        )
        return out["hessian"]

    # ------------------------------------------------------------------
    # Potential surface  (visualisation / analysis)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_potential_surface(
        self,
        graphs: Any,
        mu_range: Tuple[float, float] = (-20.0, 5.0),
        n_points: int = 50,
        T: float = 313.0,
    ) -> Dict[str, Any]:
        """
        Compute Ω and loadings on a dense 3-D μ grid (for one MOF).
        Returns numpy arrays for plotting.

        Note: @no_grad() is on the outer scope, but _icnn_forward uses
        torch.enable_grad() internally so autograd.grad still works.
        """
        import numpy as np
        device = next(self.parameters()).device
        mu = torch.linspace(mu_range[0], mu_range[1], n_points, device=device)
        mu1, mu2, mu3 = torch.meshgrid(mu, mu, mu, indexing="ij")
        S = n_points
        cond = torch.stack(
            [mu1.flatten(), mu2.flatten(), mu3.flatten(),
             torch.full((S**3,), T, device=device)],
            dim=1,
        ).unsqueeze(0)                               # [1, S^3, 4]

        out = self.forward(
            graphs, cond,
            return_potential=True,
            return_uncertainty=False,
            return_hessian=False,
        )
        return {
            "mu":    mu.cpu().numpy(),
            "omega": out["omega"].squeeze(0).reshape(S, S, S).cpu().numpy(),
            "q_co2": out["q_pred"][0, :, 0].reshape(S, S, S).cpu().numpy(),
            "q_n2":  out["q_pred"][0, :, 1].reshape(S, S, S).cpu().numpy(),
            "q_h2o": out["q_pred"][0, :, 2].reshape(S, S, S).cpu().numpy(),
        }


# ---------------------------------------------------------------------------
# TPNOEnsemble
# ---------------------------------------------------------------------------

class TPNOEnsemble(nn.Module):
    """
    Ensemble of M independent TPNO models for epistemic UQ.

    When share_encoder=True:
    - The encoder runs ONCE per forward call (not M times).
    - Each ensemble member's ICNN + FiLM are independent.
    - ~M× speedup in encoder inference vs. running all M models separately.

    With share_encoder=True the shared encoder is kept as a plain attribute
    (not a registered submodule) so its parameters appear only once in
    self.parameters(); it is still trained through self.models, which all
    hold the same encoder object.
    """

    def __init__(
        self,
        config:        TPNOConfig,
        encoder:       nn.Module,
        n_models:      int = 5,
        share_encoder: bool = False,
    ):
        super().__init__()
        self.n_models      = n_models
        self.share_encoder = share_encoder

        if share_encoder:
            # plain attribute, NOT a registered submodule, to avoid
            # duplicate parameter registration in self.parameters().
            # The encoder IS still registered inside self.models[0].encoder
            # (all models share the same object), so it will be trained.
            self._shared_encoder_ref: Optional[nn.Module] = encoder
        else:
            self._shared_encoder_ref = None

        self.models = nn.ModuleList()
        for _ in range(n_models):
            enc = encoder if share_encoder else copy.deepcopy(encoder)
            self.models.append(ThermodynamicPotentialNO(enc, config))

    def set_normalization(
        self,
        mu_mean: torch.Tensor,
        mu_std:  torch.Tensor,
        q_mean:  torch.Tensor,
        q_std:   torch.Tensor,
    ) -> None:
        for model in self.models:
            model.set_normalization(mu_mean, mu_std, q_mean, q_std)

    def normalize_q(self, q: torch.Tensor) -> torch.Tensor:
        """Delegate to member 0 (all members share the same normalization)."""
        return self.models[0].normalize_q(q)

    def denormalize_q(self, q_norm: torch.Tensor) -> torch.Tensor:
        return self.models[0].denormalize_q(q_norm)

    def forward(
        self,
        graphs:     Any,
        conditions: torch.Tensor,
        *,
        return_all: bool = False,
        return_uncertainty: bool = True,
        return_potential: bool = False,
        return_hessian: bool = False,
    ) -> Dict[str, torch.Tensor]:
        preds:  List[torch.Tensor] = []
        sigmas: List[torch.Tensor] = []
        auxs:   List[torch.Tensor] = []
        logvs:  List[torch.Tensor] = []

        if self.share_encoder and self._shared_encoder_ref is not None:
            # Run encoder once, pass embedding to each member's ICNN
            h = self._shared_encoder_ref(graphs)
            for model in self.models:
                out = model.forward_with_embedding(
                    h, conditions,
                    return_uncertainty=True,
                    return_potential=False,
                )
                preds.append(out["q_pred"])
                sigmas.append(out.get("sigma", torch.zeros_like(out["q_pred"])))
                if "q_aux" in out:
                    auxs.append(out["q_aux"])
                if "log_var" in out:
                    logvs.append(out["log_var"])
        else:
            for model in self.models:
                out = model.forward(
                    graphs, conditions,
                    return_uncertainty=True,
                    return_potential=False,
                )
                preds.append(out["q_pred"])
                sigmas.append(out.get("sigma", torch.zeros_like(out["q_pred"])))
                if "q_aux" in out:
                    auxs.append(out["q_aux"])
                if "log_var" in out:
                    logvs.append(out["log_var"])

        pred_stack  = torch.stack(preds,  dim=0)   # [M, B, P, C]
        sigma_stack = torch.stack(sigmas, dim=0)   # [M, B, P, C]

        mean_pred = pred_stack.mean(dim=0)          # [B, P, C]
        epistemic = pred_stack.std(dim=0)           # [B, P, C]  mol/kg
        aleatoric = sigma_stack.mean(dim=0)         # [B, P, C]  mol/kg (physical σ)
        # both terms in mol/kg
        total     = (epistemic.pow(2) + aleatoric.pow(2)).sqrt()

        result: Dict[str, torch.Tensor] = {
            "q_aux":             (torch.stack(auxs, dim=0).mean(dim=0)
                                  if auxs else mean_pred),
            "q_pred":            mean_pred,
            "epistemic":         epistemic,
            "aleatoric":         aleatoric,
            "total_uncertainty": total,
        }
        # per-member outputs are returned so the loss trains every member on
        # its OWN prediction (a deep ensemble); training only the mean would
        # let the members co-adapt and make their spread meaningless.
        result["all_predictions"] = pred_stack
        result["all_sigma"]       = sigma_stack
        if auxs:
            result["all_q_aux"] = torch.stack(auxs, dim=0)
        if logvs:
            result["all_log_var"] = torch.stack(logvs, dim=0)
        return result

    def get_hessian(
        self, graphs: Any, conditions: torch.Tensor
    ) -> torch.Tensor:
        """Average Hessian across ensemble members."""
        return torch.stack(
            [m.get_hessian(graphs, conditions) for m in self.models]
        ).mean(dim=0)

    @property
    def num_parameters(self) -> Dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ---------------------------------------------------------------------------
# Thermodynamic validator  (run every N epochs — no_grad on outer scope)
# ---------------------------------------------------------------------------

class ThermodynamicValidator:
    """
    Post-hoc validation of thermodynamic consistency.

    NOTE: methods do NOT use @torch.no_grad().  TPNO derives loadings via
    autograd.grad(omega, mu) internally; torch.no_grad() would silently
    prevent requires_grad_(True) on mu from working and crash that call.
    Call model.eval() before using this validator — that is sufficient
    to disable dropout / batchnorm tracking without breaking autograd.
    """

    def __init__(self, n_test_points: int = 100):
        self.n_test_points = n_test_points

    def check_convexity(
        self,
        model:      nn.Module,
        graphs:     Any,
        conditions: torch.Tensor,
        n_pairs:    int = 200,
    ) -> Dict[str, float]:
        """
        Jensen's inequality check on Φ = −Ω (convex in μ at fixed MOF and T):
        Φ(λc1 + (1-λ)c2) ≤ λΦ(c1) + (1-λ)Φ(c2).
        Pairs must share T; pass conditions at a single temperature.
        """
        B, P = conditions.shape[:2]
        D    = conditions.shape[-1]
        device = conditions.device

        out   = model(graphs, conditions, return_potential=True, return_uncertainty=False)
        omega = out["phi"]                            # [B, P, 1]  Φ (convex)

        idx1 = torch.randint(0, P, (B, n_pairs), device=device)
        idx2 = torch.randint(0, P, (B, n_pairs), device=device)
        lam  = torch.rand(B, n_pairs, 1, device=device)

        c1 = conditions.gather(1, idx1.unsqueeze(-1).expand(-1, -1, D))
        c2 = conditions.gather(1, idx2.unsqueeze(-1).expand(-1, -1, D))
        o1 = omega.gather(1, idx1.unsqueeze(-1))
        o2 = omega.gather(1, idx2.unsqueeze(-1))

        c_interp   = lam * c1 + (1 - lam) * c2
        out_interp = model(graphs, c_interp,
                           return_potential=True, return_uncertainty=False)
        o_interp   = out_interp["phi"]

        bound      = lam * o1 + (1 - lam) * o2
        violations = F.relu(o_interp - bound - 1e-4).squeeze(-1)

        return {
            "convexity_violation_rate": (violations > 0).float().mean().item(),
            "mean_convexity_violation": violations.mean().item(),
            "max_convexity_violation":  violations.max().item(),
        }

    def check_monotonicity(
        self,
        model:      nn.Module,
        graphs:     Any,
        conditions: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Check ∂n_i/∂μ_i ≥ 0 for each component.
        """
        out = model(graphs, conditions, return_uncertainty=False)
        q   = out["q_pred"]
        mu  = conditions[..., : q.shape[-1]]

        mu_diff    = mu[:, 1:] - mu[:, :-1]
        q_diff     = q[:, 1:]  - q[:, :-1]
        mask       = (mu_diff > 0).float()
        violations = (q_diff < -1e-6).float() * mask

        return {
            "monotonicity_violation_rate":   violations.mean().item(),
            "monotonicity_by_component":     violations.mean(dim=(0, 1)).cpu().tolist(),
        }

    def check_henry_region(
        self,
        model:         nn.Module,
        graphs:        Any,
        low_pressure:  float = 1e-3,
        high_pressure: float = 1e-2,
        T:             float = 313.0,
        n_components:  int   = 3,
    ) -> Dict[str, float]:
        """
        Verify Henry's law: q(P_high)/q(P_low) ≈ P_high/P_low.
        Only checks CO2 and N2 (H2O at very low μ is numerically unstable).
        """
        device = next(model.parameters()).device
        # μ in the model is RT·ln(f / 1 bar) in kJ/mol (src.utils.chemistry).
        # Ideal-gas fugacity ≈ pressure at these dilute conditions.
        RT = 8.314462618e-3 * T
        mu_lo  = RT * math.log(low_pressure)
        mu_hi  = RT * math.log(high_pressure)

        n_cond = model.config.n_conditions
        cond   = torch.zeros(2, n_cond, device=device)
        cond[0, :n_components] = mu_lo
        cond[1, :n_components] = mu_hi
        cond[:, -1] = T
        # Set H2O very low to avoid numerical instability in ratio check
        if n_components >= 3:
            cond[:, 2] = RT * math.log(1e-10)   # water absent (trace), as in the data
        cond = cond.unsqueeze(0)                    # [1, 2, n_cond]

        out = model(graphs, cond, return_uncertainty=False)
        q   = out["q_pred"]                         # [1, 2, n_comp]

        expected = high_pressure / low_pressure
        # Only check CO2 (idx 0) and N2 (idx 1) — both physically meaningful
        ratio = q[0, 1, :2] / (q[0, 0, :2] + 1e-8)   # [2]
        error = (ratio - expected).abs() / expected

        return {
            "henry_mean_error_co2_n2": error.mean().item(),
            "henry_co2_ratio":         ratio[0].item(),
            "henry_n2_ratio":          ratio[1].item(),
            "expected_ratio":          expected,
        }

    def check_maxwell_relations(
        self,
        model:      nn.Module,
        graphs:     Any,
        conditions: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Check Hessian symmetry ∂n_i/∂μ_j ≈ ∂n_j/∂μ_i.
        Requires model to support return_hessian=True.
        """
        out  = model(graphs, conditions,
                     return_hessian=True, return_uncertainty=False)
        hess = out["hessian"]                        # [B, P, C, C]
        asym = (hess - hess.transpose(-1, -2)).abs()
        return {
            "maxwell_mean_asymmetry": asym.mean().item(),
            "maxwell_max_asymmetry":  asym.max().item(),
        }


# ---------------------------------------------------------------------------

__all__ = [
    "TPNOConfig",
    "Swish",
    "ICNN",
    "FiLMConditioning",
    "ThermodynamicPotentialNO",
    "TPNOEnsemble",
    "ThermodynamicValidator",
]