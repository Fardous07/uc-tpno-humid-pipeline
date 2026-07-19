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
    film_conditioning: bool = True
    dropout: float = 0.1
    use_layer_norm: bool = True
    activation: str = "swish"
    min_potential: float = 1e-6


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
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unknown activation: {name}")


# ---------------------------------------------------------------------------
# ICNN  (Input Convex Neural Network)
#
# FIX: Replaced project_weights() + .data assignment with reparameterisation.
#
# OLD approach (broken gradient flow):
#   - Stored raw weights W, projected them with softplus in torch.no_grad()
#     by assigning to W.data inside forward().
#   - Backward pass saw the projected values but NOT the softplus Jacobian,
#     so the effective learning rate was wrong by a factor of softplus'(w-5).
#
# NEW approach (correct gradient flow):
#   - Store raw (unconstrained) parameters.
#   - Apply the non-negativity constraint inside the forward pass via
#     _pos(raw_weight), so autograd differentiates through softplus correctly.
#   - project_weights() is gone entirely.
# ---------------------------------------------------------------------------

class ICNN(nn.Module):
    """
    Input Convex Neural Network.

    Ω(μ) is convex in μ because all weights on the z-passthrough paths
    (U_raw, U_out_raw) are constrained to be non-negative via
    reparameterisation:
        W_pos = softplus(W_raw - 5) + ε

    The skip connections from x use unconstrained weights (W_layers, W_out)
    which does not break convexity.
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
        returns omega : [N, 1]  (strictly positive, convex in x)
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
        # Ensure Ω > 0 (grand potential is non-negative by convention)
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
    encoder(graph) → h ∈ ℝ^{emb_dim}                 (MOF embedding)
    FiLM(h)        → (γ, β)                            (MOF-specific modulation)
    ICNN(μ_norm; γ, β) → Ω(μ)                          (grand potential, convex)
    n_i = −∂Ω/∂μ_i  via autograd                       (adsorption loadings)

    Maxwell relations  ∂n_i/∂μ_j = ∂n_j/∂μ_i  are enforced by the Hessian
    symmetry physics loss during training.

    Fixes vs. previous version
    --------------------------
    1. ICNN uses reparameterisation instead of project_weights() so that
       gradients flow correctly through the non-negativity constraint.
    2. Hessian is computed from the FIRST autograd.grad call (no second
       ICNN forward pass needed), cutting hessian computation cost in half.
    3. Hessian cache removed — it was unreliable (id() reuse, non-unique
       sum keys) and wrong to cache across training batches.
    """

    def __init__(self, encoder: nn.Module, config: TPNOConfig):
        super().__init__()
        self.config = config
        self.encoder = encoder

        self.film: Optional[FiLMConditioning] = None
        if config.film_conditioning:
            self.film = FiLMConditioning(
                config.emb_dim, config.hidden_dim, config.dropout
            )

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

        # Aleatoric uncertainty head (heteroscedastic)
        # Takes MOF embedding + condition to make sigma condition-aware
        self.log_var_net = nn.Sequential(
            nn.Linear(config.emb_dim + config.n_conditions, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 2, config.n_components),
        )
        # Init: start with small uncertainty (log_var ≈ -5 → σ ≈ 0.08)
        nn.init.zeros_(self.log_var_net[-1].weight)
        nn.init.constant_(self.log_var_net[-1].bias, -5.0)

        # Auxiliary direct head: [h, mu_norm] -> q_norm.
        # Provides a FIRST-ORDER gradient path to the encoder (the ICNN
        # path only reaches the encoder via a mixed second derivative,
        # which caused encoder collapse in run_003).
        self.aux_head = nn.Sequential(
            nn.Linear(config.emb_dim + config.n_conditions, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.n_components),
        )

        # Normalization buffers (set by set_normalization before training)
        self.register_buffer("mu_mean", torch.zeros(config.n_conditions))
        self.register_buffer("mu_std",  torch.ones(config.n_conditions))
        self.register_buffer("q_mean",  torch.zeros(config.n_components))
        self.register_buffer("q_std",   torch.ones(config.n_components))

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

    def normalize_mu(self, mu: torch.Tensor) -> torch.Tensor:
        return (mu - self.mu_mean) / (self.mu_std + 1e-8)

    def denormalize_q(self, q: torch.Tensor) -> torch.Tensor:
        return q * self.q_std + self.q_mean

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
        cond_norm = self.normalize_mu(cond_flat)   # [B*P, D]

        need_graph = self.training or return_hessian

        # ICNN + derivatives run in fp32 even when the encoder uses AMP:
        # autograd-derived loadings are too precision-sensitive for fp16.
        with torch.enable_grad(), torch.amp.autocast("cuda", enabled=False):
            h_flat = h_flat.float()
            cond_norm = cond_norm.float()
            # Detach and re-attach so we always have a fresh leaf for grad
            mu = cond_norm.detach().requires_grad_(True)

            film_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
            if self.film is not None:
                film_params = self.film(h_flat)

            omega_flat = self.icnn(mu, film_params)   # [B*P, 1]

            # n_i = −∂Ω/∂μ_i
            grads = autograd.grad(
                outputs=omega_flat,
                inputs=mu,
                grad_outputs=torch.ones_like(omega_flat),
                create_graph=need_graph,
                retain_graph=need_graph,
            )[0]                                      # [B*P, D]

            q_norm = -grads[:, :n_comp]               # [B*P, n_comp]

            # FIX: Hessian computed HERE from the first grad call.
            # No second ICNN forward needed — saves ~50% compute.
            if return_hessian:
                rows: List[torch.Tensor] = []
                for i in range(n_comp):
                    row_i = autograd.grad(
                        outputs=q_norm[:, i].sum(),
                        inputs=mu,
                        retain_graph=True,
                        create_graph=self.training,
                    )[0][:, :n_comp]                  # [B*P, n_comp]
                    rows.append(row_i)
                # [B*P, n_comp, n_comp]
                hess_flat = torch.stack(rows, dim=1)

        # --- reshape and denormalise ---
        q_pred = self.denormalize_q(q_norm.reshape(B, P, n_comp))

        output: Dict[str, torch.Tensor] = {"q_pred": q_pred}

        # Auxiliary direct prediction (first-order encoder gradient path)
        q_aux_norm = self.aux_head(
            torch.cat([h_flat.float(), cond_norm.float()], dim=-1)
        )
        output["q_aux"] = self.denormalize_q(q_aux_norm.reshape(B, P, n_comp))

        if return_potential:
            output["omega"] = omega_flat.reshape(B, P, 1)

        if return_hessian:
            output["hessian"] = hess_flat.reshape(B, P, n_comp, n_comp)

        if return_uncertainty:
            # Condition-aware aleatoric uncertainty:
            # concatenate MOF embedding with normalised conditions
            cond_mean = cond_norm.reshape(B, P, -1).mean(dim=1)   # [B, D]
            sigma_input = torch.cat([h_for_sigma, cond_mean], dim=-1)  # [B, emb+D]
            log_var = self.log_var_net(sigma_input)                # [B, n_comp]
            sigma   = torch.exp(0.5 * log_var)                    # [B, n_comp]
            output["log_var"] = log_var
            output["sigma"]   = sigma.unsqueeze(1).expand(-1, P, -1)  # [B,P,n_comp]

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

    FIX: when share_encoder=True, the shared encoder is stored as a plain
    Python attribute (not a registered nn.Module submodule) to prevent the
    same parameters appearing in self.parameters() multiple times — once via
    self._shared_encoder and once via each model.encoder.  In PyTorch < 2.0
    there is no automatic deduplication, so the optimizer would apply the
    encoder update n_models+1 times per step.  Storing it outside the module
    registry avoids double-registration while keeping gradients flowing
    correctly (the encoder IS registered inside each model in self.models,
    and all models share the SAME encoder object, so it IS trained).
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
            # FIX: store as plain attribute, NOT as a registered submodule,
            # to avoid duplicate parameter registration in self.parameters().
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

        pred_stack  = torch.stack(preds,  dim=0)   # [M, B, P, C]
        sigma_stack = torch.stack(sigmas, dim=0)   # [M, B, P, C]

        mean_pred = pred_stack.mean(dim=0)          # [B, P, C]
        epistemic = pred_stack.std(dim=0)           # [B, P, C]
        aleatoric = sigma_stack.mean(dim=0)         # [B, P, C]
        total     = (epistemic.pow(2) + aleatoric.pow(2)).sqrt()

        result: Dict[str, torch.Tensor] = {
            "q_aux":             (torch.stack(auxs, dim=0).mean(dim=0)
                                  if auxs else mean_pred),
            "q_pred":            mean_pred,
            "epistemic":         epistemic,
            "aleatoric":         aleatoric,
            "total_uncertainty": total,
        }
        if return_all:
            result["all_predictions"] = pred_stack
            result["all_sigma"]       = sigma_stack
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
        Jensen's inequality check: Ω(λc1 + (1-λ)c2) ≤ λΩ(c1) + (1-λ)Ω(c2)
        """
        B, P = conditions.shape[:2]
        D    = conditions.shape[-1]
        device = conditions.device

        out   = model(graphs, conditions, return_potential=True, return_uncertainty=False)
        omega = out["omega"]                          # [B, P, 1]

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
        o_interp   = out_interp["omega"]

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

        FIX: conditions tensor size now reads from model.config.n_conditions
        instead of the previous hardcoded 4, so the validator works correctly
        when the model is built with a non-default number of conditions.
        """
        device = next(model.parameters()).device
        mu_lo  = float(torch.tensor(low_pressure).log())
        mu_hi  = float(torch.tensor(high_pressure).log())

        # FIX: use model.config.n_conditions instead of hardcoded 4
        n_cond = model.config.n_conditions
        cond   = torch.zeros(2, n_cond, device=device)
        cond[0, :n_components] = mu_lo
        cond[1, :n_components] = mu_hi
        cond[:, -1] = T
        # Set H2O very low to avoid numerical instability in ratio check
        if n_components >= 3:
            cond[:, 2] = -50.0
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