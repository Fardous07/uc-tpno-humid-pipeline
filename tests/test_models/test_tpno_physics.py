"""
Physics guarantees of the UC-TPNO operator (run_009).

These are properties of the ARCHITECTURE, so they must hold for any
parameter values, not just after training. Each test randomises the
parameters and checks:

  1. Maxwell symmetry   dn_i/dmu_j == dn_j/dmu_i         (physical units)
  2. Stability / PSD    the Jacobian dn/dmu is positive semi-definite,
                        hence dn_i/dmu_i >= 0 (monotone isotherms)
  3. Convexity of Phi   Jensen's inequality along random segments in mu
  4. The ICNN rejects settings that break convexity (swish, LayerNorm)
  5. The variance head depends on the condition, and sigma is in mol/kg
  6. The ensemble exposes per-member outputs and the loss trains each one
  7. Error-bar weights, NaN handling and the Henry term behave as specified

Run:  python tests/test_models/test_tpno_physics.py   (or with pytest)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.operator.tpno import ICNN, TPNOConfig, ThermodynamicPotentialNO, TPNOEnsemble  # noqa: E402
from src.models.operator.losses import (  # noqa: E402
    LossConfig, ThermodynamicLoss, error_bar_weights, henry_law_loss,
)

EMB = 16
R_KJ = 8.314462618e-3


class FixedEncoder(nn.Module):
    """Stand-in encoder: one learnable embedding per MOF index in `graphs`."""

    def __init__(self, n_mofs: int = 4, emb: int = EMB):
        super().__init__()
        self.table = nn.Embedding(n_mofs, emb)

    def forward(self, graphs: torch.Tensor) -> torch.Tensor:
        return self.table(graphs)


def make_model(seed: int, perturb: float = 0.5) -> ThermodynamicPotentialNO:
    torch.manual_seed(seed)
    cfg = TPNOConfig(emb_dim=EMB, hidden_dim=32, n_layers=3, dropout=0.0)
    m = ThermodynamicPotentialNO(FixedEncoder(), cfg)
    m.set_normalization(
        mu_mean=torch.tensor([-8.0, -2.0, -25.0, 313.0]),
        mu_std=torch.tensor([6.0, 4.0, 12.0, 15.0]),
        q_mean=torch.tensor([0.5, 0.2, 0.1]),
        q_std=torch.tensor([0.6, 0.3, 0.4]),
    )
    with torch.no_grad():                      # move away from the init
        for p in m.parameters():
            p.add_(torch.randn_like(p) * perturb)
    m.eval()
    return m


def random_conditions(B: int, P: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    mu = torch.stack([
        -8 + 6 * torch.randn(B, P, generator=g),
        -2 + 4 * torch.randn(B, P, generator=g),
        -25 + 12 * torch.randn(B, P, generator=g),
    ], dim=-1)
    T = torch.tensor([298.15, 313.15, 333.15])[torch.randint(0, 3, (B, P), generator=g)]
    return torch.cat([mu, T.unsqueeze(-1)], dim=-1)


# ---------------------------------------------------------------- 1, 2
def test_maxwell_symmetry_and_psd():
    worst_asym, worst_eig = 0.0, 0.0
    for seed in range(8):
        m = make_model(seed)
        cond = random_conditions(4, 64, seed)
        out = m(torch.arange(4), cond, return_hessian=True, return_uncertainty=False)
        H = out["hessian"].double()                          # [B, P, 3, 3]  dn_i/dmu_j
        scale = H.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-12)
        asym = ((H - H.transpose(-1, -2)).abs() / scale).max().item()
        eig = torch.linalg.eigvalsh(0.5 * (H + H.transpose(-1, -2)))
        min_rel_eig = (eig.min(dim=-1).values / scale.squeeze(-1).squeeze(-1)).min().item()
        worst_asym = max(worst_asym, asym)
        worst_eig = min(worst_eig, min_rel_eig)
    assert worst_asym < 1e-4, f"Maxwell relations violated: relative asymmetry {worst_asym:.2e}"
    assert worst_eig > -1e-5, f"Jacobian not PSD: relative min eigenvalue {worst_eig:.2e}"


def test_monotone_isotherms_finite_difference():
    """dn_i/dmu_i >= 0 checked independently of autograd, by finite differences."""
    for seed in range(5):
        m = make_model(seed)
        cond = random_conditions(4, 32, seed)
        n0 = m(torch.arange(4), cond, return_uncertainty=False)["q_pred"]
        for i in range(3):
            c2 = cond.clone()
            c2[..., i] += 0.05                                  # +0.05 kJ/mol
            n1 = m(torch.arange(4), c2, return_uncertainty=False)["q_pred"]
            assert (n1[..., i] - n0[..., i] >= -1e-6).all(), f"n_{i} decreased with its own mu (seed {seed})"


# ---------------------------------------------------------------- 3
def test_phi_convex_in_mu():
    for seed in range(5):
        m = make_model(seed)
        g = torch.Generator().manual_seed(100 + seed)
        c1 = random_conditions(4, 64, seed)
        c2 = random_conditions(4, 64, seed + 50)
        c2[..., 3] = c1[..., 3]                                 # same T: convexity is in mu only
        lam = torch.rand(4, 64, 1, generator=g)
        cm = lam * c1 + (1 - lam) * c2
        f = lambda c: m(torch.arange(4), c, return_potential=True, return_uncertainty=False)["phi"]
        gap = f(cm) - (lam * f(c1) + (1 - lam) * f(c2))
        assert gap.max().item() < 1e-5, f"Jensen violated by {gap.max().item():.2e} (seed {seed})"


# ---------------------------------------------------------------- 4
def test_icnn_rejects_non_convex_settings():
    for kwargs in ({"activation": "swish"}, {"use_layer_norm": True}):
        try:
            ICNN(input_dim=3, hidden_dim=8, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"ICNN accepted a non-convex setting: {kwargs}")


# ---------------------------------------------------------------- 5
def test_sigma_depends_on_condition_and_is_physical():
    m = make_model(0, perturb=0.1)
    cond = random_conditions(2, 16, 0)
    out = m(torch.arange(2), cond, return_uncertainty=True)
    s = out["sigma_norm"]
    assert s.shape == (2, 16, 3)
    assert (s.std(dim=1) > 1e-6).all(), "sigma is constant across conditions"
    assert (out["log_var"].abs() <= 10.0 + 1e-6).all(), "log variance is not bounded"
    n = out["q_pred"].detach().clamp(min=0)
    expected = s * (1 + n) * m.q_std                            # delta method, log1p
    assert torch.allclose(out["sigma"], expected, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------- 6
def test_ensemble_members_are_trained_individually():
    torch.manual_seed(0)
    cfg = TPNOConfig(emb_dim=EMB, hidden_dim=32, n_layers=3, dropout=0.0)
    ens = TPNOEnsemble(cfg, FixedEncoder(), n_models=3, share_encoder=True)
    ens.set_normalization(torch.tensor([-8.0, -2.0, -25.0, 313.0]), torch.tensor([6.0, 4.0, 12.0, 15.0]),
                          torch.tensor([0.5, 0.2, 0.1]), torch.tensor([0.6, 0.3, 0.4]))
    ens.train()
    cond = random_conditions(2, 10, 1)
    out = ens(torch.arange(2), cond)
    for k in ("all_predictions", "all_log_var", "all_q_aux"):
        assert k in out and out[k].shape[0] == 3, k
    assert out["aleatoric"].shape == out["epistemic"].shape
    targets = torch.rand(2, 10, 3)
    loss = ThermodynamicLoss(LossConfig(lambda_hessian=0, lambda_monotonic=0,
                                        lambda_competition=0, use_nll=False))
    res = loss(out, targets, ens, torch.arange(2), cond, mask=torch.ones(2, 10, dtype=torch.bool))
    res["total"].backward()
    for i, member in enumerate(ens.models):
        g = sum(p.grad.abs().sum() for p in member.icnn.parameters() if p.grad is not None)
        assert g > 0, f"member {i} received no gradient"
    assert "nll_sigma" in res["physics"]
    lv_grad = sum(p.grad.abs().sum() for p in ens.models[0].log_var_net.parameters() if p.grad is not None)
    assert lv_grad > 0, "variance head is not trained"


# ---------------------------------------------------------------- 7
def test_error_weights():
    targets = torch.tensor([[[1.0, 0.5, 0.0], [2.0, 1.0, 0.3]]])
    err = torch.tensor([[[0.1, float("nan"), 0.0], [0.4, 0.05, 0.03]]])
    w = error_bar_weights(targets, err, torch.ones(3), log1p=True, floor=0.05,
                          mask=torch.ones(1, 2, dtype=torch.bool))
    assert torch.isfinite(w).all()
    assert abs(w.mean().item() - 1.0) < 1e-5
    assert w[0, 0, 0] > w[0, 1, 0], "the noisier point must get the smaller weight"


def test_henry_term_units():
    """Exact Henry data n = K(T) f with f = exp(mu/RT) must give zero loss."""
    T = torch.tensor([298.15] * 3 + [333.15] * 3)
    f = torch.tensor([1e-3, 3e-3, 1e-2] * 2)
    mu = R_KJ * T * torch.log(f)
    K = torch.where(T < 300, 4.0, 1.5)                        # K_H depends on T
    cond = torch.stack([mu, mu, mu, T], dim=-1).unsqueeze(0)
    q = (K * f).unsqueeze(-1).repeat(1, 3).unsqueeze(0)
    assert henry_law_loss(q, cond).item() < 1e-10
    q_bad = q.clone(); q_bad[0, 0] *= 3
    assert henry_law_loss(q_bad, cond).item() > 1e-3


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
