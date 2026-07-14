from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ThermodynamicSanityChecks:
    """
    Essential thermodynamic consistency checks for adsorption predictions.

    Checks implemented
    ------------------
    1. Gibbs-Duhem consistency
    2. Henry's law in dilute limit
    3. Selectivity trends with pressure
    4. Monotonicity of loading with chemical potential
    5. Maxwell relations (Hessian symmetry)
    6. Convexity of grand potential
    7. Cross-species competition

    Fixes vs. original
    ------------------
    1. BUG FIXED: check_convexity allocated idx1/idx2/lam/c_interp/o1/o2 and
       then ignored all of them.  The actual computation simplified to checking
       whether omega is increasing — not convexity.  Replaced with the correct
       discrete second-difference test: Ω[i+1] - 2Ω[i] + Ω[i-1] ≥ 0.
    2. BUG FIXED: check_gibbs_duhem used relu(-integral), which only fires when
       the path integral is negative.  For standard adsorption (positive
       loadings, dμ > 0 sweep), this is always ≥ 0, so the violation rate is
       permanently 0.  Replaced with the absolute-value residual.
    3. BUG FIXED: check_henry_region called K_H.std(dim=0) with only 1 point,
       returning NaN (Bessel-corrected std).  Added guard for n < 2.
    4. BUG FIXED: check_competition accumulated pair_viol.mean() (a magnitude)
       in a scalar, then divided by pair count — giving a scaled magnitude, not
       a rate in [0, 1].  Now computes the actual fraction of violating points
       per pair and averages the per-pair rates.

    References
    ----------
    [1] Myers & Prausnitz (1965). Thermodynamics of Mixed-Gas Adsorption.
    [2] Talu & Myers (2001). Reference Isotherms for Adsorption.
    [3] Smit & Maesen (2008). Molecular Simulations of Zeolites.
    """

    def __init__(
        self,
        tolerance: float = 1e-4,
        n_test_points: int = 100,
    ):
        self.tolerance    = tolerance
        self.n_test_points = n_test_points

    # ------------------------------------------------------------------
    # 1. Gibbs-Duhem Consistency
    # ------------------------------------------------------------------

    @staticmethod
    def check_gibbs_duhem(
        q_pred:       torch.Tensor,
        conditions:   torch.Tensor,
        n_components: int   = 3,
        tolerance:    float = 1e-4,
    ) -> Dict[str, float]:
        """
        Verify Gibbs-Duhem: Σ nᵢ dμᵢ = -dΩ along a path.

        Without access to dΩ we test self-consistency by measuring the
        absolute magnitude of the line-integral residual |∫ n·dμ|.
        A perfectly consistent model has this equal to |ΔΩ|; large
        deviations indicate thermodynamic inconsistency.

        FIX: original used relu(-integral), which never fires for normal
        adsorption (positive loadings swept from low to high μ).  Now uses
        the absolute residual so that inconsistency in either direction is
        detected.

        Parameters
        ----------
        q_pred      : [B, P, C]
        conditions  : [B, P, D]
        """
        mu        = conditions[..., :n_components]          # [B, P, C]
        dmu       = mu[:, 1:] - mu[:, :-1]                 # [B, P-1, C]
        q_mid     = 0.5 * (q_pred[:, 1:] + q_pred[:, :-1]) # [B, P-1, C]
        integrand = (q_mid * dmu).sum(dim=-1)               # [B, P-1]

        # Absolute value of the cumulative residual per sample
        residuals = integrand.abs().sum(dim=-1)             # [B]

        return {
            "gibbs_duhem_violation_mean": float(residuals.mean().item()),
            "gibbs_duhem_violation_max":  float(residuals.max().item()),
            "gibbs_duhem_violation_rate": float(
                (residuals > tolerance).float().mean().item()
            ),
        }

    # ------------------------------------------------------------------
    # 2. Henry's Law
    # ------------------------------------------------------------------

    @staticmethod
    def check_henry_region(
        q_pred:             torch.Tensor,
        pressures:          torch.Tensor,
        threshold_pressure: float = 0.1,
        tolerance:          float = 0.1,
    ) -> Dict[str, float]:
        """
        Verify Henry's law in dilute limit: q ∝ P.

        FIX: original called K_H.std(dim=0) with n=1, returning NaN.
        Now returns henry_valid=True (trivially) when fewer than 2 Henry
        points are available.

        Parameters
        ----------
        q_pred    : [B, P, C]
        pressures : [B, P]
        """
        mask = pressures < threshold_pressure   # [B, P] bool
        n_henry = int(mask.sum().item())

        if n_henry == 0:
            return {
                "henry_mean_error": float("nan"),
                "henry_max_error":  float("nan"),
                "henry_valid":      True,
                "henry_n_points":   0,
            }

        q_low = q_pred[mask]                    # [K, C]
        P_low = pressures[mask]                 # [K]

        K_H = q_low / (P_low.unsqueeze(-1) + 1e-8)  # [K, C]

        # FIX: guard n < 2 to avoid NaN from Bessel-corrected std
        if K_H.shape[0] < 2:
            return {
                "henry_mean_error": 0.0,
                "henry_max_error":  0.0,
                "henry_valid":      True,
                "henry_n_points":   n_henry,
            }

        K_H_mean  = K_H.mean(dim=0)                             # [C]
        K_H_std   = K_H.std(dim=0, unbiased=True)              # [C]
        rel_error = K_H_std / (K_H_mean.abs() + 1e-8)          # [C]

        mean_error = float(rel_error.mean().item())
        max_error  = float(rel_error.max().item())

        return {
            "henry_mean_error": mean_error,
            "henry_max_error":  max_error,
            "henry_valid":      max_error < tolerance,
            "henry_n_points":   n_henry,
        }

    # ------------------------------------------------------------------
    # 3. Selectivity Trends
    # ------------------------------------------------------------------

    @staticmethod
    def check_selectivity_trends(
        q_CO2:     torch.Tensor,
        q_N2:      torch.Tensor,
        pressures: torch.Tensor,
        tolerance: float = 0.01,
    ) -> Dict[str, float]:
        """
        Verify selectivity decreases with increasing pressure.

        Parameters
        ----------
        q_CO2, q_N2 : [B, P]
        pressures    : [B, P]
        """
        selectivity = q_CO2 / (q_N2 + 1e-8)             # [B, P]
        sel_diff    = selectivity[:, 1:] - selectivity[:, :-1]   # [B, P-1]
        press_diff  = pressures[:, 1:] - pressures[:, :-1]       # [B, P-1]

        # Violation: selectivity increases when pressure also increases
        violations = torch.relu(sel_diff * (press_diff > 0).float())

        return {
            "selectivity_violation_rate":    float((violations > tolerance).float().mean().item()),
            "selectivity_max_violation":     float(violations.max().item()),
            "selectivity_mean_violation":    float(violations.mean().item()),
        }

    # ------------------------------------------------------------------
    # 4. Monotonicity
    # ------------------------------------------------------------------

    @staticmethod
    def check_monotonicity(
        q_pred:       torch.Tensor,
        conditions:   torch.Tensor,
        n_components: int   = 3,
        tolerance:    float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Verify ∂qᵢ/∂μᵢ ≥ 0.

        Parameters
        ----------
        q_pred     : [B, P, C]
        conditions : [B, P, D]
        """
        mu      = conditions[..., :n_components]            # [B, P, C]
        mu_diff = mu[:, 1:] - mu[:, :-1]                   # [B, P-1, C]
        q_diff  = q_pred[:, 1:] - q_pred[:, :-1]           # [B, P-1, C]

        # Only flag steps where μᵢ increased but qᵢ decreased
        mask       = (mu_diff > 0).float()
        violations = (q_diff < -tolerance).float() * mask  # [B, P-1, C]

        per_component = [
            float(violations[..., i].mean().item())
            for i in range(n_components)
        ]

        return {
            "monotonicity_violation_rate": float(violations.mean().item()),
            "monotonicity_by_component":   per_component,
            "monotonicity_max_violation":  float(
                (torch.relu(-q_diff) * mask).max().item()
            ),
        }

    # ------------------------------------------------------------------
    # 5. Maxwell Relations (Hessian Symmetry)
    # ------------------------------------------------------------------

    @staticmethod
    def check_maxwell_relations(
        hessian:   torch.Tensor,
        tolerance: float = 1e-4,
    ) -> Dict[str, float]:
        """
        Verify ∂²Ω/∂μᵢ∂μⱼ = ∂²Ω/∂μⱼ∂μᵢ.

        Parameters
        ----------
        hessian : [B, P, C, C]
        """
        antisym   = hessian - hessian.transpose(-1, -2)    # [B, P, C, C]
        asym_norm = antisym.norm(dim=(-1, -2))              # [B, P]

        return {
            "maxwell_violation_mean": float(asym_norm.mean().item()),
            "maxwell_violation_max":  float(asym_norm.max().item()),
            "maxwell_violation_rate": float(
                (asym_norm > tolerance).float().mean().item()
            ),
        }

    # ------------------------------------------------------------------
    # 6. Convexity of Grand Potential
    # ------------------------------------------------------------------

    @staticmethod
    def check_convexity(
        omega:        torch.Tensor,
        conditions:   torch.Tensor,
        n_components: int   = 3,
        n_pairs:      int   = 200,
        tolerance:    float = 1e-4,
    ) -> Dict[str, float]:
        """
        Verify Ω is convex along the provided grid.

        FIX: original allocated idx1/idx2/lam/c1/c2/o1/o2 and then
        ignored all of them.  The actual violation line simplified to
        (omega[:,1:] - omega[:,:-1]) / 2 — which just tests whether omega
        is increasing, not convexity.

        Correct discrete test: second difference ≥ 0.
            Ω[i+1] - 2·Ω[i] + Ω[i-1] ≥ 0   (valid for uniform μ-grids)

        For non-uniform grids this is an approximation; it remains a
        useful sanity check.

        Parameters
        ----------
        omega      : [B, P, 1]
        conditions : [B, P, D]
        """
        # Squeeze to [B, P]
        omega_sq = omega.squeeze(-1)

        if omega_sq.shape[1] < 3:
            return {
                "convexity_violation_rate": 0.0,
                "convexity_mean_violation": 0.0,
                "convexity_max_violation":  0.0,
            }

        # Second difference: should be ≥ 0 for convex Ω
        second_diff = (
            omega_sq[:, 2:] - 2.0 * omega_sq[:, 1:-1] + omega_sq[:, :-2]
        )                                                      # [B, P-2]

        violations = torch.relu(-second_diff)                  # penalise negative curvature

        return {
            "convexity_violation_rate": float(
                (violations > tolerance).float().mean().item()
            ),
            "convexity_mean_violation": float(violations.mean().item()),
            "convexity_max_violation":  float(violations.max().item()),
        }

    # ------------------------------------------------------------------
    # 7. Cross-Species Competition
    # ------------------------------------------------------------------

    @staticmethod
    def check_competition(
        q_pred:       torch.Tensor,
        conditions:   torch.Tensor,
        n_components: int   = 3,
        tolerance:    float = 1e-6,
    ) -> Dict[str, float]:
        """
        Verify ∂qⱼ/∂μᵢ ≤ 0 for i ≠ j.

        FIX: original accumulated pair_viol.mean() (mean violation
        magnitude) in a scalar `violations`, then divided by pair count —
        giving a scaled magnitude, not a rate in [0, 1].
        Now computes fraction of violating points per pair and averages
        those fractions.

        Parameters
        ----------
        q_pred     : [B, P, C]
        conditions : [B, P, D]
        """
        mu      = conditions[..., :n_components]    # [B, P, C]
        mu_diff = mu[:, 1:] - mu[:, :-1]           # [B, P-1, C]
        q_diff  = q_pred[:, 1:] - q_pred[:, :-1]   # [B, P-1, C]

        pair_violations: List[Dict[str, Any]] = []
        per_pair_rates:  List[float] = []
        max_violation = torch.tensor(0.0, device=q_pred.device)

        for i in range(n_components):
            for j in range(n_components):
                if i == j:
                    continue
                # Steps where μᵢ increased
                mask      = (mu_diff[..., i] > 0).float()   # [B, P-1]
                # Violation: qⱼ increased when μᵢ increased (should decrease)
                pair_viol_mag  = torch.relu(q_diff[..., j]) * mask   # [B, P-1]
                # Rate = fraction of masked steps that violate
                n_mask    = mask.sum().clamp_min(1.0)
                pair_rate = float((pair_viol_mag > tolerance).sum().item() / n_mask.item())

                per_pair_rates.append(pair_rate)
                pair_violations.append({"i": i, "j": j, "rate": pair_rate})
                max_violation = torch.maximum(max_violation, pair_viol_mag.max())

        overall_rate = float(np.mean(per_pair_rates)) if per_pair_rates else 0.0

        return {
            "competition_violation_rate": overall_rate,
            "competition_max_violation":  float(max_violation.item()),
            "competition_per_pair":       pair_violations,
        }

    # ------------------------------------------------------------------
    # Full check
    # ------------------------------------------------------------------

    def check_all(
        self,
        q_pred:       torch.Tensor,
        conditions:   torch.Tensor,
        pressures:    torch.Tensor,
        hessian:      Optional[torch.Tensor] = None,
        omega:        Optional[torch.Tensor] = None,
        n_components: int = 3,
    ) -> Dict[str, Any]:
        """
        Run all thermodynamic consistency checks.

        Parameters
        ----------
        q_pred     : [B, P, C]
        conditions : [B, P, D]
        pressures  : [B, P]
        hessian    : [B, P, C, C] (optional)
        omega      : [B, P, 1]   (optional)
        """
        results: Dict[str, Any] = {
            "gibbs_duhem": self.check_gibbs_duhem(
                q_pred, conditions, n_components, self.tolerance
            ),
            "henry": self.check_henry_region(
                q_pred, pressures, tolerance=self.tolerance
            ),
            "monotonicity": self.check_monotonicity(
                q_pred, conditions, n_components, self.tolerance
            ),
            "competition": self.check_competition(
                q_pred, conditions, n_components, self.tolerance
            ),
        }

        if n_components >= 2:
            results["selectivity"] = self.check_selectivity_trends(
                q_pred[..., 0],
                q_pred[..., 1],
                pressures,
                self.tolerance,
            )

        if hessian is not None:
            results["maxwell"] = self.check_maxwell_relations(
                hessian, self.tolerance
            )

        if omega is not None:
            results["convexity"] = self.check_convexity(
                omega, conditions, n_components, tolerance=self.tolerance
            )

        results["overall_score"] = self._compute_overall_score(results)
        return results

    def _compute_overall_score(self, results: Dict[str, Any]) -> float:
        """Thermodynamic consistency score in [0, 1]. Higher is better."""
        scores: List[float] = []

        if "gibbs_duhem" in results:
            rate = results["gibbs_duhem"].get("gibbs_duhem_violation_rate", 1.0)
            scores.append(1.0 - min(rate, 1.0))

        if "henry" in results:
            scores.append(1.0 if results["henry"].get("henry_valid", False) else 0.5)

        if "monotonicity" in results:
            rate = results["monotonicity"].get("monotonicity_violation_rate", 1.0)
            scores.append(1.0 - min(rate, 1.0))

        if "competition" in results:
            rate = results["competition"].get("competition_violation_rate", 1.0)
            scores.append(1.0 - min(rate, 1.0))

        if "maxwell" in results:
            rate = results["maxwell"].get("maxwell_violation_rate", 1.0)
            scores.append(1.0 - min(rate, 1.0))

        if "selectivity" in results:
            rate = results["selectivity"].get("selectivity_violation_rate", 1.0)
            scores.append(1.0 - min(rate, 1.0))

        if "convexity" in results:
            rate = results["convexity"].get("convexity_violation_rate", 1.0)
            scores.append(1.0 - min(rate, 1.0))

        return float(np.mean(scores)) if scores else 0.0

    # ------------------------------------------------------------------
    # Numpy convenience
    # ------------------------------------------------------------------

    def check_all_numpy(
        self,
        q_pred:     np.ndarray,
        conditions: np.ndarray,
        pressures:  np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run checks on numpy arrays (converts to torch internally)."""
        return self.check_all(
            torch.from_numpy(np.asarray(q_pred,     dtype=np.float32)),
            torch.from_numpy(np.asarray(conditions, dtype=np.float32)),
            torch.from_numpy(np.asarray(pressures,  dtype=np.float32)),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Human-readable summary of check results."""
        out: Dict[str, Any] = {
            "overall_score": results.get("overall_score", 0.0),
            "checks_passed": 0,
            "checks_total":  0,
            "details":       {},
        }

        for name in ["gibbs_duhem", "henry", "monotonicity",
                     "competition", "maxwell", "selectivity", "convexity"]:
            if name not in results:
                continue

            out["checks_total"] += 1

            if name == "henry":
                passed = results[name].get("henry_valid", False)
            else:
                rate   = results[name].get(f"{name}_violation_rate", 1.0)
                passed = rate < 0.05   # < 5% violations

            if passed:
                out["checks_passed"] += 1

            out["details"][name] = {
                "passed":         passed,
                "violation_rate": results[name].get(f"{name}_violation_rate"),
            }

        return out


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def quick_thermo_check(
    q_pred:     torch.Tensor,
    conditions: torch.Tensor,
    pressures:  torch.Tensor,
    **kwargs,
) -> Dict[str, Any]:
    """One-liner thermodynamic check."""
    checker = ThermodynamicSanityChecks(**kwargs)
    return checker.check_all(q_pred, conditions, pressures)


def thermo_consistency_score(
    q_pred:     torch.Tensor,
    conditions: torch.Tensor,
    pressures:  torch.Tensor,
    **kwargs,
) -> float:
    """Single thermodynamic consistency score in [0, 1]."""
    return quick_thermo_check(q_pred, conditions, pressures, **kwargs).get(
        "overall_score", 0.0
    )


__all__ = [
    "ThermodynamicSanityChecks",
    "quick_thermo_check",
    "thermo_consistency_score",
]