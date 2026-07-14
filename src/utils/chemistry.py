"""
Chemical conversion utilities for the UC-TPNO humid flue-gas pipeline.

This module provides the thermodynamic plumbing that connects raw
experimental conditions (pressure, temperature, composition, relative
humidity) to the internal representation consumed by the TPNO operator
(chemical potentials μ_i, temperature T).

Key capabilities
────────────────
•  Peng–Robinson cubic EOS — fully implemented solver.
•  Pure-component fugacity via ideal-gas, Peng–Robinson, or truncated
   second-virial methods.
•  *Mixture* fugacity via Peng–Robinson with van-der-Waals one-fluid
   mixing rules and binary interaction parameters (k_ij) for the
   CO₂ / N₂ / H₂O ternary system.
•  Bidirectional P ↔ f ↔ μ conversion chains, both scalar and
   numpy-vectorised.
•  Relative-humidity ↔ mole-fraction utilities (Antoine-equation-based
   and Buck-equation cross-check).
•  Adsorption loading unit conversions (mmol/g, cm³(STP)/g, mg/g,
   molecules/uc, wt%).

BUG FIX (this version)
──────────────────────
Previous version: build_condition_vector used mu_dict.get("H2O", 0.0).
μ = 0 corresponds to f = 1 bar of water — physically wrong for dry
conditions, which poisoned the grand-potential surface with a false
"1 bar water" signal on every dry training point.

Fix: when a species has zero or absent mole fraction, replace with
TRACE_Y = 1e-10 before computing fugacities so the resulting μ is
very negative (physically meaningful "essentially absent") rather than
zero (physically "1 bar present").

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT

References
──────────
[1] Peng, D.-Y.; Robinson, D.B. Ind. Eng. Chem. Fundam. 1976, 15, 59.
[2] Poling, B.E.; Prausnitz, J.M.; O'Connell, J.P.
    "The Properties of Gases and Liquids", 5th ed., McGraw-Hill, 2001.
[3] Prausnitz, J.M.; Lichtenthaler, R.N.; de Azevedo, E.G.
    "Molecular Thermodynamics of Fluid-Phase Equilibria", 3rd ed., 1999.
[4] Buck, A.L. J. Appl. Meteor. 1981, 20, 1527.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .constants import (
    R,
    R_kJ,
    EPS,
    LOG_EPS,
    BAR_TO_PA,
    GAS_REGISTRY,
    GasSpecies,
    CO2,
    N2,
    H2O,
    gas_properties,
)

# Type alias for "scalar or array"
ArrayLike = Union[float, np.ndarray]

# Trace mole fraction used for "absent" species so that the resulting μ
# is physically very negative rather than the erroneous μ = 0 (= 1 bar).
# 1e-10 → f ≈ 1e-10 bar → μ ≈ RT·ln(1e-10) ≈ −59 kJ/mol at 313 K.
_TRACE_Y: float = 1e-10

# ═══════════════════════════════════════════════════════════════════════
# 1.  PENG–ROBINSON EQUATION OF STATE
# ═══════════════════════════════════════════════════════════════════════

# R in bar·L·mol⁻¹·K⁻¹ for the PR EOS when P is in bar.
_R_PR: float = 8.314462618e-2   # 0.08314… bar·L/(mol·K)

# ── Binary interaction parameters k_ij ────────────────────────────────
# Fitted to VLE data.  Symmetric: k_ij = k_ji, k_ii = 0.
# Source: Poling et al. Table 8-1; Tsivintzelis et al. (2011) for H₂O pairs.
_KIJ: Dict[Tuple[str, str], float] = {
    ("CO2", "N2"):  -0.017,
    ("CO2", "H2O"):  0.190,
    ("N2",  "H2O"):  0.490,
}


def _get_kij(gas_i: str, gas_j: str) -> float:
    """Return binary interaction parameter k_ij (symmetric, zero for i==j)."""
    if gas_i == gas_j:
        return 0.0
    key = (gas_i, gas_j) if (gas_i, gas_j) in _KIJ else (gas_j, gas_i)
    return _KIJ.get(key, 0.0)


def _pr_params_pure(species: GasSpecies, temperature: float) -> Tuple[float, float]:
    """
    Compute Peng–Robinson attractive parameter *a(T)* and co-volume *b*
    for a single species.

    Returns
    -------
    a : float   [bar·L²·mol⁻²]
    b : float   [L·mol⁻¹]
    """
    Tc = species.critical_temperature
    Pc = species.critical_pressure
    w  = species.acentric_factor
    Tr = temperature / Tc
    kappa = 0.37464 + 1.54226 * w - 0.26992 * w ** 2
    alpha = (1.0 + kappa * (1.0 - math.sqrt(max(Tr, 1e-6)))) ** 2
    a = 0.45724 * (_R_PR ** 2) * (Tc ** 2) / Pc * alpha
    b = 0.07780 * _R_PR * Tc / Pc
    return a, b


def solve_pr_cubic(A: float, B: float) -> float:
    """
    Solve the Peng–Robinson cubic in Z:
        Z³ − (1 − B)Z² + (A − 3B² − 2B)Z − (AB − B² − B³) = 0

    and return the *vapour-phase* (largest real positive) root.

    Parameters
    ----------
    A : float   dimensionless  a·P / (R·T)²
    B : float   dimensionless  b·P / (R·T)

    Returns
    -------
    Z_vap : float  compressibility factor (vapour root)
    """
    c2 = -(1.0 - B)
    c1 = A - 3.0 * B * B - 2.0 * B
    c0 = -(A * B - B * B - B ** 3)

    roots = np.roots([1.0, c2, c1, c0])

    # Keep real, positive roots above co-volume B (vapour phase)
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10 and r.real > B:
            real_roots.append(r.real)

    if not real_roots:
        warnings.warn(
            "PR cubic yielded no valid roots — returning Z = 1 (ideal-gas).",
            RuntimeWarning,
            stacklevel=2,
        )
        return 1.0

    return float(max(real_roots))


# ═══════════════════════════════════════════════════════════════════════
# 2.  PURE-COMPONENT FUGACITY
# ═══════════════════════════════════════════════════════════════════════

def _fugacity_pr_pure(
    pressure: float,
    temperature: float,
    species: GasSpecies,
) -> float:
    """Fugacity [bar] of a pure species via Peng–Robinson EOS."""
    a, b = _pr_params_pure(species, temperature)
    RT  = _R_PR * temperature
    A   = a * pressure / RT ** 2
    B   = b * pressure / RT
    Z   = solve_pr_cubic(A, B)

    sqrt2 = math.sqrt(2.0)
    arg_num = Z + (1.0 + sqrt2) * B
    arg_den = Z + (1.0 - sqrt2) * B

    if arg_num <= 0 or arg_den <= 0 or (Z - B) <= 0:
        return pressure  # degenerate → ideal

    ln_phi = (
        Z
        - 1.0
        - math.log(Z - B)
        - A / (2.0 * sqrt2 * B) * math.log(arg_num / arg_den)
    )
    return pressure * math.exp(ln_phi)


def _fugacity_virial_pure(
    pressure: float,
    temperature: float,
    species: GasSpecies,
) -> float:
    """
    Fugacity [bar] via the truncated second-virial (Pitzer) correlation.
        B·Pc/(R·Tc) = B⁰ + ω·B¹
    where  B⁰ = 0.083 − 0.422/Tr^1.6
           B¹ = 0.139 − 0.172/Tr^4.2
    Ref: Poling et al. §4-5.
    """
    Tc = species.critical_temperature
    Pc = species.critical_pressure
    w  = species.acentric_factor
    Tr = max(temperature / Tc, 0.3)  # clamp for sub-critical
    B0 = 0.083 - 0.422 / (Tr ** 1.6)
    B1 = 0.139 - 0.172 / (Tr ** 4.2)
    B_over_RTc = (B0 + w * B1) / Pc
    B = B_over_RTc * _R_PR * Tc
    ln_phi = B * pressure / (_R_PR * temperature)
    return pressure * math.exp(ln_phi)


def pressure_to_fugacity(
    pressure: ArrayLike,
    temperature: ArrayLike,
    gas: str = "CO2",
    method: str = "peng_robinson",
) -> ArrayLike:
    """
    Convert partial pressure → fugacity for a pure component.

    Parameters
    ----------
    pressure    : Partial pressure [bar].  Scalar or array.
    temperature : Temperature [K].  Scalar or array (broadcast with *pressure*).
    gas         : Gas formula (``'CO2'``, ``'N2'``, ``'H2O'``).
    method      : ``'ideal'``, ``'peng_robinson'``, or ``'virial'``.

    Returns
    -------
    Fugacity [bar], same shape as *pressure*.
    """
    species = GAS_REGISTRY[gas]

    if method == "ideal":
        return pressure  # f = P

    p_arr = np.atleast_1d(np.asarray(pressure, dtype=np.float64))
    t_arr = np.broadcast_to(
        np.atleast_1d(np.asarray(temperature, dtype=np.float64)), p_arr.shape
    )

    if method == "peng_robinson":
        fn = _fugacity_pr_pure
    elif method == "virial":
        fn = _fugacity_virial_pure
    else:
        raise ValueError(f"Unknown fugacity method: {method!r}")

    result = np.array(
        [fn(float(p), float(t), species) for p, t in zip(p_arr.ravel(), t_arr.ravel())]
    ).reshape(p_arr.shape)

    if np.ndim(pressure) == 0 and result.ndim >= 1 and result.size == 1:
        return float(result.item())
    return result


def fugacity_to_pressure(
    fugacity: ArrayLike,
    temperature: ArrayLike,
    gas: str = "CO2",
    method: str = "peng_robinson",
    tol: float = 1e-8,
    max_iter: int = 50,
) -> ArrayLike:
    """
    Invert fugacity → pressure via Newton iteration on the EOS.
    Uses *pressure_to_fugacity* as the forward model.
    """
    f_target = np.atleast_1d(np.asarray(fugacity, dtype=np.float64))
    t_arr = np.broadcast_to(
        np.atleast_1d(np.asarray(temperature, dtype=np.float64)), f_target.shape
    )

    P = f_target.copy()
    for _ in range(max_iter):
        f_calc = np.atleast_1d(
            pressure_to_fugacity(P, t_arr, gas=gas, method=method)
        )
        residual = f_calc - f_target
        if np.max(np.abs(residual)) < tol:
            break
        dfdP = np.where(np.abs(P) > EPS, f_calc / P, 1.0)
        P = P - residual / (dfdP + EPS)
        P = np.clip(P, EPS, None)

    if np.ndim(fugacity) == 0 and P.size == 1:
        return float(P.item())
    return P


# ═══════════════════════════════════════════════════════════════════════
# 3.  MIXTURE FUGACITY  (Peng–Robinson + vdW1f mixing rules)
# ═══════════════════════════════════════════════════════════════════════

def mixture_fugacity_pr(
    y: Dict[str, float],
    pressure: float,
    temperature: float,
) -> Dict[str, float]:
    """
    Component fugacities [bar] in a gas mixture via PR EOS with
    van-der-Waals one-fluid (vdW1f) mixing rules.

    Parameters
    ----------
    y           : Mole-fraction dict, e.g. ``{'CO2': 0.15, 'N2': 0.75, 'H2O': 0.10}``.
                  Species with mole fraction < _TRACE_Y are treated as trace
                  (replaced with _TRACE_Y) so that the resulting μ is very
                  negative rather than the erroneous zero.
    pressure    : Total pressure [bar].
    temperature : Temperature [K].

    Returns
    -------
    Dict mapping each species formula to its fugacity [bar].
    """
    # --- Sanitise composition: replace zeros with trace so μ ≠ 0 -------
    y_safe: Dict[str, float] = {}
    for s, yi in y.items():
        y_safe[s] = max(float(yi), _TRACE_Y)
    total = sum(y_safe.values())
    y_safe = {s: v / total for s, v in y_safe.items()}

    species_list = list(y_safe.keys())

    # Pure parameters
    a_pure: Dict[str, float] = {}
    b_pure: Dict[str, float] = {}
    for s in species_list:
        sp = GAS_REGISTRY[s]
        a_pure[s], b_pure[s] = _pr_params_pure(sp, temperature)

    # Mixture parameters (vdW1f mixing rules)
    a_mix = 0.0
    for si in species_list:
        for sj in species_list:
            kij = _get_kij(si, sj)
            aij = math.sqrt(a_pure[si] * a_pure[sj]) * (1.0 - kij)
            a_mix += y_safe[si] * y_safe[sj] * aij

    b_mix = sum(y_safe[s] * b_pure[s] for s in species_list)

    RT = _R_PR * temperature
    A_mix = a_mix * pressure / RT ** 2
    B_mix = b_mix * pressure / RT
    Z = solve_pr_cubic(A_mix, B_mix)

    sqrt2 = math.sqrt(2.0)
    arg_num = Z + (1.0 + sqrt2) * B_mix
    arg_den = Z + (1.0 - sqrt2) * B_mix

    fugacities: Dict[str, float] = {}
    for k in species_list:
        if arg_num <= 0 or arg_den <= 0 or (Z - B_mix) <= 0:
            fugacities[k] = y_safe[k] * pressure
            continue

        # ∂(n²a_mix)/∂n_k = 2 Σⱼ yⱼ a_kj
        sum_ya_k = 0.0
        for j in species_list:
            kij = _get_kij(k, j)
            akj = math.sqrt(a_pure[k] * a_pure[j]) * (1.0 - kij)
            sum_ya_k += y_safe[j] * akj

        bk = b_pure[k]
        ln_phi_k = (
            bk / b_mix * (Z - 1.0)
            - math.log(Z - B_mix)
            - A_mix / (2.0 * sqrt2 * B_mix)
            * (2.0 * sum_ya_k / a_mix - bk / b_mix)
            * math.log(arg_num / arg_den)
        )
        phi_k = math.exp(ln_phi_k)
        fugacities[k] = y_safe[k] * phi_k * pressure

    return fugacities


# ═══════════════════════════════════════════════════════════════════════
# 4.  CHEMICAL-POTENTIAL CONVERSIONS
# ═══════════════════════════════════════════════════════════════════════

# μ = μ° + R·T·ln(f / f°)   with f° = 1 bar  (ideal-gas standard state)

def fugacity_to_chemical_potential(
    fugacity: ArrayLike,
    temperature: ArrayLike,
) -> ArrayLike:
    """
    Convert fugacity [bar] → excess chemical potential [kJ mol⁻¹].

    μ − μ° = R·T·ln(f / 1 bar)

    Note: for trace species (f ≈ 1e-10 bar), μ ≈ -59 kJ/mol at 313 K,
    which correctly represents "essentially absent".
    """
    f = np.asarray(fugacity, dtype=np.float64)
    T = np.asarray(temperature, dtype=np.float64)
    # Clamp to LOG_EPS (1e-12) so log never hits -inf
    mu = R_kJ * T * np.log(np.maximum(f, LOG_EPS))
    if np.ndim(fugacity) == 0 and np.ndim(mu) >= 1 and mu.size == 1:
        return float(mu.item())
    return mu


def chemical_potential_to_fugacity(
    mu: ArrayLike,
    temperature: ArrayLike,
) -> ArrayLike:
    """
    Convert excess chemical potential [kJ mol⁻¹] → fugacity [bar].

    f = exp(μ / (R·T))   where R is in kJ mol⁻¹ K⁻¹.
    """
    mu_arr = np.asarray(mu, dtype=np.float64)
    T = np.asarray(temperature, dtype=np.float64)
    f = np.exp(mu_arr / (R_kJ * T))
    if np.ndim(mu) == 0 and np.ndim(f) >= 1 and f.size == 1:
        return float(f.item())
    return f


def pressure_to_chemical_potential(
    pressure: ArrayLike,
    temperature: ArrayLike,
    gas: str = "CO2",
    method: str = "peng_robinson",
) -> ArrayLike:
    """Shortcut:  P [bar] → μ [kJ mol⁻¹]."""
    f = pressure_to_fugacity(pressure, temperature, gas=gas, method=method)
    return fugacity_to_chemical_potential(f, temperature)


def chemical_potential_to_pressure(
    mu: ArrayLike,
    temperature: ArrayLike,
    gas: str = "CO2",
    method: str = "peng_robinson",
) -> ArrayLike:
    """Shortcut:  μ [kJ mol⁻¹] → P [bar]."""
    f = chemical_potential_to_fugacity(mu, temperature)
    return fugacity_to_pressure(f, temperature, gas=gas, method=method)


def mixture_pressure_to_chemical_potentials(
    y: Dict[str, float],
    pressure: float,
    temperature: float,
) -> Dict[str, float]:
    """
    Convert total pressure + composition → per-species μ [kJ mol⁻¹]
    using mixture Peng–Robinson fugacities.

    This is the *primary entry point* used by the data pipeline to
    convert raw (P, T, y) conditions into the μ-vector fed to TPNO.

    Zero or absent mole fractions are treated as _TRACE_Y (1e-10) so
    that absent species receive a physically meaningful very-negative μ
    rather than μ = 0 (which would incorrectly imply f = 1 bar).

    Parameters
    ----------
    y           : Mole fractions  {'CO2': …, 'N2': …, 'H2O': …}.
                  Missing species are treated as absent (trace).
    pressure    : Total pressure [bar].
    temperature : Temperature [K].

    Returns
    -------
    Dict of chemical potentials [kJ mol⁻¹], keyed by species formula.
    """
    fug = mixture_fugacity_pr(y, pressure, temperature)
    return {
        gas: float(fugacity_to_chemical_potential(f, temperature))
        for gas, f in fug.items()
    }


# ═══════════════════════════════════════════════════════════════════════
# 5.  BUILD TPNO CONDITION VECTOR
# ═══════════════════════════════════════════════════════════════════════

# All three species the TPNO operator tracks, in fixed order.
_TPNO_SPECIES: List[str] = ["CO2", "N2", "H2O"]


def build_condition_vector(
    pressure: float,
    temperature: float,
    y: Dict[str, float],
    method: str = "peng_robinson",
) -> np.ndarray:
    """
    Build the 4-D condition vector  [μ_CO₂, μ_N₂, μ_H₂O, T]  that the
    TPNO operator expects as input.

    BUG FIX: previous version used mu_dict.get("H2O", 0.0), which gave
    μ = 0 (≡ f = 1 bar) for dry conditions.  This version always computes
    all three μ values using a trace mole fraction (_TRACE_Y = 1e-10) for
    absent species, yielding a physically meaningful very-negative μ.

    Parameters
    ----------
    pressure    : Total pressure [bar].
    temperature : Temperature [K].
    y           : Mole fractions  {'CO2': …, 'N2': …, 'H2O': …}.
                  Missing keys are treated as absent (mole fraction → 0).
    method      : Fugacity method (``'peng_robinson'``, ``'virial'``,
                  ``'ideal'``).

    Returns
    -------
    np.ndarray of shape (4,).
    """
    # Ensure all three species are present with at least trace concentration
    y_full: Dict[str, float] = {}
    for s in _TPNO_SPECIES:
        yi = float(y.get(s, 0.0))
        y_full[s] = max(yi, _TRACE_Y)  # absent → trace, not zero

    # Normalise so mole fractions sum to 1
    total = sum(y_full.values())
    y_full = {s: v / total for s, v in y_full.items()}

    if method == "peng_robinson":
        mu_dict = mixture_pressure_to_chemical_potentials(
            y_full, pressure, temperature
        )
    else:
        # Pure-component fallback: partial pressure × EOS correction
        mu_dict = {}
        for s in _TPNO_SPECIES:
            pi = y_full[s] * pressure
            fi = pressure_to_fugacity(
                max(pi, LOG_EPS), temperature, gas=s, method=method
            )
            mu_dict[s] = float(fugacity_to_chemical_potential(fi, temperature))

    return np.array(
        [mu_dict["CO2"], mu_dict["N2"], mu_dict["H2O"], temperature],
        dtype=np.float64,
    )


def build_condition_grid(
    pressures: np.ndarray,
    temperature: float,
    y: Dict[str, float],
    method: str = "peng_robinson",
) -> np.ndarray:
    """
    Vectorised version of :func:`build_condition_vector` over a pressure
    array.  Returns shape ``(len(pressures), 4)``.
    """
    rows = [
        build_condition_vector(float(p), temperature, y, method)
        for p in pressures
    ]
    return np.stack(rows, axis=0)


def build_condition_vector_from_rh(
    rh: float,
    temperature: float,
    total_pressure: float = 1.013,
    y_co2_dry: float = 0.15,
    method: str = "peng_robinson",
) -> np.ndarray:
    """
    Convert relative humidity + operating conditions → TPNO condition vector.

    This is the convenience entry point for humid flue-gas scenarios.
    Internally calls :func:`relative_humidity_to_mole_fraction` then
    :func:`build_condition_vector`.

    Parameters
    ----------
    rh              : Relative humidity ∈ [0, 1].
    temperature     : Temperature [K].
    total_pressure  : Total pressure [bar].  Default 1.013 bar (post-FGD).
    y_co2_dry       : CO₂ mole fraction in the *dry* flue gas.  Default 0.15.
    method          : Fugacity method.

    Returns
    -------
    np.ndarray of shape (4,):  [μ_CO₂, μ_N₂, μ_H₂O, T]
    """
    y_h2o = float(relative_humidity_to_mole_fraction(rh, temperature, total_pressure))
    y_dry = 1.0 - y_h2o
    y_co2 = y_co2_dry * y_dry
    y_n2  = y_dry - y_co2
    y = {"CO2": y_co2, "N2": y_n2, "H2O": y_h2o}
    return build_condition_vector(total_pressure, temperature, y, method)


# ═══════════════════════════════════════════════════════════════════════
# 6.  HUMIDITY ↔ MOLE-FRACTION CONVERSIONS
# ═══════════════════════════════════════════════════════════════════════

def water_saturation_pressure_antoine(temperature: ArrayLike) -> ArrayLike:
    """
    Saturation pressure of water [bar] via Antoine equation.
    Uses the constants stored in :pydata:`H2O.antoine_*` from
    ``constants.py`` (NIST, valid ~255–373 K).
    """
    T = np.asarray(temperature, dtype=np.float64)
    P_sat = 10.0 ** (H2O.antoine_A - H2O.antoine_B / (T + H2O.antoine_C))
    if np.ndim(temperature) == 0:
        return float(P_sat.item()) if P_sat.ndim >= 1 else float(P_sat)
    return P_sat


def water_saturation_pressure_buck(temperature: ArrayLike) -> ArrayLike:
    """
    Saturation pressure of water [bar] via the Buck (1981) equation.
    More accurate than Antoine above ~320 K.

        P_sat [hPa] = 6.1121 · exp[ (18.678 − T_C/234.5) · T_C / (257.14 + T_C) ]

    where T_C is temperature in °C.  Converted to bar.  [4]
    """
    T = np.asarray(temperature, dtype=np.float64)
    Tc = T - 273.15
    P_hPa = 6.1121 * np.exp((18.678 - Tc / 234.5) * Tc / (257.14 + Tc))
    P_bar = P_hPa / 1000.0
    if np.ndim(temperature) == 0:
        return float(P_bar.item()) if P_bar.ndim >= 1 else float(P_bar)
    return P_bar


def relative_humidity_to_mole_fraction(
    rh: ArrayLike,
    temperature: ArrayLike,
    total_pressure: float = 1.013,
    saturation_method: str = "antoine",
) -> ArrayLike:
    """
    Convert relative humidity (0–1) → water mole fraction y_H₂O.

    Parameters
    ----------
    rh                : Relative humidity ∈ [0, 1].
    temperature       : Temperature [K].
    total_pressure    : Total pressure [bar].
    saturation_method : ``'antoine'`` or ``'buck'``.
    """
    if saturation_method == "buck":
        P_sat = water_saturation_pressure_buck(temperature)
    else:
        P_sat = water_saturation_pressure_antoine(temperature)

    rh = np.asarray(rh, dtype=np.float64)
    P_w = rh * P_sat
    y_w = P_w / total_pressure
    y_w = np.clip(y_w, 0.0, 1.0)

    if np.ndim(rh) == 0 and np.ndim(y_w) >= 1 and y_w.size == 1:
        return float(y_w.item())
    return y_w


def mole_fraction_to_relative_humidity(
    y_w: ArrayLike,
    temperature: ArrayLike,
    total_pressure: float = 1.013,
    saturation_method: str = "antoine",
) -> ArrayLike:
    """Convert water mole fraction y_H₂O → relative humidity (0–1)."""
    if saturation_method == "buck":
        P_sat = water_saturation_pressure_buck(temperature)
    else:
        P_sat = water_saturation_pressure_antoine(temperature)

    y_w = np.asarray(y_w, dtype=np.float64)
    P_w = y_w * total_pressure
    rh  = P_w / np.maximum(P_sat, EPS)
    rh  = np.clip(rh, 0.0, 1.0)

    if np.ndim(y_w) == 0 and np.ndim(rh) >= 1 and rh.size == 1:
        return float(rh.item())
    return rh


def flue_gas_composition(
    rh: float,
    temperature: float = 313.15,
    total_pressure: float = 1.013,
    y_CO2_dry: float = 0.15,
) -> Dict[str, float]:
    """
    Build a normalised flue-gas composition dict from a relative-humidity
    value, assuming fixed CO₂/N₂ dry ratio.

    Returns
    -------
    ``{'CO2': …, 'N2': …, 'H2O': …}`` summing to 1.0.
    """
    y_H2O = float(
        relative_humidity_to_mole_fraction(rh, temperature, total_pressure)
    )
    y_dry = 1.0 - y_H2O
    y_CO2 = y_CO2_dry * y_dry
    y_N2  = y_dry - y_CO2
    return {"CO2": y_CO2, "N2": y_N2, "H2O": y_H2O}


# ═══════════════════════════════════════════════════════════════════════
# 7.  ADSORPTION LOADING UNIT CONVERSIONS
# ═══════════════════════════════════════════════════════════════════════

def mmol_g_to_mol_kg(q: ArrayLike) -> ArrayLike:
    """mmol g⁻¹ → mol kg⁻¹ (identity: 1 mmol/g = 1 mol/kg)."""
    return np.asarray(q, dtype=np.float64)


def cm3stp_g_to_mol_kg(q: ArrayLike) -> ArrayLike:
    """cm³(STP) g⁻¹ → mol kg⁻¹.  1 mol gas at STP = 22 414 cm³."""
    return np.asarray(q, dtype=np.float64) / 22.414


def mol_kg_to_cm3stp_g(q: ArrayLike) -> ArrayLike:
    """mol kg⁻¹ → cm³(STP) g⁻¹."""
    return np.asarray(q, dtype=np.float64) * 22.414


def mg_g_to_mol_kg(q: ArrayLike, molar_mass: float = 44.009) -> ArrayLike:
    """
    mg-adsorbate per g-adsorbent → mol kg⁻¹.

    Parameters
    ----------
    q          : Loading in mg g⁻¹.
    molar_mass : Molar mass of the adsorbate [g mol⁻¹].  Default = CO₂.
    """
    return np.asarray(q, dtype=np.float64) / molar_mass


def mol_kg_to_mg_g(q: ArrayLike, molar_mass: float = 44.009) -> ArrayLike:
    """mol kg⁻¹ → mg g⁻¹."""
    return np.asarray(q, dtype=np.float64) * molar_mass


def wt_pct_to_mol_kg(
    wt: ArrayLike,
    molar_mass: float = 44.009,
) -> ArrayLike:
    """
    Weight-percent loading → mol kg⁻¹.
    wt% = 100 · m_ads / (m_ads + m_sorbent)
    """
    wt = np.asarray(wt, dtype=np.float64)
    mg_g = wt / (100.0 - wt + EPS) * 1000.0
    return mg_g / molar_mass


def molecules_per_uc_to_mol_kg(
    n_molec: ArrayLike,
    uc_mass_g_mol: float,
) -> ArrayLike:
    """
    Molecules per unit cell → mol kg⁻¹.

    Parameters
    ----------
    n_molec      : Loading in molecules per unit cell.
    uc_mass_g_mol: Molar mass of one unit cell [g mol⁻¹].
    """
    n = np.asarray(n_molec, dtype=np.float64)
    return n * 1000.0 / uc_mass_g_mol


# ═══════════════════════════════════════════════════════════════════════
# 8.  COMPRESSIBILITY & DENSITY HELPERS
# ═══════════════════════════════════════════════════════════════════════

def compressibility_factor_pr(
    pressure: float,
    temperature: float,
    gas: str = "CO2",
) -> float:
    """Return Z from PR EOS for a pure gas."""
    species = GAS_REGISTRY[gas]
    a, b = _pr_params_pure(species, temperature)
    RT = _R_PR * temperature
    A = a * pressure / RT ** 2
    B = b * pressure / RT
    return solve_pr_cubic(A, B)


def gas_density(
    pressure: float,
    temperature: float,
    gas: str = "CO2",
    method: str = "ideal",
) -> float:
    """
    Molar density [mol L⁻¹] of a gas at given (P, T).
    Uses PR EOS if ``method='peng_robinson'``, else ideal-gas law.
    """
    if method == "peng_robinson":
        Z = compressibility_factor_pr(pressure, temperature, gas)
    else:
        Z = 1.0
    return pressure / (Z * _R_PR * temperature)


# ═══════════════════════════════════════════════════════════════════════
# 9.  ISOSTERIC HEAT HELPERS
# ═══════════════════════════════════════════════════════════════════════

def clausius_clapeyron_qst(
    ln_p: np.ndarray,
    inv_T: np.ndarray,
) -> float:
    """
    Estimate isosteric heat of adsorption Q_st [kJ mol⁻¹] from a set of
    (ln P, 1/T) points at constant loading via Clausius–Clapeyron:
        Q_st = −R · d(ln P) / d(1/T)

    A simple least-squares linear fit is used.
    """
    if len(ln_p) < 2:
        return 0.0
    coeffs = np.polyfit(inv_T, ln_p, 1)
    slope = coeffs[0]
    return -R_kJ * slope   # kJ mol⁻¹


# ═══════════════════════════════════════════════════════════════════════
# 10.  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    # EOS internals
    "solve_pr_cubic",
    "compressibility_factor_pr",
    "gas_density",
    # Pure-component fugacity
    "pressure_to_fugacity",
    "fugacity_to_pressure",
    # Mixture fugacity
    "mixture_fugacity_pr",
    # Chemical-potential chains
    "fugacity_to_chemical_potential",
    "chemical_potential_to_fugacity",
    "pressure_to_chemical_potential",
    "chemical_potential_to_pressure",
    "mixture_pressure_to_chemical_potentials",
    # Condition-vector builders (TPNO entry points)
    "build_condition_vector",
    "build_condition_grid",
    "build_condition_vector_from_rh",
    # Humidity
    "water_saturation_pressure_antoine",
    "water_saturation_pressure_buck",
    "relative_humidity_to_mole_fraction",
    "mole_fraction_to_relative_humidity",
    "flue_gas_composition",
    # Loading units
    "mmol_g_to_mol_kg",
    "cm3stp_g_to_mol_kg",
    "mol_kg_to_cm3stp_g",
    "mg_g_to_mol_kg",
    "mol_kg_to_mg_g",
    "wt_pct_to_mol_kg",
    "molecules_per_uc_to_mol_kg",
    # Thermodynamic helpers
    "clausius_clapeyron_qst",
]