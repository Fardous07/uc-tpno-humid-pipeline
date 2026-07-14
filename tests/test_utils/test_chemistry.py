"""
Tests for src/utils/chemistry.py — CRITICAL for model correctness.

These tests verify:
1.  Peng-Robinson EOS solver
2.  Pure-component fugacity calculations
3.  Mixture fugacity with vdW1f mixing rules
4.  Chemical potential conversions
5.  TPNO condition vector building
6.  Humidity conversions
7.  Unit conversions for adsorption loadings
8.  Compressibility and density helpers
9.  Isosteric heat calculation
10. Integration: mixture pressure → chemical potentials
11. Edge cases and error handling
12. Performance / sanity
13. Physical consistency checks

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import time

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════
# 1.  PENG-ROBINSON EOS
# ═══════════════════════════════════════════════════════════════════════

class TestPengRobinson:
    """Test the Peng-Robinson EOS solver."""

    def test_ideal_gas_limit(self):
        """At A→0, B→0 the gas-phase root Z should approach 1."""
        from src.utils.chemistry import solve_pr_cubic
        Z = solve_pr_cubic(A=1e-6, B=1e-6)
        assert 0.99 < Z < 1.01, f"Z should be ~1 for ideal gas, got {Z}"

    def test_co2_313k_1bar(self):
        """CO₂ at 313 K, 1 bar → Z ≈ 0.995."""
        from src.utils.chemistry import solve_pr_cubic
        # CO₂: a ≈ 3.96 bar·L²/mol², b ≈ 0.0427 L/mol
        A = 3.96 * 1.0 / (0.08314 * 313.15) ** 2
        B = 0.0427 * 1.0 / (0.08314 * 313.15)
        Z = solve_pr_cubic(A, B)
        assert 0.98 < Z < 1.01, f"Z should be ~0.995 for CO₂ at 1 bar, got {Z}"

    def test_co2_313k_50bar(self):
        """CO₂ at 313 K, 50 bar → Z < 1 (attractive forces dominate)."""
        from src.utils.chemistry import solve_pr_cubic
        A = 3.96 * 50.0 / (0.08314 * 313.15) ** 2
        B = 0.0427 * 50.0 / (0.08314 * 313.15)
        Z = solve_pr_cubic(A, B)
        assert Z < 0.95, f"Z should be <0.95 for CO₂ at 50 bar, got {Z}"

    def test_water_vapour_373k_1bar(self):
        """Water vapour at 373 K, 1 bar → Z ≈ 0.98."""
        from src.utils.chemistry import solve_pr_cubic
        # Water: a ≈ 5.54 bar·L²/mol², b ≈ 0.0305 L/mol
        A = 5.54 * 1.0 / (0.08314 * 373.15) ** 2
        B = 0.0305 * 1.0 / (0.08314 * 373.15)
        Z = solve_pr_cubic(A, B)
        assert 0.96 < Z < 1.00, f"Z should be ~0.98 for water vapour, got {Z}"

    def test_fallback_to_ideal(self):
        """Unphysical parameters → fallback Z = 1."""
        from src.utils.chemistry import solve_pr_cubic
        Z = solve_pr_cubic(A=1000.0, B=100.0)
        assert Z == 1.0, f"Fallback should return 1.0, got {Z}"


# ═══════════════════════════════════════════════════════════════════════
# 2.  PURE-COMPONENT FUGACITY
# ═══════════════════════════════════════════════════════════════════════

class TestPureFugacity:
    """Test pure-component fugacity calculations."""

    def test_ideal_gas_fugacity_equals_pressure(self):
        """Ideal-gas: f = P for all species."""
        from src.utils.chemistry import pressure_to_fugacity
        pressures = np.array([0.1, 1.0, 10.0])
        for gas in ["CO2", "N2", "H2O"]:
            f = pressure_to_fugacity(pressures, 313.15, gas=gas, method="ideal")
            np.testing.assert_allclose(f, pressures, rtol=1e-6,
                                       err_msg=f"Ideal fugacity != P for {gas}")

    def test_scalar_input_returns_float(self):
        """Scalar P → scalar f."""
        from src.utils.chemistry import pressure_to_fugacity
        for method in ("ideal", "peng_robinson"):
            f = pressure_to_fugacity(1.0, 313.15, gas="CO2", method=method)
            assert isinstance(f, float), \
                f"Expected float for method={method}, got {type(f)}"

    def test_array_input_returns_array(self):
        """Array P → array f with matching shape."""
        from src.utils.chemistry import pressure_to_fugacity
        pressures = np.array([0.1, 1.0, 10.0])
        f = pressure_to_fugacity(pressures, 313.15, gas="CO2",
                                 method="peng_robinson")
        assert isinstance(f, np.ndarray)
        assert f.shape == pressures.shape

    def test_pr_low_pressure_approaches_ideal(self):
        """At 1 mbar, PR fugacity ≈ pressure."""
        from src.utils.chemistry import pressure_to_fugacity
        P = 0.001
        f = pressure_to_fugacity(P, 313.15, gas="CO2", method="peng_robinson")
        assert 0.99 < f / P < 1.01, \
            f"Fugacity should be ~P at very low P, got f/P = {f/P}"

    def test_pr_high_pressure_fugacity_exceeds_pressure(self):
        """At 100 bar, repulsive forces → f > P."""
        from src.utils.chemistry import pressure_to_fugacity
        P = 100.0
        f = pressure_to_fugacity(P, 313.15, gas="CO2", method="peng_robinson")
        assert f > P * 1.1, \
            f"Fugacity should exceed P at 100 bar, got f={f}, P={P}"

    def test_virial_agrees_with_pr_at_1bar(self):
        """Virial and PR should agree within 2 % at 1 bar."""
        from src.utils.chemistry import pressure_to_fugacity
        f_pr  = pressure_to_fugacity(1.0, 313.15, gas="CO2",
                                     method="peng_robinson")
        f_vir = pressure_to_fugacity(1.0, 313.15, gas="CO2", method="virial")
        assert abs(f_pr - f_vir) / f_pr < 0.02, \
            f"PR and virial disagree: {f_pr:.4f} vs {f_vir:.4f}"

    def test_fugacity_pressure_roundtrip(self):
        """pressure_to_fugacity → fugacity_to_pressure recovers P."""
        from src.utils.chemistry import fugacity_to_pressure, pressure_to_fugacity
        P_orig = 5.0
        T = 313.15
        f = pressure_to_fugacity(P_orig, T, gas="CO2", method="peng_robinson")
        P_back = fugacity_to_pressure(f, T, gas="CO2", method="peng_robinson")
        assert abs(P_back - P_orig) / P_orig < 1e-5, \
            f"Round-trip failed: {P_orig} → {f} → {P_back}"


# ═══════════════════════════════════════════════════════════════════════
# 3.  MIXTURE FUGACITY
# ═══════════════════════════════════════════════════════════════════════

class TestMixtureFugacity:
    """Test mixture fugacity with vdW1f mixing rules."""

    def test_pure_co2_matches_pure_component(self):
        """Pure-CO₂ mixture fugacity = pure-component PR fugacity."""
        from src.utils.chemistry import mixture_fugacity_pr, pressure_to_fugacity
        y = {"CO2": 1.0}
        P, T = 1.0, 313.15
        f_mix  = mixture_fugacity_pr(y, P, T)
        f_pure = pressure_to_fugacity(P, T, gas="CO2", method="peng_robinson")
        assert "CO2" in f_mix
        assert abs(f_mix["CO2"] - f_pure) / f_pure < 1e-4, \
            f"Pure-limit mismatch: {f_mix['CO2']:.6f} vs {f_pure:.6f}"

    def test_binary_mixture_positive_fugacities(self):
        """Both components must have positive fugacity."""
        from src.utils.chemistry import mixture_fugacity_pr
        f = mixture_fugacity_pr({"CO2": 0.15, "N2": 0.85}, 1.0, 313.15)
        assert f["CO2"] > 0
        assert f["N2"] > 0

    def test_ternary_humid_flue_gas(self):
        """Ternary CO₂/N₂/H₂O: all fugacities positive, sum ≈ P."""
        from src.utils.chemistry import mixture_fugacity_pr
        y = {"CO2": 0.15, "N2": 0.75, "H2O": 0.10}
        P, T = 1.0, 313.15
        f = mixture_fugacity_pr(y, P, T)
        assert all(f[k] > 0 for k in ("CO2", "N2", "H2O"))
        f_sum = sum(f.values())
        assert 0.9 * P < f_sum < 1.1 * P, \
            f"Fugacity sum should be ≈ P, got {f_sum:.4f}"

    def test_trace_species_nonzero_fugacity(self):
        """Trace H₂O (y=1e-12) → very small but non-zero fugacity."""
        from src.utils.chemistry import mixture_fugacity_pr
        f = mixture_fugacity_pr(
            {"CO2": 0.15, "N2": 0.85, "H2O": 1e-12}, 1.0, 313.15
        )
        assert 1e-15 < f["H2O"] < 1e-6, \
            f"Trace H₂O fugacity out of range: {f['H2O']}"

    def test_absent_species_treated_as_trace(self):
        """H₂O absent from y → returned as trace, not zero."""
        from src.utils.chemistry import mixture_fugacity_pr
        f = mixture_fugacity_pr({"CO2": 0.15, "N2": 0.85}, 1.0, 313.15)
        assert "H2O" in f, "H₂O should appear even when absent from input"
        assert 1e-15 < f["H2O"] < 1e-6, \
            f"Absent H₂O fugacity out of range: {f['H2O']}"


# ═══════════════════════════════════════════════════════════════════════
# 4.  CHEMICAL POTENTIAL CONVERSIONS
# ═══════════════════════════════════════════════════════════════════════

class TestChemicalPotentials:
    """Test chemical potential conversions."""

    def test_fugacity_1bar_gives_zero_mu(self):
        """f = 1 bar → μ = 0 kJ/mol (standard state)."""
        from src.utils.chemistry import fugacity_to_chemical_potential
        mu = fugacity_to_chemical_potential(1.0, 313.15)
        assert abs(mu) < 1e-10, f"μ should be 0 at f=1 bar, got {mu}"

    def test_scalar_input(self):
        from src.utils.chemistry import fugacity_to_chemical_potential
        mu = fugacity_to_chemical_potential(1.0, 313.15)
        assert isinstance(mu, float)

    def test_array_input(self):
        from src.utils.chemistry import fugacity_to_chemical_potential
        f = np.array([0.1, 1.0, 10.0])
        mu = fugacity_to_chemical_potential(f, 313.15)
        assert isinstance(mu, np.ndarray)
        assert mu.shape == f.shape

    def test_zero_mu_gives_fugacity_one(self):
        """μ = 0 → f = 1 bar."""
        from src.utils.chemistry import chemical_potential_to_fugacity
        f = chemical_potential_to_fugacity(0.0, 313.15)
        assert abs(f - 1.0) < 1e-10, f"f should be 1 bar at μ=0, got {f}"

    def test_mu_fugacity_roundtrip(self):
        """μ → f → μ should be identity."""
        from src.utils.chemistry import (
            chemical_potential_to_fugacity,
            fugacity_to_chemical_potential,
        )
        f_orig = 0.5
        mu = fugacity_to_chemical_potential(f_orig, 313.15)
        f_back = chemical_potential_to_fugacity(mu, 313.15)
        assert abs(f_back - f_orig) / f_orig < 1e-10

    def test_mu_increases_with_pressure(self):
        """Higher pressure → higher μ."""
        from src.utils.chemistry import pressure_to_chemical_potential
        T = 313.15
        mu_lo = pressure_to_chemical_potential(0.01, T, gas="CO2")
        mu_hi = pressure_to_chemical_potential(1.0,  T, gas="CO2")
        assert mu_hi > mu_lo


# ═══════════════════════════════════════════════════════════════════════
# 5.  TPNO CONDITION VECTOR
# ═══════════════════════════════════════════════════════════════════════

class TestConditionVector:
    """Test building TPNO condition vectors."""

    def test_ternary_gives_4d_vector(self):
        """Ternary mixture → 4-element vector [μ_CO2, μ_N2, μ_H2O, T]."""
        from src.utils.chemistry import build_condition_vector
        cond = build_condition_vector(
            pressure=1.0, temperature=313.15,
            y={"CO2": 0.15, "N2": 0.75, "H2O": 0.10},
        )
        assert len(cond) == 4
        assert cond[3] == pytest.approx(313.15), "Last element should be T"

    def test_dry_conditions_very_negative_mu_h2o(self):
        """
        CRITICAL: absent H₂O must give very negative μ, NOT zero.

        If μ_H2O = 0 the model thinks water is at standard state (1 bar),
        which is physically wrong and causes ~3 kJ/mol error per missing
        water molecule.
        """
        from src.utils.chemistry import build_condition_vector
        cond = build_condition_vector(
            pressure=1.0, temperature=313.15,
            y={"CO2": 0.15, "N2": 0.85},   # H₂O absent
        )
        assert cond[2] < -30.0, (
            f"μ_H2O should be << 0 for dry conditions, got {cond[2]:.2f} kJ/mol. "
            "This is the μ=0 bug: absent species must be treated as trace, "
            "not as zero chemical potential."
        )

    def test_humid_conditions_moderate_mu_h2o(self):
        """10 % H₂O at 1 bar → μ_H2O ≈ RT ln(0.1) ≈ −6 kJ/mol."""
        from src.utils.chemistry import build_condition_vector
        cond = build_condition_vector(
            pressure=1.0, temperature=313.15,
            y={"CO2": 0.15, "N2": 0.75, "H2O": 0.10},
        )
        assert -15.0 < cond[2] < -2.0, \
            f"μ_H2O should be ≈ −6 kJ/mol at y_H2O=0.10, got {cond[2]:.2f}"

    def test_returns_numpy_array(self):
        from src.utils.chemistry import build_condition_vector
        cond = build_condition_vector(
            pressure=1.0, temperature=313.15,
            y={"CO2": 0.15, "N2": 0.85},
        )
        assert isinstance(cond, np.ndarray)

    def test_condition_grid_monotonic_in_pressure(self):
        """μ must increase with P across a pressure sweep."""
        from src.utils.chemistry import build_condition_grid
        pressures = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
        grid = build_condition_grid(
            pressures=pressures,
            temperature=313.15,
            y={"CO2": 0.15, "N2": 0.85},
        )
        assert grid.shape == (len(pressures), 4)
        assert np.all(np.diff(grid[:, 0]) > 0), "μ_CO2 must increase with P"
        assert np.all(np.diff(grid[:, 1]) > 0), "μ_N2 must increase with P"

    def test_from_rh_dry_very_negative_mu(self):
        """RH=0 → μ_H2O very negative (not zero)."""
        from src.utils.chemistry import build_condition_vector_from_rh
        cond = build_condition_vector_from_rh(
            rh=0.0, temperature=313.15, total_pressure=1.013, y_co2_dry=0.15,
        )
        assert cond[2] < -30.0, \
            f"RH=0 should give μ_H2O << 0, got {cond[2]:.2f}"

    def test_from_rh_50pct(self):
        """RH=50 % → moderate μ_H2O."""
        from src.utils.chemistry import build_condition_vector_from_rh
        cond = build_condition_vector_from_rh(
            rh=0.5, temperature=313.15, total_pressure=1.013, y_co2_dry=0.15,
        )
        assert -15.0 < cond[2] < -2.0, \
            f"RH=50 % should give μ_H2O ≈ −8 kJ/mol, got {cond[2]:.2f}"

    def test_from_rh_100pct(self):
        """RH=100 % → saturated μ_H2O."""
        from src.utils.chemistry import build_condition_vector_from_rh
        cond = build_condition_vector_from_rh(
            rh=1.0, temperature=313.15, total_pressure=1.013, y_co2_dry=0.15,
        )
        assert -15.0 < cond[2] < -2.0, \
            f"RH=100 % should give μ_H2O ≈ −7 kJ/mol, got {cond[2]:.2f}"


# ═══════════════════════════════════════════════════════════════════════
# 6.  HUMIDITY CONVERSIONS
# ═══════════════════════════════════════════════════════════════════════

class TestHumidity:
    """Test humidity ↔ mole-fraction conversions."""

    def test_water_saturation_pressure_at_313k(self):
        """Antoine equation: P_sat(313 K) ≈ 0.073 bar."""
        from src.utils.chemistry import water_saturation_pressure_antoine
        P_sat = water_saturation_pressure_antoine(313.15)
        assert 0.06 < P_sat < 0.09, \
            f"P_sat(313 K) should be ≈ 0.073 bar, got {P_sat:.4f}"

    def test_antoine_and_buck_agree(self):
        """Antoine and Buck equations agree within 5 % at 313 K."""
        from src.utils.chemistry import (
            water_saturation_pressure_antoine,
            water_saturation_pressure_buck,
        )
        P_a = water_saturation_pressure_antoine(313.15)
        P_b = water_saturation_pressure_buck(313.15)
        assert abs(P_a - P_b) / P_a < 0.05

    def test_rh_100pct_gives_psat_over_ptotal(self):
        """y_H2O(RH=100 %) ≈ P_sat / P_total."""
        from src.utils.chemistry import relative_humidity_to_mole_fraction
        y = relative_humidity_to_mole_fraction(1.0, 313.15, total_pressure=1.013)
        assert 0.06 < y < 0.08, \
            f"y_H2O at 100 % RH should be ≈ 0.072, got {y:.4f}"

    def test_rh_to_y_scalar(self):
        from src.utils.chemistry import relative_humidity_to_mole_fraction
        y = relative_humidity_to_mole_fraction(0.5, 313.15, total_pressure=1.013)
        assert isinstance(y, float)

    def test_rh_to_y_array(self):
        from src.utils.chemistry import relative_humidity_to_mole_fraction
        rh = np.array([0.0, 0.5, 1.0])
        y = relative_humidity_to_mole_fraction(rh, 313.15, total_pressure=1.013)
        assert isinstance(y, np.ndarray)
        assert y.shape == rh.shape
        assert y[0] == 0.0
        assert y[1] > 0.0
        assert y[2] > y[1]

    def test_rh_y_roundtrip(self):
        """RH → y → RH should be identity."""
        from src.utils.chemistry import (
            mole_fraction_to_relative_humidity,
            relative_humidity_to_mole_fraction,
        )
        T, P = 313.15, 1.013
        rh_orig = 0.5
        y = relative_humidity_to_mole_fraction(rh_orig, T, P)
        rh_back = mole_fraction_to_relative_humidity(y, T, P)
        assert abs(rh_back - rh_orig) < 0.01

    def test_flue_gas_composition_sums_to_one(self):
        from src.utils.chemistry import flue_gas_composition
        comp = flue_gas_composition(rh=0.5, temperature=313.15)
        assert abs(sum(comp.values()) - 1.0) < 1e-6
        assert {"CO2", "N2", "H2O"} == set(comp.keys())

    def test_flue_gas_composition_dry(self):
        """RH=0 → y_H2O=0, standard 15 %/85 % split."""
        from src.utils.chemistry import flue_gas_composition
        comp = flue_gas_composition(rh=0.0, temperature=313.15)
        assert comp["H2O"] == 0.0
        assert comp["CO2"] == pytest.approx(0.15, abs=1e-6)
        assert comp["N2"]  == pytest.approx(0.85, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# 7.  ADSORPTION LOADING UNIT CONVERSIONS
# ═══════════════════════════════════════════════════════════════════════

class TestLoadingUnits:
    """Test adsorption loading unit conversions."""

    def test_mmol_g_to_mol_kg_identity(self):
        """1 mmol/g = 1 mol/kg."""
        from src.utils.chemistry import mmol_g_to_mol_kg
        assert mmol_g_to_mol_kg(1.0) == 1.0
        arr = mmol_g_to_mol_kg(np.array([1.0, 2.0]))
        assert arr[0] == 1.0

    def test_cm3stp_g_to_mol_kg(self):
        """22.414 cm³(STP)/g = 1 mol/kg."""
        from src.utils.chemistry import cm3stp_g_to_mol_kg
        assert abs(cm3stp_g_to_mol_kg(22.414) - 1.0) < 1e-6

    def test_mol_kg_to_cm3stp_g(self):
        """1 mol/kg = 22.414 cm³(STP)/g."""
        from src.utils.chemistry import mol_kg_to_cm3stp_g
        assert abs(mol_kg_to_cm3stp_g(1.0) - 22.414) < 1e-6

    def test_mg_g_to_mol_kg_co2(self):
        """44.01 mg/g CO₂ = 1 mol/kg."""
        from src.utils.chemistry import mg_g_to_mol_kg
        assert abs(mg_g_to_mol_kg(44.01, molar_mass=44.01) - 1.0) < 1e-6

    def test_mol_kg_to_mg_g_co2(self):
        """1 mol/kg CO₂ = 44.01 mg/g."""
        from src.utils.chemistry import mol_kg_to_mg_g
        assert abs(mol_kg_to_mg_g(1.0, molar_mass=44.01) - 44.01) < 1e-6

    def test_wt_pct_to_mol_kg_monotonic(self):
        """Higher wt% → higher mol/kg."""
        from src.utils.chemistry import wt_pct_to_mol_kg
        q1 = wt_pct_to_mol_kg(10.0, molar_mass=44.01)
        q2 = wt_pct_to_mol_kg(20.0, molar_mass=44.01)
        assert q2 > q1


# ═══════════════════════════════════════════════════════════════════════
# 8.  COMPRESSIBILITY & DENSITY
# ═══════════════════════════════════════════════════════════════════════

class TestCompressibility:
    """Test compressibility factor and density helpers."""

    def test_co2_z_at_1bar(self):
        """Z(CO₂, 313 K, 1 bar) ≈ 0.995."""
        from src.utils.chemistry import compressibility_factor_pr
        Z = compressibility_factor_pr(1.0, 313.15, gas="CO2")
        assert 0.99 < Z < 1.01

    def test_ideal_gas_density(self):
        """n/V = P/(RT) at 313 K, 1 bar ≈ 0.038 mol/L."""
        from src.utils.chemistry import gas_density
        rho = gas_density(1.0, 313.15, gas="CO2", method="ideal")
        expected = 1.0 / (0.08314 * 313.15)
        assert abs(rho - expected) / expected < 0.01

    def test_pr_density_close_to_ideal_at_1bar(self):
        """PR and ideal densities agree within 2 % at 1 bar."""
        from src.utils.chemistry import gas_density
        rho_i = gas_density(1.0, 313.15, gas="CO2", method="ideal")
        rho_p = gas_density(1.0, 313.15, gas="CO2", method="peng_robinson")
        assert abs(rho_p - rho_i) / rho_i < 0.02


# ═══════════════════════════════════════════════════════════════════════
# 9.  ISOSTERIC HEAT
# ═══════════════════════════════════════════════════════════════════════

class TestIsostericHeat:
    """Test isosteric heat of adsorption via Clausius-Clapeyron."""

    def test_synthetic_qst_recovery(self):
        """Synthetic data with Q_st = 30 kJ/mol should be recovered."""
        from src.utils.chemistry import clausius_clapeyron_qst
        R = 0.008314  # kJ mol⁻¹ K⁻¹
        Q_true = 30.0
        T = np.array([300.0, 310.0, 320.0, 330.0, 340.0])
        inv_T = 1.0 / T
        ln_p = -Q_true / R * inv_T + 5.0
        Q_calc = clausius_clapeyron_qst(ln_p, inv_T)
        assert abs(Q_calc - Q_true) < 0.1, \
            f"Expected Q_st ≈ {Q_true}, got {Q_calc:.3f}"

    def test_insufficient_points_returns_zero(self):
        """< 2 data points → return 0."""
        from src.utils.chemistry import clausius_clapeyron_qst
        assert clausius_clapeyron_qst(np.array([1.0]), np.array([1.0])) == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 10. INTEGRATION: MIXTURE PRESSURE → CHEMICAL POTENTIALS
# ═══════════════════════════════════════════════════════════════════════

class TestMixturePressureToChemicalPotentials:
    """Integration tests for the full P → μ pipeline."""

    def test_ternary_output_structure(self):
        """Full pipeline returns all three species with negative μ."""
        from src.utils.chemistry import mixture_pressure_to_chemical_potentials
        mu = mixture_pressure_to_chemical_potentials(
            {"CO2": 0.15, "N2": 0.75, "H2O": 0.10}, 1.0, 313.15
        )
        assert {"CO2", "N2", "H2O"} <= set(mu.keys())
        assert mu["CO2"] < 0
        assert mu["N2"]  < 0
        assert mu["H2O"] < 0

    def test_co2_mu_greater_than_n2_mu(self):
        """CO₂ partial pressure > N₂ partial pressure → μ_CO2 > μ_N2."""
        from src.utils.chemistry import mixture_pressure_to_chemical_potentials
        mu = mixture_pressure_to_chemical_potentials(
            {"CO2": 0.15, "N2": 0.75, "H2O": 0.10}, 1.0, 313.15
        )
        assert mu["CO2"] > mu["N2"]

    def test_trace_water_very_negative_mu(self):
        """
        CRITICAL: trace H₂O (y=1e-12) must give μ_H2O << 0, NOT zero.

        This is the μ=0 bug.  If μ_H2O = 0, the model interprets dry
        conditions as saturated (f_H2O = 1 bar), shifting CO₂ loading
        predictions by 10–30 %.
        """
        from src.utils.chemistry import mixture_pressure_to_chemical_potentials
        mu = mixture_pressure_to_chemical_potentials(
            {"CO2": 0.15, "N2": 0.85, "H2O": 1e-12}, 1.0, 313.15
        )
        assert mu["H2O"] < -30.0, (
            f"Trace H₂O should give μ << 0, got {mu['H2O']:.2f} kJ/mol. "
            "This is the μ=0 bug."
        )

    def test_absent_water_very_negative_mu(self):
        """
        CRITICAL: absent H₂O (not in y) must give μ_H2O << 0, NOT zero.
        """
        from src.utils.chemistry import mixture_pressure_to_chemical_potentials
        mu = mixture_pressure_to_chemical_potentials(
            {"CO2": 0.15, "N2": 0.85}, 1.0, 313.15  # H₂O absent
        )
        assert "H2O" in mu, "H₂O must appear in output even when absent from input"
        assert mu["H2O"] < -30.0, (
            f"Absent H₂O should give μ << 0, got {mu['H2O']:.2f} kJ/mol. "
            "This is the μ=0 bug."
        )


# ═══════════════════════════════════════════════════════════════════════
# 11. EDGE CASES & ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases and error handling."""

    def test_zero_pressure_gives_zero_fugacity(self):
        from src.utils.chemistry import pressure_to_fugacity
        f = pressure_to_fugacity(0.0, 313.15, gas="CO2", method="peng_robinson")
        assert f == 0.0

    def test_negative_pressure_clipped(self):
        from src.utils.chemistry import pressure_to_fugacity
        f = pressure_to_fugacity(-1.0, 313.15, gas="CO2", method="peng_robinson")
        assert f >= 0.0

    def test_zero_temperature_no_exception(self):
        from src.utils.chemistry import pressure_to_fugacity
        # Should not raise; result may be 0 or inf — just no exception
        f = pressure_to_fugacity(1.0, 0.0, gas="CO2", method="peng_robinson")
        assert f >= 0.0

    def test_unknown_gas_raises_key_error(self):
        from src.utils.chemistry import pressure_to_fugacity
        with pytest.raises(KeyError):
            pressure_to_fugacity(1.0, 313.15, gas="Krypton")

    def test_unknown_method_raises_value_error(self):
        from src.utils.chemistry import pressure_to_fugacity
        with pytest.raises(ValueError):
            pressure_to_fugacity(1.0, 313.15, gas="CO2", method="magic")

    def test_invalid_composition_handled_gracefully(self):
        """Negative mole fractions are clipped/normalised internally."""
        from src.utils.chemistry import build_condition_vector
        cond = build_condition_vector(
            pressure=1.0, temperature=313.15,
            y={"CO2": -0.1, "N2": 1.1},
        )
        assert len(cond) == 4
        assert np.all(np.isfinite(cond)), \
            "Condition vector must be finite even for invalid composition"


# ═══════════════════════════════════════════════════════════════════════
# 12. PERFORMANCE / SANITY
# ═══════════════════════════════════════════════════════════════════════

class TestPerformance:
    """Sanity and performance checks."""

    def test_vectorised_pr_fugacity_speed(self):
        """1 000-point PR fugacity sweep should finish in < 1 s."""
        from src.utils.chemistry import pressure_to_fugacity
        pressures = np.logspace(-3, 2, 1000)
        t0 = time.time()
        f = pressure_to_fugacity(pressures, 313.15, gas="CO2",
                                 method="peng_robinson")
        elapsed = time.time() - t0
        assert len(f) == 1000
        assert np.all(np.isfinite(f))
        assert elapsed < 1.0, f"1 000-point sweep took {elapsed:.2f} s"

    def test_condition_grid_monotone(self):
        """Pressure sweep → μ must increase for all species."""
        from src.utils.chemistry import build_condition_grid
        pressures = np.logspace(-2, 1, 20)
        grid = build_condition_grid(
            pressures=pressures,
            temperature=313.15,
            y={"CO2": 0.15, "N2": 0.85},
        )
        assert np.all(np.diff(grid[:, 0]) > 0), "μ_CO2 not monotone with P"
        assert np.all(np.diff(grid[:, 1]) > 0), "μ_N2 not monotone with P"


# ═══════════════════════════════════════════════════════════════════════
# 13. PHYSICAL CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════

class TestPhysicalConsistency:
    """Physical consistency checks for thermodynamic quantities."""

    def test_gibbs_duhem_fugacity_sum(self):
        """
        Σ y_i · f_i ≈ P for near-ideal mixtures.

        For an ideal mixture f_i = y_i · P, so this sum equals P exactly.
        For a real mixture it's approximate but should be within 10 %.
        """
        from src.utils.chemistry import mixture_fugacity_pr
        y = {"CO2": 0.15, "N2": 0.75, "H2O": 0.10}
        P, T = 1.0, 313.15
        f = mixture_fugacity_pr(y, P, T)
        f_sum = sum(y[s] * f[s] for s in y)
        assert 0.9 * P < f_sum < 1.1 * P, \
            f"Σ y_i f_i should be ≈ P, got {f_sum:.4f}"

    def test_henry_law_mu_linearity(self):
        """
        In the dilute limit μ = μ° + RT ln(P) → slope = RT.

        Δμ / Δ(ln P) should equal RT.
        """
        from src.utils.chemistry import pressure_to_chemical_potential
        T = 313.15
        P_lo, P_hi = 1e-6, 1e-4
        mu_lo = pressure_to_chemical_potential(P_lo, T, gas="CO2")
        mu_hi = pressure_to_chemical_potential(P_hi, T, gas="CO2")
        RT = 0.008314 * T
        expected = RT * np.log(P_hi / P_lo)
        actual   = mu_hi - mu_lo
        assert abs(actual - expected) / expected < 0.01, \
            f"Henry slope: expected {expected:.4f}, got {actual:.4f}"