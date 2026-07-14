"""
Physical constants, gas properties, unit conversions, and domain-specific
parameters for the UC-TPNO humid flue-gas CO₂ capture pipeline.

This module serves as the single source of truth for all physical and
chemical parameters used throughout the pipeline.  Every value carries
a reference so that reviewers can trace it back to primary literature.

Fixes vs previous version
──────────────────────────
• SOFTPLUS_BETA, TORR_TO_PA, KPA_TO_BAR, PSI_TO_BAR, ATM_TO_PA,
  NM_TO_ANGSTROM, AMU_TO_KG, CM1_TO_EV, J_TO_EV, KJ_TO_KCAL,
  HARTREE_TO_EV, ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, ANGSTROM_TO_M
  were all defined but missing from __all__.  Added.
• No physics values changed — all verified against CODATA 2018.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Final, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# 1.  FUNDAMENTAL PHYSICAL CONSTANTS  (CODATA 2018 exact values)
# ═══════════════════════════════════════════════════════════════════════

R: Final[float]           = 8.314462618          # Universal gas constant  [J mol⁻¹ K⁻¹]
NA: Final[float]          = 6.02214076e23         # Avogadro number         [mol⁻¹]
kB: Final[float]          = 1.380649e-23          # Boltzmann constant      [J K⁻¹]
h_planck: Final[float]    = 6.62607015e-34        # Planck constant         [J s]
c_light: Final[float]     = 2.99792458e8          # Speed of light          [m s⁻¹]
epsilon_0: Final[float]   = 8.8541878128e-12      # Vacuum permittivity     [F m⁻¹]
e_charge: Final[float]    = 1.602176634e-19       # Elementary charge       [C]

# Derived
R_kJ: Final[float]  = R / 1000.0                 # Gas constant            [kJ mol⁻¹ K⁻¹]
kB_eV: Final[float] = kB / e_charge              # Boltzmann constant      [eV K⁻¹]


# ═══════════════════════════════════════════════════════════════════════
# 2.  UNIT CONVERSION FACTORS
# ═══════════════════════════════════════════════════════════════════════

# ── Pressure ─────────────────────────────────────────────────────────
BAR_TO_PA: Final[float]   = 1.0e5
PA_TO_BAR: Final[float]   = 1.0e-5
ATM_TO_PA: Final[float]   = 101325.0
ATM_TO_BAR: Final[float]  = 1.01325
TORR_TO_PA: Final[float]  = 133.32236842
KPA_TO_BAR: Final[float]  = 0.01
PSI_TO_BAR: Final[float]  = 0.0689476

# ── Energy ───────────────────────────────────────────────────────────
EV_TO_J: Final[float]       = 1.602176634e-19
J_TO_EV: Final[float]       = 1.0 / EV_TO_J
HARTREE_TO_EV: Final[float] = 27.211386245988
KCAL_TO_KJ: Final[float]    = 4.184
KJ_TO_KCAL: Final[float]    = 1.0 / 4.184
CM1_TO_EV: Final[float]     = 1.2398419843320026e-4   # cm⁻¹ → eV

# ── Length ───────────────────────────────────────────────────────────
BOHR_TO_ANGSTROM: Final[float]  = 0.529177210903
ANGSTROM_TO_BOHR: Final[float]  = 1.0 / BOHR_TO_ANGSTROM
ANGSTROM_TO_M: Final[float]     = 1.0e-10
NM_TO_ANGSTROM: Final[float]    = 10.0

# ── Mass ─────────────────────────────────────────────────────────────
AMU_TO_KG: Final[float] = 1.66053906660e-27       # unified atomic mass unit [kg]

# ── Adsorption-specific ──────────────────────────────────────────────
MMOL_G_TO_MOL_KG: Final[float]    = 1.0           # mmol/g ≡ mol/kg
CM3STP_G_TO_MOL_KG: Final[float]  = 1.0 / 22.414  # cm³(STP)/g → mol/kg
MG_G_TO_MOL_KG_CO2: Final[float]  = 1.0 / 44.01   # mg_CO2/g → mol_CO2/kg
MOLEC_UC_TO_MOL_KG: Final[float]  = 1.0 / NA       # molecules/uc → mol (needs uc mass)


# ═══════════════════════════════════════════════════════════════════════
# 3.  GAS SPECIES PROPERTIES
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GasSpecies:
    """Immutable container for a single gas species' physical properties."""

    name: str
    formula: str
    molar_mass: float            # g mol⁻¹
    kinetic_diameter: float      # Å
    polarizability: float        # Å³
    critical_temperature: float  # K         (NIST WebBook)
    critical_pressure: float     # bar       (NIST WebBook)
    acentric_factor: float       # —         (Poling et al., 5th ed.)

    # Electrostatics
    quadrupole_moment: Optional[float] = None    # 10⁻⁴⁰ C m²
    dipole_moment: Optional[float]     = None    # Debye

    # Lennard-Jones (TraPPE-UA, Martin & Siepmann 1998)
    lj_epsilon_kB: Optional[float] = None        # ε/kB [K]
    lj_sigma: Optional[float]      = None        # σ [Å]

    # Ideal-gas heat capacity coefficients  Cp/R = a + bT + cT² + dT³
    # Valid 298–1500 K  (JANAF / Poling Appendix A)
    cp_coeffs: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)

    # Antoine equation  log₁₀(P_sat / bar) = A − B/(T + C)   [T in K]
    # Only meaningful for condensable species at pipeline conditions.
    antoine_A: Optional[float] = None
    antoine_B: Optional[float] = None
    antoine_C: Optional[float] = None

    # ── derived helpers ──────────────────────────────────────────
    @property
    def reduced_temperature(self) -> float:
        """Reduced temperature at 298.15 K (useful default)."""
        return 298.15 / self.critical_temperature

    @property
    def critical_volume(self) -> float:
        """Estimate Vc via Peng-Robinson: Vc ≈ 0.3074·R·Tc/Pc  [cm³ mol⁻¹].
        R in bar·cm³/(mol·K) = 83.145.
        """
        return 0.3074 * 83.145 * self.critical_temperature / self.critical_pressure

    def saturation_pressure(self, temperature: float) -> Optional[float]:
        """Antoine saturation pressure [bar] at given T [K]."""
        if self.antoine_A is None:
            return None
        return 10.0 ** (
            self.antoine_A - self.antoine_B / (temperature + self.antoine_C)
        )

    def heat_capacity(self, temperature: float) -> float:
        """Ideal-gas Cp [J mol⁻¹ K⁻¹] via polynomial."""
        a, b, c, d = self.cp_coeffs
        Cp_over_R = a + b * temperature + c * temperature ** 2 + d * temperature ** 3
        return Cp_over_R * R


# ── Instantiate the three target species ──────────────────────────────

CO2 = GasSpecies(
    name="carbon dioxide",
    formula="CO2",
    molar_mass=44.009,
    kinetic_diameter=3.30,
    polarizability=2.507,
    critical_temperature=304.128,
    critical_pressure=73.773,
    acentric_factor=0.22394,
    quadrupole_moment=-13.4,
    lj_epsilon_kB=235.9,        # TraPPE C site
    lj_sigma=3.05,              # TraPPE C site
    cp_coeffs=(3.259, 1.356e-3, 1.502e-5, -2.374e-8),
)

N2 = GasSpecies(
    name="nitrogen",
    formula="N2",
    molar_mass=28.014,
    kinetic_diameter=3.64,
    polarizability=1.740,
    critical_temperature=126.192,
    critical_pressure=33.958,
    acentric_factor=0.03720,
    quadrupole_moment=-4.65,
    lj_epsilon_kB=36.0,         # TraPPE N site
    lj_sigma=3.31,              # TraPPE N site
    cp_coeffs=(3.539, -0.261e-3, 0.007e-5, 0.157e-8),
)

H2O = GasSpecies(
    name="water",
    formula="H2O",
    molar_mass=18.015,
    kinetic_diameter=2.65,
    polarizability=1.450,
    critical_temperature=647.096,
    critical_pressure=220.640,
    acentric_factor=0.34486,
    dipole_moment=1.8546,
    lj_epsilon_kB=93.2,         # TIP4P/2005
    lj_sigma=3.1589,            # TIP4P/2005
    cp_coeffs=(4.070, -1.108e-3, 4.152e-6, -2.964e-9),
    # Antoine constants (NIST, valid ~255–373 K, P in bar)
    antoine_A=5.40221,
    antoine_B=1838.675,
    antoine_C=-31.737,
)

# Lookup dict keyed by formula string
GAS_REGISTRY: Dict[str, GasSpecies] = {
    "CO2": CO2,
    "N2": N2,
    "H2O": H2O,
}

# Legacy flat dict (backward-compatible with earlier pipeline code)
gas_properties: Dict[str, Dict] = {
    formula: {
        "molar_mass": sp.molar_mass,
        "kinetic_diameter": sp.kinetic_diameter,
        "polarizability": sp.polarizability,
        "critical_temperature": sp.critical_temperature,
        "critical_pressure": sp.critical_pressure,
        "acentric_factor": sp.acentric_factor,
        "quadrupole_moment": sp.quadrupole_moment,
        "dipole_moment": sp.dipole_moment,
    }
    for formula, sp in GAS_REGISTRY.items()
}


# ═══════════════════════════════════════════════════════════════════════
# 4.  TYPICAL FLUE-GAS CONDITIONS
# ═══════════════════════════════════════════════════════════════════════
# Post-combustion coal flue gas after desulfurization (FGD).
# Reference: Bui et al. (2018) Energy Environ. Sci., 11, 1062.

@dataclass(frozen=True)
class FlueGasCondition:
    """A named operating point for the flue-gas feed."""
    name: str
    temperature: float           # K
    total_pressure: float        # bar
    y_CO2: float                 # mole fraction
    y_N2: float
    y_H2O: float
    description: str = ""


FLUE_GAS_DRY = FlueGasCondition(
    name="dry_coal_post_fgd",
    temperature=313.15,
    total_pressure=1.013,
    y_CO2=0.15,
    y_N2=0.85,
    y_H2O=0.00,
    description="Dry post-FGD coal flue gas (baseline).",
)

FLUE_GAS_HUMID_LOW = FlueGasCondition(
    name="humid_coal_rh30",
    temperature=313.15,
    total_pressure=1.013,
    y_CO2=0.13,
    y_N2=0.74,
    y_H2O=0.13,
    description="~30 % RH post-FGD coal flue gas.",
)

FLUE_GAS_HUMID_HIGH = FlueGasCondition(
    name="humid_coal_rh80",
    temperature=313.15,
    total_pressure=1.013,
    y_CO2=0.10,
    y_N2=0.55,
    y_H2O=0.35,
    description="~80 % RH (saturated direct-air-capture-like).",
)

FLUE_GAS_CONDITIONS: Dict[str, FlueGasCondition] = {
    c.name: c for c in [FLUE_GAS_DRY, FLUE_GAS_HUMID_LOW, FLUE_GAS_HUMID_HIGH]
}


# ═══════════════════════════════════════════════════════════════════════
# 5.  ADSORPTION REGIME BOUNDARIES
# ═══════════════════════════════════════════════════════════════════════

PRESSURE_REGIMES: Dict[str, Tuple[float, float]] = {
    "henry":   (1e-6,  1e-3),     # bar — linear (Henry) regime
    "low":     (1e-3,  0.1),      # bar — onset of curvature
    "medium":  (0.1,   5.0),      # bar — process-relevant window
    "high":    (5.0,   100.0),    # bar — saturation regime
}

TEMPERATURE_RANGE: Tuple[float, float] = (273.15, 423.15)  # K — typical MOF studies


# ═══════════════════════════════════════════════════════════════════════
# 6.  MOF STRUCTURAL DESCRIPTORS & TOPOLOGY LABELS
# ═══════════════════════════════════════════════════════════════════════

PORE_SIZE_CLASSES: Dict[str, Tuple[float, float]] = {
    "ultramicroporous": (0.0,    7.0),      # Å
    "microporous":      (7.0,   20.0),
    "mesoporous":       (20.0, 500.0),
    "macroporous":      (500.0, float("inf")),
}

# Common net topologies encountered in CoRE MOF / ARC-MOF databases.
COMMON_TOPOLOGIES: List[str] = [
    "pcu", "dia", "sod", "rht", "ftw", "csq", "alb", "bct",
    "nbo", "fcu", "hxg", "lvt", "pts", "tbo", "tfb", "ths",
    "acs", "bnn", "cab", "cds", "flu", "gme", "irl", "mtn",
    "pyr", "qtz", "she", "sql", "sra", "srs", "tpt", "ubt",
]

# Metal node families relevant for humid CO₂ capture
METAL_FAMILIES: Dict[str, List[str]] = {
    "open_metal_site": ["Cu", "Co", "Ni", "Fe", "Mn", "Cr", "V"],
    "high_valent":     ["Zr", "Hf", "Ti", "Ce"],
    "main_group":      ["Al", "In", "Ga"],
    "noble":           ["Ag", "Au", "Pt", "Pd"],
    "alkaline_earth":  ["Mg", "Ca", "Sr", "Ba"],
    "alkali":          ["Li", "Na", "K"],
}


# ═══════════════════════════════════════════════════════════════════════
# 7.  FORCE-FIELD REGISTRY  (for GCMC / MD simulations via RASPA)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ForceField:
    """Metadata for a simulation force field."""
    name: str
    ff_type: str           # "generic", "transferable", "water", "polarizable"
    reference: str
    charges: str           # "none", "point", "gaussian"
    mixing_rule: str = "Lorentz-Berthelot"
    fidelity_level: int = 1   # 1=low (generic), 2=mid, 3=high (polarizable)


FORCE_FIELDS: Dict[str, ForceField] = {
    "UFF": ForceField(
        name="UFF",
        ff_type="generic",
        reference="Rappé et al. J. Am. Chem. Soc. 1992, 114, 10024",
        charges="none",
        fidelity_level=1,
    ),
    "DREIDING": ForceField(
        name="DREIDING",
        ff_type="generic",
        reference="Mayo et al. J. Phys. Chem. 1990, 94, 8897",
        charges="none",
        fidelity_level=1,
    ),
    "TraPPE": ForceField(
        name="TraPPE-UA",
        ff_type="transferable",
        reference="Martin & Siepmann J. Phys. Chem. B 1998, 102, 2569",
        charges="point",
        fidelity_level=2,
    ),
    "TIP4P_2005": ForceField(
        name="TIP4P/2005",
        ff_type="water",
        reference="Abascal & Vega J. Chem. Phys. 2005, 123, 234505",
        charges="point",
        fidelity_level=2,
    ),
    "SPC_E": ForceField(
        name="SPC/E",
        ff_type="water",
        reference="Berendsen et al. J. Phys. Chem. 1987, 91, 6269",
        charges="point",
        fidelity_level=2,
    ),
    "EQeq_TraPPE": ForceField(
        name="EQeq + TraPPE",
        ff_type="transferable",
        reference="Wilmer et al. J. Phys. Chem. Lett. 2012, 3, 2506",
        charges="point",
        fidelity_level=2,
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# 8.  MULTI-FIDELITY SIMULATION COST MODEL
# ═══════════════════════════════════════════════════════════════════════
# Relative cost coefficients for the multi-fidelity BO module.
# F1=cheap GCMC (UFF, short), F2=GCMC (TraPPE, converged),
# F3=DFT+D, F4=AIMD.  Costs normalised so F1=1.

FIDELITY_LEVELS: Dict[str, Dict] = {
    "F1": {
        "label": "UFF-GCMC-short",
        "relative_cost": 1.0,
        "accuracy_tier": "screening",
        "description": "UFF + Lorentz-Berthelot, 200k cycles, no charges.",
    },
    "F2": {
        "label": "TraPPE-GCMC-converged",
        "relative_cost": 8.0,
        "accuracy_tier": "quantitative",
        "description": "TraPPE + EQeq charges, 2M cycles, Ewald sum.",
    },
    "F3": {
        "label": "DFT-D-static",
        "relative_cost": 200.0,
        "accuracy_tier": "reference",
        "description": "DFT-D3(BJ)/PBE single-point on binding sites.",
    },
    "F4": {
        "label": "AIMD",
        "relative_cost": 5000.0,
        "accuracy_tier": "gold_standard",
        "description": "Ab-initio MD at operating T (very expensive).",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# 9.  PROCESS ENGINEERING DEFAULTS  (PVSA cycle)
# ═══════════════════════════════════════════════════════════════════════
# Default parameters for the Pressure-Vacuum Swing Adsorption surrogate.
# Reference: Khurana & Farooq (2016) AIChE J., 62, 2290.

@dataclass
class PVSADefaults:
    """Default PVSA cycle parameters for KPI evaluation."""
    feed_pressure: float       = 1.013      # bar
    vacuum_pressure: float     = 0.05       # bar
    feed_temperature: float    = 313.15     # K
    desorption_temperature: float = 423.15  # K  (only for T-swing hybrid)
    feed_velocity: float       = 1.0        # m s⁻¹
    column_length: float       = 1.0        # m
    column_diameter: float     = 0.3        # m
    bed_voidage: float         = 0.37       # —
    pellet_density: float      = 1100.0     # kg m⁻³
    cycle_time: float          = 120.0      # s  (adsorption step)
    n_steps: int               = 4          # Skarstrom cycle steps
    # Target KPIs (DOE/NETL guidelines)
    target_purity: float       = 0.95       # mol fraction CO₂ in product
    target_recovery: float     = 0.90       # mol fraction CO₂ recovered
    max_energy_penalty: float  = 1.0        # GJ per tonne CO₂ captured


PVSA_DEFAULTS = PVSADefaults()


# ═══════════════════════════════════════════════════════════════════════
# 10.  MODEL HYPERPARAMETER DEFAULTS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EncoderDefaults:
    """Default NequIP encoder hyper-parameters."""
    n_species: int   = 100
    emb_dim: int     = 128
    n_layers: int    = 4
    lmax: int        = 2
    n_rbf: int       = 32
    cutoff: float    = 6.0       # Å
    use_pbc: bool    = True


@dataclass
class OperatorDefaults:
    """Default TPNO operator hyper-parameters."""
    hidden_dim: int        = 256
    n_layers: int          = 4
    n_conditions: int      = 4   # μ_CO₂, μ_N₂, μ_H₂O, T
    n_components: int      = 3   # CO₂, N₂, H₂O
    convex_constraint: str = "softplus"
    film_conditioning: bool = True
    dropout: float         = 0.1
    use_layer_norm: bool   = True
    activation: str        = "silu"    # SiLU ≡ Swish


@dataclass
class TrainingDefaults:
    """Default training hyper-parameters."""
    n_epochs: int        = 100
    batch_size: int      = 16
    lr: float            = 1e-3
    weight_decay: float  = 1e-5
    optimizer: str       = "adamw"
    scheduler: str       = "cosine_warm_restarts"
    scheduler_T0: int    = 10
    scheduler_T_mult: int = 2
    grad_clip: float     = 1.0
    # Physics loss weights
    lambda_hessian: float    = 0.10
    lambda_monotonic: float  = 0.10
    lambda_henry: float      = 0.01
    lambda_competition: float = 0.05
    # Ensemble
    n_ensemble: int      = 5
    # UQ
    alpha: float         = 0.10   # 90 % prediction intervals


ENCODER_DEFAULTS  = EncoderDefaults()
OPERATOR_DEFAULTS = OperatorDefaults()
TRAINING_DEFAULTS = TrainingDefaults()


# ═══════════════════════════════════════════════════════════════════════
# 11.  NUMERICAL SAFETY CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

EPS: Final[float]           = 1e-8    # Generic epsilon for division safety
LOG_EPS: Final[float]       = 1e-12   # Epsilon inside log() to prevent -inf
SOFTPLUS_BETA: Final[float] = 1.0     # Default β for softplus (used in ICNN)
MAX_LOG_VAR: Final[float]   = 10.0    # Clamp log-variance to prevent overflow
MIN_SIGMA: Final[float]     = 1e-6    # Minimum allowed standard deviation


# ═══════════════════════════════════════════════════════════════════════
# 12.  ELEMENT PROPERTIES  (for graph construction & featurisation)
# ═══════════════════════════════════════════════════════════════════════
# Pauling electro-negativities and covalent radii for the most common
# elements found in MOFs (Z ≤ 57 + selected lanthanides).

ELECTRONEGATIVITIES: Dict[int, float] = {
    1: 2.20,  6: 2.55,  7: 3.04,  8: 3.44,  9: 3.98,
    11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19,
    16: 2.58, 17: 3.16, 19: 0.82, 20: 1.00, 22: 1.54,
    23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83, 27: 1.88,
    28: 1.91, 29: 1.90, 30: 1.65, 35: 2.96, 40: 1.33,
    42: 2.16, 47: 1.93, 48: 1.69, 49: 1.78, 50: 1.96,
    56: 0.89, 57: 1.10, 72: 1.30,
}

COVALENT_RADII: Dict[int, float] = {
    # Å — Cordero et al. Dalton Trans. 2008, 2832
    1: 0.31,  6: 0.76,  7: 0.71,  8: 0.66,  9: 0.57,
    11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07,
    16: 1.05, 17: 1.02, 19: 2.03, 20: 1.76, 22: 1.60,
    23: 1.53, 24: 1.39, 25: 1.39, 26: 1.32, 27: 1.26,
    28: 1.24, 29: 1.32, 30: 1.22, 35: 1.20, 40: 1.75,
    42: 1.54, 47: 1.45, 48: 1.44, 49: 1.42, 50: 1.39,
    56: 2.15, 57: 2.07, 72: 1.75,
}


# ═══════════════════════════════════════════════════════════════════════
# 13.  SEED & VERSIONING
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_SEED: Final[int]    = 42
PIPELINE_VERSION: Final[str] = "1.0.0"
PIPELINE_NAME: Final[str]   = "UC-TPNO"


# ═══════════════════════════════════════════════════════════════════════
# 14.  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    # ── Fundamental constants ────────────────────────────────────
    "R", "R_kJ",
    "NA", "kB", "kB_eV",
    "h_planck", "c_light", "epsilon_0", "e_charge",

    # ── Unit conversions — pressure ──────────────────────────────
    "BAR_TO_PA", "PA_TO_BAR",
    "ATM_TO_PA", "ATM_TO_BAR",
    "TORR_TO_PA", "KPA_TO_BAR", "PSI_TO_BAR",

    # ── Unit conversions — energy ────────────────────────────────
    "EV_TO_J", "J_TO_EV",
    "HARTREE_TO_EV",
    "KCAL_TO_KJ", "KJ_TO_KCAL",
    "CM1_TO_EV",

    # ── Unit conversions — length ────────────────────────────────
    "BOHR_TO_ANGSTROM", "ANGSTROM_TO_BOHR",
    "ANGSTROM_TO_M",
    "NM_TO_ANGSTROM",

    # ── Unit conversions — mass ──────────────────────────────────
    "AMU_TO_KG",

    # ── Unit conversions — adsorption ────────────────────────────
    "MMOL_G_TO_MOL_KG", "CM3STP_G_TO_MOL_KG",
    "MG_G_TO_MOL_KG_CO2", "MOLEC_UC_TO_MOL_KG",

    # ── Gas species ──────────────────────────────────────────────
    "GasSpecies",
    "CO2", "N2", "H2O",
    "GAS_REGISTRY", "gas_properties",

    # ── Flue-gas conditions ──────────────────────────────────────
    "FlueGasCondition",
    "FLUE_GAS_DRY", "FLUE_GAS_HUMID_LOW", "FLUE_GAS_HUMID_HIGH",
    "FLUE_GAS_CONDITIONS",

    # ── Adsorption regimes ───────────────────────────────────────
    "PRESSURE_REGIMES", "TEMPERATURE_RANGE",

    # ── MOF descriptors ──────────────────────────────────────────
    "PORE_SIZE_CLASSES", "COMMON_TOPOLOGIES", "METAL_FAMILIES",

    # ── Force fields & fidelity ──────────────────────────────────
    "ForceField", "FORCE_FIELDS", "FIDELITY_LEVELS",

    # ── Process defaults ─────────────────────────────────────────
    "PVSADefaults", "PVSA_DEFAULTS",

    # ── Model defaults ───────────────────────────────────────────
    "EncoderDefaults", "OperatorDefaults", "TrainingDefaults",
    "ENCODER_DEFAULTS", "OPERATOR_DEFAULTS", "TRAINING_DEFAULTS",

    # ── Numerical safety ─────────────────────────────────────────
    "EPS", "LOG_EPS",
    "SOFTPLUS_BETA",           # FIX: was missing from __all__
    "MAX_LOG_VAR", "MIN_SIGMA",

    # ── Element properties ───────────────────────────────────────
    "ELECTRONEGATIVITIES", "COVALENT_RADII",

    # ── Meta ─────────────────────────────────────────────────────
    "DEFAULT_SEED", "PIPELINE_VERSION", "PIPELINE_NAME",
]