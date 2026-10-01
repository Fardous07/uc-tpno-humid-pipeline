#!/usr/bin/env python3
"""
build_raspa_forcefield.py
Generate the RASPA2 force field used for all humid CO2/N2/H2O simulations:

    forcefields/UFF_TIP4P2005/pseudo_atoms.def
    forcefields/UFF_TIP4P2005/force_field_mixing_rules.def

Framework : UFF Lennard-Jones for every element (Rappe et al., J. Am. Chem.
            Soc. 1992, 114, 10024), sigma = x_i / 2^(1/6), eps = D_i.
            Framework charges come from the CIF (UseChargesFromCIFFile yes).
CO2       : Garcia-Sanchez et al., J. Phys. Chem. C 2009, 113, 8814.
N2        : Martin-Calvo et al., Phys. Chem. Chem. Phys. 2011, 13, 11165.
H2O       : TIP4P/2005, Abascal & Vega, J. Chem. Phys. 2005, 123, 234505.
Mixing    : Lorentz-Berthelot, shifted, no tail corrections.

Why this replaces RASPA's bundled ExampleMOFsForceField
-------------------------------------------------------
1. It contains a TraPPE united-atom quaternary-carbon entry "C" (eps 0.8 K,
   sigma 6.38 A). With RemoveAtomNumberCodeFromLabel, every framework carbon
   label ("C1", "C2", ...) becomes "C" and matched that entry instead of the
   DREIDING "C_" one -- framework carbon was a huge, nearly non-attractive
   sphere in 474/500 selected MOFs (verified in RASPA's own pair table:
   O_co2 - C  eps 8.3 K, sigma 4.70 A).
2. Its entries ending in "_" are prefix wildcards, so "C_" also matched Ca,
   Ce, Cs; "N_" matched Na, Nd; "S_" matched Sr, Sm, Se; "P_" matched Pb,
   Pd, Pr; "H_" matched Ho -- and La, K, Li, Rb, W, Mo, U, I, Re and most
   lanthanides had no parameters at all.
Every name written here is an exact match (no trailing "_"), there is one
framework pseudo atom per element symbol, and no guest pseudo atom shares a
name with an element symbol.

Usage:
    python scripts/build_raspa_forcefield.py [--out forcefields/UFF_TIP4P2005]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ase.data import atomic_masses, atomic_numbers

# UFF vdW parameters per element: (x_i [A], D_i [kcal/mol]), first atom type
# listed for each element in Open Babel's data/UFF.prm (all atom types of an
# element share the same vdW parameters there).
UFF_XD = {
    "H": (2.886, 0.044), "He": (2.362, 0.056), "Li": (2.451, 0.025), "Be": (2.745, 0.085), "B": (4.083, 0.18),
    "C": (3.851, 0.105), "N": (3.66, 0.069), "O": (3.5, 0.06), "F": (3.364, 0.05), "Ne": (3.243, 0.042),
    "Na": (2.983, 0.03), "Mg": (3.021, 0.111), "Al": (4.499, 0.505), "Si": (4.295, 0.402), "P": (4.147, 0.305),
    "S": (4.035, 0.274), "Cl": (3.947, 0.227), "Ar": (3.868, 0.185), "K": (3.812, 0.035), "Ca": (3.399, 0.238),
    "Sc": (3.295, 0.019), "Ti": (3.175, 0.017), "V": (3.144, 0.016), "Cr": (3.023, 0.015), "Mn": (2.961, 0.013),
    "Fe": (2.912, 0.013), "Co": (2.872, 0.014), "Ni": (2.834, 0.015), "Cu": (3.495, 0.005), "Zn": (2.763, 0.124),
    "Ga": (4.383, 0.415), "Ge": (4.28, 0.379), "As": (4.23, 0.309), "Se": (4.205, 0.291), "Br": (4.189, 0.251),
    "Kr": (4.141, 0.22), "Rb": (4.114, 0.04), "Sr": (3.641, 0.235), "Y": (3.345, 0.072), "Zr": (3.124, 0.069),
    "Nb": (3.165, 0.059), "Mo": (3.052, 0.056), "Tc": (2.998, 0.048), "Ru": (2.963, 0.056), "Rh": (2.929, 0.053),
    "Pd": (2.899, 0.048), "Ag": (3.148, 0.036), "Cd": (2.848, 0.228), "In": (4.463, 0.599), "Sn": (4.392, 0.567),
    "Sb": (4.42, 0.449), "Te": (4.47, 0.398), "I": (4.5, 0.339), "Xe": (4.404, 0.332), "Cs": (4.517, 0.045),
    "Ba": (3.703, 0.364), "La": (3.522, 0.017), "Ce": (3.556, 0.013), "Pr": (3.606, 0.01), "Nd": (3.575, 0.01),
    "Pm": (3.547, 0.009), "Sm": (3.52, 0.008), "Eu": (3.493, 0.008), "Gd": (3.368, 0.009), "Tb": (3.451, 0.007),
    "Dy": (3.428, 0.007), "Ho": (3.409, 0.007), "Er": (3.391, 0.007), "Tm": (3.374, 0.006), "Yb": (3.355, 0.228),
    "Lu": (3.64, 0.041), "Hf": (3.141, 0.072), "Ta": (3.17, 0.081), "W": (3.069, 0.067), "Re": (2.954, 0.066),
    "Os": (3.12, 0.037), "Ir": (2.84, 0.073), "Pt": (2.754, 0.08), "Au": (3.293, 0.039), "Hg": (2.705, 0.385),
    "Tl": (4.347, 0.68), "Pb": (4.297, 0.663), "Bi": (4.37, 0.518), "Po": (4.709, 0.325), "At": (4.75, 0.284),
    "Rn": (4.765, 0.248), "Fr": (4.9, 0.05), "Ra": (3.677, 0.404), "Ac": (3.478, 0.033), "Th": (3.396, 0.026),
    "Pa": (3.424, 0.022), "U": (3.395, 0.022), "Np": (3.424, 0.019), "Pu": (3.424, 0.016), "Am": (3.381, 0.014),
    "Cm": (3.326, 0.013), "Bk": (3.339, 0.013), "Cf": (3.313, 0.013), "Es": (3.299, 0.012), "Fm": (3.286, 0.012),
    "Md": (3.274, 0.011), "No": (3.248, 0.011), "Lr": (3.236, 0.011),
}

KCALMOL_TO_K = 503.2195
SIX_ROOT_TWO = 2.0 ** (1.0 / 6.0)

# Guest pseudo atoms: (name, print, as, chem, mass, charge, radius) -- names
# and charges must match the ExampleDefinitions CO2/N2/H2O .def files.
GUEST_PSEUDO = [
    ("He",    "yes", "He", "He", 4.002602,  0.0,    1.0),
    ("C_co2", "yes", "C",  "C",  12.0,      0.6512, 0.720),
    ("O_co2", "yes", "O",  "O",  15.9994,  -0.3256, 0.68),
    ("N_n2",  "yes", "N",  "N",  14.00674, -0.405,  0.7),
    ("N_com", "no",  "N",  "-",  0.0,       0.810,  0.7),
    ("O_h2o", "yes", "O",  "O",  15.9994,   0.0,    0.68),
    ("H_h2o", "yes", "H",  "H",  1.00794,   0.5564, 0.4),
    ("M_h2o", "no",  "O",  "-",  0.0,      -1.1128, 0.0),
]
GUEST_LJ = [
    ("He",    "lennard-jones", 10.9,   2.64,   "Hirschfelder et al. 1954"),
    ("C_co2", "lennard-jones", 29.933, 2.745,  "Garcia-Sanchez et al. 2009"),
    ("O_co2", "lennard-jones", 85.671, 3.017,  "Garcia-Sanchez et al. 2009"),
    ("N_n2",  "lennard-jones", 38.298, 3.306,  "Martin-Calvo et al. 2011"),
    ("N_com", "none",          None,   None,   "massless charge site"),
    ("O_h2o", "lennard-jones", 93.2,   3.1589, "TIP4P/2005"),
    ("H_h2o", "none",          None,   None,   "TIP4P/2005, no LJ site"),
    ("M_h2o", "none",          None,   None,   "TIP4P/2005, massless charge site"),
]

FORCEFIELD_NAME = "UFF_TIP4P2005"


def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    elements = sorted(UFF_XD, key=lambda e: atomic_numbers[e])
    guest_names = {g[0] for g in GUEST_PSEUDO}
    clash = guest_names & set(elements)
    # "He" is both a guest (void-fraction probe) and an element; keep the
    # guest definition, which is what RASPA's helium probe expects.
    elements = [e for e in elements if e not in clash]

    pa = [
        "#number of pseudo atoms",
        str(len(GUEST_PSEUDO) + len(elements)),
        "#type      print   as    chem  oxidation   mass        charge   polarization B-factor radii  connectivity anisotropic anisotropic-type   tinker-type",
    ]
    for name, prt, as_, chem, mass, q, rad in GUEST_PSEUDO:
        pa.append(f"{name:<10} {prt:<7} {as_:<5} {chem:<5} 0           {mass:<11.6f} {q:<8.4f} 0.0          1.0      {rad:<6} 0            0           relative           0")
    for e in elements:
        mass = atomic_masses[atomic_numbers[e]]
        pa.append(f"{e:<10} yes     {e:<5} {e:<5} 0           {mass:<11.6f} 0.0      0.0          1.0      1.0    0            0           relative           0")
    (out_dir / "pseudo_atoms.def").write_text("\n".join(pa) + "\n")

    mr = [
        "# general rule for shifted vs truncated",
        "shifted",
        "# general rule tailcorrections",
        "no",
        "# number of defined interactions",
        str(len(GUEST_LJ) + len(elements)),
        "# type interaction, parameters (eps/k_B [K], sigma [A]). All names are exact matches -- no '_' wildcards.",
    ]
    for name, kind, eps, sig, ref in GUEST_LJ:
        if kind == "none":
            mr.append(f"{name:<10} none                                // {ref}")
        else:
            mr.append(f"{name:<10} lennard-jones {eps:>10.4f} {sig:>8.4f}   // {ref}")
    for e in elements:
        x, d = UFF_XD[e]
        mr.append(f"{e:<10} lennard-jones {d * KCALMOL_TO_K:>10.4f} {x / SIX_ROOT_TWO:>8.4f}   // UFF")
    mr += ["# general mixing rule for Lennard-Jones", "Lorentz-Berthelot"]
    (out_dir / "force_field_mixing_rules.def").write_text("\n".join(mr) + "\n")
    print(f"Wrote {len(elements)} framework elements + {len(GUEST_PSEUDO)} guest atoms -> {out_dir}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(root / "forcefields" / FORCEFIELD_NAME))
    build(Path(ap.parse_args().out))


if __name__ == "__main__":
    main()
