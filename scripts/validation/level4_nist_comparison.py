#!/usr/bin/env python3
"""
level4_nist_comparison.py -- Level 4: GCMC (this work) vs experiment.

Pure-component isotherms of HKUST-1, Mg-MOF-74, ZIF-8 and UiO-66 simulated at
298.15 K with exactly the training-data settings (validation_gcmc.sh) are
compared with every experimental isotherm in the NIST/ARPA-E Adsorption
Database (ISODB) measured at 298 +- 3 K. Loadings are converted to mol/kg
(mmol/g, mol/kg, mmol/kg, cm3(STP)/g, mg/g, wt%); other units are skipped.

    thesis_env/bin/python scripts/validation/level4_nist_comparison.py
Output: results/validation/level4/  (figure + CSVs)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ISODB = ROOT / "data/raw/nist_isodb/isodb-library-master/isodb-library-master/Library"
PURE = ROOT / "data/processed/validation_gcmc/level4_pure"
OUT = ROOT / "results/validation/level4"

PATTERNS = {"HKUST-1": r"hkust-?1|cu-?btc|cu3\(btc\)2|basolite c ?300|mof-199",
            "Mg-MOF-74": r"mg-?mof-?74|cpo-27-mg|mg2\(dobdc\)|mg-?dobdc",
            "ZIF-8": r"zif-?8\b|basolite z ?1200",
            "UiO-66": r"uio-?66$|uio-?66\b(?!-)"}
GASES = {"Carbon Dioxide": "CO2", "Nitrogen": "N2", "Water": "H2O"}
MOLAR_MASS = {"CO2": 44.01, "N2": 28.013, "H2O": 18.015}
COLOR = {"CO2": "#2a78d6", "N2": "#eb6834", "H2O": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def to_molkg(v: float, unit: str, gas: str) -> float | None:
    u = unit.strip().lower()
    if u in ("mmol/g", "mol/kg"):
        return v
    if u == "mmol/kg":
        return v / 1000.0
    if u == "cm3(stp)/g":
        return v / 22.414
    if u == "mg/g":
        return v / MOLAR_MASS[gas]
    if u == "wt%":
        return v * 10.0 / MOLAR_MASS[gas]
    return None


def experiments() -> pd.DataFrame:
    rows = []
    for f in ISODB.glob("*/*.json"):
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        name = j.get("adsorbent", {}).get("name", "").lower()
        mof = next((k for k, p in PATTERNS.items() if re.search(p, name)), None)
        ads = [a["name"] for a in j.get("adsorbates", [])]
        if mof is None or j.get("category") != "exp" or len(ads) != 1 or ads[0] not in GASES:
            continue
        T = j.get("temperature")
        if T is None or abs(T - 298) > 3 or j.get("pressureUnits", "").lower() != "bar":
            continue
        gas = GASES[ads[0]]
        # Drop entries with implausible units: CO2/N2 above 50 bar (Pa stored
        # as bar) and water above its saturation pressure at 298 K (0.0317
        # bar; such points are relative pressures p/p0 labelled as bar).
        p_max = 0.0317 if gas == "H2O" else 50.0
        for pt in j.get("isotherm_data", []):
            if pt.get("pressure", 0) > p_max:
                continue
            n = pt.get("total_adsorption")
            if n is None and pt.get("species_data"):
                n = pt["species_data"][0].get("adsorption")
            n = to_molkg(n, j.get("adsorptionUnits", ""), gas) if n is not None else None
            if n is not None and pt.get("pressure", 0) > 0:
                rows.append({"mof": mof, "gas": gas, "isotherm": j["filename"], "doi": j.get("DOI"),
                             "T": T, "pressure_bar": pt["pressure"], "loading_molkg": n})
    return pd.DataFrame(rows)


def simulations() -> pd.DataFrame:
    out = []
    for d in PURE.iterdir():
        p = d / "stage1_isotherms.parquet"
        if p.exists():
            s = pd.read_parquet(p)
            out.append(s.assign(mof=d.name, gas=s.component))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    exp, sim = experiments(), simulations()
    exp.to_csv(OUT / "nist_experimental_points.csv", index=False)
    sim[["mof", "gas", "temperature_k", "pressure_bar", "loading_molkg"]].to_csv(
        OUT / "gcmc_pure_298K.csv", index=False)

    # Ratio GCMC / experiment at the simulated pressures (median over all
    # experimental isotherms, log-interpolated within each isotherm's range)
    rows = []
    for (mof, gas), s in sim.groupby(["mof", "gas"]):
        e = exp[(exp.mof == mof) & (exp.gas == gas)]
        for _, x in s.iterrows():
            vals = []
            for _, iso in e.groupby("isotherm"):
                iso = iso.sort_values("pressure_bar")
                p, n = iso.pressure_bar.to_numpy(), iso.loading_molkg.to_numpy()
                if p.min() <= x.pressure_bar <= p.max() and np.all(n > 0):
                    vals.append(np.exp(np.interp(np.log(x.pressure_bar), np.log(p), np.log(n))))
            if vals:
                rows.append({"mof": mof, "gas": gas, "pressure_bar": x.pressure_bar,
                             "gcmc": x.loading_molkg, "exp_median": np.median(vals),
                             "exp_min": np.min(vals), "exp_max": np.max(vals), "n_isotherms": len(vals)})
    cmp_ = pd.DataFrame(rows)
    cmp_["ratio"] = cmp_.gcmc / cmp_.exp_median
    cmp_["inside_exp_range"] = (cmp_.gcmc >= cmp_.exp_min) & (cmp_.gcmc <= cmp_.exp_max)
    cmp_.to_csv(OUT / "gcmc_vs_experiment.csv", index=False)
    print(cmp_.round(3).to_string(index=False))

    mofs = ["HKUST-1", "Mg-MOF-74", "ZIF-8", "UiO-66"]
    gases = ["CO2", "N2", "H2O"]
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.6))
    for i, gas in enumerate(gases):
        for j, mof in enumerate(mofs):
            ax = axes[i, j]
            e = exp[(exp.mof == mof) & (exp.gas == gas)]
            for k, (_, iso) in enumerate(e.groupby("isotherm")):
                iso = iso.sort_values("pressure_bar")
                ax.plot(iso.pressure_bar, iso.loading_molkg, color=MUTED, lw=0.6, alpha=0.5,
                        marker=".", ms=2, label="experiment (NIST ISODB)" if k == 0 else None)
            s = sim[(sim.mof == mof) & (sim.gas == gas)].sort_values("pressure_bar")
            if len(s):
                ax.plot(s.pressure_bar, s.loading_molkg, color=COLOR[gas], lw=1.6, marker="o", ms=3,
                        label="GCMC, this work")
            ax.set_xscale("log")
            ax.set_xlim(*((1e-4, 0.035) if gas == "H2O" else (1e-3, 50)))
            if i == 0:
                ax.set_title(mof, color=INK)
            if j == 0:
                ax.set_ylabel(f"{gas} (mol kg$^{{-1}}$)")
            if i == 2:
                ax.set_xlabel("pressure (bar)")
            n_exp = e.isotherm.nunique()
            ax.text(0.03, 0.95, f"{n_exp} exp.", transform=ax.transAxes, va="top", color=MUTED, fontsize=7)
    h, l = [], []
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Pure-gas isotherms at 298 K: GCMC (training-data settings) vs experiment",
                 color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"level4_gcmc_vs_experiment.{ext}")
    print(f"\nfigure -> {OUT / 'level4_gcmc_vs_experiment.png'}")


if __name__ == "__main__":
    main()
