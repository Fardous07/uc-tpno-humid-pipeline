#!/usr/bin/env python3
"""
level1_data_validation.py -- Level 1 validation of the GCMC training data.

No model is involved: these checks ask whether the simulated labels
themselves are physically sound and agree with independent published data.

  1a  Physical consistency of the Stage 3 data (completeness, signs,
      monotonicity in pressure and temperature, simulation error bars).
  1b  Stage 1 water competition vs Kancharlapalli & Snurr (2023) binding
      energies (ACS Appl. Mater. Interfaces, SI am3c04079_si_002.xlsx).
  1c  Stage 1 pure-CO2 uptake vs CRAFTED 2.0 (UFF; DDEC / PACMOF / no
      charges), interpolated from 298 and 323 K to 313.15 K.
  1d  IAST from Stage 1 pure-gas runs vs Stage 3 dry CO2/N2 mixture GCMC.

Run:   .venv/bin/python scripts/validation/level1_data_validation.py
Output (figures as PNG 300 dpi + PDF, numbers as CSV):
       results/validation/level1/
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import brentq, curve_fit  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "data/processed/adsorption_v3"
STAGE3 = V3 / "adsorption_training.parquet"
STAGE1 = V3 / "stage1_isotherms.parquet"
CRAFTED = ROOT / "data/raw/crafted/CRAFTED-2.0.0/CRAFTED-2.0.0/ISOTHERM_FILES"
KS_SI = ROOT.parent / "thesis_docs/am3c04079_si_002.xlsx"
OUT = ROOT / "results/validation/level1"

T_STAGE1 = 313.15
GAS = ["co2", "n2", "h2o"]
LABEL = {"co2": "CO$_2$", "n2": "N$_2$", "h2o": "H$_2$O"}
PLAIN = {"co2": "CO2", "n2": "N2", "h2o": "H2O"}
# One fixed colour per adsorbate in every figure (categorical slots 1-3,
# validated colour-blind safe as a set).
COLOR = {"co2": "#2a78d6", "n2": "#eb6834", "h2o": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  figure -> {OUT / name}.png/.pdf")


def refcode(mof_id: str) -> str | None:
    """CSD reference code of a CoRE MOF id, e.g. ...__HAKWUM_clean__hash -> HAKWUM."""
    parts = mof_id.split("__")
    if not mof_id.startswith("core") or len(parts) < 4:
        return None
    return re.split(r"[_-]", parts[-2])[0].upper()


def parity_axes(ax, lo: float, hi: float, factor2: bool = True) -> None:
    ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8, zorder=1)
    if factor2:
        ax.fill_between([lo, hi], [lo / 2, hi / 2], [lo * 2, hi * 2],
                        color=GRID, alpha=0.5, lw=0, zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")


# ─────────────────────────────────────────────────────────────────────
# 1a  physical consistency of Stage 3
# ─────────────────────────────────────────────────────────────────────
def check_1a(d: pd.DataFrame) -> pd.DataFrame:
    print("1a  physical consistency of Stage 3")
    L = [f"{g}_loading_molkg" for g in GAS]
    rows = [("Simulations", len(d)), ("Frameworks", d.mof_id.nunique()),
            ("Frameworks with all 45 conditions", int((d.groupby("mof_id").size() == 45).sum())),
            ("Missing loadings", int(d[L].isna().sum().sum())),
            ("Negative loadings", int((d[L] < 0).sum().sum()))]

    def frac(series) -> float:
        return round(100 * float(series.mean()), 1)

    for g in ("co2", "n2"):
        col = f"{g}_loading_molkg"
        # Uptake rises with pressure (dry, each MOF and T: highest vs lowest P)
        piv = d[d.rh == 0].pivot_table(index=["mof_id", "temperature_k"],
                                       columns="pressure_bar", values=col)
        rows.append((f"{PLAIN[g]} rises with pressure, 0.1 -> 10 bar (%)",
                     frac(piv[piv.columns.max()] >= piv[piv.columns.min()])))
        # Uptake falls with temperature (dry, 1 bar)
        piv = d[(d.rh == 0) & (d.pressure_bar == 1.0)].pivot_table(
            index="mof_id", columns="temperature_k", values=col)
        rows.append((f"{PLAIN[g]} falls with temperature, 298 -> 333 K (%)",
                     frac(piv[piv.columns.min()] >= piv[piv.columns.max()])))
    piv = d[d.pressure_bar == 1.0].pivot_table(index=["mof_id", "temperature_k"],
                                              columns="rh", values="h2o_loading_molkg")
    rows.append(("H2O rises with RH, 5 -> 15 % (%)",
                 frac(piv[piv.columns.max()] >= piv[0.05])))
    for g in GAS:
        e = f"{g}_loading_molkg_err"
        m = d[f"{g}_loading_molkg"] > 0.05
        rel = (d.loc[m, e] / d.loc[m, f"{g}_loading_molkg"]).dropna()
        rows.append((f"{PLAIN[g]} median relative error (%)", round(100 * rel.median(), 1)))
    rows.append(("Simulations with error bars (%)", frac(d["co2_loading_molkg_err"].notna())))
    t = pd.DataFrame(rows, columns=["check", "value"])
    t.to_csv(OUT / "1a_physics_checks.csv", index=False)
    print(t.to_string(index=False))

    # Figure: relative simulation errors per adsorbate
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.0), sharey=True)
    bins = np.logspace(-3, 0.5, 40)
    for ax, g in zip(axes, GAS):
        m = d[f"{g}_loading_molkg"] > 0.05
        rel = (d.loc[m, f"{g}_loading_molkg_err"] / d.loc[m, f"{g}_loading_molkg"]).dropna()
        ax.hist(rel, bins=bins, color=COLOR[g], edgecolor="white", linewidth=0.5)
        ax.axvline(rel.median(), color=INK, lw=0.8, ls="--")
        ax.text(0.04, 0.95, f"{LABEL[g]}\nmedian {100 * rel.median():.0f} %",
                transform=ax.transAxes, ha="left", va="top", color=INK)
        ax.set_xscale("log"); ax.set_xlabel(r"relative error $\delta n/n$")
    axes[0].set_ylabel("simulations")
    save(fig, "1a_error_bars")

    # Figure: humidity effect -- CO2 retained vs water taken up (313 K, 1 bar)
    k = ["mof_id", "temperature_k", "pressure_bar"]
    dry = d[d.rh == 0].set_index(k)
    wet = d[d.rh == 0.15].set_index(k)
    j = dry.join(wet, rsuffix="_w", how="inner").reset_index()
    j = j[(j.temperature_k == T_STAGE1) & (j.pressure_bar == 1.0) & (j.co2_loading_molkg > 0.01)]
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.scatter(j.h2o_loading_molkg_w.clip(lower=1e-4), j.co2_loading_molkg_w / j.co2_loading_molkg,
               s=10, color=COLOR["co2"], alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axhline(1, color=INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("H$_2$O uptake at RH 15 % (mol kg$^{-1}$)")
    ax.set_ylabel("CO$_2$ uptake, RH 15 % / dry")
    ax.set_title("313 K, 1 bar, 15:85 CO$_2$:N$_2$", color=MUTED)
    j[["mof_id", "co2_loading_molkg", "co2_loading_molkg_w", "h2o_loading_molkg_w"]].to_csv(
        OUT / "1a_humidity_effect.csv", index=False)
    save(fig, "1a_humidity_effect")
    return t


# ─────────────────────────────────────────────────────────────────────
# 1b  water competition vs Kancharlapalli & Snurr 2023
# ─────────────────────────────────────────────────────────────────────
def henry_like(s1: pd.DataFrame) -> pd.DataFrame:
    """Low-pressure slope n/p of the lowest-pressure Stage 1 point, per gas."""
    out = []
    for m, g in s1.groupby("mof_id"):
        r = {"mof_id": m, "ref": refcode(m)}
        for comp in ("CO2", "H2O"):
            c = g[g.component == comp].sort_values("pressure_bar")
            r[f"K_{comp}"] = c.loading_molkg.iloc[0] / c.pressure_bar.iloc[0] if len(c) else np.nan
        out.append(r)
    return pd.DataFrame(out)


def check_1b(s1: pd.DataFrame) -> dict:
    print("1b  water competition vs Kancharlapalli & Snurr 2023")
    if not KS_SI.exists():
        print(f"  skipped: {KS_SI} not found"); return {}
    be = pd.read_excel(KS_SI, header=1)
    be.columns = ["ref", "be_co2", "be_h2o", "be_n2", "be_diff"]
    be["ref"] = be.ref.astype(str).str.upper()
    for c in ("be_co2", "be_h2o", "be_diff"):
        be[c] = pd.to_numeric(be[c], errors="coerce")
    k = henry_like(s1).dropna(subset=["ref"])
    j = k.merge(be.dropna(), on="ref")
    j = j[(j.K_CO2 > 0) & (j.K_H2O > 0)].drop_duplicates("mof_id").copy()
    j["ln_ratio"] = np.log(j.K_H2O / j.K_CO2)
    rho = spearmanr(j.ln_ratio, j.be_diff).statistic
    ours_co2 = j.K_CO2 > j.K_H2O
    theirs_co2 = j.be_co2 < j.be_h2o
    agree = float((ours_co2 == theirs_co2).mean())
    j.to_csv(OUT / "1b_ks2023_comparison.csv", index=False)
    print(f"  {len(j)} shared frameworks, Spearman {rho:+.2f}, "
          f"same CO2-vs-water preference for {100 * agree:.0f} %")

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.scatter(j.be_diff, j.ln_ratio, s=12, color=COLOR["h2o"], alpha=0.8,
               edgecolor="white", linewidth=0.3)
    ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel(r"BE(H$_2$O) $-$ BE(CO$_2$) (kJ mol$^{-1}$), K&S 2023")
    ax.set_ylabel(r"$\ln(K_{\mathrm{H_2O}}/K_{\mathrm{CO_2}})$, this work")
    ax.text(0.04, 0.04, f"n = {len(j)}\nSpearman $\\rho$ = {rho:.2f}\nagreement {100 * agree:.0f} %",
            transform=ax.transAxes, ha="left", va="bottom", color=INK)
    save(fig, "1b_ks2023_water_competition")
    return {"n": len(j), "spearman": rho, "agreement": agree}


# ─────────────────────────────────────────────────────────────────────
# 1c  pure CO2 vs CRAFTED 2.0
# ─────────────────────────────────────────────────────────────────────
def crafted_isotherm(charges: str, name: str, T: int):
    f = CRAFTED / f"{charges}_{name}_UFF_CO2_{T}.csv"
    if not f.exists():
        return None
    a = np.loadtxt(f, delimiter=",")
    return a[:, 0] / 1e5, np.clip(a[:, 1], 1e-6, None)


def check_1c(s1: pd.DataFrame) -> pd.DataFrame:
    print("1c  pure CO2 vs CRAFTED 2.0 (UFF)")
    if not CRAFTED.exists():
        print(f"  skipped: {CRAFTED} not found"); return pd.DataFrame()
    names = {}
    for f in os.listdir(CRAFTED):
        if f.startswith("DDEC_") and f.endswith("_UFF_CO2_298.csv"):
            n = f[len("DDEC_"):-len("_UFF_CO2_298.csv")]
            names.setdefault(re.split(r"[_-]", n)[0].upper(), n)
    # ln n interpolated linearly in 1/T between 298 and 323 K
    w = (1 / T_STAGE1 - 1 / 298) / (1 / 323 - 1 / 298)
    rows = []
    co2 = s1[s1.component == "CO2"]
    for m, g in co2.groupby("mof_id"):
        r = refcode(m)
        if r not in names:
            continue
        for q in ("PACMOF", "DDEC", "NEUTRAL"):
            a, b = crafted_isotherm(q, names[r], 298), crafted_isotherm(q, names[r], 323)
            if a is None or b is None:
                continue
            for _, x in g.iterrows():
                p = x.pressure_bar
                n1 = np.exp(np.interp(np.log(p), np.log(a[0]), np.log(a[1])))
                n2 = np.exp(np.interp(np.log(p), np.log(b[0]), np.log(b[1])))
                rows.append({"mof_id": m, "ref": r, "charges": q, "pressure_bar": p,
                             "this_work": x.loading_molkg,
                             "crafted": float(np.exp((1 - w) * np.log(n1) + w * np.log(n2)))})
    c = pd.DataFrame(rows)
    c.to_csv(OUT / "1c_crafted_comparison.csv", index=False)
    summ = []
    for (q, p), h in c.groupby(["charges", "pressure_bar"]):
        r = h.this_work / h.crafted
        summ.append({"charges": q, "pressure_bar": p, "n": len(h),
                     "spearman": spearmanr(h.this_work, h.crafted).statistic,
                     "median_ratio": r.median(), "within_factor_2_pct": 100 * ((r > 0.5) & (r < 2)).mean()})
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "1c_crafted_summary.csv", index=False)
    print(summ.round(2).to_string(index=False))

    title = {"PACMOF": "PACMOF charges", "DDEC": "DDEC charges", "NEUTRAL": "no charges"}
    marker = {0.1: "o", 1.0: "s", 10.0: "^"}
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), sharey=True)
    for ax, q in zip(axes, ("PACMOF", "DDEC", "NEUTRAL")):
        h = c[c.charges == q]
        parity_axes(ax, 1e-2, 2e1)
        for p, mk in marker.items():
            s = h[h.pressure_bar == p]
            ax.scatter(s.crafted, s.this_work, s=12, marker=mk, color=COLOR["co2"],
                       alpha=0.75, edgecolor="white", linewidth=0.3, label=f"{p:g} bar")
        rho = spearmanr(h.this_work, h.crafted).statistic
        ax.set_title(title[q], color=MUTED)
        ax.text(0.05, 0.95, f"$\\rho$ = {rho:.2f}", transform=ax.transAxes, va="top")
        ax.set_xlabel("CRAFTED CO$_2$ (mol kg$^{-1}$)")
    axes[0].set_ylabel("this work CO$_2$ (mol kg$^{-1}$)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1))
    save(fig, "1c_crafted_parity")
    return summ


# ─────────────────────────────────────────────────────────────────────
# 1d  IAST (Stage 1 pure gas) vs Stage 3 dry binary GCMC
# ─────────────────────────────────────────────────────────────────────
def langmuir(p, qs, b):
    return qs * b * p / (1 + b * p)


def fit_langmuir(p: np.ndarray, n: np.ndarray):
    """Single-site Langmuir through the three Stage 1 points (relative errors)."""
    if np.any(n <= 0):
        return None
    try:
        (qs, b), _ = curve_fit(langmuir, p, n, p0=[max(n.max() * 1.5, 1e-3), 0.5],
                               sigma=n, bounds=([1e-6, 1e-8], [1e3, 1e4]), maxfev=20000)
    except RuntimeError:
        return None
    return qs, b


def iast_binary(P: float, y1: float, f1, f2):
    """IAST for two Langmuir components; returns (n1, n2) in mol/kg."""
    (qs1, b1), (qs2, b2) = f1, f2
    y2 = 1 - y1

    def g(x1):  # equal reduced spreading pressures
        p1, p2 = P * y1 / x1, P * y2 / (1 - x1)
        return qs1 * np.log1p(b1 * p1) - qs2 * np.log1p(b2 * p2)

    x1 = brentq(g, 1e-12, 1 - 1e-12)
    n1o = langmuir(P * y1 / x1, qs1, b1)
    n2o = langmuir(P * y2 / (1 - x1), qs2, b2)
    nt = 1 / (x1 / n1o + (1 - x1) / n2o)
    return x1 * nt, (1 - x1) * nt


def check_1d(s1: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    print("1d  IAST (Stage 1) vs Stage 3 dry mixture")
    dry = d[(d.rh == 0) & ((d.temperature_k - T_STAGE1).abs() < 0.01)]
    rows = []
    s1 = s1[s1.mof_id.isin(dry.mof_id.unique())]
    for m, g in s1.groupby("mof_id"):
        fits = {}
        for comp in ("CO2", "N2"):
            c = g[g.component == comp].sort_values("pressure_bar")
            fits[comp] = fit_langmuir(c.pressure_bar.to_numpy(), c.loading_molkg.to_numpy())
        if None in fits.values():
            continue
        for _, x in dry[dry.mof_id == m].iterrows():
            try:
                n1, n2 = iast_binary(x.pressure_bar, x.y_co2_dry, fits["CO2"], fits["N2"])
            except ValueError:
                continue
            rows.append({"mof_id": m, "pressure_bar": x.pressure_bar,
                         "co2_gcmc": x.co2_loading_molkg, "co2_iast": n1,
                         "n2_gcmc": x.n2_loading_molkg, "n2_iast": n2})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "1d_iast_vs_gcmc.csv", index=False)
    summ = []
    for g in ("co2", "n2"):
        a, b = r[f"{g}_gcmc"], r[f"{g}_iast"]
        ok = (a > 1e-3) & (b > 1e-3)
        ratio = b[ok] / a[ok]
        summ.append({"gas": g, "n": int(ok.sum()), "mof": r.mof_id.nunique(),
                     "spearman": spearmanr(a, b).statistic, "median_ratio_iast_gcmc": ratio.median(),
                     "within_factor_2_pct": 100 * ((ratio > 0.5) & (ratio < 2)).mean(),
                     "MAE_molkg": (a - b).abs().mean()})
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "1d_iast_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.5))
    for ax, g, lim in zip(axes, ("co2", "n2"), ((1e-3, 2e1), (1e-3, 5e0))):
        parity_axes(ax, *lim)
        ax.scatter(r[f"{g}_gcmc"].clip(lower=lim[0]), r[f"{g}_iast"].clip(lower=lim[0]), s=6,
                   color=COLOR[g], alpha=0.5, edgecolor="none")
        rho = spearmanr(r[f"{g}_gcmc"], r[f"{g}_iast"]).statistic
        ax.text(0.05, 0.95, f"{LABEL[g]}\n$\\rho$ = {rho:.2f}", transform=ax.transAxes, va="top")
        ax.set_xlabel(f"binary GCMC {LABEL[g]} (mol kg$^{{-1}}$)")
        ax.set_ylabel(f"IAST {LABEL[g]} (mol kg$^{{-1}}$)")
    fig.tight_layout()  # conditions (313 K, 15:85, 0.1-10 bar) go in the caption
    save(fig, "1d_iast_parity")
    return summ


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=["1a", "1b", "1c", "1d"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(STAGE3)
    s1 = pd.read_parquet(STAGE1)
    s1 = s1[(s1.temperature_k - T_STAGE1).abs() < 0.01]
    todo = a.only or ["1a", "1b", "1c", "1d"]
    if "1a" in todo:
        check_1a(d)
    if "1b" in todo:
        check_1b(s1)
    if "1c" in todo:
        check_1c(s1)
    if "1d" in todo:
        check_1d(s1, d)
    print(f"\nAll Level 1 outputs in {OUT}")


if __name__ == "__main__":
    main()
