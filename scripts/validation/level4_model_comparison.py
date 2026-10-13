#!/usr/bin/env python3
"""
level4_model_comparison.py -- UC-TPNO (run_014) on the four reference MOFs
(HKUST-1, Mg-MOF-74, ZIF-8, UiO-66) against their humid-mixture GCMC.

Together with level4_nist_comparison.py (GCMC vs experiment) this closes the
chain model -> GCMC -> experiment: the model is judged against the GCMC it
was trained to reproduce, and the GCMC against measured isotherms.

Inputs come from predict_split.py with --graph-dir data/processed/graphs_level4
and --features data/processed/validation_gcmc/level4_mof_features.parquet.

    thesis_env/bin/python scripts/validation/level4_model_comparison.py
Output: results/validation/level4/level4_model_vs_gcmc.{csv,png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/validation/level4"
GAS = ["co2", "n2", "h2o"]
LABEL = {"co2": "CO$_2$", "n2": "N$_2$", "h2o": "H$_2$O"}
COLOR = {"co2": "#2a78d6", "n2": "#eb6834", "h2o": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def main() -> None:
    d = pd.read_csv(OUT / "run_014_level4_predictions.csv")
    names = pd.read_csv(ROOT / "data/processed/validation_gcmc/level4_mofs.csv").set_index("mof_id").name
    d["name"] = d.mof_id.map(names)
    # RH level and pressure index from the chemical potentials (as in confusion_matrix.py)
    lev = d.groupby(["mof_id", "T"]).mu_h2o.rank(method="dense").astype(int)
    d["rh"] = np.select([lev <= 5, lev <= 10], [0.0, 0.05], 0.15)
    d["p_rank"] = d.groupby(["mof_id", "T", "rh"]).mu_co2.rank(method="dense").astype(int)
    d["pressure_bar"] = d.p_rank.map({1: 0.1, 2: 0.5, 3: 1.0, 4: 5.0, 5: 10.0})

    rows = []
    for (n, g), x in [((n, g), d[d.name == n]) for n in sorted(d.name.unique()) for g in GAS]:
        e = x[f"{g}_pred"] - x[f"{g}_true"]
        rows.append({"mof": n, "gas": g, "MAE": np.abs(e).mean(),
                     "mean_true": x[f"{g}_true"].mean(), "mean_pred": x[f"{g}_pred"].mean(),
                     "coverage_1.645sigma": (np.abs(e) <= 1.645 * x[f"{g}_sigma"]).mean()})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "level4_model_vs_gcmc.csv", index=False)
    print(res.round(3).to_string(index=False))

    # isotherms at 313 K: dry and RH 15 %, GCMC points vs model lines
    t = d[(d["T"] - 313.15).abs() < 0.01]
    mofs = sorted(t.name.unique())
    fig, axes = plt.subplots(2, len(mofs), figsize=(2.6 * len(mofs), 4.8), sharex=True)
    for j, n in enumerate(mofs):
        for i, rh in enumerate([0.0, 0.15]):
            ax = axes[i, j]
            x = t[(t.name == n) & (t.rh == rh)].sort_values("pressure_bar")
            for g in GAS if rh else ["co2", "n2"]:
                ax.plot(x.pressure_bar, x[f"{g}_pred"], color=COLOR[g], lw=1.4, label=f"{LABEL[g]} model")
                ax.fill_between(x.pressure_bar, x[f"{g}_pred"] - 1.645 * x[f"{g}_sigma"],
                                x[f"{g}_pred"] + 1.645 * x[f"{g}_sigma"], color=COLOR[g], alpha=0.15, lw=0)
                ax.plot(x.pressure_bar, x[f"{g}_true"], "o", ms=4, mfc="white", mec=COLOR[g], mew=1.2)
            ax.set_xscale("log")
            if i == 0:
                ax.set_title(n, color=INK, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{'dry' if rh == 0 else 'RH 15 %'}\nloading (mol kg$^{{-1}}$)")
            if i == 1:
                ax.set_xlabel("total pressure (bar)")
    h = [plt.Line2D([], [], color=COLOR[g], lw=1.4) for g in GAS] + \
        [plt.Line2D([], [], ls="", marker="o", mfc="white", mec=MUTED)]
    fig.legend(h, [LABEL[g] for g in GAS] + ["GCMC"], loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("313 K, CO$_2$:N$_2$ 15:85 -- lines: UC-TPNO (band 90 %), circles: GCMC",
                 color=MUTED, fontsize=8, y=-0.01)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"level4_model_vs_gcmc.{ext}")
    print(f"outputs -> {OUT}/level4_model_vs_gcmc.*")


if __name__ == "__main__":
    main()
