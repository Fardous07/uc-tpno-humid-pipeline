#!/usr/bin/env python3
"""
evaluate_predictions.py -- metrics and figures for a predictions CSV
(from scripts/validation/predict_split.py), used for Level 2 (held-out
validation/test frameworks) and Level 3 (unseen frameworks, fresh GCMC).

    thesis_env/bin/python scripts/validation/evaluate_predictions.py \
        --pred results/validation/level2/run_014_val_predictions.csv \
        --label "validation (48 MOFs)" --out results/validation/level2/run_014_val \
        [--metrics experiments/run_014/metrics_part1.csv experiments/run_014/metrics.csv]

Outputs in --out (prefix): _metrics.csv, _by_rh.csv, _parity.png/.pdf,
_error_by_rh.png/.pdf, _uncertainty.csv and, with --metrics,
_training_curve.png/.pdf.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

GAS = ["co2", "n2", "h2o"]
LABEL = {"co2": "CO$_2$", "n2": "N$_2$", "h2o": "H$_2$O"}
COLOR = {"co2": "#2a78d6", "n2": "#eb6834", "h2o": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
RH_NAME = {1: "0 %", 2: "5 %", 3: "15 %"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, out: str, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}_{name}.{ext}")
    plt.close(fig)


def metrics(y, p) -> dict:
    e = p - y
    ss = ((y - y.mean()) ** 2).sum()
    return {"n": len(y), "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "R2": 1 - (e ** 2).sum() / ss if ss > 0 else np.nan,
            "Spearman": spearmanr(y, p).statistic}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True, help="output path prefix")
    ap.add_argument("--metrics", nargs="*", help="training metrics.csv file(s), in order")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.pred)
    # humidity level from the water chemical potential within each (MOF, T):
    # three RH levels x five pressures -> ranks 1-5 dry, 6-10 RH 5 %, 11-15 RH 15 %
    lev = d.groupby(["mof_id", "T"]).mu_h2o.rank(method="dense").astype(int)
    d["rh_level"] = np.select([lev <= 5, lev <= 10], [1, 2], 3)

    rows = [{"gas": g, **metrics(d[f"{g}_true"], d[f"{g}_pred"])} for g in GAS]
    allt = np.concatenate([d[f"{g}_true"] for g in GAS]); allp = np.concatenate([d[f"{g}_pred"] for g in GAS])
    rows.append({"gas": "all", **metrics(allt, allp)})
    m = pd.DataFrame(rows)
    m.to_csv(f"{a.out}_metrics.csv", index=False)
    print(f"{a.label}: {d.mof_id.nunique()} MOFs, {len(d)} conditions")
    print(m.round(3).to_string(index=False))

    byrh = pd.DataFrame([{"rh": RH_NAME[r], "gas": g, **metrics(x[f"{g}_true"], x[f"{g}_pred"])}
                         for r, x in d.groupby("rh_level") for g in GAS])
    byrh.to_csv(f"{a.out}_by_rh.csv", index=False)

    # uncertainty: fraction of points inside the raw ensemble +-1.645 sigma (nominal 90 %)
    unc = []
    for g in GAS:
        s = d[f"{g}_sigma"]
        if s.notna().any():
            inside = (np.abs(d[f"{g}_true"] - d[f"{g}_pred"]) <= 1.645 * s).mean()
            unc.append({"gas": g, "nominal": 0.90, "empirical_coverage": inside,
                        "median_sigma": s.median()})
    if unc:
        pd.DataFrame(unc).to_csv(f"{a.out}_uncertainty.csv", index=False)
        print(pd.DataFrame(unc).round(3).to_string(index=False))

    # parity
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for ax, g in zip(axes, GAS):
        y, p = d[f"{g}_true"], d[f"{g}_pred"]
        hi = max(y.max(), p.max()) * 1.05
        ax.plot([0, hi], [0, hi], color=INK, lw=0.8, zorder=1)
        ax.scatter(y, p, s=5, color=COLOR[g], alpha=0.45, edgecolor="none", zorder=2)
        r = m[m.gas == g].iloc[0]
        ax.text(0.04, 0.96, f"{LABEL[g]}\nMAE {r.MAE:.3f}\nR$^2$ {r.R2:.2f}",
                transform=ax.transAxes, va="top", color=INK)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
        ax.set_xlabel("GCMC (mol kg$^{-1}$)")
    axes[0].set_ylabel("model (mol kg$^{-1}$)")
    fig.suptitle(a.label, color=MUTED, fontsize=9)
    fig.tight_layout()
    save(fig, a.out, "parity")

    # error by humidity
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(3); w = 0.26
    for k, g in enumerate(GAS):
        vals = [byrh[(byrh.rh == RH_NAME[r]) & (byrh.gas == g)].MAE.iloc[0] for r in (1, 2, 3)]
        ax.bar(x + (k - 1) * w, vals, w * 0.92, color=COLOR[g], label=LABEL[g])
    ax.set_xticks(x, [f"RH {RH_NAME[r]}" for r in (1, 2, 3)])
    ax.set_ylabel("MAE (mol kg$^{-1}$)")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_title(a.label, color=MUTED, fontsize=9)
    save(fig, a.out, "error_by_rh")

    # training curve
    if a.metrics:
        mc = pd.concat([pd.read_csv(f) for f in a.metrics]).drop_duplicates("epoch", keep="last").sort_values("epoch")
        fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
        ax[0].plot(mc.epoch + 1, mc["train/data_loss"], color=INK, lw=1, label="training (data term)")
        ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")
        ax[0].legend(frameon=False)
        for g, c in zip(["CO2", "N2", "H2O"], GAS):
            ax[1].plot(mc.epoch + 1, mc[f"val/mae_{g}"], color=COLOR[c], lw=1, label=LABEL[c])
        ax[1].set_yscale("log"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation MAE (mol kg$^{-1}$)")
        ax[1].legend(frameon=False)
        fig.tight_layout()
        save(fig, a.out, "training_curve")
    print(f"outputs -> {a.out}_*")


if __name__ == "__main__":
    main()
