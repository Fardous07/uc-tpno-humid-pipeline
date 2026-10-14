#!/usr/bin/env python3
"""
learning_curve.py -- prediction error vs number of training frameworks.

Models trained with scripts/05_train_model.py --train-fraction f (nested
subsets of the run_014 training MOFs, same validation/test MOFs and settings)
are compared with the full run_014 model. Error bars are 90 % bootstrap
intervals over frameworks (MOFs resampled with replacement).

Inputs (from predict_split.py):
    results/validation/learning_curve/f{f}_{split}_predictions.csv
    results/validation/level2/run_014_{split}_predictions.csv   (f = 1.0)

    thesis_env/bin/python scripts/validation/learning_curve.py
Output: results/validation/learning_curve/learning_curve.{csv,png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/validation/learning_curve"
N_TRAIN = 387
FRACTIONS = [0.25, 0.5, 0.75, 1.0]
SPLITS = {"val": "validation (48 MOFs)", "test": "test (48 MOFs)"}
GAS = ["co2", "n2", "h2o"]
LABEL = {"co2": "CO$_2$", "n2": "N$_2$", "h2o": "H$_2$O"}
COLOR = {"val": "#2a78d6", "test": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def path(f: float, split: str) -> Path:
    if f == 1.0:
        return ROOT / f"results/validation/level2/run_014_{split}_predictions.csv"
    return OUT / f"f{f:g}_{split}_predictions.csv"


def mae_with_ci(d: pd.DataFrame, g: str, n_boot: int = 2000, seed: int = 0):
    err = (d[f"{g}_pred"] - d[f"{g}_true"]).abs().groupby(d.mof_id)
    s, n = err.sum().to_numpy(), err.count().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boot = s[idx].sum(1) / n[idx].sum(1)
    return s.sum() / n.sum(), *np.quantile(boot, [0.05, 0.95])


def r2(d: pd.DataFrame, g: str) -> float:
    y, p = d[f"{g}_true"], d[f"{g}_pred"]
    return 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main() -> None:
    rows = []
    for f in FRACTIONS:
        n_mof = int(round(f * N_TRAIN))
        for split in SPLITS:
            d = pd.read_csv(path(f, split))
            for g in GAS:
                m, lo, hi = mae_with_ci(d, g)
                rows.append({"fraction": f, "n_train_mofs": n_mof, "split": split, "gas": g,
                             "MAE": m, "MAE_lo90": lo, "MAE_hi90": hi, "R2": r2(d, g)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "learning_curve.csv", index=False)
    print(res.pivot_table(index=["split", "n_train_mofs"], columns="gas",
                          values=["MAE", "R2"]).round(3).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8))
    for ax, g in zip(axes, GAS):
        for k, split in enumerate(SPLITS):
            x = res[(res.gas == g) & (res.split == split)].sort_values("n_train_mofs")
            xs = x.n_train_mofs + (k - 0.5) * 6        # small offset so bars don't overlap
            ax.errorbar(xs, x.MAE, yerr=[x.MAE - x.MAE_lo90, x.MAE_hi90 - x.MAE],
                        color=COLOR[split], lw=1.6, marker="o", ms=5, mfc="white", mew=1.4,
                        capsize=2, label=SPLITS[split])
        ax.set_title(LABEL[g], color=INK)
        ax.set_xlabel("training MOFs")
        ax.set_xticks([int(round(f * N_TRAIN)) for f in FRACTIONS])
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("MAE (mol kg$^{-1}$)")
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"learning_curve.{ext}")
    print(f"outputs -> {OUT}/learning_curve.*")


if __name__ == "__main__":
    main()
