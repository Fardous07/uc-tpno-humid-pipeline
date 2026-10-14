#!/usr/bin/env python3
"""
conformal_calibration.py -- split conformal calibration of the ensemble
uncertainty (Mondrian by adsorbate), as in the Methods chapter.

Calibration set: the validation frameworks (never used for fitting weights).
Score per point: s = |y - y_hat| / sigma, sigma = ensemble total uncertainty.
For each adsorbate the conformal quantile q = s_(ceil((n+1)(1-alpha))) gives
the interval y_hat +- q*sigma with marginal coverage >= 1-alpha on
exchangeable data. Coverage is then measured on other prediction files.
Because conditions of one framework are correlated, the quantile is also
reported with a framework-level (block) calibration for comparison.

    .venv/bin/python scripts/validation/conformal_calibration.py \
        --calib results/validation/level2/run_014_val_predictions.csv \
        --apply results/validation/level3/run_014_level3_predictions.csv \
        --out results/validation/uq/run_014
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GAS = ["co2", "n2", "h2o"]
LABEL = {"co2": "CO$_2$", "n2": "N$_2$", "h2o": "H$_2$O"}
COLOR = {"co2": "#2a78d6", "n2": "#eb6834", "h2o": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def conformal_q(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(scores[np.isfinite(scores)])
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float(s[min(k, len(s)) - 1])


def coverage(df: pd.DataFrame, g: str, q: float) -> float:
    return float((np.abs(df[f"{g}_true"] - df[f"{g}_pred"]) <= q * df[f"{g}_sigma"]).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--apply", nargs="*", default=[])
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cal = pd.read_csv(a.calib)
    sets = {("Level 3 (30 unseen MOFs)" if "level3" in p else "test set (48 MOFs)" if "_test" in p else Path(p).stem.replace("_predictions", "")): pd.read_csv(p) for p in a.apply}

    rows = []
    for g in GAS:
        sc = (np.abs(cal[f"{g}_true"] - cal[f"{g}_pred"]) / cal[f"{g}_sigma"]).to_numpy()
        q = conformal_q(sc, a.alpha)
        # framework-level: worst score per framework, then the same quantile rule
        per_mof = (np.abs(cal[f"{g}_true"] - cal[f"{g}_pred"]) / cal[f"{g}_sigma"]).groupby(cal.mof_id).quantile(1 - a.alpha)
        q_block = conformal_q(per_mof.to_numpy(), a.alpha)
        r = {"gas": g, "alpha": a.alpha, "q_conformal": q, "q_block": q_block,
             "raw_coverage_1.645sigma_calib": coverage(cal, g, 1.645),
             "coverage_calib": coverage(cal, g, q)}
        for name, df in sets.items():
            r[f"raw_coverage_1.645sigma_{name}"] = coverage(df, g, 1.645)
            r[f"coverage_{name}"] = coverage(df, g, q)
            r[f"coverage_block_{name}"] = coverage(df, g, q_block)
            r[f"median_halfwidth_{name}"] = float((q * df[f"{g}_sigma"]).median())
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(f"{a.out}_conformal.csv", index=False)
    print(res.round(3).T.to_string(header=False))

    # reliability diagram: empirical vs nominal coverage, raw Gaussian sigma
    levels = np.linspace(0.05, 0.99, 30)
    from scipy.stats import norm
    fig, axes = plt.subplots(1, 1 + len(sets), figsize=(3.2 * (1 + len(sets)), 3.0), squeeze=False)
    for ax, (name, df) in zip(axes[0], [("calibration (validation)", cal), *sets.items()]):
        ax.plot([0, 1], [0, 1], color=INK, lw=0.8)
        for g in GAS:
            z = np.abs(df[f"{g}_true"] - df[f"{g}_pred"]) / df[f"{g}_sigma"]
            emp = [(z <= norm.ppf(0.5 + lv / 2)).mean() for lv in levels]
            ax.plot(levels, emp, color=COLOR[g], lw=1.4, label=LABEL[g])
        ax.set_xlabel("nominal coverage"); ax.set_title(name, color=MUTED, fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    axes[0][0].set_ylabel("empirical coverage (raw ensemble $\\sigma$)")
    axes[0][0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}_reliability.{ext}")
    print(f"outputs -> {a.out}_*")


if __name__ == "__main__":
    main()
