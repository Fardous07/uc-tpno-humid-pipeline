#!/usr/bin/env python3
"""
confusion_matrix.py -- framework-level "good MOF" classification.

Definition (results/validation/level2/good_mof_definition.json, fixed before
the test set was evaluated): at 313.15 K, 1 bar, 15:85 CO2:N2 and RH 15 %,
a framework is good if humid CO2 uptake >= co2_humid_molkg_min AND
CO2(RH 15 %)/CO2(dry) >= co2_retention_humid_over_dry_min. The class is
computed once from GCMC (truth) and once from the predictions.

    thesis_env/bin/python scripts/validation/confusion_matrix.py \
        --pred results/validation/level2/run_014_test_predictions.csv \
        --label "test set (48 MOFs)" --out results/validation/level2/run_014_test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEF = ROOT / "results/validation/level2/good_mof_definition.json"
INK, MUTED = "#0b0b0b", "#52514e"


def framework_table(d: pd.DataFrame) -> pd.DataFrame:
    """CO2 at 313 K and 1 bar, dry and RH 15 %, per framework (true and predicted)."""
    d = d[(d["T"] - 313.15).abs() < 0.01].copy()
    lev = d.groupby("mof_id").mu_h2o.rank(method="dense").astype(int)
    d["rh_level"] = np.select([lev <= 5, lev <= 10], [0, 1], 2)        # dry, 5 %, 15 %
    # pressure index within (MOF, RH level): 0.1, 0.5, 1, 5, 10 bar -> rank 3 = 1 bar
    d["p_rank"] = d.groupby(["mof_id", "rh_level"]).mu_co2.rank(method="dense").astype(int)
    one = d[d.p_rank == 3]
    wet, dry = one[one.rh_level == 2].set_index("mof_id"), one[one.rh_level == 0].set_index("mof_id")
    return pd.DataFrame({"co2_wet_true": wet.co2_true, "co2_dry_true": dry.co2_true,
                         "co2_wet_pred": wet.co2_pred, "co2_dry_pred": dry.co2_pred})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rule = json.loads(DEF.read_text())["good_if"]
    t = framework_table(pd.read_csv(a.pred))
    for s in ("true", "pred"):
        t[f"retention_{s}"] = t[f"co2_wet_{s}"] / t[f"co2_dry_{s}"]
        t[f"good_{s}"] = (t[f"co2_wet_{s}"] >= rule["co2_humid_molkg_min"]) & \
                         (t[f"retention_{s}"] >= rule["co2_retention_humid_over_dry_min"])
    t.to_csv(f"{a.out}_classification.csv")
    tp = int((t.good_true & t.good_pred).sum()); tn = int((~t.good_true & ~t.good_pred).sum())
    fp = int((~t.good_true & t.good_pred).sum()); fn = int((t.good_true & ~t.good_pred).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / den if den else np.nan
    res = {"label": a.label, "n": len(t), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
           "accuracy": (tp + tn) / len(t), "precision": prec, "recall": rec,
           "F1": 2 * prec * rec / (prec + rec) if prec + rec else np.nan, "MCC": mcc}
    pd.DataFrame([res]).to_csv(f"{a.out}_confusion.csv", index=False)
    print(pd.Series(res).to_string())

    fig, ax = plt.subplots(figsize=(3.0, 2.7))
    M = np.array([[tp, fn], [fp, tn]])
    ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max() * 1.6)
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, str(v), ha="center", va="center", color=INK, fontsize=12)
    ax.set_xticks([0, 1], ["good", "not good"]); ax.set_yticks([0, 1], ["good", "not good"])
    ax.set_xlabel("model"); ax.set_ylabel("GCMC")
    ax.grid(False)
    ax.set_title(f"{a.label}\nMCC {mcc:.2f}, precision {prec:.2f}, recall {rec:.2f}",
                 color=MUTED, fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}_confusion.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
