#!/usr/bin/env python3
"""
build_mof_features.py -- per-framework input features for the encoder.

The crystal graph plus 12 composition descriptors let the model fit water on
its training frameworks but not transfer water affinity to new ones
(validation water Spearman 0.24).
Two physically targeted feature groups are added:

  Stage 1 single-component GCMC (313.15 K; already simulated for every
  admissible framework, cost of minutes per framework):
      log1p n_CO2 at 0.1, 1, 10 bar;  log1p n_N2 at 0.1, 1, 10 bar;
      log1p n_H2O at RH 5 % and 15 %                                (8)
  Framework partial charges (from the graph, i.e. the charges used in GCMC):
      mean |q|, std q, max |q|, fraction of atoms with |q| > 0.5 e   (4)

Each column is standardised over all frameworks in the table (input
statistics only; no labels are used). Frameworks without Stage 1 results
get NaN and are reported.

    .venv/bin/python scripts/build_mof_features.py
Output: data/processed/adsorption_v3/mof_features.parquet
        (mof_id + 12 standardised columns f_*, raw values in r_*)

New frameworks (e.g. the Level 4 reference MOFs) are added without changing
the training inputs: --reference standardises them with the mean/std of the
reference table and writes reference + new rows to --out.

    .venv/bin/python scripts/build_mof_features.py \
        --stage1 data/processed/validation_gcmc/level4_stage1/*/stage1_isotherms.parquet \
        --graph-dir data/processed/graphs_level4 \
        --reference data/processed/adsorption_v3/mof_features.parquet \
        --out data/processed/validation_gcmc/level4_mof_features.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "data/processed/adsorption_v3/stage1_isotherms.parquet"
GRAPHS = ROOT / "data/processed/graphs"
OUT = ROOT / "data/processed/adsorption_v3/mof_features.parquet"

FEATURES = ["co2_p0.1", "co2_p1", "co2_p10", "n2_p0.1", "n2_p1", "n2_p10",
            "h2o_rh5", "h2o_rh15", "q_mean_abs", "q_std", "q_max_abs", "q_frac_gt0.5"]


def stage1_features(path: Path) -> pd.DataFrame:
    s = pd.read_parquet(path)
    s = s[(s.temperature_k - 313.15).abs() < 0.01]
    rows = {}
    for (m, comp), g in s.groupby(["mof_id", "component"]):
        g = g.sort_values("pressure_bar")
        v = np.log1p(g.loading_molkg.clip(lower=0).to_numpy())
        r = rows.setdefault(m, {})
        if comp == "CO2" and len(v) == 3:
            r.update({"co2_p0.1": v[0], "co2_p1": v[1], "co2_p10": v[2]})
        elif comp == "N2" and len(v) == 3:
            r.update({"n2_p0.1": v[0], "n2_p1": v[1], "n2_p10": v[2]})
        elif comp == "H2O" and len(v) == 2:  # RH 5 % and 15 % (sorted by pressure)
            r.update({"h2o_rh5": v[0], "h2o_rh15": v[1]})
    return pd.DataFrame.from_dict(rows, orient="index")


def charge_features(graph_dir: Path, mofs) -> pd.DataFrame:
    rows = {}
    for m in mofs:
        p = graph_dir / f"{m}.pt"
        if not p.exists():
            continue
        q = torch.load(p, weights_only=False)["q"].float().flatten()
        a = q.abs()
        rows[m] = {"q_mean_abs": a.mean().item(), "q_std": q.std().item(),
                   "q_max_abs": a.max().item(), "q_frac_gt0.5": (a > 0.5).float().mean().item()}
    return pd.DataFrame.from_dict(rows, orient="index")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", type=Path, default=[STAGE1], nargs="+",
                    help="one or more Stage 1 parquet files (e.g. extra validation runs)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--graph-dir", type=Path, default=GRAPHS)
    ap.add_argument("--reference", type=Path, default=None,
                    help="existing feature table whose statistics standardise the new rows")
    a = ap.parse_args()

    s1 = pd.concat([stage1_features(p) for p in a.stage1])
    s1 = s1[~s1.index.duplicated(keep="first")]
    ch = charge_features(a.graph_dir, s1.index)
    raw = s1.join(ch, how="left")[FEATURES]
    if a.reference is not None:
        ref = pd.read_parquet(a.reference).set_index("mof_id")
        ref_raw = ref[[f"r_{c}" for c in FEATURES]].set_axis(FEATURES, axis=1)
        raw = raw[~raw.index.isin(ref.index)]
        std = (raw - ref_raw.mean()) / ref_raw.std().replace(0, 1)
    else:
        std = (raw - raw.mean()) / raw.std().replace(0, 1)
    out = pd.concat([std.add_prefix("f_"), raw.add_prefix("r_")], axis=1)
    if a.reference is not None:
        out = pd.concat([ref, out])
    out.index.name = "mof_id"
    out.reset_index().to_parquet(a.out, index=False)
    missing = raw.isna().any(axis=1)
    print(f"{len(out)} frameworks, {len(FEATURES)} features -> {a.out}")
    print(f"incomplete rows: {int(missing.sum())}")
    print(raw.describe().T[["mean", "std", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
