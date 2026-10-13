#!/usr/bin/env python3
"""
baseline_models.py -- simple baselines with the SAME inputs as UC-TPNO run_014.

Inputs per (framework, condition): the 12 per-framework features of
data/processed/adsorption_v3/mof_features.parquet (Stage 1 single-component
GCMC + charge statistics) and the conditions (mu_CO2, mu_N2, mu_H2O, T).
Targets: CO2, N2, H2O loadings, fitted on log1p(n) and back-transformed.
Training frameworks = run_014 train split. Models:
  * ridge      linear regression (with pairwise products of the inputs)
  * gbt        histogram gradient boosting, one model per adsorbate
No thermodynamic structure is imposed, so the baselines are free to violate
Maxwell symmetry or give negative loadings.

    thesis_env/bin/python scripts/validation/baseline_models.py \
        --splits experiments/run_014/splits_with_level3.json --eval val level3
Output: results/validation/baselines/{split}_{model}_predictions.csv +
        baseline_summary.csv (alongside UC-TPNO when its predictions exist)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data/processed/adsorption_v3/adsorption_training.parquet"
VAL3 = ROOT / "data/processed/validation_gcmc/validation_humid.parquet"
FEAT = ROOT / "data/processed/adsorption_v3/mof_features.parquet"
OUT = ROOT / "results/validation/baselines"
GAS = ["co2", "n2", "h2o"]
COND = ["mu_CO2", "mu_N2", "mu_H2O", "temperature_k"]
TPNO = {"val": ROOT / "results/validation/level2/run_014_val_predictions.csv",
        "level3": ROOT / "results/validation/level3/run_014_level3_predictions.csv",
        "test": ROOT / "results/validation/level2/run_014_test_predictions.csv"}


def design(df: pd.DataFrame, feats: pd.DataFrame) -> np.ndarray:
    f = feats.loc[df.mof_id, [c for c in feats.columns if c.startswith("f_")]].to_numpy()
    c = df[COND].to_numpy().copy()
    # the trace water mu at RH 0 is an arbitrary stand-in for 'absent': give
    # the models an explicit indicator instead
    dry = (df.rh.to_numpy() == 0)[:, None].astype(float)
    c[:, 2] = np.where(dry[:, 0] > 0, c[:, 2][~dry[:, 0].astype(bool)].mean() if (~dry[:, 0].astype(bool)).any() else 0, c[:, 2])
    return np.hstack([f, c, dry])


def metrics(y, p) -> dict:
    e = p - y
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "R2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--eval", nargs="+", default=["val", "level3"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    splits = json.loads(Path(a.splits).read_text())
    feats = pd.read_parquet(FEAT).set_index("mof_id")
    data = pd.concat([pd.read_parquet(TRAIN), pd.read_parquet(VAL3)], ignore_index=True)
    tr = data[data.mof_id.isin(splits["train"])]
    Xtr = design(tr, feats)

    models = {
        "ridge": lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1.0)),
        "gbt": lambda: HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                                     max_leaf_nodes=31, random_state=0),
    }
    fitted = {}
    for name, make in models.items():
        for g in GAS:
            m = make()
            m.fit(Xtr, np.log1p(tr[f"{g}_loading_molkg"].clip(lower=0)))
            fitted[(name, g)] = m

    rows = []
    for split in a.eval:
        ev = data[data.mof_id.isin(splits[split])]
        Xev = design(ev, feats)
        for name in models:
            pred = ev[["mof_id", "temperature_k", "pressure_bar", "rh"]].copy()
            for g in GAS:
                pred[f"{g}_true"] = ev[f"{g}_loading_molkg"].to_numpy()
                pred[f"{g}_pred"] = np.expm1(fitted[(name, g)].predict(Xev))
                rows.append({"split": split, "model": name, "gas": g,
                             **metrics(pred[f"{g}_true"], pred[f"{g}_pred"]),
                             "negative_pred_pct": 100 * (pred[f"{g}_pred"] < 0).mean()})
            pred.to_csv(OUT / f"{split}_{name}_predictions.csv", index=False)
        if TPNO[split].exists():
            t = pd.read_csv(TPNO[split])
            for g in GAS:
                rows.append({"split": split, "model": "UC-TPNO (run_014)", "gas": g,
                             **metrics(t[f"{g}_true"], t[f"{g}_pred"]),
                             "negative_pred_pct": 100 * (t[f"{g}_pred"] < 0).mean()})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "baseline_summary.csv", index=False)
    print(res.pivot_table(index=["split", "model"], columns="gas", values=["MAE", "R2"]).round(3).to_string())


if __name__ == "__main__":
    main()
