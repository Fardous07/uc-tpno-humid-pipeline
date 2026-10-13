#!/usr/bin/env python3
"""
physics_checks.py -- thermodynamic consistency of a trained UC-TPNO model on
real frameworks and conditions.

For every condition of the chosen frameworks it evaluates the response
matrix J_ij = dn_i/dmu_j (mol kg^-1 per kJ mol^-1) of the ensemble mean and
reports:
  * Maxwell symmetry   |J - J^T|_F / |J|_F            (exact by construction)
  * stability          smallest eigenvalue of (J+J^T)/2 (>= 0: PSD)
  * self-response      dn_i/dmu_i >= 0
  * sign               fraction of negative predicted loadings

    thesis_env/bin/python scripts/validation/physics_checks.py \
        --checkpoint experiments/run_014/checkpoints/best_model.pt \
        --splits experiments/run_014/splits.json --split val \
        --out results/validation/level2/run_014_physics.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ps", ROOT / "scripts/validation/predict_split.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-mofs", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = ps.train05.load_config(str(ROOT / "configs/pipeline.yaml"))
    nf = int(cfg.get("model", {}).get("n_mof_features", 0) or 0)
    fpath = str(ROOT / cfg["data"]["mof_features"]) if nf else None
    ds = ps.AdsorptionDataset(registry_path=str(ROOT / "data/mof_registry.parquet"),
                              adsorption_path=str(ROOT / "data/processed/adsorption_v3/adsorption_training.parquet"),
                              graph_dir=str(ROOT / "data/processed/graphs"), feature_path=fpath)
    wanted = json.loads(Path(a.splits).read_text())[a.split]
    if a.max_mofs:
        wanted = wanted[: a.max_mofs]
    pos = {m: i for i, m in enumerate(ds.mof_ids)}
    idx = [pos[m] for m in wanted]
    model = ps.build_model(cfg)
    st = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(st.get("model_state_dict", st), strict=True)
    model.eval()

    rows = []
    for i, batch in zip(idx, ds.get_dataloader(indices=idx, batch_size=1, shuffle=False)):
        g, c, t, m = ps.TPNOTrainer._to_device(batch, torch.device("cpu"))
        with torch.set_grad_enabled(True):
            q = model(g, c, return_uncertainty=False)["q_pred"][0].detach().numpy()
            H = model.get_hessian(g, c)[0].detach().numpy()          # [P, C, C]
        for p in range(H.shape[0]):
            J = H[p]
            sym = np.linalg.norm(J - J.T) / max(np.linalg.norm(J), 1e-12)
            lam = np.linalg.eigvalsh(0.5 * (J + J.T)).min()
            rows.append({"mof_id": ds.mof_ids[i], "point": p, "asym": sym, "min_eig": lam,
                         "min_self": float(np.diag(J).min()), "scale": float(np.abs(J).max()),
                         "n_neg": int((q[p] < 0).sum())})
    r = pd.DataFrame(rows)
    r.to_csv(a.out, index=False)
    tol = 1e-6 * r.scale.clip(lower=1e-12)
    print(f"{r.mof_id.nunique()} MOFs, {len(r)} conditions")
    print(f"  Maxwell symmetry: max relative asymmetry {r.asym.max():.2e}")
    print(f"  stability (PSD):  {100 * (r.min_eig >= -tol).mean():.2f} % of conditions, "
          f"most negative eigenvalue {r.min_eig.min():.2e}")
    print(f"  self-response dn_i/dmu_i >= 0: {100 * (r.min_self >= -tol).mean():.2f} %")
    print(f"  negative loadings: {r.n_neg.sum()} of {3 * len(r)} predictions")


if __name__ == "__main__":
    main()
