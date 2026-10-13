#!/usr/bin/env python3
"""
predict_split.py -- predictions of a trained UC-TPNO checkpoint for one split.

Rebuilds the model exactly as scripts/05_train_model.py does (same config
keys and defaults), loads the checkpoint strictly, and writes one row per
(MOF, condition) with the GCMC labels and the ensemble prediction.

    thesis_env/bin/python scripts/validation/predict_split.py \
        --checkpoint experiments/run_010/checkpoints/best_model.pt \
        --splits experiments/run_010/splits.json --split val \
        --out results/validation/level2/run_010_val_predictions.csv

Use --split val while developing; the test split is evaluated once, at the end.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.datasets.adsorption_dataset import AdsorptionDataset  # noqa: E402
from src.models.encoder.adapter import EncoderAdapter  # noqa: E402
from src.models.operator.tpno import TPNOConfig, TPNOEnsemble, ThermodynamicPotentialNO  # noqa: E402
from src.training.trainer import TPNOTrainer  # noqa: E402

spec = importlib.util.spec_from_file_location("train05", ROOT / "scripts/05_train_model.py")
train05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train05)

GAS = ["co2", "n2", "h2o"]


def build_model(cfg: dict) -> torch.nn.Module:
    """Same construction and defaults as scripts/05_train_model.py."""
    m = dict(cfg.get("model", {}))
    encoder = EncoderAdapter.from_config({
        "encoder": m.get("encoder", "nequip"), "n_species": m.get("n_species", 100),
        "emb_dim": m.get("emb_dim", 64), "n_layers": m.get("n_encoder_layers", 2),
        "lmax": m.get("lmax", 1), "cutoff": m.get("cutoff", 6.0), "n_rbf": m.get("n_rbf", 32),
        "use_pbc": m.get("use_pbc", True), "use_charges": m.get("use_charges", False),
        "avg_num_neighbours": m.get("avg_num_neighbours", 15.0),
        "n_extra_features": int(m.get("n_mof_features", 0) or 0),
    }, target_dim=int(m.get("emb_dim", 64)), normalize=True)
    tc = TPNOConfig(
        emb_dim=int(m.get("emb_dim", 64)), n_conditions=int(m.get("n_conditions", 4)),
        n_components=int(m.get("n_components", 3)), hidden_dim=int(m.get("hidden_dim", 128)),
        n_layers=int(m.get("n_tpno_layers", 3)),
        convex_constraint=m.get("convex_constraint", "softplus"),
        film_conditioning=bool(m.get("film_conditioning", True)),
        dropout=float(m.get("dropout", 0.15)), use_layer_norm=bool(m.get("use_layer_norm", False)),
        activation=m.get("activation", "softplus"), min_potential=float(m.get("min_potential", 1e-6)),
    )
    if not bool(m.get("ensemble", False)):
        return ThermodynamicPotentialNO(encoder=encoder, config=tc)
    ens = cfg.get("ensemble", {}) if isinstance(cfg.get("ensemble"), dict) else {}
    return TPNOEnsemble(config=tc, encoder=encoder,
                        n_models=int(ens.get("n_models", m.get("n_ensemble", 5))),
                        share_encoder=bool(ens.get("share_encoder", m.get("share_encoder", True))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split", default="val", help="key in the splits file (train/val/test or e.g. level3)")
    ap.add_argument("--config", default=str(ROOT / "configs/pipeline.yaml"))
    ap.add_argument("--registry", default=str(ROOT / "data/mof_registry.parquet"))
    ap.add_argument("--adsorption-data",
                    default=str(ROOT / "data/processed/adsorption_v3/adsorption_training.parquet"))
    ap.add_argument("--graph-dir", default=str(ROOT / "data/processed/graphs"))
    ap.add_argument("--features", default=None,
                    help="per-MOF feature table (default: data.mof_features of the config)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-mofs", type=int, default=0, help="0 = all MOFs of the split")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = train05.load_config(a.config)
    nf = int(cfg.get("model", {}).get("n_mof_features", 0) or 0)
    fpath = (a.features or cfg.get("data", {}).get("mof_features")) if nf else None
    if fpath and not Path(fpath).is_absolute():
        fpath = str(ROOT / fpath)
    ds = AdsorptionDataset(registry_path=a.registry, adsorption_path=a.adsorption_data,
                           graph_dir=a.graph_dir, feature_path=fpath)
    wanted = json.loads(Path(a.splits).read_text())[a.split]
    if a.max_mofs:
        wanted = wanted[:a.max_mofs]
    pos = {m: i for i, m in enumerate(ds.mof_ids)}
    idx = [pos[m] for m in wanted]

    model = build_model(cfg).to(device)
    state = torch.load(a.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=True)
    model.eval()

    raw = pd.read_parquet(a.adsorption_data)
    rows = []
    loader = ds.get_dataloader(indices=idx, batch_size=1, shuffle=False)
    with torch.set_grad_enabled(True):  # the forward differentiates the potential
        for i, batch in zip(idx, loader):
            graphs, cond, targets, mask = TPNOTrainer._to_device(batch, device)
            out = model(graphs, cond, return_uncertainty=True,
                        return_potential=False, return_hessian=False)
            q = out["q_pred"][0].detach().cpu().numpy()
            # ensemble: total = sqrt(epistemic^2 + aleatoric^2), all in mol/kg
            unc = {}
            for key, name in (("total_uncertainty", "sigma"), ("epistemic", "epistemic"),
                              ("aleatoric", "aleatoric")):
                v = out.get(key, out.get("sigma") if name == "sigma" else None)
                unc[name] = v[0].detach().cpu().numpy() if v is not None else np.full_like(q, np.nan)
            sig = unc["sigma"]
            y = targets[0].cpu().numpy()
            keep = mask[0].cpu().numpy().astype(bool) if mask is not None else np.ones(len(q), bool)
            c = cond[0].cpu().numpy()
            for p in np.where(keep)[0]:
                r = {"mof_id": ds.mof_ids[i], "point": int(p),
                     "mu_co2": c[p, 0], "mu_n2": c[p, 1], "mu_h2o": c[p, 2], "T": c[p, 3]}
                for k, g in enumerate(GAS):
                    r[f"{g}_true"], r[f"{g}_pred"], r[f"{g}_sigma"] = y[p, k], q[p, k], sig[p, k]
                    r[f"{g}_epistemic"] = unc["epistemic"][p, k]
                    r[f"{g}_aleatoric"] = unc["aleatoric"][p, k]
                rows.append(r)
    df = pd.DataFrame(rows)
    # attach the physical conditions (pressure, RH) by matching T and mu_H2O order
    lab = raw[raw.mof_id.isin(wanted)][["mof_id", "temperature_k", "pressure_bar", "rh",
                                        "co2_loading_molkg", "n2_loading_molkg", "h2o_loading_molkg"]]
    df = df.merge(lab, left_on=["mof_id", "co2_true", "n2_true", "h2o_true"],
                  right_on=["mof_id", "co2_loading_molkg", "n2_loading_molkg", "h2o_loading_molkg"],
                  how="left").drop(columns=[f"{g}_loading_molkg" for g in GAS])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} points, {df.mof_id.nunique()} MOFs -> {a.out}")
    for g in GAS:
        print(f"  {g}: MAE {np.mean(np.abs(df[f'{g}_true'] - df[f'{g}_pred'])):.4f} mol/kg")


if __name__ == "__main__":
    main()
