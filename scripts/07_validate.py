#!/usr/bin/env python3
"""
07_validate.py — Comprehensive validation of trained UC-TPNO model.

Validation tiers
----------------
1. Regression metrics on held-out test set (overall + per component)
2. Uncertainty quantification metrics
3. Statistical validation / data-quality checks
4. Thermodynamic consistency checks on a representative test MOF
5. NIST ISODB external validation (model vs experiment)
6. Publication-quality visualisations

NIST ISODB validation (tier 5)
-------------------------------
Loads all JSON files from --nist-dir (default: data/raw/nist_isodb).
For each experimental isotherm:
  - Matches adsorbent name to MOF registry by fuzzy string overlap
  - Converts units to mol/kg and bar
  - Builds pure-component condition vector, queries the model
  - Computes MAE/RMSE between prediction and experiment
  - Saves comparison table and summary JSON

Usage:
    python scripts/07_validate.py \
        --model-checkpoint experiments/run_001/final_model.pt \
        --config configs/pipeline.yaml \
        --registry data/mof_registry.parquet \
        --adsorption-data data/processed/adsorption/adsorption_training.parquet \
        --graph-dir data/processed/graphs \
        --splits-file experiments/run_001/splits.json \
        --output-dir results/validation \
        --nist-dir data/raw/nist_isodb

    # Skip NIST if no experimental data yet:
    python scripts/07_validate.py ... --skip-nist
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datasets.adsorption_dataset import AdsorptionDataset
from src.models.encoder.adapter import EncoderAdapter
from src.models.operator.tpno import ThermodynamicPotentialNO, TPNOConfig
from src.evaluation.metrics import (
    compute_regression_metrics,
    compute_uncertainty_metrics,
)
from src.evaluation.validator import ModelValidator
from src.evaluation.visualizer import ResultVisualizer
from src.utils.reproducibility import set_seed
from src.utils.io_utils import parse_nist_isodb_json


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Model loading helpers
# ─────────────────────────────────────────────────────────────

def load_yaml_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_splits(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model_from_config(model_cfg: Dict[str, Any], device: torch.device) -> ThermodynamicPotentialNO:
    encoder_config = {
        "encoder": model_cfg.get("encoder", "nequip"),
        "n_species": model_cfg.get("n_species", 100),
        "emb_dim": model_cfg.get("emb_dim", 128),
        "n_layers": model_cfg.get("n_encoder_layers", 4),
        "lmax": model_cfg.get("lmax", 2),
        "cutoff": model_cfg.get("cutoff", 6.0),
        "n_rbf": model_cfg.get("n_rbf", 32),
        "use_pbc": model_cfg.get("use_pbc", True),
    }
    encoder = EncoderAdapter.from_config(
        encoder_config, target_dim=model_cfg.get("emb_dim", 128), normalize=True,
    )
    tpno_cfg = TPNOConfig(
        emb_dim=int(model_cfg.get("emb_dim", 128)),
        n_conditions=int(model_cfg.get("n_conditions", 4)),
        n_components=int(model_cfg.get("n_components", 3)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        n_layers=int(model_cfg.get("n_tpno_layers", 4)),
        convex_constraint=model_cfg.get("convex_constraint", "softplus"),
        film_conditioning=bool(model_cfg.get("film_conditioning", True)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_layer_norm=bool(model_cfg.get("use_layer_norm", True)),
        activation=model_cfg.get("activation", "swish"),
        min_potential=float(model_cfg.get("min_potential", 1e-6)),
    )
    model = ThermodynamicPotentialNO(encoder=encoder, config=tpno_cfg)
    model.to(device)
    return model


def load_checkpoint_into_model(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)


# ─────────────────────────────────────────────────────────────
# Prediction collection helpers
# ─────────────────────────────────────────────────────────────

def _safe_sigma_fallback(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    resid = np.abs(y_true - y_pred)
    scale = max(float(np.std(resid)), 1e-3)
    return np.full_like(y_pred, scale, dtype=np.float64)


def collect_predictions(model, loader, device, log_prefix="TEST", log_interval=10):
    model.eval()
    all_true, all_pred, all_std, all_cond, all_mof_ids = [], [], [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            graphs = batch["graphs"].to(device)
            conditions = batch["conditions"].to(device)
            targets = batch["loadings"].to(device)
            mask = batch["mask"].to(device)
            mof_ids = batch["mof_ids"]

            out = model(graphs, conditions, return_uncertainty=True, return_potential=False, return_hessian=False)
            y_pred = out["q_pred"] if isinstance(out, dict) else out
            y_std = out.get("sigma", None) if isinstance(out, dict) else None

            valid = mask.bool()
            true_pts = targets[valid].detach().cpu().numpy().astype(np.float64)
            pred_pts = y_pred[valid].detach().cpu().numpy().astype(np.float64)
            cond_pts = conditions[valid].detach().cpu().numpy().astype(np.float64)
            std_pts = (y_std[valid].detach().cpu().numpy().astype(np.float64)
                       if y_std is not None else _safe_sigma_fallback(true_pts, pred_pts))
            std_pts = np.maximum(std_pts, 1e-6)

            counts = mask.sum(dim=1).detach().cpu().numpy().astype(int)
            mof_pts = []
            for m_id, cnt in zip(mof_ids, counts):
                mof_pts.extend([m_id] * int(cnt))

            all_true.append(true_pts); all_pred.append(pred_pts)
            all_std.append(std_pts); all_cond.append(cond_pts)
            all_mof_ids.append(np.array(mof_pts, dtype=object))

            if batch_idx == 1 or batch_idx % max(log_interval, 1) == 0:
                logger.info("%s batches: %d | valid points: %d", log_prefix, batch_idx, sum(a.shape[0] for a in all_true))

    if not all_true:
        raise RuntimeError(f"No predictions collected for: {log_prefix}")

    return {
        "y_true": np.concatenate(all_true, axis=0),
        "y_pred": np.concatenate(all_pred, axis=0),
        "y_std":  np.concatenate(all_std,  axis=0),
        "conditions": np.concatenate(all_cond, axis=0),
        "mof_ids": np.concatenate(all_mof_ids, axis=0),
    }


def collect_conditions_targets(loader, device, log_prefix="DATA", log_interval=10):
    all_cond, all_tgt = [], []
    for batch_idx, batch in enumerate(loader, start=1):
        conditions = batch["conditions"].to(device)
        targets = batch["loadings"].to(device)
        mask = batch["mask"].to(device)
        valid = mask.bool()
        all_cond.append(conditions[valid].detach().cpu().numpy().astype(np.float64))
        all_tgt.append(targets[valid].detach().cpu().numpy().astype(np.float64))
        if batch_idx == 1 or batch_idx % max(log_interval, 1) == 0:
            logger.info("%s batches: %d | points: %d", log_prefix, batch_idx, sum(a.shape[0] for a in all_cond))
    if not all_cond:
        raise RuntimeError(f"No data collected for: {log_prefix}")
    return {"conditions": np.concatenate(all_cond, 0), "targets": np.concatenate(all_tgt, 0)}


def build_representative_thermo_batch(dataset, indices, device):
    if not indices:
        raise RuntimeError("No indices for thermodynamic validation.")
    sample = dataset[indices[0]]
    batch = AdsorptionDataset.collate_fn([sample])
    for k in ("graphs", "conditions", "loadings", "mask"):
        batch[k] = batch[k].to(device)
    return batch


# ─────────────────────────────────────────────────────────────
# NIST ISODB — unit conversion tables
# ─────────────────────────────────────────────────────────────

# adsorbate name → component index [CO2=0, N2=1, H2O=2]
_ADS_TO_COMP: Dict[str, int] = {
    "carbon dioxide": 0, "co2": 0,
    "nitrogen": 1, "n2": 1,
    "water": 2, "h2o": 2, "water vapor": 2,
}

# pressure unit → bar
_P_TO_BAR: Dict[str, float] = {
    "bar": 1.0, "kpa": 0.01, "pa": 1e-5, "mpa": 10.0,
    "atm": 1.01325, "psi": 0.0689476, "mmhg": 0.00133322, "torr": 0.00133322,
}

# molar mass g/mol
_MM: Dict[str, float] = {"co2": 44.01, "n2": 28.014, "h2o": 18.015}


def _to_bar(value: float, unit: str) -> Optional[float]:
    f = _P_TO_BAR.get(unit.lower().strip())
    return float(value) * f if f is not None else None


def _to_molkg(value: float, unit: str, ads: str) -> Optional[float]:
    u = unit.lower().strip()
    if u in ("mol/kg", "mmol/g"):
        return float(value)
    if u == "mmol/kg":
        return float(value) * 1e-3
    if u == "mol/g":
        return float(value) * 1000.0
    if u in ("cm3(stp)/g", "cm3 (stp)/g", "cm3/g"):
        return float(value) / 22.414  # cm3(STP)/g -> mol/kg
    if u == "mg/g":
        mm = _MM.get(ads.lower())
        return (float(value) / mm) if mm else None
    return None


def _fuzzy_match(name: str, registry_ids: List[str], min_overlap: int = 4) -> Optional[str]:
    """Match NIST adsorbent name to registry MOF ID by token overlap."""
    tokens = set(
        t for t in name.lower().replace("-", " ").replace("_", " ").split()
        if len(t) >= min_overlap
    )
    if not tokens:
        return None
    best_id, best_score = None, 0
    for rid in registry_ids:
        score = sum(1 for t in tokens if t in rid.lower())
        if score > best_score:
            best_score, best_id = score, rid
    return best_id if best_score > 0 else None


# ─────────────────────────────────────────────────────────────
# NIST ISODB — main validation
# ─────────────────────────────────────────────────────────────

def run_nist_validation(
    nist_dir: Path,
    model: torch.nn.Module,
    dataset: AdsorptionDataset,
    device: torch.device,
    normalization: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Compare model predictions against NIST ISODB experimental isotherms.

    Steps per JSON file:
      1. Parse using parse_nist_isodb_json (already implemented in io_utils)
      2. Fuzzy-match adsorbent name to registry MOF ID
      3. Convert pressure → bar, loading → mol/kg
      4. Build normalised pure-component condition vector [mu_CO2, mu_N2, mu_H2O, T]
      5. Query model, extract component loading, denormalize
      6. Accumulate MAE / RMSE
    """
    nist_dir = Path(nist_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(nist_dir.glob("*.json"))
    if not json_files:
        return {"n_files": 0, "n_matched_isotherms": 0, "n_points": 0,
                "mae_molkg": None, "rmse_molkg": None}

    registry_ids: List[str] = dataset.mof_ids

    mu_mean = np.array(normalization.get("mu_mean", [0, 0, 0, 298.15]), dtype=np.float64)
    mu_std  = np.array(normalization.get("mu_std",  [1, 1, 1, 1]),      dtype=np.float64)
    q_mean  = np.array(normalization.get("q_mean",  [0, 0, 0]),          dtype=np.float64)
    q_std   = np.array(normalization.get("q_std",   [1, 1, 1]),          dtype=np.float64)
    mu_std  = np.where(mu_std > 0, mu_std, 1.0)
    q_std   = np.where(q_std  > 0, q_std,  1.0)

    comp_names = ["CO2", "N2", "H2O"]
    eps = 1e-10

    all_rows: List[Dict] = []
    n_matched = 0
    n_skip_unit = 0
    n_skip_match = 0

    model.eval()

    for jf in json_files:
        try:
            points = parse_nist_isodb_json(jf)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", jf.name, e)
            continue

        # Group by (adsorbent, adsorbate, temperature, units)
        groups: Dict[Tuple, List] = {}
        for pt in points:
            if pt.get("pressure") is None or pt.get("loading") is None:
                continue
            key = (
                str(pt.get("adsorbent", "unknown")),
                str(pt.get("adsorbate", "unknown")).lower(),
                float(pt.get("temperature") or 298.15),
                str(pt.get("units_pressure", "bar")),
                str(pt.get("units_loading", "mmol/g")),
            )
            groups.setdefault(key, []).append(pt)

        for (adsorbent, adsorbate, temperature, p_unit, q_unit), grp in groups.items():

            comp_idx = _ADS_TO_COMP.get(adsorbate.lower().strip())
            if comp_idx is None:
                continue

            matched_mof = _fuzzy_match(adsorbent, registry_ids)
            if matched_mof is None:
                n_skip_match += 1
                continue

            try:
                mof_idx = registry_ids.index(matched_mof)
            except ValueError:
                n_skip_match += 1
                continue

            pressures_bar, loadings_molkg = [], []
            for pt in grp:
                p = _to_bar(float(pt["pressure"]), p_unit)
                q = _to_molkg(float(pt["loading"]), q_unit, adsorbate)
                if p is None or q is None:
                    n_skip_unit += 1
                    continue
                if p <= 0 or not np.isfinite(p) or not np.isfinite(q) or q < 0:
                    continue
                pressures_bar.append(p)
                loadings_molkg.append(q)

            if not pressures_bar:
                continue

            n_matched += 1

            # Load MOF graph once
            sample = dataset[mof_idx]
            batch = AdsorptionDataset.collate_fn([sample])
            graphs = batch["graphs"].to(device)

            for p_bar, q_exp in zip(pressures_bar, loadings_molkg):
                # Pure-component chemical potentials: mu_i = ln(y_i * P)
                # All non-target species get y≈0 → ln(eps)
                mu_raw = np.array([
                    math.log(eps), math.log(eps), math.log(eps), float(temperature)
                ], dtype=np.float64)
                mu_raw[comp_idx] = math.log(p_bar + eps)

                mu_norm = (mu_raw - mu_mean) / mu_std
                cond_t = torch.tensor(
                    mu_norm[np.newaxis, np.newaxis, :],
                    dtype=torch.float32, device=device,
                )  # [1, 1, 4]

                with torch.no_grad():
                    out = model(graphs, cond_t, return_uncertainty=False,
                                return_potential=False, return_hessian=False)

                q_norm = float(
                    (out["q_pred"] if isinstance(out, dict) else out)[0, 0, comp_idx]
                    .detach().cpu().numpy()
                )
                q_pred = max(float(q_norm * q_std[comp_idx] + q_mean[comp_idx]), 0.0)

                all_rows.append({
                    "file": jf.name,
                    "adsorbent_nist": adsorbent,
                    "matched_mof": matched_mof,
                    "adsorbate": adsorbate,
                    "component": comp_names[comp_idx],
                    "temperature_k": temperature,
                    "pressure_bar": p_bar,
                    "loading_exp_molkg": q_exp,
                    "loading_pred_molkg": q_pred,
                    "abs_error_molkg": abs(q_pred - q_exp),
                })

    # ── Aggregate ────────────────────────────────────────────────
    base_summary = {
        "n_files": len(json_files),
        "n_matched_isotherms": n_matched,
        "n_skipped_unit_conversion": n_skip_unit,
        "n_skipped_no_mof_match": n_skip_match,
        "n_points": len(all_rows),
    }

    if not all_rows:
        logger.warning("NIST: no comparable points. matched=%d skip_unit=%d skip_match=%d",
                       n_matched, n_skip_unit, n_skip_match)
        with open(out_dir / "nist_validation_summary.json", "w") as f:
            json.dump({**base_summary, "mae_molkg": None, "rmse_molkg": None}, f, indent=2)
        return base_summary

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "nist_comparison_points.csv", index=False)
    df.to_parquet(out_dir / "nist_comparison_points.parquet", index=False)

    errs = df["abs_error_molkg"].values
    mae  = float(np.mean(errs))
    rmse = float(np.sqrt(np.mean(errs ** 2)))

    comp_mae = {}
    for c in ["CO2", "N2", "H2O"]:
        sub = df[df["component"] == c]["abs_error_molkg"].values
        comp_mae[f"mae_{c}_molkg"] = float(np.mean(sub)) if len(sub) > 0 else None

    mof_summary = (
        df.groupby("matched_mof")
        .agg(n_points=("abs_error_molkg","count"),
             mae=("abs_error_molkg","mean"),
             rmse=("abs_error_molkg", lambda x: float(np.sqrt(np.mean(x**2)))))
        .reset_index()
    )
    mof_summary.to_csv(out_dir / "nist_per_mof_summary.csv", index=False)

    summary = {
        **base_summary,
        "n_mofs_matched": int(df["matched_mof"].nunique()),
        "mae_molkg": mae,
        "rmse_molkg": rmse,
        **comp_mae,
    }

    with open(out_dir / "nist_validation_summary.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, (float, np.floating)) else v)
                   for k, v in summary.items()}, f, indent=2)

    logger.info("NIST: %d points from %d isotherms | MAE=%.4f mol/kg | RMSE=%.4f mol/kg",
                len(df), n_matched, mae, rmse)
    return summary


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate trained UC-TPNO model")
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--adsorption-data", required=True)
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--splits-file", required=True)
    parser.add_argument("--output-dir", default="results/validation")
    parser.add_argument("--nist-dir", default="data/raw/nist_isodb",
                        help="Directory containing NIST ISODB JSON files.")
    parser.add_argument("--skip-nist", action="store_true",
                        help="Skip NIST ISODB external validation.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MODEL VALIDATION")
    print(f"Device:      {device}")
    print(f"Checkpoint:  {args.model_checkpoint}")
    print(f"NIST dir:    {args.nist_dir}")
    print(f"Output dir:  {out_dir}")
    print("=" * 70)

    # ── Data ─────────────────────────────────────────────
    print("\n=== Loading Data ===")
    dataset = AdsorptionDataset(
        registry_path=args.registry,
        adsorption_path=args.adsorption_data,
        graph_dir=args.graph_dir,
    )
    splits = load_splits(args.splits_file)
    train_ids = set(splits["train"])
    test_ids  = set(splits["test"])
    train_idx = [i for i, m in enumerate(dataset.mof_ids) if m in train_ids]
    test_idx  = [i for i, m in enumerate(dataset.mof_ids) if m in test_ids]

    if len(test_idx) == 0:
        raise RuntimeError("Test split is empty.")

    train_loader = dataset.get_dataloader(indices=train_idx, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader  = dataset.get_dataloader(indices=test_idx,  batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    print(f"Train MOFs: {len(train_idx)} | Test MOFs: {len(test_idx)}")

    # ── Model ────────────────────────────────────────────
    print("\n=== Loading Model ===")
    cfg = load_yaml_config(args.config)
    model_cfg = dict(cfg.get("model", {}))

    # Load normalization: try resolved_config.json → checkpoint → fallback
    normalization: Dict[str, Any] = {}
    rc_path = Path(args.output_dir).parent / "resolved_config.json"
    if rc_path.exists():
        with open(rc_path) as f:
            normalization = json.load(f).get("normalization", {})
    if not normalization:
        ckpt = torch.load(args.model_checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            normalization = ckpt.get("normalization", {})
    if not normalization:
        logger.warning("No normalization stats found — using identity.")
        normalization = {
            "mu_mean": [0.0, 0.0, 0.0, 298.15],
            "mu_std":  [1.0, 1.0, 1.0,   1.0],
            "q_mean":  [0.0, 0.0, 0.0],
            "q_std":   [1.0, 1.0, 1.0],
        }

    model = build_model_from_config(model_cfg, device=device)
    load_checkpoint_into_model(model, args.model_checkpoint, device=device)
    model.eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Collect test predictions ─────────────────────────
    print("\n=== Running Inference on Test Set ===")
    pred_data = collect_predictions(model=model, loader=test_loader, device=device,
                                    log_prefix="TEST", log_interval=args.log_interval)
    if not np.all(np.isfinite(pred_data["y_std"])) or np.any(pred_data["y_std"] <= 0):
        pred_data["y_std"] = _safe_sigma_fallback(pred_data["y_true"], pred_data["y_pred"])
    print(f"Valid test points: {pred_data['y_true'].shape[0]}")

    train_data = collect_conditions_targets(loader=train_loader, device=device,
                                            log_prefix="TRAIN", log_interval=args.log_interval)
    y_true = pred_data["y_true"]
    y_pred = pred_data["y_pred"]
    y_std  = pred_data["y_std"]

    # ── 1. Regression metrics ────────────────────────────
    print("\n" + "=" * 50)
    print("1. REGRESSION METRICS (simulated test set)")
    print("=" * 50)
    components = ["CO2", "N2", "H2O"]
    reg = compute_regression_metrics(y_true=y_true, y_pred=y_pred,
                                     component_names=components, prefix="reg_")
    for k, v in sorted(reg.items()):
        print(f"  {k}: {v:.6f}")

    # ── 2. UQ metrics ────────────────────────────────────
    print("\n" + "=" * 50)
    print("2. UNCERTAINTY QUANTIFICATION")
    print("=" * 50)
    uq = compute_uncertainty_metrics(y_true=y_true, y_pred=y_pred, y_std=y_std, prefix="uq_")
    for k, v in sorted(uq.items()):
        print(f"  {k}: {v:.6f}")

    # ── 3. Statistical validation ────────────────────────
    print("\n" + "=" * 50)
    print("3. STATISTICAL VALIDATION")
    print("=" * 50)
    mv = ModelValidator()
    report = mv.full_report(
        y_true=y_true, y_pred=y_pred, y_std=y_std,
        X_train=train_data["conditions"],
        X_test=pred_data["conditions"],
        y_train=train_data["targets"],
    )
    for section, data in report.items():
        if isinstance(data, dict):
            print(f"\n  [{section}]")
            for k, v in data.items():
                if isinstance(v, (int, float, bool, np.bool_)):
                    print(f"    {k}: {v}")

    # ── 4. Thermodynamic consistency ─────────────────────
    print("\n" + "=" * 50)
    print("4. THERMODYNAMIC CONSISTENCY")
    print("=" * 50)
    thermo_batch = build_representative_thermo_batch(dataset=dataset, indices=test_idx, device=device)
    thermo_report = mv.thermo_wrapper.check_all(
        model=model, graphs=thermo_batch["graphs"], conditions=thermo_batch["conditions"],
    )
    report["thermodynamic"] = thermo_report
    for k, v in thermo_report.items():
        if isinstance(v, (int, float, bool, np.bool_)):
            print(f"  {k}: {v}")

    # ── 5. NIST ISODB external validation ────────────────
    print("\n" + "=" * 50)
    print("5. NIST ISODB EXTERNAL VALIDATION")
    print("=" * 50)
    nist_summary: Dict[str, Any] = {}

    if args.skip_nist:
        print("  Skipped (--skip-nist).")
    else:
        nist_dir = Path(args.nist_dir)
        if not nist_dir.exists():
            print(f"  NIST directory not found: {nist_dir}")
            print("  Place NIST ISODB JSON files there to enable experimental validation.")
            print("  Download from: https://adsorption.nist.gov/isodb/api/isotherms")
        else:
            n_jsons = len(list(nist_dir.glob("*.json")))
            print(f"  Found {n_jsons} JSON file(s) in {nist_dir}")
            if n_jsons == 0:
                print("  No JSON files — download isotherms and rerun.")
            else:
                nist_out = out_dir / "nist_validation"
                nist_summary = run_nist_validation(
                    nist_dir=nist_dir, model=model, dataset=dataset,
                    device=device, normalization=normalization, out_dir=nist_out,
                )
                print(f"  Files scanned:       {nist_summary['n_files']}")
                print(f"  Isotherms matched:   {nist_summary['n_matched_isotherms']}")
                print(f"  Points compared:     {nist_summary['n_points']}")
                if nist_summary.get("mae_molkg") is not None:
                    print(f"  MAE  [mol/kg]:       {nist_summary['mae_molkg']:.4f}")
                    print(f"  RMSE [mol/kg]:       {nist_summary['rmse_molkg']:.4f}")
                    for c in ["CO2", "N2", "H2O"]:
                        val = nist_summary.get(f"mae_{c}_molkg")
                        if val is not None:
                            print(f"  MAE {c} [mol/kg]:     {val:.4f}")
                    print(f"  Points CSV  -> {nist_out}/nist_comparison_points.csv")
                    print(f"  Per-MOF CSV -> {nist_out}/nist_per_mof_summary.csv")
                else:
                    print("  No comparable points found (check MOF name matching).")

    report["nist_external"] = nist_summary

    # ── 6. Visualisations ────────────────────────────────
    print("\n" + "=" * 50)
    print("6. GENERATING PLOTS")
    print("=" * 50)
    viz = ResultVisualizer(save_dir=str(out_dir), component_names=components)
    figs = viz.full_report(y_true=y_true, y_pred=y_pred, y_std=y_std)
    print(f"  Generated {len(figs)} figures -> {out_dir}")

    # ── Save outputs ─────────────────────────────────────
    print("\n=== Saving Outputs ===")
    summary_metrics = {**reg, **uq}
    if nist_summary.get("mae_molkg") is not None:
        summary_metrics["nist_mae_molkg"]  = nist_summary["mae_molkg"]
        summary_metrics["nist_rmse_molkg"] = nist_summary["rmse_molkg"]

    with open(out_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) if isinstance(v, (float, np.floating)) else v
                   for k, v in summary_metrics.items()}, f, indent=2)

    mv.save_report(report, out_dir / "full_report.json")
    np.save(out_dir / "y_pred.npy", y_pred)
    np.save(out_dir / "y_true.npy", y_true)
    np.save(out_dir / "y_std.npy",  y_std)
    np.save(out_dir / "conditions.npy", pred_data["conditions"])
    np.save(out_dir / "mof_ids.npy", pred_data["mof_ids"].astype(str))

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_checkpoint": args.model_checkpoint,
            "config": args.config,
            "registry": args.registry,
            "adsorption_data": args.adsorption_data,
            "graph_dir": args.graph_dir,
            "splits_file": args.splits_file,
            "output_dir": str(out_dir),
            "nist_dir": args.nist_dir,
            "device": str(device),
            "seed": args.seed,
            "n_test_mofs": len(test_idx),
            "n_test_points": int(y_true.shape[0]),
            "nist_n_points": nist_summary.get("n_points", 0),
            "nist_mae_molkg": nist_summary.get("mae_molkg"),
        }, f, indent=2)

    print(f"\nValidation complete -> {out_dir}")
    print("Next: python scripts/08_active_learning_loop.py")


if __name__ == "__main__":
    main()