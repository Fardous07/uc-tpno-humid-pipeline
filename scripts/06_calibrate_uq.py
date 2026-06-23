#!/usr/bin/env python3
"""
06_calibrate_uq.py — Calibrate uncertainty with conformal prediction.

This version fixes:
1. Model/encoder API mismatches.
2. Padded batch handling via batch mask.
3. Split-file handling for either MOF IDs or integer indices.
4. MOF tracking so saved MOF labels are real IDs, not batch-local indices.
5. Wrapped checkpoint loading compatibility.

Usage:
    python scripts/06_calibrate_uq.py \
        --model-checkpoint experiments/run_001/final_model.pt \
        --registry data/processed/mof_registry.parquet \
        --adsorption-data data/processed/adsorption/adsorption_training.parquet \
        --graph-dir data/processed/graphs \
        --splits-file experiments/run_001/splits.json \
        --output-dir results/calibration
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datasets.adsorption_dataset import AdsorptionDataset
from src.models.encoder.adapter import EncoderAdapter
from src.models.operator.tpno import ThermodynamicPotentialNO, TPNOConfig
from src.models.uq.conformal import (
    ConformalCalibrator,
    ConformalConfig,
    evaluate_coverage,
)
from src.evaluation.metrics import (
    compute_regression_metrics,
    compute_uncertainty_metrics,
)
from src.utils.reproducibility import set_seed


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
# Helpers
# ─────────────────────────────────────────────────────────────

def load_yaml_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_splits(path: str) -> Dict[str, List[Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_loaded_splits(
    dataset_mof_ids: Sequence[str],
    loaded_train: Sequence[Any],
    loaded_val: Sequence[Any],
    loaded_test: Sequence[Any],
) -> Tuple[List[int], List[int], List[int]]:
    """
    Handle split files whether they store:
    - MOF IDs
    - integer indices
    """
    dataset_mof_ids = list(dataset_mof_ids)

    def _all_ints(xs: Sequence[Any]) -> bool:
        return all(isinstance(x, int) for x in xs)

    if _all_ints(loaded_train) and _all_ints(loaded_val) and _all_ints(loaded_test):
        max_idx = len(dataset_mof_ids) - 1
        train_idx = [i for i in loaded_train if 0 <= i <= max_idx]
        val_idx = [i for i in loaded_val if 0 <= i <= max_idx]
        test_idx = [i for i in loaded_test if 0 <= i <= max_idx]
        return train_idx, val_idx, test_idx

    id_to_idx = {m: i for i, m in enumerate(dataset_mof_ids)}
    train_idx = [id_to_idx[m] for m in loaded_train if m in id_to_idx]
    val_idx = [id_to_idx[m] for m in loaded_val if m in id_to_idx]
    test_idx = [id_to_idx[m] for m in loaded_test if m in id_to_idx]
    return train_idx, val_idx, test_idx


def build_model_from_config(
    model_cfg: Dict[str, Any],
    device: torch.device,
) -> ThermodynamicPotentialNO:
    """
    Build the same model family used in 05_train_model.py.
    """
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
        encoder_config,
        target_dim=model_cfg.get("emb_dim", 128),
        normalize=True,
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


def load_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    """
    Load either a raw state_dict or a wrapped checkpoint.
    """
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


def _safe_sigma_fallback(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Fallback uncertainty if the model did not emit sigma.
    """
    resid = np.abs(y_true - y_pred)
    scale = float(np.std(resid))
    scale = max(scale, 1e-3)
    return np.full_like(y_pred, scale, dtype=np.float64)


def collect_predictions(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    log_prefix: str = "",
    log_interval: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Run masked inference and collect only real (unpadded) points.

    Returns flattened scalar-level arrays:
        y_true      : [N]
        y_pred      : [N]
        y_std       : [N]
        conditions  : [N, D]
        component   : [N]
        mof_id      : [N] object array of MOF IDs
    """
    model.eval()

    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_std: List[np.ndarray] = []
    all_cond: List[np.ndarray] = []
    all_comp: List[np.ndarray] = []
    all_mof_id: List[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            graphs = batch["graphs"].to(device)
            conditions = batch["conditions"].to(device)     # [B, P, D]
            targets = batch["loadings"].to(device)          # [B, P, C]
            mask = batch["mask"].to(device)                 # [B, P]
            batch_mof_ids = batch["mof_ids"]                # list[str]

            out = model(
                graphs,
                conditions,
                return_uncertainty=True,
                return_potential=False,
                return_hessian=False,
            )

            y_pred = out["q_pred"] if isinstance(out, dict) else out
            y_std = out.get("sigma", None) if isinstance(out, dict) else None

            B, P, C = targets.shape
            D = conditions.shape[-1]

            valid = mask.unsqueeze(-1).expand(B, P, C)  # [B, P, C]
            cond_rep = conditions.unsqueeze(2).expand(B, P, C, D)  # [B, P, C, D]

            comp_idx = (
                torch.arange(C, device=device)
                .view(1, 1, C)
                .expand(B, P, C)
            )

            pred_flat = y_pred[valid].detach().cpu().numpy().astype(np.float64)
            true_flat = targets[valid].detach().cpu().numpy().astype(np.float64)

            if y_std is not None:
                std_flat = y_std[valid].detach().cpu().numpy().astype(np.float64)
            else:
                std_flat = _safe_sigma_fallback(true_flat, pred_flat)

            cond_flat = cond_rep[valid].detach().cpu().numpy().astype(np.float64)
            comp_flat = comp_idx[valid].detach().cpu().numpy().astype(np.int64)

            # Real MOF IDs, not batch-local indices
            valid_cpu = valid.detach().cpu().numpy()
            mof_flat_list: List[str] = []
            for b in range(B):
                n_valid_scalars = int(valid_cpu[b].sum())
                mof_flat_list.extend([batch_mof_ids[b]] * n_valid_scalars)
            mof_flat = np.array(mof_flat_list, dtype=object)

            std_flat = np.maximum(std_flat, 1e-6)

            all_true.append(true_flat)
            all_pred.append(pred_flat)
            all_std.append(std_flat)
            all_cond.append(cond_flat)
            all_comp.append(comp_flat)
            all_mof_id.append(mof_flat)

            if batch_idx == 1 or batch_idx % max(log_interval, 1) == 0:
                logger.info(
                    "%s batches processed: %d | scalar points collected so far: %d",
                    log_prefix,
                    batch_idx,
                    sum(len(x) for x in all_true),
                )

    if not all_true:
        raise RuntimeError(f"No predictions collected for loader: {log_prefix}")

    return {
        "y_true": np.concatenate(all_true, axis=0),
        "y_pred": np.concatenate(all_pred, axis=0),
        "y_std": np.concatenate(all_std, axis=0),
        "conditions": np.concatenate(all_cond, axis=0),   # [N, D]
        "component": np.concatenate(all_comp, axis=0),    # [N]
        "mof_id": np.concatenate(all_mof_id, axis=0),     # [N]
    }


def build_covariates(pred_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Covariates for weighted conformal:
      [mu_CO2, mu_N2, mu_H2O, T, component_index]
    """
    comp = pred_dict["component"].reshape(-1, 1).astype(np.float64)
    return np.hstack([pred_dict["conditions"], comp])


def build_mondrian_groups(pred_dict: Dict[str, np.ndarray], group_by: str) -> np.ndarray:
    """
    Build group labels for Mondrian conformal.
    """
    if group_by == "component":
        return pred_dict["component"].astype(str)

    if group_by == "temperature_bin":
        T = pred_dict["conditions"][:, -1]
        bins = np.digitize(T, bins=np.array([305.0, 323.0]), right=False)
        return bins.astype(str)

    if group_by == "component_temperature":
        T = pred_dict["conditions"][:, -1]
        bins = np.digitize(T, bins=np.array([305.0, 323.0]), right=False)
        return np.array(
            [f"c{c}_t{b}" for c, b in zip(pred_dict["component"], bins)],
            dtype=object,
        )

    raise ValueError(f"Unknown group_by='{group_by}'.")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate uncertainty with conformal prediction")
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--adsorption-data", required=True)
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--splits-file", required=True)
    parser.add_argument("--output-dir", default="results/calibration")

    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage rate (0.1 = 90% intervals)")
    parser.add_argument("--method", default="split", choices=["split", "weighted", "mondrian", "cv"])
    parser.add_argument(
        "--score-method",
        default="normalized",
        choices=["absolute", "squared", "normalized", "studentized"],
    )
    parser.add_argument(
        "--group-by",
        default="component",
        choices=["component", "temperature_bin", "component_temperature"],
        help="Only used when method=mondrian",
    )

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
    print("CONFORMAL CALIBRATION")
    print(f"Device:            {device}")
    print(f"Checkpoint:        {args.model_checkpoint}")
    print(f"Splits file:       {args.splits_file}")
    print(f"Method:            {args.method}")
    print(f"Score method:      {args.score_method}")
    print(f"Alpha:             {args.alpha}")
    print(f"Output dir:        {out_dir}")
    print("=" * 70)

    # ── Load dataset ────────────────────────────────────────
    print("\n=== Loading Data ===")
    dataset = AdsorptionDataset(
        registry_path=args.registry,
        adsorption_path=args.adsorption_data,
        graph_dir=args.graph_dir,
    )

    splits = load_splits(args.splits_file)
    train_idx, val_idx, test_idx = resolve_loaded_splits(
        dataset.mof_ids,
        splits["train"],
        splits["val"],
        splits["test"],
    )

    if len(val_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError(
            f"Invalid calibration/test split sizes: val={len(val_idx)}, test={len(test_idx)}"
        )

    val_loader = dataset.get_dataloader(
        indices=val_idx,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = dataset.get_dataloader(
        indices=test_idx,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print(f"Validation MOFs: {len(val_idx)}")
    print(f"Test MOFs:       {len(test_idx)}")
    print(f"Batch size:      {args.batch_size}")

    # ── Load model ───────────────────────────────────────
    print("\n=== Loading Model ===")
    cfg = load_yaml_config(args.config)
    model_cfg = dict(cfg.get("model", {}))

    model = build_model_from_config(model_cfg, device=device)
    load_checkpoint_into_model(model, args.model_checkpoint, device=device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ── Collect predictions ──────────────────────────────
    print("\n=== Collecting Validation Predictions ===")
    val_data = collect_predictions(
        model=model,
        loader=val_loader,
        device=device,
        log_prefix="VAL",
        log_interval=args.log_interval,
    )

    print("\n=== Collecting Test Predictions ===")
    test_data = collect_predictions(
        model=model,
        loader=test_loader,
        device=device,
        log_prefix="TEST",
        log_interval=args.log_interval,
    )

    print(f"Validation scalar points: {len(val_data['y_true'])}")
    print(f"Test scalar points:       {len(test_data['y_true'])}")

    if not np.all(np.isfinite(val_data["y_std"])) or np.any(val_data["y_std"] <= 0):
        logger.warning("Validation sigma invalid; using fallback residual scale.")
        val_data["y_std"] = _safe_sigma_fallback(val_data["y_true"], val_data["y_pred"])

    if not np.all(np.isfinite(test_data["y_std"])) or np.any(test_data["y_std"] <= 0):
        logger.warning("Test sigma invalid; using fallback residual scale.")
        test_data["y_std"] = _safe_sigma_fallback(test_data["y_true"], test_data["y_pred"])

    # ── Calibrate ────────────────────────────────────────
    print("\n=== Calibrating ===")
    cal_cfg = ConformalConfig(
        alpha=args.alpha,
        method=args.method,
        score_method=args.score_method,
    )
    calibrator = ConformalCalibrator(cal_cfg)

    calibration_payload: Dict[str, np.ndarray] = {
        "y_true": val_data["y_true"],
        "y_pred": val_data["y_pred"],
        "y_std": val_data["y_std"],
    }

    prediction_payload: Dict[str, np.ndarray] = {
        "y_pred": test_data["y_pred"],
        "y_std": test_data["y_std"],
    }

    if args.method == "weighted":
        calibration_payload["covariates"] = build_covariates(val_data)
        calibration_payload["target_covariates"] = build_covariates(test_data)

    if args.method == "mondrian":
        calibration_payload["groups"] = build_mondrian_groups(val_data, args.group_by)
        prediction_payload["groups"] = build_mondrian_groups(test_data, args.group_by)

    calibrator.calibrate(calibration_payload)
    intervals = calibrator.predict_intervals(prediction_payload)

    # ── Evaluate ─────────────────────────────────────────
    print("\n=== Evaluating Calibration ===")
    coverage_results = evaluate_coverage(
        intervals=intervals,
        y_true=test_data["y_true"],
        groups=prediction_payload.get("groups"),
    )

    uq_metrics = compute_uncertainty_metrics(
        y_true=test_data["y_true"],
        y_pred=test_data["y_pred"],
        y_std=test_data["y_std"],
        prefix="uq_",
    )

    reg_metrics = compute_regression_metrics(
        y_true=test_data["y_true"],
        y_pred=test_data["y_pred"],
        prefix="reg_",
    )

    lower = intervals["lower"]
    upper = intervals["upper"]
    widths = upper - lower
    covered = (test_data["y_true"] >= lower) & (test_data["y_true"] <= upper)

    quantile_value = None
    predictor = getattr(calibrator, "_predictor", None)
    if predictor is not None and hasattr(predictor, "quantile"):
        quantile_value = getattr(predictor, "quantile")

    results: Dict[str, Any] = {
        "alpha": float(args.alpha),
        "method": args.method,
        "score_method": args.score_method,
        "group_by": args.group_by if args.method == "mondrian" else None,
        "n_val_points": int(len(val_data["y_true"])),
        "n_test_points": int(len(test_data["y_true"])),
        "quantile": float(quantile_value) if quantile_value is not None else None,
        **{k: float(v) for k, v in coverage_results.items() if not isinstance(v, dict)},
        **{k: float(v) for k, v in uq_metrics.items()},
        **{k: float(v) for k, v in reg_metrics.items()},
    }

    if "conditional_coverage" in coverage_results:
        results["conditional_coverage"] = {
            str(k): float(v) for k, v in coverage_results["conditional_coverage"].items()
        }

    print("\n" + "=" * 70)
    print(f"Target coverage:      {1 - args.alpha:.1%}")
    print(f"Empirical coverage:   {coverage_results['overall_coverage']:.1%}")
    print(f"Coverage error:       {coverage_results['coverage_error']:.4f}")
    print(f"Mean interval width:  {coverage_results['mean_width']:.4f}")
    print(f"Median width:         {coverage_results['median_width']:.4f}")
    print(f"Regression MAE:       {reg_metrics['reg_mae']:.4f}")
    print(f"Regression RMSE:      {reg_metrics['reg_rmse']:.4f}")
    print(f"UQ ECE:               {uq_metrics['uq_ece']:.4f}")
    print("=" * 70)

    # ── Save ─────────────────────────────────────────────
    print("\n=== Saving Outputs ===")
    np.save(out_dir / "val_y_true.npy", val_data["y_true"])
    np.save(out_dir / "val_y_pred.npy", val_data["y_pred"])
    np.save(out_dir / "val_y_std.npy", val_data["y_std"])

    np.save(out_dir / "test_y_true.npy", test_data["y_true"])
    np.save(out_dir / "test_y_pred.npy", test_data["y_pred"])
    np.save(out_dir / "test_y_std.npy", test_data["y_std"])

    np.save(out_dir / "intervals_lower.npy", lower)
    np.save(out_dir / "intervals_upper.npy", upper)
    np.save(out_dir / "interval_widths.npy", widths)
    np.save(out_dir / "covered.npy", covered.astype(np.uint8))

    np.save(out_dir / "test_conditions.npy", test_data["conditions"])
    np.save(out_dir / "test_component.npy", test_data["component"])
    np.save(out_dir / "test_mof_id.npy", test_data["mof_id"])

    with open(out_dir / "calibration_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    manifest = {
        "model_checkpoint": args.model_checkpoint,
        "config": args.config,
        "registry": args.registry,
        "adsorption_data": args.adsorption_data,
        "graph_dir": args.graph_dir,
        "splits_file": args.splits_file,
        "output_dir": str(out_dir),
        "device": str(device),
        "seed": args.seed,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved to: {out_dir}")
    print("Next: python scripts/07_validate.py")


if __name__ == "__main__":
    main()