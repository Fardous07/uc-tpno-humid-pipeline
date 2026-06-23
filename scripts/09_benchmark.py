#!/usr/bin/env python3
"""
09_benchmark.py — Benchmark UC-TPNO vs baselines + process optimisation.

This version fixes:
- wrong model/encoder imports
- checkpoint loading mismatches
- padded-batch contamination by using pointwise, mask-free collection
- Pareto analysis at the MOF level instead of per-condition points
- process optimisation using the actual PVSA simulator + KPI calculator

Runs
----
1. Benchmark against:
   - TPNO
   - Linear baseline
   - MLP baseline (optional)
   - GCMC oracle reference
2. Pairwise statistical significance tests
3. PVSA process simulation for test-set MOFs
4. Pareto front analysis using MOF-level objectives
5. Benchmark / Pareto plots

Usage
-----
python scripts/09_benchmark.py \
    --model-checkpoint experiments/run_001/final_model.pt \
    --registry data/processed/mof_registry.parquet \
    --adsorption-data data/processed/adsorption/adsorption_training.parquet \
    --graph-dir data/processed/graphs \
    --splits-file experiments/run_001/splits.json \
    --output-dir results/benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datasets.adsorption_dataset import AdsorptionDataset
from src.models.encoder.adapter import EncoderAdapter
from src.models.operator.tpno import ThermodynamicPotentialNO, TPNOConfig
from src.evaluation.benchmarking import Benchmarker
from src.evaluation.metrics import (
    compute_pareto_metrics,
    compute_regression_metrics,
    pareto_front_indices,
)
from src.evaluation.visualizer import plot_benchmark, plot_pareto
from src.models.process.pvsa import PVSASimulator
from src.models.process.kpi import KPICalculator
from src.utils.chemistry import mixture_pressure_to_chemical_potentials
from src.utils.reproducibility import set_seed


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Config / model helpers
# ---------------------------------------------------------------------

def load_yaml_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_splits(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model_from_config(
    model_cfg: Dict[str, Any],
    device: torch.device,
) -> ThermodynamicPotentialNO:
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
    ckpt = torch.load(checkpoint_path, map_location=device)

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


# ---------------------------------------------------------------------
# Feature / pointwise collection
# ---------------------------------------------------------------------

def _safe_tensor_stats(x: Optional[torch.Tensor], n_dims: int = 4) -> np.ndarray:
    """
    Fixed-size summary statistics from a node feature tensor.
    """
    if x is None or not isinstance(x, torch.Tensor) or x.numel() == 0:
        return np.zeros(2 * n_dims, dtype=np.float32)

    x = x.detach().cpu().float()
    if x.ndim == 1:
        x = x.unsqueeze(-1)

    d = min(x.shape[1], n_dims)
    mean = x[:, :d].mean(dim=0).numpy()
    std = x[:, :d].std(dim=0).numpy()

    if d < n_dims:
        mean = np.pad(mean, (0, n_dims - d))
        std = np.pad(std, (0, n_dims - d))

    return np.concatenate([mean, std]).astype(np.float32)


def extract_graph_descriptor(graph) -> np.ndarray:
    """
    Build a simple fixed-length descriptor for linear/MLP baselines.
    """
    num_nodes = float(getattr(graph, "num_nodes", 0) or 0)
    if hasattr(graph, "edge_index") and graph.edge_index is not None:
        num_edges = float(graph.edge_index.shape[1])
    else:
        num_edges = 0.0

    x_stats = _safe_tensor_stats(getattr(graph, "x", None), n_dims=4)

    pos = getattr(graph, "pos", None)
    if pos is not None and isinstance(pos, torch.Tensor) and pos.numel() > 0:
        pos = pos.detach().cpu().float()
        mean_norm = float(torch.norm(pos, dim=1).mean().item())
        std_norm = float(torch.norm(pos, dim=1).std().item())
    else:
        mean_norm = 0.0
        std_norm = 0.0

    descriptor = np.concatenate(
        [
            np.array([num_nodes, num_edges, mean_norm, std_norm], dtype=np.float32),
            x_stats,
        ]
    )
    return descriptor.astype(np.float32)


def _sigma_fallback(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    resid = np.abs(y_true - y_pred)
    scale = float(np.std(resid))
    scale = max(scale, 1e-3)
    return np.full_like(y_pred, scale, dtype=np.float64)


@torch.no_grad()
def collect_pointwise_data(
    dataset: AdsorptionDataset,
    indices: List[int],
    device: torch.device,
    model: Optional[torch.nn.Module] = None,
    log_prefix: str = "DATA",
    log_interval: int = 25,
) -> Dict[str, Any]:
    """
    Collect pointwise rows from dataset samples without padding.

    Returns
    -------
    Dict containing:
        X        : [N, d] baseline features
        y_true   : [N, 3]
        y_pred   : [N, 3] if model provided
        y_std    : [N, 3] if model provided
        mof_ids  : [N]
        conditions : [N, 4]
    """
    X_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    pred_rows: List[np.ndarray] = []
    std_rows: List[np.ndarray] = []
    mof_ids: List[str] = []
    cond_rows: List[np.ndarray] = []

    if model is not None:
        model.eval()

    for count, idx in enumerate(indices, start=1):
        sample = dataset[idx]
        mof_id = sample["mof_id"]
        graph = sample["graphs"]
        conditions = sample["conditions"]          # [P, 4]
        loadings = sample["loadings"]              # [P, 3]

        graph_desc = extract_graph_descriptor(graph)
        cond_np = conditions.detach().cpu().numpy().astype(np.float32)
        y_np = loadings.detach().cpu().numpy().astype(np.float32)

        desc_rep = np.repeat(graph_desc[None, :], cond_np.shape[0], axis=0)
        X = np.concatenate([desc_rep, cond_np], axis=1)

        X_rows.append(X)
        y_rows.append(y_np)
        cond_rows.append(cond_np)
        mof_ids.extend([mof_id] * cond_np.shape[0])

        if model is not None:
            graph_batch = Batch.from_data_list([graph]).to(device)
            cond_t = conditions.unsqueeze(0).to(device)  # [1, P, 4]

            out = model(
                graph_batch,
                cond_t,
                return_uncertainty=True,
                return_potential=False,
                return_hessian=False,
            )
            y_pred = out["q_pred"][0].detach().cpu().numpy().astype(np.float64)

            sigma = out.get("sigma", None)
            if sigma is not None:
                y_std = sigma[0].detach().cpu().numpy().astype(np.float64)
            else:
                y_std = np.full_like(y_pred, 0.1, dtype=np.float64)

            pred_rows.append(y_pred)
            std_rows.append(np.maximum(y_std, 1e-6))

        if count == 1 or count % max(log_interval, 1) == 0:
            logger.info("%s progress: %d/%d MOFs", log_prefix, count, len(indices))

    result = {
        "X": np.concatenate(X_rows, axis=0),
        "y_true": np.concatenate(y_rows, axis=0),
        "conditions": np.concatenate(cond_rows, axis=0),
        "mof_ids": np.array(mof_ids, dtype=object),
    }

    if model is not None:
        y_pred = np.concatenate(pred_rows, axis=0)
        y_std = np.concatenate(std_rows, axis=0)

        if (not np.all(np.isfinite(y_std))) or np.any(y_std <= 0):
            y_std = _sigma_fallback(result["y_true"], y_pred)

        result["y_pred"] = y_pred
        result["y_std"] = y_std

    return result


# ---------------------------------------------------------------------
# Process / MOF-level helpers
# ---------------------------------------------------------------------

def make_model_isotherm_fn(
    model: ThermodynamicPotentialNO,
    graph,
    device: torch.device,
):
    """
    Wrap the trained TPNO model as an isotherm function:
        (y, P_total, T) -> loadings [3]
    """
    graph_batch = Batch.from_data_list([graph]).to(device)

    @torch.no_grad()
    def _isotherm_fn(y: np.ndarray, P_total: float, T: float) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if len(y) < 3:
            y_full = np.zeros(3, dtype=np.float64)
            y_full[: len(y)] = y
            y = y_full

        total = float(np.sum(y))
        if total <= 0:
            y = np.array([0.15, 0.75, 0.10], dtype=np.float64)
        else:
            y = y / total

        comp = {"CO2": float(y[0]), "N2": float(y[1]), "H2O": float(y[2])}
        mus = mixture_pressure_to_chemical_potentials(
            y=comp,
            pressure=float(P_total),
            temperature=float(T),
        )

        cond = torch.tensor(
            [[[mus["CO2"], mus["N2"], mus["H2O"], float(T)]]],
            dtype=torch.float32,
            device=device,
        )  # [1, 1, 4]

        out = model(
            graph_batch,
            cond,
            return_uncertainty=False,
            return_potential=False,
            return_hessian=False,
        )
        q = out["q_pred"][0, 0].detach().cpu().numpy().astype(np.float64)
        return q

    return _isotherm_fn


def run_process_screening(
    dataset: AdsorptionDataset,
    indices: List[int],
    model: ThermodynamicPotentialNO,
    device: torch.device,
    log_interval: int = 10,
) -> pd.DataFrame:
    """
    Run PVSA + KPI evaluation for each MOF in the given index set.
    """
    kpi_calc = KPICalculator()
    rows: List[Dict[str, Any]] = []

    for count, idx in enumerate(indices, start=1):
        sample = dataset[idx]
        mof_id = sample["mof_id"]
        graph = sample["graphs"]

        try:
            iso_fn = make_model_isotherm_fn(model, graph, device=device)
            pvsa = PVSASimulator(iso_fn, n_components=3)
            cycle = pvsa.run_cycle()
            kpis = kpi_calc.from_cycle_result(cycle)

            row = {
                "mof_id": mof_id,
                "valid_cycle": bool(getattr(cycle, "valid", True)),
                "cycle_message": str(getattr(cycle, "message", "")),
                **kpis,
            }
        except Exception as e:
            logger.exception("PVSA/KPI evaluation failed for %s", mof_id)
            row = {
                "mof_id": mof_id,
                "valid_cycle": False,
                "cycle_message": str(e),
                "delta_q_CO2": 0.0,
                "delta_q_N2": 0.0,
                "delta_q_H2O": 0.0,
                "selectivity_CO2_N2": 0.0,
                "regenerability": 0.0,
                "API": 0.0,
                "SSP": 0.0,
                "purity": 0.0,
                "recovery": 0.0,
                "productivity_mol_kg_s": 0.0,
                "energy_kJ_mol": 0.0,
                "energy_MJ_ton": 0.0,
                "capture_cost_USD_ton": float("inf"),
                "meets_purity": False,
                "meets_recovery": False,
                "meets_energy": False,
                "meets_all": False,
                "composite_score": 0.0,
            }

        rows.append(row)

        if count == 1 or count % max(log_interval, 1) == 0:
            logger.info("PVSA screening progress: %d/%d MOFs", count, len(indices))

    df = pd.DataFrame(rows).sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark UC-TPNO and process-optimize top MOFs")
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--adsorption-data", required=True)
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--splits-file", required=True)
    parser.add_argument("--output-dir", default="results/benchmark")

    parser.add_argument("--top-k", type=int, default=20, help="Top MOFs to highlight in process ranking")
    parser.add_argument("--skip-mlp", action="store_true", help="Skip MLP baseline if you want a faster benchmark")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BENCHMARK + PROCESS OPTIMISATION")
    print(f"Device:            {device}")
    print(f"Checkpoint:        {args.model_checkpoint}")
    print(f"Output dir:        {out_dir}")
    print(f"Top-K process:     {args.top_k}")
    print("=" * 80)

    # ── Data ─────────────────────────────────────────────
    print("\n=== Loading Data ===")
    dataset = AdsorptionDataset(
        registry_path=args.registry,
        adsorption_path=args.adsorption_data,
        graph_dir=args.graph_dir,
    )

    splits = load_splits(args.splits_file)
    train_ids = set(splits["train"])
    test_ids = set(splits["test"])

    train_idx = [i for i, m in enumerate(dataset.mof_ids) if m in train_ids]
    test_idx = [i for i, m in enumerate(dataset.mof_ids) if m in test_ids]

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError(
            f"Invalid split sizes for benchmarking: train={len(train_idx)}, test={len(test_idx)}"
        )

    print(f"Train MOFs: {len(train_idx)}")
    print(f"Test MOFs:  {len(test_idx)}")

    # ── Model ────────────────────────────────────────────
    print("\n=== Loading TPNO Model ===")
    cfg = load_yaml_config(args.config)
    model_cfg = dict(cfg.get("model", {}))

    model = build_model_from_config(model_cfg, device=device)
    load_checkpoint_into_model(model, args.model_checkpoint, device=device)
    model.eval()

    # ── Collect pointwise train/test data ────────────────
    print("\n=== Collecting Pointwise Train/Test Data ===")
    train_data = collect_pointwise_data(
        dataset=dataset,
        indices=train_idx,
        device=device,
        model=None,
        log_prefix="TRAIN",
        log_interval=args.log_interval,
    )
    test_data = collect_pointwise_data(
        dataset=dataset,
        indices=test_idx,
        device=device,
        model=model,
        log_prefix="TEST",
        log_interval=args.log_interval,
    )

    X_train = train_data["X"]
    y_train = train_data["y_true"]
    X_test = test_data["X"]
    y_test = test_data["y_true"]
    tpno_pred = test_data["y_pred"]
    tpno_std = test_data["y_std"]

    print(f"Train point rows: {len(X_train)}")
    print(f"Test point rows:  {len(X_test)}")

    # ── 1. Benchmark ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. BENCHMARKING")
    print("=" * 60)

    component_names = ["CO2", "N2", "H2O"]
    benchmarker = Benchmarker(y_true=y_test, component_names=component_names)

    benchmarker.add_model("TPNO", lambda X: (tpno_pred, tpno_std), has_uq=True)
    benchmarker.add_baseline_linear(X_train, y_train)

    if not args.skip_mlp:
        try:
            benchmarker.add_baseline_mlp(X_train, y_train)
        except Exception as e:
            logger.warning("MLP baseline failed to initialize/train and will be skipped: %s", e)

    # Oracle reference using actual GCMC labels on the test set
    benchmarker.add_model("GCMC_oracle", lambda X: y_test, has_uq=False)

    results = benchmarker.run(X_test)
    table = benchmarker.comparison_table(results)
    tests = benchmarker.pairwise_tests(results)
    summary = benchmarker.summary(results)

    print("\nComparison table:")
    print(f"{'Model':<15} {'R2':>10} {'MAE':>10} {'RMSE':>10} {'Time(s)':>10}")
    print("-" * 60)
    for name, metrics in table.items():
        print(
            f"{name:<15} "
            f"{metrics.get('r2', 0.0):>10.4f} "
            f"{metrics.get('mae', 0.0):>10.4f} "
            f"{metrics.get('rmse', 0.0):>10.4f} "
            f"{metrics.get('time_s', 0.0):>10.2f}"
        )

    print("\nPairwise tests:")
    for key, result in tests.items():
        print(
            f"  {key}: p={result['p_value']:.4g}, "
            f"significant={result['significant']}, "
            f"better={result['better_model']}"
        )

    tpno_reg = compute_regression_metrics(
        y_true=y_test,
        y_pred=tpno_pred,
        component_names=component_names,
        prefix="tpno_",
    )
    print("\nTPNO detailed regression metrics:")
    for k, v in sorted(tpno_reg.items()):
        print(f"  {k}: {v:.6f}")

    # ── 2. Process screening + Pareto analysis ───────────
    print("\n" + "=" * 60)
    print("2. PVSA PROCESS SCREENING + PARETO ANALYSIS")
    print("=" * 60)

    process_df = run_process_screening(
        dataset=dataset,
        indices=test_idx,
        model=model,
        device=device,
        log_interval=max(1, args.log_interval // 2),
    )

    if process_df.empty:
        raise RuntimeError("Process screening produced no rows.")

    valid_process_df = process_df.copy()
    valid_process_df["selectivity_CO2_N2"] = valid_process_df["selectivity_CO2_N2"].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    objectives = valid_process_df[["selectivity_CO2_N2", "delta_q_CO2"]].to_numpy(dtype=np.float64)
    pareto_idx = pareto_front_indices(objectives)
    pareto_metrics = compute_pareto_metrics(objectives)

    valid_process_df["is_pareto"] = False
    valid_process_df.loc[pareto_idx, "is_pareto"] = True

    print(f"Pareto front size: {len(pareto_idx)}/{len(valid_process_df)} MOFs")
    for k, v in sorted(pareto_metrics.items()):
        if isinstance(v, (float, np.floating)):
            print(f"  {k}: {float(v):.6f}")
        else:
            print(f"  {k}: {v}")

    # ── 3. Top-K ranking ─────────────────────────────────
    print("\n" + "=" * 60)
    print(f"3. TOP-{args.top_k} PROCESS RANKING")
    print("=" * 60)

    top_k = min(args.top_k, len(valid_process_df))
    top_df = valid_process_df.head(top_k).copy()

    print(f"{'Rank':<6} {'MOF':<16} {'Score':>9} {'Purity':>9} {'Recovery':>9} {'Energy':>10}")
    print("-" * 70)
    for _, row in top_df.iterrows():
        print(
            f"{int(row['rank']):<6} "
            f"{str(row['mof_id'])[:16]:<16} "
            f"{float(row['composite_score']):>9.4f} "
            f"{float(row['purity']):>9.4f} "
            f"{float(row['recovery']):>9.4f} "
            f"{float(row['energy_MJ_ton']):>10.2f}"
        )

    # ── 4. Figures ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. GENERATING FIGURES")
    print("=" * 60)

    plot_benchmark(
        table,
        metric="r2",
        title="Benchmark Comparison (R²)",
        save_path=out_dir / "benchmark_r2.png",
    )
    plot_benchmark(
        table,
        metric="rmse",
        title="Benchmark Comparison (RMSE)",
        save_path=out_dir / "benchmark_rmse.png",
    )
    plot_pareto(
        objectives,
        pareto_idx=pareto_idx,
        labels=("CO2/N2 Selectivity", "CO2 Working Capacity"),
        title="MOF Pareto Front",
        save_path=out_dir / "pareto_front.png",
    )

    # ── Save outputs ─────────────────────────────────────
    print("\n=== Saving Outputs ===")

    with open(out_dir / "benchmark_table.json", "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, default=float)

    with open(out_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    with open(out_dir / "significance_tests.json", "w", encoding="utf-8") as f:
        json.dump(tests, f, indent=2, default=float)

    with open(out_dir / "pareto_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in pareto_metrics.items()},
            f,
            indent=2,
        )

    with open(out_dir / "tpno_regression_metrics.json", "w", encoding="utf-8") as f:
        json.dump(tpno_reg, f, indent=2, default=float)

    process_df.to_csv(out_dir / "process_screening_all.csv", index=False)
    top_df.to_csv(out_dir / "process_topk.csv", index=False)

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test.npy", y_test)
    np.save(out_dir / "tpno_pred.npy", tpno_pred)
    np.save(out_dir / "tpno_std.npy", tpno_std)
    np.save(out_dir / "pareto_indices.npy", pareto_idx)

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
        "n_train_mofs": len(train_idx),
        "n_test_mofs": len(test_idx),
        "n_train_points": int(len(X_train)),
        "n_test_points": int(len(X_test)),
        "skip_mlp": bool(args.skip_mlp),
        "top_k": int(top_k),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print(f"Results saved to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()