#!/usr/bin/env python3
"""
08_active_learning_loop.py — Practical active learning loop for UC-TPNO.

This version fixes the broken imports/API mismatches and turns the file into
a working orchestration loop that:

1. Loads or bootstraps a model
2. Scores candidate MOFs by predictive uncertainty / UCB
3. Runs new GCMC simulations for selected MOFs
4. Appends successful points to the adsorption training data
5. Retrains the model
6. Repeats for multiple iterations

Notes
-----
- This is a WORKING active-learning loop, not a full BoTorch/qEHVI pipeline.
- It uses model uncertainty on a small screening condition grid.
- It uses the corrected runner/model/trainer stack already fixed in this project.

Usage
-----
python scripts/08_active_learning_loop.py \
    --model-checkpoint experiments/run_001/final_model.pt \
    --registry data/processed/mof_registry.parquet \
    --adsorption-data data/processed/adsorption/adsorption_training.parquet \
    --graph-dir data/processed/graphs \
    --cif-dir data/intermediate/cifs_sanitized \
    --output-dir experiments/active_learning
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch_geometric.data import Batch

from src.data.datasets.adsorption_dataset import AdsorptionDataset
from src.data.datasets.splitter import DataSplitter
from src.models.encoder.adapter import EncoderAdapter
from src.models.operator.tpno import ThermodynamicPotentialNO, TPNOConfig
from src.models.operator.losses import ThermodynamicLoss, LossConfig
from src.simulation.gcmc.parser import GCMCRunner, GCMCConfig
from src.training.trainer import TPNOTrainer, TrainConfig
from src.utils.chemistry import (
    mixture_pressure_to_chemical_potentials,
    relative_humidity_to_mole_fraction,
)
from src.utils.reproducibility import set_seed


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

SCREEN_TEMPERATURES = [313.15]        # K
SCREEN_PRESSURES = [0.1, 1.0, 10.0]  # bar
SCREEN_RHS = [0.00, 0.10]            # relative humidity
DRY_CO2_FRACTION = 0.15


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
# Condition helpers
# ---------------------------------------------------------------------

def build_flue_gas_composition(
    rh: float,
    temperature: float,
    total_pressure: float,
    y_co2_dry: float = DRY_CO2_FRACTION,
) -> Optional[Dict[str, float]]:
    """
    Convert RH -> humid flue-gas mole fractions.
    """
    if total_pressure <= 0:
        return None

    y_h2o = float(
        relative_humidity_to_mole_fraction(
            rh,
            temperature,
            total_pressure=total_pressure,
            saturation_method="antoine",
        )
    )

    if not (0.0 <= y_h2o < 1.0):
        return None

    y_dry = 1.0 - y_h2o
    y_co2 = y_co2_dry * y_dry
    y_n2 = y_dry - y_co2

    if y_co2 < 0.0 or y_n2 < 0.0:
        return None

    comp = {"CO2": y_co2, "N2": y_n2, "H2O": y_h2o}
    total = sum(comp.values())
    if total <= 0.0:
        return None

    return {k: v / total for k, v in comp.items()}


def build_screening_conditions(
    temperatures: Sequence[float] = SCREEN_TEMPERATURES,
    pressures: Sequence[float] = SCREEN_PRESSURES,
    rhs: Sequence[float] = SCREEN_RHS,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Return:
      conditions_array : [P, 4] = [mu_CO2, mu_N2, mu_H2O, T]
      condition_meta   : list of dicts with T, P, RH, composition
    """
    rows: List[List[float]] = []
    meta: List[Dict[str, Any]] = []

    for T in temperatures:
        for P in pressures:
            for rh in rhs:
                comp = build_flue_gas_composition(rh, T, P)
                if comp is None:
                    continue

                mus = mixture_pressure_to_chemical_potentials(
                    y=comp,
                    pressure=P,
                    temperature=T,
                )

                rows.append([
                    float(mus["CO2"]),
                    float(mus["N2"]),
                    float(mus["H2O"]),
                    float(T),
                ])
                meta.append(
                    {
                        "temperature": float(T),
                        "pressure": float(P),
                        "relative_humidity": float(rh),
                        "composition": comp,
                    }
                )

    if not rows:
        raise RuntimeError("No valid screening conditions could be constructed.")

    return np.asarray(rows, dtype=np.float32), meta


# ---------------------------------------------------------------------
# Candidate discovery / scoring
# ---------------------------------------------------------------------

def discover_candidate_cifs(
    cif_dir: Path,
    graph_dir: Path,
    trained_mof_ids: Sequence[str],
) -> List[Path]:
    """
    Candidate CIFs must:
    - exist under cif_dir
    - have a matching graph .pt file
    - not already appear in adsorption training data
    """
    trained = set(trained_mof_ids)
    graph_mofs = {p.stem for p in graph_dir.glob("*.pt")}

    candidates = []
    for cif in sorted(cif_dir.rglob("*.cif")):
        if cif.stem in trained:
            continue
        if cif.stem not in graph_mofs:
            continue
        candidates.append(cif)
    return candidates


def load_graph_for_mof(graph_dir: Path, mof_id: str):
    path = graph_dir / f"{mof_id}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Graph file missing for MOF '{mof_id}': {path}")
    return torch.load(str(path), weights_only=False)


@torch.no_grad()
def score_candidates_with_model(
    model: ThermodynamicPotentialNO,
    candidate_cifs: List[Path],
    graph_dir: Path,
    device: torch.device,
    acquisition: str = "ucb",
    beta: float = 1.0,
    candidate_batch_size: int = 32,
) -> pd.DataFrame:
    """
    Score candidate MOFs on a fixed screening grid.

    Acquisition choices:
      - random
      - uncertainty
      - exploitation
      - ucb
    """
    model.eval()
    cond_np, _ = build_screening_conditions()
    cond_base = torch.tensor(cond_np, dtype=torch.float32, device=device)  # [P, 4]

    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(12345)

    for start in range(0, len(candidate_cifs), max(candidate_batch_size, 1)):
        batch_cifs = candidate_cifs[start:start + max(candidate_batch_size, 1)]
        graph_list = [load_graph_for_mof(graph_dir, cif.stem) for cif in batch_cifs]
        graph_batch = Batch.from_data_list(graph_list).to(device)

        B = len(batch_cifs)
        P = cond_base.shape[0]
        conditions = cond_base.unsqueeze(0).expand(B, P, -1).contiguous()

        out = model(
            graph_batch,
            conditions,
            return_uncertainty=True,
            return_potential=False,
            return_hessian=False,
        )

        q_pred = out["q_pred"].detach().cpu().numpy()   # [B, P, 3]
        sigma = out.get("sigma", None)
        if sigma is not None:
            sigma_np = sigma.detach().cpu().numpy()
        else:
            sigma_np = np.full_like(q_pred, 0.1, dtype=np.float32)

        for i, cif in enumerate(batch_cifs):
            mean_q_co2 = float(np.mean(q_pred[i, :, 0]))
            mean_sigma_co2 = float(np.mean(sigma_np[i, :, 0]))
            max_sigma_co2 = float(np.max(sigma_np[i, :, 0]))

            if acquisition == "uncertainty":
                score = mean_sigma_co2
            elif acquisition == "exploitation":
                score = mean_q_co2
            elif acquisition == "random":
                score = float(rng.random())
            else:  # default ucb
                score = mean_q_co2 + beta * mean_sigma_co2

            rows.append(
                {
                    "mof_id": cif.stem,
                    "cif_path": str(cif),
                    "score": float(score),
                    "mean_q_CO2": mean_q_co2,
                    "mean_sigma_CO2": mean_sigma_co2,
                    "max_sigma_CO2": max_sigma_co2,
                }
            )

        logger.info(
            "Candidate scoring progress: %d/%d",
            min(start + len(batch_cifs), len(candidate_cifs)),
            len(candidate_cifs),
        )

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------

def run_selected_gcmc(
    selected_cifs: List[Path],
    gcmc_runner: GCMCRunner,
    keep_workspaces: bool,
) -> List[Dict[str, Any]]:
    """
    Simulate selected MOFs on the fixed screening condition grid.
    """
    _, condition_meta = build_screening_conditions()

    all_results: List[Dict[str, Any]] = []
    total_jobs = len(selected_cifs) * len(condition_meta)
    job_counter = 0

    for cif in selected_cifs:
        for meta in condition_meta:
            job_counter += 1
            T = meta["temperature"]
            P = meta["pressure"]
            rh = meta["relative_humidity"]
            comp = meta["composition"]

            print(
                f"  GCMC [{job_counter}/{total_jobs}] {cif.stem} | "
                f"T={T:.2f} K, P={P:.4g} bar, RH={rh:.2%}"
            )

            result = gcmc_runner.run_single(
                mof_cif=cif,
                temperature=T,
                pressure=P,
                composition=comp,
                clean_after=not keep_workspaces,
            )
            result["relative_humidity"] = rh
            all_results.append(result)

    return all_results


def results_to_adsorption_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert successful GCMC results into adsorption_training.parquet rows.
    """
    rows: List[Dict[str, Any]] = []

    for r in results:
        if not r.get("success", False):
            continue

        T = float(r["temperature"])
        P = float(r["pressure"])
        rh = float(r.get("relative_humidity", 0.0))
        comp = r.get("composition", {})

        mus = mixture_pressure_to_chemical_potentials(
            y=comp,
            pressure=P,
            temperature=T,
        )

        rows.append(
            {
                "mof_id": r["mof_id"],
                "temperature": T,
                "pressure": P,
                "relative_humidity": rh,
                "mu_CO2": float(mus["CO2"]),
                "mu_N2": float(mus["N2"]),
                "mu_H2O": float(mus["H2O"]),
                "T": T,
                "y_CO2": float(comp.get("CO2", 0.0)),
                "y_N2": float(comp.get("N2", 0.0)),
                "y_H2O": float(comp.get("H2O", 0.0)),
                "co2_loading_molkg": float(r.get("loadings", {}).get("CO2", 0.0)),
                "n2_loading_molkg": float(r.get("loadings", {}).get("N2", 0.0)),
                "h2o_loading_molkg": float(r.get("loadings", {}).get("H2O", 0.0)),
                "fidelity": "gcmc_al",
            }
        )

    return rows


# ---------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------

def prepare_indices(
    dataset: AdsorptionDataset,
    splits_file: Optional[str],
    seed: int,
) -> Tuple[List[int], List[int], List[int], Dict[str, List[str]]]:
    """
    If a splits file exists:
      - keep existing val/test fixed
      - assign any new MOFs to train
    Otherwise create random splits.
    """
    if splits_file is not None and Path(splits_file).exists():
        with open(splits_file, "r", encoding="utf-8") as f:
            splits = json.load(f)

        train_ids = set(splits.get("train", []))
        val_ids = set(splits.get("val", []))
        test_ids = set(splits.get("test", []))

        # New MOFs not listed in val/test automatically go to training
        train_idx, val_idx, test_idx = [], [], []
        for i, mof_id in enumerate(dataset.mof_ids):
            if mof_id in val_ids:
                val_idx.append(i)
            elif mof_id in test_ids:
                test_idx.append(i)
            else:
                train_idx.append(i)

        final_splits = {
            "train": [dataset.mof_ids[i] for i in train_idx],
            "val": [dataset.mof_ids[i] for i in val_idx],
            "test": [dataset.mof_ids[i] for i in test_idx],
        }
        return train_idx, val_idx, test_idx, final_splits

    splitter = DataSplitter(
        method="random",
        test_size=0.1,
        val_size=0.1,
        random_state=seed,
    )
    train_idx, val_idx, test_idx = splitter.split(dataset.mof_ids)
    final_splits = {
        "train": [dataset.mof_ids[i] for i in train_idx],
        "val": [dataset.mof_ids[i] for i in val_idx],
        "test": [dataset.mof_ids[i] for i in test_idx],
    }
    return train_idx, val_idx, test_idx, final_splits


def train_model_once(
    config_path: str,
    registry: str,
    adsorption_path: str,
    graph_dir: str,
    output_dir: Path,
    seed: int,
    device: torch.device,
    checkpoint_init: Optional[str] = None,
    splits_file: Optional[str] = None,
    epochs: int = 30,
    batch_size: int = 1,
    num_workers: int = 0,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Train or fine-tune the TPNO model on the current adsorption dataset.
    """
    cfg = load_yaml_config(config_path)
    model_cfg = dict(cfg.get("model", {}))
    train_cfg = dict(cfg.get("training", {}))
    pin_memory = device.type == "cuda"

    dataset = AdsorptionDataset(
        registry_path=registry,
        adsorption_path=adsorption_path,
        graph_dir=graph_dir,
    )

    train_idx, val_idx, test_idx, splits = prepare_indices(
        dataset=dataset,
        splits_file=splits_file,
        seed=seed,
    )

    if len(train_idx) == 0:
        raise RuntimeError("Training split is empty during active-learning retraining.")
    if len(val_idx) == 0:
        raise RuntimeError("Validation split is empty during active-learning retraining.")

    train_loader = dataset.get_dataloader(
        indices=train_idx,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = dataset.get_dataloader(
        indices=val_idx,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = build_model_from_config(model_cfg, device=device)
    if checkpoint_init is not None and Path(checkpoint_init).exists():
        logger.info("Warm-starting retraining from checkpoint: %s", checkpoint_init)
        load_checkpoint_into_model(model, checkpoint_init, device=device)

    loss_cfg = LossConfig(
        lambda_data=float(train_cfg.get("lambda_data", 1.0)),
        lambda_hessian=float(train_cfg.get("lambda_hessian", 0.1)),
        lambda_monotonic=float(train_cfg.get("lambda_monotonic", 0.1)),
        lambda_henry=float(train_cfg.get("lambda_henry", 0.01)),
        lambda_competition=float(train_cfg.get("lambda_competition", 0.05)),
        lambda_gibbs_duhem=float(train_cfg.get("lambda_gibbs_duhem", 0.0)),
        henry_mu_threshold=float(train_cfg.get("henry_mu_threshold", -5.0)),
        use_nll=bool(train_cfg.get("use_nll", True)),
    )
    criterion = ThermodynamicLoss(config=loss_cfg)

    trainer_cfg = TrainConfig(
        n_epochs=int(epochs),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        optimizer=train_cfg.get("optimizer", "adamw"),
        scheduler=train_cfg.get("scheduler", "cosine_warm_restarts"),
        scheduler_T0=int(train_cfg.get("scheduler_T0", 10)),
        scheduler_T_mult=int(train_cfg.get("scheduler_T_mult", 2)),
        step_size=int(train_cfg.get("step_size", 30)),
        step_gamma=float(train_cfg.get("step_gamma", 0.5)),
        warmup_epochs=int(train_cfg.get("warmup_epochs", 5)),
        physics_warmup=int(train_cfg.get("physics_warmup", 20)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        use_amp=bool(train_cfg.get("use_amp", False)),
        early_stopping=bool(train_cfg.get("early_stopping", True)),
        patience=int(train_cfg.get("patience", 20)),
        checkpoint_dir=str(output_dir / "checkpoints"),
        checkpoint_freq=int(train_cfg.get("checkpoint_freq", 10)),
        use_wandb=False,
        use_tensorboard=False,
        log_interval=int(train_cfg.get("log_interval", 10)),
    )

    print("\n=== Retraining / Fine-tuning Model ===")
    print(f"Train MOFs: {len(train_idx)} | Val MOFs: {len(val_idx)} | Test MOFs: {len(test_idx)}")
    print(f"Epochs: {epochs} | Batch size: {batch_size}")

    trainer = TPNOTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_cfg,
        criterion=criterion,
        device=device,
    )
    history = trainer.fit(n_epochs=epochs)

    final_model_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=float)

    with open(output_dir / "splits_used.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    best_metrics = trainer.get_best_metrics()
    if best_metrics is not None:
        with open(output_dir / "best_metrics.json", "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2, default=float)

    return str(final_model_path), history, splits


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Active learning loop for UC-TPNO")
    parser.add_argument("--model-checkpoint", default=None)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--adsorption-data", required=True)
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--cif-dir", required=True)
    parser.add_argument("--splits-file", default=None)
    parser.add_argument("--output-dir", default="experiments/active_learning")

    parser.add_argument("--n-iterations", type=int, default=5)
    parser.add_argument("--batch-size-al", type=int, default=5, help="MOFs selected per AL iteration")
    parser.add_argument("--candidate-batch-size", type=int, default=32, help="Batch size for candidate scoring")
    parser.add_argument("--retrain-epochs", type=int, default=20)
    parser.add_argument("--retrain-batch-size", type=int, default=1, help="Use 1 to avoid padding contamination")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--acquisition",
        default="ucb",
        choices=["ucb", "uncertainty", "exploitation", "random"],
    )
    parser.add_argument("--beta", type=float, default=1.0, help="UCB beta")

    parser.add_argument("--raspa-path", default="simulate")
    parser.add_argument("--n-cycles", type=int, default=100_000)
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument("--bootstrap-if-missing", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_dir = Path(args.graph_dir)
    cif_dir = Path(args.cif_dir)

    print("=" * 80)
    print("ACTIVE LEARNING LOOP")
    print(f"Device:            {device}")
    print(f"Acquisition:       {args.acquisition}")
    print(f"Iterations:        {args.n_iterations}")
    print(f"Batch size (AL):   {args.batch_size_al}")
    print(f"Retrain epochs:    {args.retrain_epochs}")
    print(f"Output dir:        {out_dir}")
    print("=" * 80)

    # ── Load current adsorption data ───────────────────────────
    ads_path = Path(args.adsorption_data)
    if not ads_path.exists():
        raise FileNotFoundError(f"Adsorption data not found: {ads_path}")

    ads_df = pd.read_parquet(ads_path) if ads_path.suffix == ".parquet" else pd.read_csv(ads_path)
    trained_mofs = set(ads_df["mof_id"].astype(str).unique())

    candidate_cifs = discover_candidate_cifs(
        cif_dir=cif_dir,
        graph_dir=graph_dir,
        trained_mof_ids=sorted(trained_mofs),
    )

    print(f"Current training MOFs: {len(trained_mofs)}")
    print(f"Candidate MOFs:        {len(candidate_cifs)}")

    if len(candidate_cifs) == 0:
        print("No candidate MOFs remain. Exiting.")
        return

    # Copy starting adsorption file into AL directory
    working_ads_path = out_dir / "adsorption_iter00.parquet"
    ads_df.to_parquet(working_ads_path, index=False)

    # ── Bootstrap / load model ─────────────────────────────────
    current_model_ckpt = args.model_checkpoint
    if current_model_ckpt is None or not Path(current_model_ckpt).exists():
        print("\nNo existing model checkpoint found.")
        if args.bootstrap_if_missing:
            print("Bootstrapping a model from the current adsorption dataset...")
            bootstrap_dir = out_dir / "bootstrap_model"
            bootstrap_dir.mkdir(parents=True, exist_ok=True)

            current_model_ckpt, _, _ = train_model_once(
                config_path=args.config,
                registry=args.registry,
                adsorption_path=str(working_ads_path),
                graph_dir=args.graph_dir,
                output_dir=bootstrap_dir,
                seed=args.seed,
                device=device,
                checkpoint_init=None,
                splits_file=args.splits_file,
                epochs=args.retrain_epochs,
                batch_size=args.retrain_batch_size,
                num_workers=args.num_workers,
            )
        else:
            print("No checkpoint and bootstrap disabled. Exiting.")
            return

    gcmc_runner = GCMCRunner(
        GCMCConfig(
            raspa_path=args.raspa_path,
            work_dir=str(out_dir / "gcmc_work"),
            n_cycles=args.n_cycles,
        )
    )

    history: List[Dict[str, Any]] = []
    failed_mofs_all: List[Dict[str, Any]] = []

    # ── Main AL loop ────────────────────────────────────────────
    for iteration in range(1, args.n_iterations + 1):
        if len(candidate_cifs) == 0:
            print("\nAll candidate MOFs exhausted.")
            break

        iter_dir = out_dir / f"iter_{iteration:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"ACTIVE LEARNING ITERATION {iteration}/{args.n_iterations}")
        print("=" * 80)

        # Load current model for scoring
        cfg = load_yaml_config(args.config)
        model_cfg = dict(cfg.get("model", {}))

        model = build_model_from_config(model_cfg, device=device)
        load_checkpoint_into_model(model, current_model_ckpt, device=device)
        model.eval()

        # 1) Score candidates
        print("\n=== Scoring Candidate MOFs ===")
        score_df = score_candidates_with_model(
            model=model,
            candidate_cifs=candidate_cifs,
            graph_dir=graph_dir,
            device=device,
            acquisition=args.acquisition,
            beta=args.beta,
            candidate_batch_size=args.candidate_batch_size,
        )
        score_df.to_csv(iter_dir / "candidate_scores.csv", index=False)

        n_select = min(args.batch_size_al, len(score_df))
        selected_ids = score_df.head(n_select)["mof_id"].tolist()
        selected_map = {cif.stem: cif for cif in candidate_cifs}
        selected_cifs = [selected_map[mof_id] for mof_id in selected_ids]

        print(f"Selected {len(selected_cifs)} MOFs:")
        for rank, mof_id in enumerate(selected_ids, start=1):
            row = score_df.iloc[rank - 1]
            print(
                f"  {rank:02d}. {mof_id} | "
                f"score={row['score']:.4f} | "
                f"mean_q_CO2={row['mean_q_CO2']:.4f} | "
                f"mean_sigma_CO2={row['mean_sigma_CO2']:.4f}"
            )

        with open(iter_dir / "selected_mofs.json", "w", encoding="utf-8") as f:
            json.dump(selected_ids, f, indent=2)

        # 2) Run new GCMC simulations
        print("\n=== Running GCMC on Selected MOFs ===")
        sim_results = run_selected_gcmc(
            selected_cifs=selected_cifs,
            gcmc_runner=gcmc_runner,
            keep_workspaces=args.keep_workspaces,
        )

        with open(iter_dir / "raw_simulation_results.json", "w", encoding="utf-8") as f:
            json.dump(sim_results, f, indent=2, default=str)

        success_count = sum(bool(r.get("success", False)) for r in sim_results)
        print(f"Successful GCMC jobs: {success_count}/{len(sim_results)}")

        # Record failures
        for r in sim_results:
            if not r.get("success", False):
                failed_mofs_all.append(
                    {
                        "iteration": iteration,
                        "mof_id": r.get("mof_id"),
                        "temperature": r.get("temperature"),
                        "pressure": r.get("pressure"),
                        "error": r.get("error"),
                    }
                )

        # 3) Append successful points
        print("\n=== Updating Adsorption Dataset ===")
        new_rows = results_to_adsorption_rows(sim_results)

        n_rows_added = 0
        if new_rows:
            new_df = pd.DataFrame(new_rows)

            before = len(ads_df)
            ads_df = pd.concat([ads_df, new_df], ignore_index=True)

            dedup_subset = ["mof_id", "mu_CO2", "mu_N2", "mu_H2O", "T"]
            ads_df = ads_df.drop_duplicates(subset=dedup_subset, keep="last").reset_index(drop=True)

            after = len(ads_df)
            n_rows_added = after - before

            working_ads_path = out_dir / f"adsorption_iter{iteration:02d}.parquet"
            ads_df.to_parquet(working_ads_path, index=False)

            print(f"Added {n_rows_added} new adsorption rows.")
            print(f"Total adsorption rows: {len(ads_df)}")
        else:
            print("No successful new adsorption rows were produced.")
            working_ads_path = out_dir / f"adsorption_iter{iteration:02d}.parquet"
            ads_df.to_parquet(working_ads_path, index=False)

        # 4) Retrain / fine-tune
        retrain_metrics: Dict[str, Any] = {}
        if n_rows_added > 0:
            model_dir = iter_dir / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            current_model_ckpt, retrain_history, splits_used = train_model_once(
                config_path=args.config,
                registry=args.registry,
                adsorption_path=str(working_ads_path),
                graph_dir=args.graph_dir,
                output_dir=model_dir,
                seed=args.seed + iteration,
                device=device,
                checkpoint_init=current_model_ckpt,
                splits_file=args.splits_file,
                epochs=args.retrain_epochs,
                batch_size=args.retrain_batch_size,
                num_workers=args.num_workers,
            )

            if retrain_history:
                retrain_metrics = dict(retrain_history[-1])
        else:
            print("Skipping retraining because no new successful data were added.")

        # 5) Update candidate pool
        selected_set = {c.stem for c in selected_cifs}
        candidate_cifs = [c for c in candidate_cifs if c.stem not in selected_set]

        trained_mofs.update(selected_set)

        iter_record = {
            "iteration": iteration,
            "selected_mofs": selected_ids,
            "n_selected": len(selected_ids),
            "n_candidate_remaining": len(candidate_cifs),
            "n_sim_jobs": len(sim_results),
            "n_sim_success": success_count,
            "n_rows_added": n_rows_added,
            "n_total_rows": int(len(ads_df)),
            "current_model_checkpoint": current_model_ckpt,
            "train_summary": {
                k: float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in retrain_metrics.items()
                if isinstance(v, (int, float, np.floating))
            },
        }
        history.append(iter_record)

        with open(iter_dir / "iteration_summary.json", "w", encoding="utf-8") as f:
            json.dump(iter_record, f, indent=2)

        print("\nIteration summary:")
        print(f"  Selected MOFs:        {len(selected_ids)}")
        print(f"  Successful GCMC jobs: {success_count}/{len(sim_results)}")
        print(f"  Rows added:           {n_rows_added}")
        print(f"  Candidate remaining:  {len(candidate_cifs)}")

    # ── Save final outputs ──────────────────────────────────────
    with open(out_dir / "al_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    ads_df.to_parquet(out_dir / "adsorption_final.parquet", index=False)

    if failed_mofs_all:
        pd.DataFrame(failed_mofs_all).to_csv(out_dir / "failed_simulations.csv", index=False)

    manifest = {
        "config": args.config,
        "registry": args.registry,
        "graph_dir": args.graph_dir,
        "cif_dir": args.cif_dir,
        "initial_adsorption_data": args.adsorption_data,
        "final_adsorption_data": str(out_dir / "adsorption_final.parquet"),
        "final_model_checkpoint": current_model_ckpt,
        "n_iterations_requested": args.n_iterations,
        "n_iterations_completed": len(history),
        "seed": args.seed,
        "device": str(device),
        "acquisition": args.acquisition,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("ACTIVE LEARNING COMPLETE")
    print(f"Iterations completed: {len(history)}")
    print(f"Final adsorption file: {out_dir / 'adsorption_final.parquet'}")
    print(f"Final model:           {current_model_ckpt}")
    print(f"Next: python scripts/09_benchmark.py")
    print("=" * 80)


if __name__ == "__main__":
    main()