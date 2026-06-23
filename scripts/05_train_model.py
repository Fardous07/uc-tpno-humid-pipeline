#!/usr/bin/env python3
"""
05_train_model.py — Train the UC-TPNO model.

This version is aligned with the repaired codebase and fixes:
1. Uses the real classes/APIs:
   - EncoderAdapter.from_config(...)
   - ThermodynamicPotentialNO
   - TrainConfig
   - trainer.fit(...)
2. Computes and sets input/output normalization statistics on the model.
3. Avoids callback/trainer conflicts by using only lightweight epoch-end callbacks.
4. Keeps terminal progress visible.
5. Supports existing split files whether they store MOF IDs or indices.

Usage
-----
python scripts/05_train_model.py \
    --registry data/processed/mof_registry.parquet \
    --adsorption-data data/processed/adsorption/adsorption_training.parquet \
    --graph-dir data/processed/graphs \
    --output-dir experiments/run_001

For more frequent batch logs:
python scripts/05_train_model.py ... --verbose-batches --log-interval 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datasets.adsorption_dataset import AdsorptionDataset
from src.data.datasets.splitter import DataSplitter
from src.models.encoder.adapter import EncoderAdapter
from src.models.operator.tpno import ThermodynamicPotentialNO, TPNOConfig
from src.models.operator.losses import ThermodynamicLoss, LossConfig
from src.training.trainer import TPNOTrainer, TrainConfig
from src.training.callbacks import CallbackRunner, MetricLogger, TimingCallback
from src.utils.reproducibility import set_seed


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

def configure_logging(verbose_batches: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    trainer_logger = logging.getLogger("src.training.trainer")
    trainer_logger.setLevel(logging.INFO if not verbose_batches else logging.INFO)

    logging.getLogger("src.training.callbacks").setLevel(logging.INFO)
    logging.getLogger("src.data.datasets.adsorption_dataset").setLevel(logging.INFO)


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ─────────────────────────────────────────────────────────────
# Split helpers
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# Normalization helpers
# ─────────────────────────────────────────────────────────────

def compute_normalization_stats(
    dataset: AdsorptionDataset,
    train_indices: Sequence[int],
) -> Dict[str, torch.Tensor]:
    """
    Compute mu/q normalization stats from the training subset only.
    """
    cond_list = []
    q_list = []

    for idx in train_indices:
        sample = dataset[idx]
        cond_list.append(sample["conditions"])
        q_list.append(sample["loadings"])

    if not cond_list or not q_list:
        raise RuntimeError("Cannot compute normalization stats: training subset is empty.")

    all_cond = torch.cat(cond_list, dim=0).float()
    all_q = torch.cat(q_list, dim=0).float()

    mu_mean = all_cond.mean(dim=0)
    mu_std = all_cond.std(dim=0, unbiased=False).clamp_min(1e-6)

    q_mean = all_q.mean(dim=0)
    q_std = all_q.std(dim=0, unbiased=False).clamp_min(1e-6)

    return {
        "mu_mean": mu_mean,
        "mu_std": mu_std,
        "q_mean": q_mean,
        "q_std": q_std,
    }


# ─────────────────────────────────────────────────────────────
# Callback bridge
# ─────────────────────────────────────────────────────────────

def make_epoch_callback(callback_runner: CallbackRunner, trainer: TPNOTrainer):
    """
    Bridge trainer.fit(callback=...) to CallbackRunner.on_epoch_end(...).
    """
    def _callback(epoch: int, metrics: Dict[str, Any]) -> None:
        callback_runner.on_epoch_end(epoch, metrics, trainer)
    return _callback


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the UC-TPNO model")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--adsorption-data", required=True)
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--splits-file", default=None)
    parser.add_argument("--output-dir", default="experiments/run_001")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--verbose-batches", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision if CUDA is available.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging(verbose_batches=args.verbose_batches)
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    # ── Load config ──────────────────────────────────────────
    cfg = load_config(args.config) if Path(args.config).exists() else {}
    model_cfg = dict(cfg.get("model", {}))
    train_cfg = dict(cfg.get("training", {}))
    data_cfg = dict(cfg.get("data", {}))

    # CLI overrides
    if args.epochs is not None:
        train_cfg["n_epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.log_interval is not None:
        train_cfg["log_interval"] = args.log_interval
    if args.amp:
        train_cfg["use_amp"] = True

    # Force experiment artifacts into chosen output dir
    train_cfg["checkpoint_dir"] = str(out_dir / "checkpoints")
    train_cfg["tb_log_dir"] = str(out_dir / "tensorboard")

    print("=" * 70)
    print("UC-TPNO TRAINING")
    print(f"Device:           {device}")
    print(f"Registry:         {args.registry}")
    print(f"Adsorption data:  {args.adsorption_data}")
    print(f"Graph dir:        {args.graph_dir}")
    print(f"Output dir:       {out_dir}")
    print(f"Seed:             {args.seed}")
    print("=" * 70)

    logger.info("Resolved training config: %s", json.dumps(train_cfg, indent=2, default=str))

    # ── Data ─────────────────────────────────────────────────
    print("\n=== Loading Data ===")
    dataset = AdsorptionDataset(
        registry_path=args.registry,
        adsorption_path=args.adsorption_data,
        graph_dir=args.graph_dir,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "AdsorptionDataset is empty. "
            "Check that graph_dir contains .pt files matching mof_id in registry/adsorption data."
        )

    logger.info("Dataset loaded: %d MOFs", len(dataset))
    logger.info("First 5 MOF IDs: %s", dataset.mof_ids[:5])

    if args.splits_file and Path(args.splits_file).exists():
        print(f"Using existing splits: {args.splits_file}")
        loaded_train, loaded_val, loaded_test = DataSplitter.load_splits(args.splits_file)
        train_idx, val_idx, test_idx = resolve_loaded_splits(
            dataset.mof_ids,
            loaded_train,
            loaded_val,
            loaded_test,
        )
    else:
        split_method = data_cfg.get("split_method", "random")
        test_size = float(data_cfg.get("test_size", 0.1))
        val_size = float(data_cfg.get("val_size", 0.1))

        splitter = DataSplitter(
            method=split_method,
            test_size=test_size,
            val_size=val_size,
            random_state=args.seed,
        )
        train_idx, val_idx, test_idx = splitter.split(dataset.mof_ids)
        splitter.save_splits(
            dataset.mof_ids,
            train_idx,
            val_idx,
            test_idx,
            out_dir / "splits.json",
        )

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError(
            f"Invalid split sizes: train={len(train_idx)}, val={len(val_idx)}. "
            "Need at least one sample in both train and validation."
        )

    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(args.num_workers)

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

    print(f"Train MOFs: {len(train_idx)}")
    print(f"Val MOFs:   {len(val_idx)}")
    print(f"Test MOFs:  {len(test_idx)}")
    print(f"Batch size: {batch_size}")
    print(f"Train batches/epoch: {len(train_loader)}")
    print(f"Val batches/epoch:   {len(val_loader)}")

    # ── Model ────────────────────────────────────────────────
    print("\n=== Building Model ===")

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

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Encoder backend: {encoder_config['encoder']}")
    print(f"Total parameters:     {n_params:,}")
    print(f"Trainable parameters: {n_trainable:,}")

    # ── Normalization ────────────────────────────────────────
    print("\n=== Computing Normalization Stats ===")
    norm_stats = compute_normalization_stats(dataset, train_idx)
    model.set_normalization(
        mu_mean=norm_stats["mu_mean"],
        mu_std=norm_stats["mu_std"],
        q_mean=norm_stats["q_mean"],
        q_std=norm_stats["q_std"],
    )

    print("mu_mean:", norm_stats["mu_mean"].tolist())
    print("mu_std: ", norm_stats["mu_std"].tolist())
    print("q_mean: ", norm_stats["q_mean"].tolist())
    print("q_std:  ", norm_stats["q_std"].tolist())

    # ── Loss ─────────────────────────────────────────────────
    print("\n=== Building Loss ===")
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

    print("Loss weights:")
    print(f"  lambda_data:        {loss_cfg.lambda_data}")
    print(f"  lambda_hessian:     {loss_cfg.lambda_hessian}")
    print(f"  lambda_monotonic:   {loss_cfg.lambda_monotonic}")
    print(f"  lambda_henry:       {loss_cfg.lambda_henry}")
    print(f"  lambda_competition: {loss_cfg.lambda_competition}")
    print(f"  lambda_gibbs_duhem: {loss_cfg.lambda_gibbs_duhem}")

    # ── Trainer config ───────────────────────────────────────
    print("\n=== Preparing Trainer ===")
    trainer_cfg = TrainConfig(
        n_epochs=int(train_cfg.get("n_epochs", 100)),
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
        checkpoint_dir=str(out_dir / "checkpoints"),
        checkpoint_freq=int(train_cfg.get("checkpoint_freq", 10)),
        use_wandb=bool(train_cfg.get("use_wandb", False)),
        wandb_project=train_cfg.get("wandb_project", "uc-tpno"),
        wandb_run_name=train_cfg.get("wandb_run_name", out_dir.name),
        use_tensorboard=bool(train_cfg.get("use_tensorboard", False)),
        tb_log_dir=str(out_dir / "tensorboard"),
        log_interval=int(train_cfg.get("log_interval", 50)),
    )

    trainer = TPNOTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_cfg,
        criterion=criterion,
        device=device,
    )

    # ── Lightweight callbacks only ──────────────────────────
    # Trainer already handles:
    # - warmup
    # - scheduler
    # - checkpointing
    # - early stopping
    callbacks = CallbackRunner(
        [
            MetricLogger(
                log_every=1,
                csv_path=str(out_dir / "metrics.csv"),
            ),
            TimingCallback(),
        ]
    )
    epoch_callback = make_epoch_callback(callbacks, trainer)

    # Save resolved config before training
    resolved = {
        "config_path": args.config,
        "seed": args.seed,
        "device": str(device),
        "registry": args.registry,
        "adsorption_data": args.adsorption_data,
        "graph_dir": args.graph_dir,
        "output_dir": str(out_dir),
        "model": model_cfg,
        "training": vars(trainer_cfg),
        "loss": vars(loss_cfg),
        "data": data_cfg,
        "normalization": {
            "mu_mean": norm_stats["mu_mean"].tolist(),
            "mu_std": norm_stats["mu_std"].tolist(),
            "q_mean": norm_stats["q_mean"].tolist(),
            "q_std": norm_stats["q_std"].tolist(),
        },
    }
    with open(out_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2, default=str)

    # ── Train ────────────────────────────────────────────────
    print("\n=== Training ===")
    print("You will see per-epoch updates in the terminal.")
    if args.verbose_batches:
        print(
            f"Frequent batch logs enabled. "
            f"Trainer will report every {trainer_cfg.log_interval} batches."
        )

    callbacks.on_train_begin(trainer)
    try:
        history = trainer.fit(
            n_epochs=trainer_cfg.n_epochs,
            callback=epoch_callback,
        )
    finally:
        callbacks.on_train_end(trainer)

    # ── Save final artifacts ─────────────────────────────────
    final_model_path = out_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)

    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=float)

    best_metrics = trainer.get_best_metrics()
    if best_metrics is not None:
        with open(out_dir / "best_metrics.json", "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Final model:      {final_model_path}")
    print(f"History:          {out_dir / 'training_history.json'}")
    print(f"Metrics CSV:      {out_dir / 'metrics.csv'}")
    print(f"Checkpoints dir:  {out_dir / 'checkpoints'}")
    print(f"Next: python scripts/06_calibrate_uq.py --model-checkpoint {final_model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()