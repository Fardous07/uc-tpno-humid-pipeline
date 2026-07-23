#!/usr/bin/env python3
"""
05_train_model.py — Train the UC-TPNO model.

Fixes vs. original
------------------
1. BUG FIXED: lazy_loading parameter removed from AdsorptionDataset call
   (we removed that parameter in the dataset fix).
2. BUG FIXED: compute_normalization_stats now applies the mask so padded
   zero-positions are excluded from mean/std computation.  Uses the method
   we added to AdsorptionDataset instead of a duplicated standalone function.
3. BUG FIXED: physics_cfg now reads from train_cfg.get("physics_loss", {})
   instead of cfg.get("physics_loss", {}).  The YAML nests physics weights
   under training.physics_loss, not at the top level.

Usage
-----
python scripts/05_train_model.py \\
    --registry   data/processed/mof_registry.parquet \\
    --adsorption-data data/processed/adsorption/adsorption_training.parquet \\
    --graph-dir  data/processed/graphs \\
    --output-dir experiments/run_001

Optional flags:
  --config  configs/pipeline.yaml   (default)
  --epochs  500
  --batch-size 4
  --lr 3e-4
  --amp                             (mixed precision, CUDA only)
  --grad-accum 2                    (gradient accumulation steps)
  --verbose-batches                 (log every --log-interval batches)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datasets.adsorption_dataset import AdsorptionDataset
from src.data.datasets.splitter import DataSplitter
from src.models.encoder.adapter import EncoderAdapter
from src.models.operator.tpno import (
    ThermodynamicPotentialNO,
    TPNOConfig,
    TPNOEnsemble,
)
from src.models.operator.losses import (
    ThermodynamicLoss,
    LossConfig,
    PhysicsLossScheduler,
    AdaptiveLossWeighting,
)
from src.training.trainer import TPNOTrainer, TrainConfig
from src.training.callbacks import CallbackRunner, MetricLogger, TimingCallback
from src.utils.reproducibility import set_seed

try:
    import yaml
except ImportError:
    yaml = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(verbose_batches: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("src.training.trainer").setLevel(logging.INFO)
    logging.getLogger("src.training.callbacks").setLevel(logging.INFO)
    logging.getLogger("src.data.datasets.adsorption_dataset").setLevel(logging.INFO)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        logger.warning(f"Config file not found: {path}. Using defaults.")
        return {}
    if yaml is None:
        logger.warning("PyYAML not installed. pip install pyyaml")
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def resolve_loaded_splits(
    dataset_mof_ids: Sequence[str],
    loaded_train: Sequence[Any],
    loaded_val:   Sequence[Any],
    loaded_test:  Sequence[Any],
) -> Tuple[List[int], List[int], List[int]]:
    """Accept split files that store either MOF IDs or integer indices."""
    mof_ids = list(dataset_mof_ids)

    def _all_ints(xs: Sequence[Any]) -> bool:
        return all(isinstance(x, int) for x in xs)

    if _all_ints(loaded_train) and _all_ints(loaded_val) and _all_ints(loaded_test):
        n = len(mof_ids) - 1
        return (
            [i for i in loaded_train if 0 <= i <= n],
            [i for i in loaded_val   if 0 <= i <= n],
            [i for i in loaded_test  if 0 <= i <= n],
        )

    id_to_idx = {m: i for i, m in enumerate(mof_ids)}
    return (
        [id_to_idx[m] for m in loaded_train if m in id_to_idx],
        [id_to_idx[m] for m in loaded_val   if m in id_to_idx],
        [id_to_idx[m] for m in loaded_test  if m in id_to_idx],
    )


# ---------------------------------------------------------------------------
# Normalization  (uses the method added to AdsorptionDataset)
# ---------------------------------------------------------------------------

def compute_normalization_stats(
    dataset:       AdsorptionDataset,
    train_indices: Sequence[int],
) -> Dict[str, torch.Tensor]:
    """
    Delegate to dataset.compute_normalization_stats() which correctly
    excludes padded positions using the per-sample row lookup.

    FIX vs. original standalone function:
    - Original accumulated full collate batches which include padded zeros,
      corrupting mu_mean / q_mean / q_std with fake zero-valued samples.
    - This version uses the item-level access in compute_normalization_stats
      so only real (unpadded) data points are included.
    """
    if len(train_indices) == 0:
        raise RuntimeError("Training subset is empty — cannot compute normalization stats.")
    logger.info(
        "Computing normalization stats from %d training MOFs …", len(train_indices)
    )
    stats = dataset.compute_normalization_stats(list(train_indices))
    logger.info("mu_mean : %s", stats["mu_mean"].tolist())
    logger.info("mu_std  : %s", stats["mu_std"].tolist())
    logger.info("q_mean  : %s", stats["q_mean"].tolist())
    logger.info("q_std   : %s", stats["q_std"].tolist())
    return stats


# ---------------------------------------------------------------------------
# Epoch callback bridge
# ---------------------------------------------------------------------------

def make_epoch_callback(
    callback_runner:    CallbackRunner,
    trainer:            TPNOTrainer,
    physics_scheduler:  Optional[PhysicsLossScheduler] = None,
    adaptive_weights:   Optional[AdaptiveLossWeighting] = None,
):
    def _callback(epoch: int, metrics: Dict[str, Any]) -> None:
        if physics_scheduler is not None:
            physics_scheduler.step(epoch)
            weights = physics_scheduler.get_current_weights()
            logger.debug("Physics weights @ epoch %d: %s", epoch, weights)

        if adaptive_weights is not None:
            physics_dict = {
                k.replace("train/physics_", ""): v
                for k, v in metrics.items()
                if k.startswith("train/physics_")
            }
            if physics_dict:
                adaptive_weights.update_weights(physics_dict)

        callback_runner.on_epoch_end(epoch, metrics, trainer)

    return _callback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the UC-TPNO model")
    # Paths
    parser.add_argument("--config",           default="configs/pipeline.yaml")
    parser.add_argument("--registry",         required=True)
    parser.add_argument("--adsorption-data",  required=True)
    parser.add_argument("--graph-dir",        required=True)
    parser.add_argument("--splits-file",      default=None)
    parser.add_argument("--output-dir",       default="experiments/run_001")
    # Training overrides
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--batch-size",   type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--num-workers",  type=int,   default=0)
    parser.add_argument("--log-interval", type=int,   default=None)
    parser.add_argument("--verbose-batches", action="store_true")
    parser.add_argument("--amp",          action="store_true")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--no-early-stopping", dest="early_stopping", action="store_false")
    parser.add_argument("--patience",     type=int,   default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--grad-accum",   type=int,   default=None)
    args = parser.parse_args()

    configure_logging(args.verbose_batches)
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    # ── Config ──────────────────────────────────────────────────────────────
    cfg        = load_config(args.config)
    model_cfg  = dict(cfg.get("model",    {}))
    train_cfg  = dict(cfg.get("training", {}))
    data_cfg   = dict(cfg.get("data",     {}))

    # FIX: physics_loss is nested under training in pipeline.yaml
    physics_cfg = dict(train_cfg.get("physics_loss", {}))

    # CLI overrides
    if args.epochs      is not None: train_cfg["n_epochs"]   = args.epochs
    if args.batch_size  is not None: train_cfg["batch_size"] = args.batch_size
    if args.lr          is not None: train_cfg["lr"]         = args.lr
    if args.log_interval is not None: train_cfg["log_interval"] = args.log_interval
    if args.amp:                      train_cfg["use_amp"]    = True
    if not args.early_stopping:       train_cfg["early_stopping"] = False
    if args.patience    is not None:  train_cfg["patience"]   = args.patience
    if args.grad_accum  is not None:  train_cfg["gradient_accumulation_steps"] = args.grad_accum

    train_cfg["checkpoint_dir"] = str(out_dir / "checkpoints")
    train_cfg["tb_log_dir"]     = str(out_dir / "tensorboard")

    print("=" * 70)
    print("UC-TPNO TRAINING")
    print(f"  Device          : {device}")
    print(f"  Registry        : {args.registry}")
    print(f"  Adsorption data : {args.adsorption_data}")
    print(f"  Graph dir       : {args.graph_dir}")
    print(f"  Output dir      : {out_dir}")
    print(f"  Seed            : {args.seed}")
    print("=" * 70)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n=== Loading Data ===")

    # FIX: lazy_loading parameter was removed from AdsorptionDataset
    dataset = AdsorptionDataset(
        registry_path=args.registry,
        adsorption_path=args.adsorption_data,
        graph_dir=args.graph_dir,
        # condition_columns / target_columns use sensible defaults
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "AdsorptionDataset is empty. "
            "Verify that graph_dir contains .pt files matching mof_id entries."
        )
    logger.info("Dataset: %d MOFs | first 5: %s", len(dataset), dataset.mof_ids[:5])

    # ── Splits ───────────────────────────────────────────────────────────────
    splits_path = Path(args.splits_file) if args.splits_file else None
    if splits_path and splits_path.exists():
        print(f"Using existing splits: {splits_path}")
        loaded_train, loaded_val, loaded_test = DataSplitter.load_splits(str(splits_path))
        train_idx, val_idx, test_idx = resolve_loaded_splits(
            dataset.mof_ids, loaded_train, loaded_val, loaded_test
        )
    else:
        splitter = DataSplitter(
            method=data_cfg.get("split_method", "random"),
            test_size=float(data_cfg.get("test_size", 0.1)),
            val_size=float(data_cfg.get("val_size",  0.1)),
            random_state=args.seed,
        )
        train_idx, val_idx, test_idx = splitter.split(dataset.mof_ids)
        splitter.save_splits(
            dataset.mof_ids, train_idx, val_idx, test_idx,
            str(out_dir / "splits.json"),
        )

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError(
            f"Invalid split: train={len(train_idx)}, val={len(val_idx)}. "
            "Need at least 1 sample in both."
        )

    batch_size  = int(train_cfg.get("batch_size", 4))
    num_workers = int(args.num_workers)

    train_loader = dataset.get_dataloader(
        indices=train_idx, batch_size=batch_size,
        shuffle=True,  num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = dataset.get_dataloader(
        indices=val_idx,   batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )

    print(f"  Train MOFs          : {len(train_idx)}")
    print(f"  Val MOFs            : {len(val_idx)}")
    print(f"  Test MOFs           : {len(test_idx)}")
    print(f"  Batch size          : {batch_size}")
    print(f"  Train batches/epoch : {len(train_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n=== Building Model ===")

    encoder_config = {
        "encoder": model_cfg.get("encoder",          "nequip"),
        "n_species": model_cfg.get("n_species",       100),
        "emb_dim":   model_cfg.get("emb_dim",         64),
        "n_layers":  model_cfg.get("n_encoder_layers", 2),
        "lmax":      model_cfg.get("lmax",             1),
        "cutoff":    model_cfg.get("cutoff",           6.0),
        "n_rbf":     model_cfg.get("n_rbf",            32),
        "use_pbc":   model_cfg.get("use_pbc",          True),
        "use_charges": model_cfg.get("use_charges", False),
        "avg_num_neighbours": model_cfg.get("avg_num_neighbours", 15.0),
    }
    encoder = EncoderAdapter.from_config(
        encoder_config,
        target_dim=int(model_cfg.get("emb_dim", 64)),
        normalize=True,
    )

    tpno_cfg = TPNOConfig(
        emb_dim=int(model_cfg.get("emb_dim",           64)),
        n_conditions=int(model_cfg.get("n_conditions",  4)),
        n_components=int(model_cfg.get("n_components",  3)),
        hidden_dim=int(model_cfg.get("hidden_dim",      128)),
        n_layers=int(model_cfg.get("n_tpno_layers",     3)),
        convex_constraint=model_cfg.get("convex_constraint", "softplus"),
        film_conditioning=bool(model_cfg.get("film_conditioning", True)),
        dropout=float(model_cfg.get("dropout",          0.15)),
        use_layer_norm=bool(model_cfg.get("use_layer_norm", True)),
        activation=model_cfg.get("activation",          "swish"),
        min_potential=float(model_cfg.get("min_potential", 1e-6)),
    )

    use_ensemble    = bool(model_cfg.get("ensemble", False))
    ensemble_cfg    = cfg.get("ensemble", {}) if isinstance(cfg.get("ensemble"), dict) else {}
    n_ensemble      = int(ensemble_cfg.get("n_models", model_cfg.get("n_ensemble", 5)))
    share_encoder   = bool(ensemble_cfg.get("share_encoder", model_cfg.get("share_encoder", True)))

    if use_ensemble:
        model = TPNOEnsemble(
            config=tpno_cfg, encoder=encoder,
            n_models=n_ensemble, share_encoder=share_encoder,
        )
        print(f"  Ensemble: {n_ensemble} models, share_encoder={share_encoder}")
    else:
        model = ThermodynamicPotentialNO(encoder=encoder, config=tpno_cfg)

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Encoder backend     : {encoder_config['encoder']}")
    print(f"  Total parameters    : {n_total:,}")
    print(f"  Trainable parameters: {n_trainable:,}")

    # ── Normalization ─────────────────────────────────────────────────────────
    print("\n=== Computing Normalization Stats ===")

    # FIX: uses dataset.compute_normalization_stats which correctly excludes
    # padded positions — the standalone function in the original script did not.
    norm_stats = compute_normalization_stats(dataset, train_idx)

    model.set_normalization(
        mu_mean=norm_stats["mu_mean"],
        mu_std =norm_stats["mu_std"],
        q_mean =norm_stats["q_mean"],
        q_std  =norm_stats["q_std"],
    )
    print("  mu_mean :", norm_stats["mu_mean"].tolist())
    print("  mu_std  :", norm_stats["mu_std"].tolist())
    print("  q_mean  :", norm_stats["q_mean"].tolist())
    print("  q_std   :", norm_stats["q_std"].tolist())

    # ── Loss ──────────────────────────────────────────────────────────────────
    print("\n=== Building Loss ===")

    # FIX: physics_cfg comes from train_cfg.get("physics_loss") not cfg root
    lambda_hessian     = float(physics_cfg.get("lambda_hessian",     0.01))
    lambda_monotonic   = float(physics_cfg.get("lambda_monotonic",   0.05))
    lambda_henry       = float(physics_cfg.get("lambda_henry",       0.005))
    lambda_competition = float(physics_cfg.get("lambda_competition", 0.05))
    lambda_gibbs_duhem = float(physics_cfg.get("lambda_gibbs_duhem", 0.0))

    loss_cfg = LossConfig(
        lambda_data=float(train_cfg.get("lambda_data", 1.0)),
        lambda_hessian=lambda_hessian,
        lambda_monotonic=lambda_monotonic,
        lambda_henry=lambda_henry,
        lambda_competition=lambda_competition,
        lambda_gibbs_duhem=lambda_gibbs_duhem,
        henry_mu_threshold=float(physics_cfg.get("henry_mu_threshold",
                                 train_cfg.get("henry_mu_threshold", -5.0))),
        use_nll=bool(train_cfg.get("use_nll", True)),
    )
    criterion = ThermodynamicLoss(config=loss_cfg)

    print("  Loss weights:")
    print(f"    lambda_data        : {loss_cfg.lambda_data}")
    print(f"    lambda_hessian     : {loss_cfg.lambda_hessian}")
    print(f"    lambda_monotonic   : {loss_cfg.lambda_monotonic}")
    print(f"    lambda_henry       : {loss_cfg.lambda_henry}")
    print(f"    lambda_competition : {loss_cfg.lambda_competition}")
    print(f"    lambda_gibbs_duhem : {loss_cfg.lambda_gibbs_duhem}")
    print(f"    use_nll            : {loss_cfg.use_nll}")

    # ── Physics warm-up scheduler ─────────────────────────────────────────────
    physics_warmup   = int(train_cfg.get("warmup_epochs",    30))
    physics_scheduler = PhysicsLossScheduler(criterion, warmup_epochs=physics_warmup)
    print(f"\n  Physics warmup: {physics_warmup} epochs")

    # ── Adaptive loss weighting ───────────────────────────────────────────────
    use_adaptive    = bool(physics_cfg.get("adaptive_weights", False))
    adaptive_weights: Optional[AdaptiveLossWeighting] = None
    if use_adaptive:
        adaptive_weights = AdaptiveLossWeighting(
            initial_weights={
                "hessian":     lambda_hessian,
                "monotonic":   lambda_monotonic,
                "henry":       lambda_henry,
                "competition": lambda_competition,
            },
            smoothing=float(physics_cfg.get("weight_smoothing",  0.9)),
            update_freq=int(physics_cfg.get("weight_update_freq", 10)),
        )
        print("  Adaptive loss weighting: enabled")

    # ── Trainer ───────────────────────────────────────────────────────────────
    print("\n=== Preparing Trainer ===")

    trainer_cfg = TrainConfig(
        n_epochs=int(train_cfg.get("n_epochs",                    500)),
        lr=float(train_cfg.get("lr",                              3e-4)),
        weight_decay=float(train_cfg.get("weight_decay",          1e-4)),
        optimizer=train_cfg.get("optimizer",                      "adamw"),
        scheduler=train_cfg.get("scheduler",                      "cosine_warm_restarts"),
        scheduler_T0=int(train_cfg.get("scheduler_T0",            30)),
        scheduler_T_mult=int(train_cfg.get("scheduler_T_mult",    2)),
        step_size=int(train_cfg.get("step_size",                  30)),
        step_gamma=float(train_cfg.get("step_gamma",              0.5)),
        warmup_epochs=int(train_cfg.get("warmup_epochs",          5)),
        physics_warmup=physics_warmup,
        grad_clip=float(train_cfg.get("grad_clip",                1.0)),
        use_amp=bool(train_cfg.get("use_amp",                     False)),
        early_stopping=bool(train_cfg.get("early_stopping",       True)),
        patience=int(train_cfg.get("patience",                    40)),
        checkpoint_dir=str(out_dir / "checkpoints"),
        checkpoint_freq=int(train_cfg.get("checkpoint_freq",      10)),
        use_wandb=bool(train_cfg.get("use_wandb",                 False)),
        wandb_project=train_cfg.get("wandb_project",              "uc-tpno"),
        wandb_run_name=train_cfg.get("wandb_run_name",            out_dir.name),
        use_tensorboard=bool(train_cfg.get("use_tensorboard",     False)),
        tb_log_dir=str(out_dir / "tensorboard"),
        log_interval=int(train_cfg.get("log_interval",            50)),
        gradient_accumulation_steps=int(
            train_cfg.get("gradient_accumulation_steps", 1)
        ),
    )

    trainer = TPNOTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_cfg,
        criterion=criterion,
        device=device,
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    callbacks = CallbackRunner([
        MetricLogger(log_every=1, csv_path=str(out_dir / "metrics.csv")),
        TimingCallback(),
    ])

    epoch_callback = make_epoch_callback(
        callbacks, trainer,
        physics_scheduler=physics_scheduler,
        adaptive_weights=adaptive_weights,
    )

    # ── Save resolved config ──────────────────────────────────────────────────
    resolved = {
        "config_path":     args.config,
        "seed":            args.seed,
        "device":          str(device),
        "registry":        args.registry,
        "adsorption_data": args.adsorption_data,
        "graph_dir":       args.graph_dir,
        "output_dir":      str(out_dir),
        "model":           model_cfg,
        "training":        {k: v for k, v in vars(trainer_cfg).items()},
        "loss":            {k: v for k, v in vars(loss_cfg).items()},
        "physics_loss":    physics_cfg,
        "data":            data_cfg,
        "normalization": {
            "mu_mean": norm_stats["mu_mean"].tolist(),
            "mu_std":  norm_stats["mu_std"].tolist(),
            "q_mean":  norm_stats["q_mean"].tolist(),
            "q_std":   norm_stats["q_std"].tolist(),
        },
        "ensemble": {
            "enabled":       use_ensemble,
            "n_models":      n_ensemble if use_ensemble else 0,
            "share_encoder": share_encoder if use_ensemble else False,
        },
    }
    with open(out_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2, default=str)
    logger.info("Resolved config saved → %s", out_dir / "resolved_config.json")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n=== Training ===")
    print(f"  Epochs            : {trainer_cfg.n_epochs}")
    print(f"  LR                : {trainer_cfg.lr}")
    print(f"  Grad accumulation : {trainer_cfg.gradient_accumulation_steps}")
    print(f"  Mixed precision   : {trainer_cfg.use_amp}")
    print(f"  Early stopping    : {trainer_cfg.early_stopping} (patience={trainer_cfg.patience})")
    if args.verbose_batches:
        print(f"  Batch log every   : {trainer_cfg.log_interval} batches")
    print()

    callbacks.on_train_begin(trainer)
    try:
        if args.resume:
            trainer.load_checkpoint(args.resume)
            print(f"Resumed from {args.resume}")
        history = trainer.fit(
            n_epochs=trainer_cfg.n_epochs,
            callback=epoch_callback,
        )
    finally:
        callbacks.on_train_end(trainer)

    # ── Save artifacts ────────────────────────────────────────────────────────
    final_path = out_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)

    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=float)

    best = trainer.get_best_metrics()
    if best is not None:
        with open(out_dir / "best_metrics.json", "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"  Final model    : {final_path}")
    print(f"  History        : {out_dir / 'training_history.json'}")
    print(f"  Metrics CSV    : {out_dir / 'metrics.csv'}")
    print(f"  Checkpoints    : {out_dir / 'checkpoints'}")
    if use_ensemble:
        print(f"  Ensemble size  : {n_ensemble}")
    print()
    print("Next steps:")
    print(f"  python scripts/06_calibrate_uq.py  --model-checkpoint {final_path}")
    print(f"  python scripts/07_evaluate.py       --model-checkpoint {final_path}")
    print(f"  python scripts/08_active_learning.py --model-checkpoint {final_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()