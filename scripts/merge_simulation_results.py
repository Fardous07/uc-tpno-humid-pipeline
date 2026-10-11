#!/usr/bin/env python3
"""
merge_simulation_results.py
Merge parquet outputs from all parallel GCMC workers into one file.

Usage:
    python scripts/merge_simulation_results.py
    python scripts/merge_simulation_results.py --worker-dir data/processed/adsorption
    python scripts/merge_simulation_results.py --output data/processed/adsorption/adsorption_training.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_worker_parquets(worker_dir: Path) -> list[Path]:
    """
    Find each worker's own final adsorption_training.parquet.

    FIX: this used to glob("*.parquet") inside each worker_* subdirectory,
    which also picks up gcmc_adsorption.parquet (pre-dedup, same rows as
    adsorption_training.parquet but not cleaned) and gcmc_failures.parquet
    (a completely different schema: mof_id/temperature_k/pressure_bar/rh/
    returncode/error_tail, no loading columns at all). Concatenating all
    three together per worker silently produced a merged file mixing rows
    from three different, incompatible tables. Now only collects each
    worker's adsorption_training.parquet -- the one clean, deduplicated
    file 04_run_simulations.py's build_training_dataset() actually writes.
    """
    files: list[Path] = []

    # Worker subdirectories: worker_0/, worker_1/, ...
    for sub in sorted(worker_dir.glob("worker_*")):
        if not sub.is_dir():
            continue
        pq = sub / "adsorption_training.parquet"
        if pq.exists():
            files.append(pq)

    return sorted(set(files))


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic sanity checks on merged adsorption data.
    Drops rows with NaN loadings or non-positive pressures.
    """
    n_before = len(df)

    # Required columns
    required = {"mof_id", "temperature_k", "pressure_bar"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("Missing expected columns: %s", missing)

    # Drop rows where all loading columns are NaN
    loading_cols = [c for c in df.columns if "loading" in c.lower() or "q_" in c.lower()]
    if loading_cols:
        df = df.dropna(subset=loading_cols, how="all")

    # Drop non-positive pressure
    if "pressure_bar" in df.columns:
        df = df[df["pressure_bar"] > 0]

    # Drop infinite values; NaN is allowed in error-bar columns (runs before
    # error bars were recorded) since training treats them as unknown.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    err_cols = [c for c in numeric_cols if c.endswith("_err")]
    value_cols = [c for c in numeric_cols if c not in err_cols]
    ok = np.isfinite(df[value_cols]).all(axis=1) & ~np.isinf(df[err_cols]).any(axis=1)
    df = df[ok]

    n_after = len(df)
    if n_before > n_after:
        logger.warning("Dropped %d invalid rows (%d → %d)", n_before - n_after, n_before, n_after)

    return df


def merge(worker_dir: Path, output_path: Path) -> None:
    logger.info("Searching for worker parquet files in: %s", worker_dir)
    files = find_worker_parquets(worker_dir)

    if not files:
        logger.error(
            "No parquet files found under %s\n"
            "Make sure at least one worker has finished.",
            worker_dir,
        )
        sys.exit(1)

    logger.info("Found %d parquet file(s):", len(files))
    for f in files:
        logger.info("  %s", f)

    # Load and concatenate
    dfs: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
            logger.info("  Loaded %s: %d rows", f.name, len(df))
        except Exception as e:
            logger.warning("  Failed to load %s: %s — skipping", f, e)

    if not dfs:
        logger.error("All parquet files failed to load.")
        sys.exit(1)

    merged = pd.concat(dfs, ignore_index=True)
    logger.info("Merged: %d rows total", len(merged))

    # ADDED: drop structural duplicates. duplicate_resolution.csv (written
    # after a pymatgen StructureMatcher check of the selected MOFs) lists,
    # for each group of identical frameworks, the one copy to keep; the
    # other copies are removed so the same structure cannot end up in both
    # the training and the test set (data leakage).
    dup_file = worker_dir / "duplicate_resolution.csv"
    if dup_file.exists():
        removed = set(pd.read_csv(dup_file).query("action == 'remove'").mof_id)
        n_before = len(merged)
        merged = merged[~merged["mof_id"].isin(removed)]
        logger.info("Removed %d rows of %d duplicate frameworks (%s).",
                    n_before - len(merged), len(removed), dup_file.name)

    # Deduplication — same MOF + conditions should not appear twice
    # FIX: was ["mof_id", "temperature", "pressure", "relative_humidity"],
    # but the actual columns 04_run_simulations.py writes are
    # temperature_k/pressure_bar/rh -- none of those three matched, so
    # id_cols silently fell back to just ["mof_id"] and drop_duplicates
    # collapsed all 45 conditions per MOF down to 1 row per MOF (verified:
    # a 4-MOF/45-condition test run merged to 4 rows total instead of 180).
    id_cols = [c for c in ["mof_id", "temperature_k", "pressure_bar", "rh"]
               if c in merged.columns]
    if id_cols:
        n_before = len(merged)
        merged = merged.drop_duplicates(subset=id_cols, keep="first")
        n_dup = n_before - len(merged)
        if n_dup > 0:
            logger.warning("Removed %d duplicate rows (same MOF+conditions).", n_dup)

    # Validate
    merged = validate_dataframe(merged)

    # Sort
    sort_cols = [c for c in ["mof_id", "temperature_k", "pressure_bar", "rh"]
                 if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)

    # Also save CSV summary (first 5 rows + stats) for quick inspection
    summary_path = output_path.with_suffix(".summary.csv")
    merged.describe(include="all").to_csv(summary_path)

    print("=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"  Input files  : {len(files)}")
    print(f"  Total rows   : {len(merged)}")
    print(f"  Unique MOFs  : {merged['mof_id'].nunique() if 'mof_id' in merged.columns else 'unknown'}")
    print(f"  Output       : {output_path}")
    print(f"  Summary CSV  : {summary_path}")
    print("=" * 60)

    if "mof_id" in merged.columns:
        print("\nRows per MOF (first 10):")
        counts = merged.groupby("mof_id").size().sort_values(ascending=False)
        print(counts.head(10).to_string())
        # FIX: was hardcoded to 36 (3T x 4P x 3RH), stale for the actual
        # condition grid this pipeline uses (3T x 5P x 3RH = 45, and this
        # script has no way to know the exact grid a given run used
        # anyway). Detect the expected per-MOF count dynamically instead
        # of hardcoding a number that silently drifts out of sync with
        # whatever --temperatures/--pressures/--rhs were actually passed.
        full_count = int(counts.max()) if not counts.empty else 0
        n_complete = (counts >= full_count).sum()
        print(f"\nMOFs with all {full_count} conditions complete (detected from data): "
              f"{n_complete}/{counts.shape[0]}")

    # FIX: this used to hardcode the OLD data/processed/adsorption path
    # regardless of --output, silently pointing users at stale/wrong data
    # whenever --output was actually used (as the parallel launcher does).
    print("\nNext step:")
    print("  python scripts/05_train_model.py \\")
    print("    --registry data/mof_registry.parquet \\")
    print(f"    --adsorption-data {output_path} \\")
    print("    --graph-dir data/processed/graphs \\")
    print("    --output-dir experiments/run_001 \\")
    print("    --config configs/pipeline.yaml --amp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge parallel GCMC worker outputs")
    parser.add_argument(
        "--worker-dir",
        default="data/processed/adsorption",
        help="Directory containing worker_* subdirectories",
    )
    parser.add_argument(
        "--output",
        default="data/processed/adsorption/adsorption_training.parquet",
        help="Path to write merged parquet",
    )
    args = parser.parse_args()

    merge(
        worker_dir=Path(args.worker_dir),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
