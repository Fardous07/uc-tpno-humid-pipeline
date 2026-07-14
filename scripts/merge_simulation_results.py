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
    """Find all parquet files produced by worker subdirectories."""
    files: list[Path] = []

    # Worker subdirectories: worker_0/, worker_1/, ...
    for sub in sorted(worker_dir.glob("worker_*")):
        if not sub.is_dir():
            continue
        for pq in sorted(sub.glob("*.parquet")):
            files.append(pq)

    # Also pick up any parquet directly in worker_dir (single-worker run)
    for pq in sorted(worker_dir.glob("*.parquet")):
        if pq.name != "adsorption_training.parquet":
            files.append(pq)

    return sorted(set(files))


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic sanity checks on merged adsorption data.
    Drops rows with NaN loadings or non-positive pressures.
    """
    n_before = len(df)

    # Required columns
    required = {"mof_id", "temperature", "pressure"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("Missing expected columns: %s", missing)

    # Drop rows where all loading columns are NaN
    loading_cols = [c for c in df.columns if "loading" in c.lower() or "q_" in c.lower()]
    if loading_cols:
        df = df.dropna(subset=loading_cols, how="all")

    # Drop non-positive pressure
    if "pressure" in df.columns:
        df = df[df["pressure"] > 0]

    # Drop infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[np.isfinite(df[numeric_cols]).all(axis=1)]

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

    # Deduplication — same MOF + conditions should not appear twice
    id_cols = [c for c in ["mof_id", "temperature", "pressure", "relative_humidity"]
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
    sort_cols = [c for c in ["mof_id", "temperature", "pressure"] if c in merged.columns]
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
        n_complete = (counts >= 36).sum()  # 3T × 4P × 3RH
        print(f"\nMOFs with all 36 conditions complete: {n_complete}/{counts.shape[0]}")

    print("\nNext step:")
    print("  python scripts/05_train_model.py \\")
    print("    --registry data/mof_registry.parquet \\")
    print("    --adsorption-data data/processed/adsorption/adsorption_training.parquet \\")
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
