#!/usr/bin/env python3
"""
03_generate_graphs.py — Build PyTorch Geometric graphs from sanitised CIFs.

Usage
-----
python scripts/03_generate_graphs.py --data-dir data --cutoff 6.0

Examples
--------
python scripts/03_generate_graphs.py --data-dir data --cutoff 6.0
python scripts/03_generate_graphs.py --data-dir data --cutoff 6.0 --skip-existing
python scripts/03_generate_graphs.py --data-dir data --cutoff 6.0 --max-neighbours 16
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing.graph_builder import GraphBuilder, GraphConfig


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PyTorch Geometric graphs from sanitised CIFs"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base data directory containing intermediate/ and processed/",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=6.0,
        help="Neighbour cutoff in Å",
    )
    parser.add_argument(
        "--max-neighbours",
        type=int,
        default=0,
        help="Maximum outgoing neighbours per node (0 = unlimited)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip graph files that already exist",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
        help="Print progress every N CIFs",
    )
    args = parser.parse_args()

    configure_logging()

    data_dir = Path(args.data_dir)
    cif_dir = data_dir / "intermediate" / "cifs_sanitized"
    graph_dir = data_dir / "processed" / "graphs"

    print("=" * 70)
    print("STEP 3: GRAPH GENERATION")
    print("=" * 70)
    print(f"  CIF dir:         {cif_dir}")
    print(f"  Graph dir:       {graph_dir}")
    print(f"  Cutoff:          {args.cutoff} Å")
    print(f"  Max neighbours:  {args.max_neighbours}")
    print(f"  Skip existing:   {args.skip_existing}")
    print(f"  Print every:     {args.print_every}")
    print("=" * 70)

    if not cif_dir.exists():
        raise FileNotFoundError(
            f"Sanitised CIF directory not found: {cif_dir}\n"
            f"Run preprocessing first."
        )

    cif_files = sorted(cif_dir.rglob("*.cif"))
    if not cif_files:
        raise FileNotFoundError(f"No CIF files found in: {cif_dir}")

    graph_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFound {len(cif_files):,} sanitised CIF files.")

    config = GraphConfig(
        cutoff=args.cutoff,
        max_neighbours=args.max_neighbours,
        edge_features=True,
    )
    builder = GraphBuilder(config)

    start = time.time()

    summary = builder.build_all(
        cif_dir=cif_dir,
        output_dir=graph_dir,
        pattern="*.cif",
        skip_existing=args.skip_existing,
        # print_every=args.print_every,
    )

    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print("GRAPH GENERATION RESULTS")
    print("=" * 70)
    print(f"  Total:    {summary.get('n_total', 0):,}")
    print(f"  Success:  {summary.get('n_success', 0):,}")
    print(f"  Skipped:  {summary.get('n_skipped', 0):,}")
    print(f"  Failed:   {summary.get('n_failed', 0):,}")
    print(f"  Runtime:  {elapsed / 60.0:.2f} min")

    failures = summary.get("failures", [])
    if failures:
        print("\nFailures (first 10):")
        for f in failures[:10]:
            print(f"  {f.get('mof_id', 'UNKNOWN')}: {str(f.get('error', ''))[:150]}")

    print("\n" + "=" * 70)
    print("GRAPH GENERATION COMPLETE")
    print("Next command:")
    print(
        "python scripts/04_run_simulations.py "
        "--cif-dir data/intermediate/cifs_sanitized "
        "--output-dir data/processed/adsorption "
        "--max-mofs 5 --n-cycles 1000 --keep-workspaces"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()