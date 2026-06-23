#!/usr/bin/env python3
"""
02_preprocess_all.py — Collect a limited number of CIFs, sanitise them
with visible progress, build a MOF registry, and create train/val/test splits.

Why this version is safer for a flexible pipeline
-------------------------------------------------
1. Selects only a controlled subset per source (e.g. 50 core + 50 arc).
2. Supports 0 = all for each source.
3. Uses exactly the selected CIF list for sanitisation.
4. Writes manifests of the selected raw and sanitised CIFs.
5. Optionally synchronises the sanitised CIF directory so stale old CIFs
   from previous runs do not leak into later pipeline steps.
6. Prints live progress so the run does not look stuck.
7. Suppresses repetitive ASE symmetry warnings by default.
8. Preserves source labels correctly.
9. Avoids filename collisions in the sanitised directory.
10. Saves a detailed sanitisation report.
11. Supports resume mode.

Examples
--------
Small test:
python scripts/02_preprocess_all.py --raw-dir data/raw --output-dir data --max-core 50 --max-arc 50

Different size later, without changing code:
python scripts/02_preprocess_all.py --raw-dir data/raw --output-dir data --max-core 20 --max-arc 80

Use everything:
python scripts/02_preprocess_all.py --raw-dir data/raw --output-dir data --max-core 0 --max-arc 0

Resume:
python scripts/02_preprocess_all.py --raw-dir data/raw --output-dir data --max-core 50 --max-arc 50 --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing.sanitize import CIFSanitizer
from src.data.datasets.splitter import DataSplitter


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def configure_warning_filters(show_ase_warnings: bool) -> None:
    """
    ASE emits a very repetitive warning:
      "crystal system '...' is not interpreted for space group ..."

    This is usually not fatal. By default we suppress it so progress
    remains readable.
    """
    if not show_ase_warnings:
        warnings.filterwarnings(
            "ignore",
            message=r"crystal system '.*' is not interpreted for space group .*",
        )


def make_unique_sanitized_name(source: str, source_root: Path, cif_path: Path) -> str:
    """
    Create a collision-safe flattened filename for the sanitised CIF directory.

    We keep:
      source + relative path stem + short hash
    so identical names from different folders do not overwrite each other.
    """
    rel = cif_path.relative_to(source_root)
    rel_no_suffix = rel.with_suffix("")
    flat_rel = "__".join(rel_no_suffix.parts)
    digest = hashlib.md5(str(rel).encode("utf-8")).hexdigest()[:10]
    return f"{source}__{flat_rel}__{digest}.cif"


def collect_limited_cifs(
    source_dir: Path,
    source_name: str,
    limit: int,
    shuffle_selection: bool,
    rng: random.Random,
) -> List[Dict[str, str]]:
    """
    Collect up to `limit` CIFs recursively from one source directory.

    limit = 0 means "all".
    """
    discovered: List[Dict[str, str]] = []

    if not source_dir.exists():
        return discovered

    found = sorted(source_dir.rglob("*.cif"))
    n_available = len(found)

    if shuffle_selection:
        rng.shuffle(found)

    if limit < 0:
        raise ValueError(f"Limit for source '{source_name}' cannot be negative: {limit}")

    if limit > 0:
        found = found[:limit]

    print(f"  {source_dir.name}: available={n_available:,} | selected={len(found):,}")

    for cif in found:
        rel = cif.relative_to(source_dir)
        discovered.append(
            {
                "source": source_name,
                "source_dir_name": source_dir.name,
                "source_root": str(source_dir),
                "relative_path": str(rel).replace("\\", "/"),
                "input_path": str(cif),
                "mof_id_raw": cif.stem,
            }
        )

    return discovered


def discover_cifs(
    raw_dir: Path,
    max_core: int,
    max_arc: int,
    shuffle_selection: bool,
    seed: int,
) -> List[Dict[str, str]]:
    """
    Discover a limited number of CIFs recursively and preserve their source.
    """
    source_dirs = [
        ("core", raw_dir / "core_mof_2019", max_core),
        ("arc", raw_dir / "arc_mof_charges", max_arc),
    ]

    discovered: List[Dict[str, str]] = []
    rng = random.Random(seed)

    print("=" * 70)
    print("STEP 1: Collecting selected CIF files")
    print("=" * 70)

    for source_name, source_dir, limit in source_dirs:
        if not source_dir.exists():
            print(f"  {source_dir.name}: folder not found")
            continue

        discovered.extend(
            collect_limited_cifs(
                source_dir=source_dir,
                source_name=source_name,
                limit=limit,
                shuffle_selection=shuffle_selection,
                rng=rng,
            )
        )

    print(f"  Total selected: {len(discovered):,} raw CIFs")
    return discovered


def sync_sanitized_directory(
    discovered: List[Dict[str, str]],
    san_dir: Path,
) -> Set[str]:
    """
    Make the sanitised CIF directory match the current selected input set.

    This prevents stale CIFs from previous runs from leaking into later steps.
    Returns the set of expected sanitised filenames for the current run.
    """
    san_dir.mkdir(parents=True, exist_ok=True)

    expected_names: Set[str] = set()
    for item in discovered:
        source = item["source"]
        source_root = Path(item["source_root"])
        input_path = Path(item["input_path"])
        expected_names.add(make_unique_sanitized_name(source, source_root, input_path))

    existing_names = {p.name for p in san_dir.glob("*.cif")}
    stale_names = sorted(existing_names - expected_names)

    if stale_names:
        print("\n" + "=" * 70)
        print("SYNC: Removing stale sanitised CIFs from previous runs")
        print("=" * 70)
        print(f"  Stale files to remove: {len(stale_names):,}")

        for name in stale_names:
            stale_path = san_dir / name
            try:
                stale_path.unlink()
            except FileNotFoundError:
                pass

    else:
        print("\n" + "=" * 70)
        print("SYNC: Sanitised CIF directory already matches current selection")
        print("=" * 70)

    print(f"  Active sanitised targets for this run: {len(expected_names):,}")
    return expected_names


def save_selection_manifest(
    discovered: List[Dict[str, str]],
    san_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Save a manifest of the exact raw CIFs selected for this run and the
    corresponding expected sanitised output paths.
    """
    rows = []
    for item in discovered:
        source = item["source"]
        source_root = Path(item["source_root"])
        input_path = Path(item["input_path"])
        out_name = make_unique_sanitized_name(source, source_root, input_path)
        output_path = san_dir / out_name

        rows.append(
            {
                "source": source,
                "source_dir_name": item["source_dir_name"],
                "source_root": item["source_root"],
                "relative_path": item["relative_path"],
                "input_path": item["input_path"],
                "mof_id_raw": item["mof_id_raw"],
                "sanitized_name": out_name,
                "sanitized_path": str(output_path),
            }
        )

    manifest_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_dir / "selected_cifs_manifest.csv", index=False)
    manifest_df.to_parquet(output_dir / "selected_cifs_manifest.parquet", index=False)
    return manifest_df


def sanitize_all(
    discovered: List[Dict[str, str]],
    san_dir: Path,
    sanitizer: CIFSanitizer,
    resume: bool,
    print_every: int,
) -> Tuple[List[Dict], int, int]:
    """
    Sanitise each CIF one by one with visible progress.
    """
    san_dir.mkdir(parents=True, exist_ok=True)

    reports: List[Dict] = []
    n_total = len(discovered)
    n_done = 0
    n_valid = 0

    print("\n" + "=" * 70)
    print("STEP 2: Sanitising selected CIFs (with live progress)")
    print("=" * 70)

    start = time.time()
    last_print = start

    for idx, item in enumerate(discovered, start=1):
        source = item["source"]
        source_root = Path(item["source_root"])
        input_path = Path(item["input_path"])

        out_name = make_unique_sanitized_name(source, source_root, input_path)
        output_path = san_dir / out_name

        if resume and output_path.exists():
            report = {
                "input": str(input_path),
                "output": str(output_path),
                "valid": True,
                "warnings": ["Skipped existing output (resume mode)."],
                "n_atoms_raw": None,
                "n_atoms_clean": None,
                "source": source,
                "relative_path": item["relative_path"],
                "mof_id_raw": item["mof_id_raw"],
                "mof_id": output_path.stem,
                "status": "skipped_existing",
            }
        else:
            report = sanitizer.sanitize(input_path, output_path)
            report["source"] = source
            report["relative_path"] = item["relative_path"]
            report["mof_id_raw"] = item["mof_id_raw"]
            report["mof_id"] = output_path.stem
            report["status"] = "sanitized"

        reports.append(report)
        n_done += 1
        if bool(report.get("valid", False)):
            n_valid += 1

        should_print = (
            idx == 1
            or idx == n_total
            or idx % max(print_every, 1) == 0
            or (time.time() - last_print) > 15
        )

        if should_print:
            elapsed = time.time() - start
            rate = n_done / elapsed if elapsed > 0 else 0.0
            remaining = (n_total - n_done) / rate if rate > 0 else float("inf")

            n_invalid = n_done - n_valid
            print(
                f"  Progress: {n_done:,}/{n_total:,} "
                f"({100.0 * n_done / max(n_total, 1):.2f}%) | "
                f"valid={n_valid:,} invalid={n_invalid:,} | "
                f"rate={rate:.2f} CIF/s | "
                f"ETA={remaining/60:.2f} min"
            )
            last_print = time.time()

    total_elapsed = time.time() - start
    print(f"\n  Sanitisation finished in {total_elapsed/60:.2f} min")
    print(f"  Valid:   {n_valid:,}/{n_total:,}")
    print(f"  Invalid: {n_total - n_valid:,}/{n_total:,}")

    return reports, n_valid, n_total


def build_registry_from_reports(
    reports: List[Dict],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Build the MOF registry directly from the sanitisation reports.
    """
    valid_reports = [r for r in reports if bool(r.get("valid", False))]

    registry_rows = []
    for r in valid_reports:
        out_path = Path(r["output"])
        if not out_path.exists():
            continue

        registry_rows.append(
            {
                "mof_id": out_path.stem,
                "mof_id_raw": r.get("mof_id_raw"),
                "cif_path": str(out_path),
                "source": r.get("source"),
                "relative_path": r.get("relative_path"),
                "file_size": out_path.stat().st_size,
                "n_atoms_raw": r.get("n_atoms_raw"),
                "n_atoms_clean": r.get("n_atoms_clean"),
                "n_warnings": len(r.get("warnings", [])),
            }
        )

    registry_df = pd.DataFrame(registry_rows)
    if not registry_df.empty:
        registry_df = registry_df.drop_duplicates(subset=["mof_id"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    registry_df.to_parquet(output_dir / "mof_registry.parquet", index=False)
    registry_df.to_csv(output_dir / "mof_registry.csv", index=False)

    return registry_df


def save_sanitization_reports(reports: List[Dict], output_dir: Path) -> None:
    """
    Save detailed reports for debugging and auditing.
    """
    rows = []
    for r in reports:
        rows.append(
            {
                "input": r.get("input"),
                "output": r.get("output"),
                "valid": bool(r.get("valid", False)),
                "source": r.get("source"),
                "relative_path": r.get("relative_path"),
                "mof_id_raw": r.get("mof_id_raw"),
                "mof_id": r.get("mof_id"),
                "status": r.get("status"),
                "n_atoms_raw": r.get("n_atoms_raw"),
                "n_atoms_clean": r.get("n_atoms_clean"),
                "warnings_json": json.dumps(r.get("warnings", []), ensure_ascii=False),
            }
        )

    rep_df = pd.DataFrame(rows)
    rep_df.to_parquet(output_dir / "sanitization_report.parquet", index=False)
    rep_df.to_csv(output_dir / "sanitization_report.csv", index=False)


def create_splits(registry_df: pd.DataFrame, out_dir: Path, seed: int) -> None:
    print("\n" + "=" * 70)
    print("STEP 4: Creating train/val/test splits")
    print("=" * 70)

    if len(registry_df) == 0:
        print("  No MOFs to split.")
        return

    mof_ids = registry_df["mof_id"].tolist()
    splitter = DataSplitter(
        method="random",
        test_size=0.1,
        val_size=0.1,
        random_state=seed,
    )
    train_idx, val_idx, test_idx = splitter.split(mof_ids)
    splitter.save_splits(mof_ids, train_idx, val_idx, test_idx, out_dir / "splits.json")

    print(f"  Train: {len(train_idx):,}")
    print(f"  Val:   {len(val_idx):,}")
    print(f"  Test:  {len(test_idx):,}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MOF data with visible progress")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--max-core",
        type=int,
        default=50,
        help="Maximum number of CIFs to take from core_mof_2019 (0 = all).",
    )
    parser.add_argument(
        "--max-arc",
        type=int,
        default=50,
        help="Maximum number of CIFs to take from arc_mof_charges (0 = all).",
    )

    parser.add_argument(
        "--shuffle-selection",
        action="store_true",
        help="Shuffle each source before taking the top N CIFs.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip CIFs whose sanitised output already exists.",
    )

    parser.add_argument(
        "--no-sync-sanitized-dir",
        dest="sync_sanitized_dir",
        action="store_false",
        help="Do not remove stale sanitised CIFs from previous runs.",
    )
    parser.set_defaults(sync_sanitized_dir=True)

    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print sanitisation progress every N CIFs.",
    )
    parser.add_argument(
        "--show-ase-warnings",
        action="store_true",
        help="Show repetitive ASE symmetry warnings instead of suppressing them.",
    )
    args = parser.parse_args()

    configure_warning_filters(show_ase_warnings=args.show_ase_warnings)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)

    # CORRECTED:
    # If output-dir is "data", this should become "data/intermediate", not "intermediate".
    inter_dir = out_dir / "intermediate"
    san_dir = inter_dir / "cifs_sanitized"

    discovered = discover_cifs(
        raw_dir=raw_dir,
        max_core=args.max_core,
        max_arc=args.max_arc,
        shuffle_selection=args.shuffle_selection,
        seed=args.seed,
    )

    if not discovered:
        raise FileNotFoundError("No CIFs selected. Check raw data folders and limits.")

    out_dir.mkdir(parents=True, exist_ok=True)
    inter_dir.mkdir(parents=True, exist_ok=True)
    san_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = save_selection_manifest(
        discovered=discovered,
        san_dir=san_dir,
        output_dir=out_dir,
    )
    print(f"  Selection manifest -> {out_dir / 'selected_cifs_manifest.csv'}")

    if args.sync_sanitized_dir:
        sync_sanitized_directory(
            discovered=discovered,
            san_dir=san_dir,
        )

    sanitizer = CIFSanitizer(
        min_atoms=5,
        max_atoms=5000,
    )

    reports, n_valid, n_total = sanitize_all(
        discovered=discovered,
        san_dir=san_dir,
        sanitizer=sanitizer,
        resume=args.resume,
        print_every=args.print_every,
    )

    print("\n" + "=" * 70)
    print("STEP 3: Building MOF registry")
    print("=" * 70)

    save_sanitization_reports(reports, out_dir)
    registry_df = build_registry_from_reports(reports, out_dir)

    print(f"  Registry: {len(registry_df):,} MOFs -> {out_dir / 'mof_registry.parquet'}")
    print(f"  Sanitisation report -> {out_dir / 'sanitization_report.parquet'}")

    create_splits(registry_df, out_dir, seed=args.seed)

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print(f"Selected raw CIFs:      {len(manifest_df):,}")
    print(f"Valid sanitised CIFs:   {len(registry_df):,}")
    print(f"Sanitised CIF folder:   {san_dir}")
    print(f"Registry file:          {out_dir / 'mof_registry.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()