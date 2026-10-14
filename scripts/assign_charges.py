#!/usr/bin/env python3
"""
assign_charges.py
Give every sanitised CIF real per-atom partial charges before GCMC.

ARC-MOF CIFs already carry REPEAT charges. CoRE MOF 2019 CIFs carry none --
the sanitiser writes their _atom_site_charge column as all 0.000000 -- so
231/500 selected MOFs were simulated (and featurised for the model) with no
framework charges at all. This script fills those in with PACMAN-charge's
REPEAT model (Zhao et al., PACMAN), i.e. the SAME charge scheme ARC-MOF uses,
so the whole dataset has one consistent charge definition. Checked against
real ARC REPEAT charges on 8 selected ARC MOFs: per-atom MAE 0.02-0.07 e,
correlation >= 0.977, net charge < 1e-5 e.

Only CIFs whose charges are all zero are touched; ARC charges are kept. Each
modified CIF is backed up first, only the charge column is rewritten (atom
order, labels and coordinates are untouched), and every decision is logged
to data/intermediate/charge_assignment.csv.

PACMAN needs torch + pymatgen + setuptools<81, so run this from its own venv
instead of .venv:
    python3 -m venv charges_env
    charges_env/bin/pip install PACMAN-charge "setuptools<81" ase pandas
    charges_env/bin/python scripts/assign_charges.py [--only-selected] [--dry-run]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils.cif_charges import CHARGE_COLUMN_NAMES, read_site_charges  # noqa: E402

CIF_DIR = ROOT / "data/intermediate/cifs_sanitized"
BACKUP_DIR = ROOT / "data/intermediate/cifs_sanitized_uncharged_backup"
MANIFEST = ROOT / "data/intermediate/charge_assignment.csv"
SELECTED = ROOT / "data/processed/adsorption_v3/gcmc_selected_mofs.txt"
CHARGE_TYPE = "REPEAT"


def symbols_in_file_order(cif: Path) -> list:
    from ase.io import read
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read(str(cif)).get_chemical_symbols()


def pacman_charges(cif: Path, n_atoms: int) -> np.ndarray:
    """Run PACMAN on a temp copy; return charges in the input's atom order."""
    from PACMANCharge import pmcharge
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / cif.name
        shutil.copy2(cif, src)
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pmcharge.predict(cif_file=str(src), charge_type=CHARGE_TYPE, digits=6,
                             atom_type=True, neutral=True, keep_connect=True)
        outs = sorted(Path(tmp).glob(f"{cif.stem}*pacman*.cif"))
        if not outs:
            raise RuntimeError("PACMAN wrote no output CIF")
        if symbols_in_file_order(outs[0]) != symbols_in_file_order(cif):
            raise RuntimeError("PACMAN output atom order differs from input")
        q = read_site_charges(outs[0], n_atoms)
        if q is None:
            raise RuntimeError("PACMAN output has no charge column for every atom")
        return q.astype(np.float64)


def rewrite_charge_column(cif: Path, charges: np.ndarray) -> None:
    """Replace the values of the atom_site charge column in place."""
    lines = cif.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            headers, j = [], i + 1
            while j < len(lines) and lines[j].strip().startswith("_"):
                headers.append(lines[j].strip())
                j += 1
            q_hdr = next((h for h in headers if h in CHARGE_COLUMN_NAMES), None)
            if "_atom_site_fract_x" in headers and q_hdr is not None:
                q_idx, k = headers.index(q_hdr), 0
                while j < len(lines):
                    s = lines[j].strip()
                    if not s or s.startswith("_") or s == "loop_" or s.startswith("#"):
                        break
                    row = lines[j].split()
                    row[q_idx] = f"{charges[k]:.6f}"
                    lines[j] = " ".join(row)
                    k += 1
                    j += 1
                if k != len(charges):
                    raise RuntimeError(f"wrote {k} charges, expected {len(charges)}")
                cif.write_text("\n".join(lines) + "\n", encoding="utf-8")
                return
            i = j
        else:
            i += 1
    raise RuntimeError("no atom_site loop with a charge column")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only-selected", action="store_true",
                    help=f"only MOFs listed in {SELECTED.relative_to(ROOT)}")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    ap.add_argument("--cif-dir", type=Path, default=CIF_DIR)
    ap.add_argument("--backup-dir", type=Path, default=BACKUP_DIR)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    a = ap.parse_args()

    cifs = sorted(a.cif_dir.glob("*.cif"))
    if a.only_selected:
        wanted = set(SELECTED.read_text().split())
        cifs = [c for c in cifs if c.stem in wanted]

    rows = []
    for n, cif in enumerate(cifs, 1):
        n_atoms = len(symbols_in_file_order(cif))
        q = read_site_charges(cif, n_atoms)
        if q is not None and np.any(np.abs(q) > 1e-4):
            rows.append({"mof_id": cif.stem, "method": "source (kept)",
                         "net_charge": float(q.sum()), "max_abs_charge": float(np.abs(q).max())})
            continue
        if a.dry_run:
            rows.append({"mof_id": cif.stem, "method": f"PACMAN-{CHARGE_TYPE} (would assign)"})
            continue
        try:
            q_new = pacman_charges(cif, n_atoms)
            a.backup_dir.mkdir(parents=True, exist_ok=True)
            backup = a.backup_dir / cif.name
            if not backup.exists():
                shutil.copy2(cif, backup)
            rewrite_charge_column(cif, q_new)
            rows.append({"mof_id": cif.stem, "method": f"PACMAN-{CHARGE_TYPE}",
                         "net_charge": float(q_new.sum()),
                         "max_abs_charge": float(np.abs(q_new).max())})
            print(f"[{n}/{len(cifs)}] {cif.stem}: assigned (net {q_new.sum():+.1e})", flush=True)
        except Exception as e:
            rows.append({"mof_id": cif.stem, "method": "FAILED", "error": str(e)})
            print(f"[{n}/{len(cifs)}] {cif.stem}: FAILED ({e})", flush=True)

    df = pd.DataFrame(rows)
    if not a.dry_run:
        df.to_csv(a.manifest, index=False)
    print("\n" + df["method"].value_counts().to_string())
    if not a.dry_run:
        print(f"Manifest -> {a.manifest}; originals -> {a.backup_dir}")


if __name__ == "__main__":
    main()
