#!/usr/bin/env python3
"""
prepare_level4_cifs.py -- sanitise the four Level 4 benchmark frameworks.

The frameworks with abundant experimental isotherms in NIST ISODB are taken
from the raw CoRE MOF 2019 download and passed through the same CIFSanitizer
and file naming as the training data (scripts/02_preprocess_all.py). The CIF
version of each was chosen by checking its composition against the ideal
formula and its shortest interatomic distance:

  HKUST-1    FIQCEN_clean     Cu12 C72 H24 O48   = 4 x Cu3(BTC)2
  Mg-MOF-74  VOGTIV_clean_h   Mg6 C24 H6 O18     = 3 x Mg2(dobdc)
  ZIF-8      OFERUN_clean     Zn6 C48 H60 N24    = 6 x Zn(mIm)2  (OFERUN06 lacks methyl H)
  UiO-66     RUBTAK02_SL      Zr6 C48 H28 O32    = Zr6O4(OH)4(BDC)6  (RUBTAK01 has a 0.81 A contact,
                                                    RUBTAK04-09 overlapping sites)

Then assign PACMAN-REPEAT charges, exactly as for the training CoRE MOFs:
    thesis_env/bin/python  scripts/validation/prepare_level4_cifs.py
    charges_env/bin/python scripts/assign_charges.py \
        --cif-dir data/intermediate/validation_cifs_level4 \
        --backup-dir data/intermediate/validation_cifs_level4_uncharged \
        --manifest data/processed/validation_gcmc/level4_charges.csv
"""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/core_mof_2019"
OUT_CIF = ROOT / "data/intermediate/validation_cifs_level4"
OUT = ROOT / "data/processed/validation_gcmc"

BENCHMARKS = {
    "HKUST-1":   "2019-07-01-ASR-public_12020/disorder_360/FIQCEN_clean.cif",
    "Mg-MOF-74": "2019-07-01-ASR-public_12020/structure_11660/VOGTIV_clean_h.cif",
    "ZIF-8":     "2019-07-01-ASR-public_12020/structure_11660/OFERUN_clean.cif",
    "UiO-66":    "2019-07-01-ASR-public_12020/structure_11660/RUBTAK02_SL.cif",
}


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from src.data.preprocessing.sanitize import CIFSanitizer

    # Reuse the training-data naming so mof_ids look the same everywhere.
    spec = importlib.util.spec_from_file_location("prep", ROOT / "scripts/02_preprocess_all.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)

    OUT_CIF.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    sanitizer = CIFSanitizer(min_atoms=5, max_atoms=5000)
    rows = []
    for name, rel in BENCHMARKS.items():
        src = RAW / rel
        out = OUT_CIF / prep.make_unique_sanitized_name("core", RAW, src)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = sanitizer.sanitize(src, out)
        ok = bool(report.get("valid", report.get("success", out.exists())))
        rows.append({"name": name, "mof_id": out.stem, "source_cif": rel, "sanitized": ok})
        print(f"{name:10s} {'ok' if ok else 'FAILED'}  {out.name}")
    pd.DataFrame(rows).to_csv(OUT / "level4_mofs.csv", index=False)
    print(f"-> {OUT / 'level4_mofs.csv'}")


if __name__ == "__main__":
    main()
