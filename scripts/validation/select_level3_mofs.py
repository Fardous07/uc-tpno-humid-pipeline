#!/usr/bin/env python3
"""
select_level3_mofs.py -- pick the Level 3 "unseen framework" validation set.

Candidates are admissible frameworks (sanitised CIFs that passed the
pre-flight checks) that were NOT selected for Stage 3. A random sample,
stratified by source (ARC-MOF / CoRE MOF 2019) in proportion to the
candidate pool, is drawn; candidates that are structural duplicates of a
Stage 3 framework, or of an already drawn candidate, are skipped
(pymatgen StructureMatcher, same settings as the Stage 3 duplicate check).

Run with charges_env (needs pymatgen):
    charges_env/bin/python scripts/validation/select_level3_mofs.py [--n 30]

Writes data/processed/validation_gcmc/level3_mofs.txt and
level3_selection.csv (every candidate examined and why it was kept/skipped).
"""
from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path

import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[2]
CIF_DIR = ROOT / "data/intermediate/cifs_sanitized"
V3 = ROOT / "data/processed/adsorption_v3"
OUT = ROOT / "data/processed/validation_gcmc"

MATCHER = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5,
                           primitive_cell=True, scale=True, attempt_supercell=False)


def load(mof: str) -> Structure:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Structure.from_file(str(CIF_DIR / f"{mof}.cif"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    all_mofs = sorted(p.stem for p in CIF_DIR.glob("*.cif"))
    rejected = set(pd.read_csv(V3 / "rejected_frameworks.csv").mof_id)
    selected = set((V3 / "gcmc_selected_mofs.txt").read_text().split())
    training = set((V3 / "gcmc_selected_mofs_dedup.txt").read_text().split())
    pool = [m for m in all_mofs if m not in rejected and m not in selected]
    print(f"{len(all_mofs)} CIFs, {len(rejected)} rejected, {len(selected)} selected "
          f"-> {len(pool)} candidates")

    rng = random.Random(a.seed)
    by_src = {s: [m for m in pool if m.startswith(s)] for s in ("arc", "core")}
    for v in by_src.values():
        rng.shuffle(v)
    quota = {"arc": round(a.n * len(by_src["arc"]) / len(pool))}
    quota["core"] = a.n - quota["arc"]

    # Reference structures grouped by reduced formula (only same-formula
    # pairs can match, so this keeps the comparisons cheap).
    ref: dict[str, list[tuple[str, Structure]]] = {}
    for m in sorted(training):
        s = load(m)
        ref.setdefault(s.composition.reduced_formula, []).append((m, s))

    log, picked = [], []
    for src, cands in by_src.items():
        n_src = 0
        for m in cands:
            if n_src == quota[src]:
                break
            s = load(m)
            f = s.composition.reduced_formula
            dup = next((r for r, rs in ref.get(f, []) if MATCHER.fit(s, rs)), None)
            if dup:
                log.append({"mof_id": m, "source": src, "kept": False,
                            "reason": f"duplicate of {dup}"})
                continue
            picked.append(m)
            n_src += 1
            ref.setdefault(f, []).append((m, s))   # also avoid duplicates within the set
            log.append({"mof_id": m, "source": src, "kept": True, "reason": ""})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "level3_mofs.txt").write_text("\n".join(picked) + "\n")
    pd.DataFrame(log).to_csv(OUT / "level3_selection.csv", index=False)
    skipped = [r for r in log if not r["kept"]]
    print(f"Picked {len(picked)} ({quota['arc']} ARC-MOF, {quota['core']} CoRE); "
          f"skipped {len(skipped)} duplicates")
    for r in skipped:
        print(f"  skipped {r['mof_id'][:60]}  ({r['reason'][:70]})")
    print(f"-> {OUT / 'level3_mofs.txt'}")


if __name__ == "__main__":
    main()
