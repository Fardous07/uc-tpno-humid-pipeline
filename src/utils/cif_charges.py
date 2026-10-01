"""
cif_charges.py
Read per-atom partial charges from a (sanitised, P1) CIF.

Shared by the GCMC label generator (scripts/04_run_simulations.py) and the
graph builder (src/data/preprocessing/graph_builder.py) so the simulation
and the model's charge node feature always come from the same numbers.
Deliberately torch-free.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

# Charge column names we accept when reading a sanitised CIF.
CHARGE_COLUMN_NAMES = (
    "_atom_site_charge",
    "_atom_type_partial_charge",
    "_atom_site_partial_charge",
    "_atom_site_charge_partial",
)


def read_site_charges(
    cif_path: Union[str, Path], n_atoms: int
) -> Optional[np.ndarray]:
    """
    Fallback per-atom charge reader.

    Parses the charge column of the atom_site loop directly from the CIF
    text, in FILE ORDER (which matches ASE's atom order for P1 cells).
    Returns a ``[n_atoms]`` float array, or ``None`` if no charge column
    is found or the atom count does not match (so the caller can fall
    back to zeros safely).
    """
    try:
        lines = Path(cif_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None

    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            headers: List[str] = []
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("_"):
                headers.append(lines[j].strip())
                j += 1

            has_coords = "_atom_site_fract_x" in headers
            charge_hdr = next((h for h in headers if h in CHARGE_COLUMN_NAMES), None)

            if has_coords and charge_hdr is not None:
                q_idx = headers.index(charge_hdr)
                vals: List[float] = []
                while j < len(lines):
                    s = lines[j].strip()
                    if not s or s.startswith("_") or s == "loop_" or s.startswith("#"):
                        break
                    row = lines[j].split()
                    if len(row) > q_idx:
                        try:
                            vals.append(float(row[q_idx]))
                        except ValueError:
                            vals.append(0.0)
                    j += 1
                arr = np.asarray(vals, dtype=np.float32)
                return arr if len(arr) == n_atoms else None
            i = j
        else:
            i += 1
    return None
