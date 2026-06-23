#!/usr/bin/env python3
"""
CIF file sanitisation for MOF structures.

Robust version for heterogeneous MOF CIF sources.

Main goals
----------
1. Read cell parameters safely, including CIF uncertainty notation like 12.345(6)
2. Parse atom rows from common atom-site loop layouts
3. Support charges from either:
      - _atom_site_charge
      - separate _atom_type_symbol / _atom_type_partial_charge loop
4. Fall back to ASE when direct parsing is not enough
5. Write a minimal P1 CIF for downstream use
6. Match RASPA's CIF examples more closely:
      - use _symmetry_space_group_name_Hall 'P 1'
      - use _symmetry_space_group_name_H-M 'P 1'
      - use _symmetry_equiv_pos_as_xyz as a SINGLE DATA ITEM, not a loop
      - keep only one atom loop
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

_ASE_AVAILABLE = False
try:
    from ase.io import read as ase_read
    _ASE_AVAILABLE = True
except ImportError:
    ase_read = None  # type: ignore


# ═══════════════════════════════════════════════════════════════════════
# 1. LOW-LEVEL CIF UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def _strip_cif_uncertainty(token: str) -> str:
    """
    Convert CIF numeric tokens like:
      13.693641(2) -> 13.693641
      -0.081179(10) -> -0.081179
    """
    token = token.strip().strip("'").strip('"')
    token = re.sub(r"\([^)]+\)$", "", token)
    return token


def _safe_float(token: str) -> Optional[float]:
    try:
        return float(_strip_cif_uncertainty(token))
    except Exception:
        return None


def _tokenize_cif_line(line: str) -> List[str]:
    """
    Tokenize a CIF row while preserving quoted chunks.
    """
    return re.findall(r"""(?:'[^']*'|"[^"]*"|\S+)""", line.strip())


def _infer_symbol_from_label(label: str) -> str:
    """
    Infer element symbol from a CIF atom label if _atom_site_type_symbol is absent.
    """
    s = label.strip().strip("'").strip('"')
    m = re.match(r"([A-Z][a-z]?)", s)
    if m:
        return m.group(1)
    return "X"


def _find_all_loops(lines: List[str]) -> List[Tuple[List[str], List[List[str]]]]:
    """
    Parse CIF loop_ blocks into:
        [(headers, rows_as_token_lists), ...]
    """
    loops: List[Tuple[List[str], List[List[str]]]] = []
    i = 0
    n = len(lines)

    while i < n:
        s = lines[i].strip()
        if s != "loop_":
            i += 1
            continue

        i += 1
        headers: List[str] = []
        while i < n and lines[i].strip().startswith("_"):
            headers.append(lines[i].strip())
            i += 1

        rows: List[List[str]] = []
        while i < n:
            s = lines[i].strip()

            if not s:
                i += 1
                continue

            if s == "loop_" or s.startswith("_") or s.startswith("data_"):
                break

            if not s.startswith("#"):
                rows.append(_tokenize_cif_line(lines[i]))
            i += 1

        loops.append((headers, rows))

    return loops


def _extract_data_name(text: str, fallback: str) -> str:
    m = re.search(r"^\s*data_(\S+)", text, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    return fallback


def read_cell_from_cif(cif_path: Union[str, Path]) -> Optional[Dict[str, float]]:
    """
    Extract unit-cell a, b, c, alpha, beta, gamma from CIF header.
    Handles uncertainty notation like 12.34(5).
    """
    text = Path(cif_path).read_text(encoding="utf-8", errors="replace")

    params: Dict[str, float] = {}
    for tag, key in [
        ("_cell_length_a", "a"),
        ("_cell_length_b", "b"),
        ("_cell_length_c", "c"),
        ("_cell_angle_alpha", "alpha"),
        ("_cell_angle_beta", "beta"),
        ("_cell_angle_gamma", "gamma"),
    ]:
        m = re.search(
            rf"^\s*{re.escape(tag)}\s+([^\s]+)",
            text,
            flags=re.MULTILINE,
        )
        if m:
            val = _safe_float(m.group(1))
            if val is not None:
                params[key] = val

    return params if len(params) == 6 else None


def _compute_cell_volume(cell: Dict[str, float]) -> float:
    """
    Compute unit-cell volume from lengths and angles.
    """
    import math

    a = cell["a"]
    b = cell["b"]
    c = cell["c"]
    alpha = math.radians(cell["alpha"])
    beta = math.radians(cell["beta"])
    gamma = math.radians(cell["gamma"])

    term = (
        1
        - math.cos(alpha) ** 2
        - math.cos(beta) ** 2
        - math.cos(gamma) ** 2
        + 2 * math.cos(alpha) * math.cos(beta) * math.cos(gamma)
    )
    term = max(term, 0.0)
    return a * b * c * math.sqrt(term)


def _guess_cell_setting(cell: Dict[str, float]) -> str:
    """
    Lightweight crystal-system guess for cleaner CIF headers.
    For P1 we only need something reasonable; triclinic is always safe.
    """
    return "triclinic"


# ═══════════════════════════════════════════════════════════════════════
# 2. ATOM EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def _build_type_charge_map(
    loops: List[Tuple[List[str], List[List[str]]]]
) -> Dict[str, float]:
    """
    Parse:
      loop_
      _atom_type_symbol
      _atom_type_partial_charge
      C -0.08
      H  0.11
    """
    charge_map: Dict[str, float] = {}

    for headers, rows in loops:
        hs = set(headers)
        if "_atom_type_symbol" in hs and "_atom_type_partial_charge" in hs:
            idx = {h: i for i, h in enumerate(headers)}
            s_idx = idx["_atom_type_symbol"]
            q_idx = idx["_atom_type_partial_charge"]

            for row in rows:
                if len(row) <= max(s_idx, q_idx):
                    continue
                sym = row[s_idx].strip().strip("'").strip('"')
                q = _safe_float(row[q_idx])
                if sym and q is not None:
                    charge_map[sym] = q

    return charge_map


def _extract_atom_rows_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Parse atom rows from common CIF atom-site loop patterns.
    """
    lines = text.splitlines()
    loops = _find_all_loops(lines)
    type_charge_map = _build_type_charge_map(loops)

    atom_rows: List[Dict[str, Any]] = []

    for headers, rows in loops:
        hs = set(headers)

        required = {
            "_atom_site_label",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        }

        if not required.issubset(hs):
            continue

        idx = {h: i for i, h in enumerate(headers)}

        has_symbol = "_atom_site_type_symbol" in idx
        has_charge = "_atom_site_charge" in idx

        needed_idx = [
            idx["_atom_site_label"],
            idx["_atom_site_fract_x"],
            idx["_atom_site_fract_y"],
            idx["_atom_site_fract_z"],
        ]
        if has_symbol:
            needed_idx.append(idx["_atom_site_type_symbol"])
        if has_charge:
            needed_idx.append(idx["_atom_site_charge"])

        parsed_rows: List[Dict[str, Any]] = []

        for row in rows:
            if len(row) <= max(needed_idx):
                continue

            label = row[idx["_atom_site_label"]].strip().strip("'").strip('"')
            if has_symbol:
                symbol = row[idx["_atom_site_type_symbol"]].strip().strip("'").strip('"')
            else:
                symbol = _infer_symbol_from_label(label)

            fx = _safe_float(row[idx["_atom_site_fract_x"]])
            fy = _safe_float(row[idx["_atom_site_fract_y"]])
            fz = _safe_float(row[idx["_atom_site_fract_z"]])

            if not label or not symbol or fx is None or fy is None or fz is None:
                continue

            charge = 0.0
            if has_charge:
                q = _safe_float(row[idx["_atom_site_charge"]])
                if q is not None:
                    charge = q
            else:
                charge = float(type_charge_map.get(symbol, 0.0))

            parsed_rows.append(
                {
                    "symbol": symbol,
                    "label": label,
                    "fx": fx,
                    "fy": fy,
                    "fz": fz,
                    "charge": charge,
                }
            )

        if parsed_rows:
            return parsed_rows

    return atom_rows


def _extract_atom_rows_with_ase(
    cif_path: Path,
) -> Tuple[Optional[Dict[str, float]], List[Dict[str, Any]]]:
    """
    Fallback atom extraction using ASE.
    """
    if not _ASE_AVAILABLE:
        return None, []

    try:
        atoms = ase_read(str(cif_path))
    except Exception:
        return None, []

    try:
        cell_lengths = atoms.cell.lengths()
        cell_angles = atoms.cell.angles()
        cell = {
            "a": float(cell_lengths[0]),
            "b": float(cell_lengths[1]),
            "c": float(cell_lengths[2]),
            "alpha": float(cell_angles[0]),
            "beta": float(cell_angles[1]),
            "gamma": float(cell_angles[2]),
        }
    except Exception:
        cell = None

    scaled = atoms.get_scaled_positions(wrap=True)
    syms = atoms.get_chemical_symbols()

    rows: List[Dict[str, Any]] = []
    counters: Dict[str, int] = {}

    for sym, pos in zip(syms, scaled):
        counters[sym] = counters.get(sym, 0) + 1
        rows.append(
            {
                "symbol": sym,
                "label": f"{sym}{counters[sym]}",
                "fx": float(pos[0]),
                "fy": float(pos[1]),
                "fz": float(pos[2]),
                "charge": 0.0,
            }
        )

    return cell, rows


def count_atoms_in_cif(cif_path: Union[str, Path]) -> int:
    """
    Count atoms using direct loop parsing first, then ASE fallback.
    """
    path = Path(cif_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = _extract_atom_rows_from_text(text)
    if rows:
        return len(rows)

    _, rows_ase = _extract_atom_rows_with_ase(path)
    return len(rows_ase)


# ═══════════════════════════════════════════════════════════════════════
# 3. CLEAN CIF WRITER
# ═══════════════════════════════════════════════════════════════════════

def _write_clean_p1_cif(
    out_path: Path,
    data_name: str,
    cell: Dict[str, float],
    atom_rows: List[Dict[str, Any]],
) -> None:
    """
    Write a minimal RASPA-friendlier CIF following the style of working
    RASPA-distributed structures.

    Important detail:
    For P1, RASPA examples often use:
        _symmetry_equiv_pos_as_xyz 'x,y,z'
    as a single data item, NOT a loop_ block.
    """
    cell_volume = _compute_cell_volume(cell)
    cell_setting = _guess_cell_setting(cell)

    lines = [
        f"data_{data_name}",
        "",
        "_audit_creation_method 'uc_tpno_humid_pipeline'",
        "",
        f"_cell_length_a {cell['a']:.6f}",
        f"_cell_length_b {cell['b']:.6f}",
        f"_cell_length_c {cell['c']:.6f}",
        f"_cell_angle_alpha {cell['alpha']:.6f}",
        f"_cell_angle_beta {cell['beta']:.6f}",
        f"_cell_angle_gamma {cell['gamma']:.6f}",
        f"_cell_volume {cell_volume:.6f}",
        "",
        f"_symmetry_cell_setting {cell_setting}",
        "_symmetry_space_group_name_Hall 'P 1'",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_symmetry_Int_Tables_number 1",
        "_symmetry_equiv_pos_as_xyz 'x,y,z'",
        "",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_charge",
    ]

    for row in atom_rows:
        lines.append(
            f"{row['label']} {row['symbol']} "
            f"{row['fx']:.6f} {row['fy']:.6f} {row['fz']:.6f} {row['charge']:.6f}"
        )

    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════
# 4. OPTIONAL OVERLAP REMOVAL
# ═══════════════════════════════════════════════════════════════════════

def _remove_overlaps_with_ase(
    cif_path: Path,
    overlap_tol: float,
) -> Tuple[Optional[Dict[str, float]], List[Dict[str, Any]]]:
    """
    Optional cleanup pass:
      - read CIF with ASE
      - remove very close duplicate/overlapping atoms
      - return cleaned scaled positions
    """
    if not _ASE_AVAILABLE:
        return None, []

    try:
        atoms = ase_read(str(cif_path))
    except Exception:
        return None, []

    if len(atoms) < 2:
        try:
            cell_lengths = atoms.cell.lengths()
            cell_angles = atoms.cell.angles()
            cell = {
                "a": float(cell_lengths[0]),
                "b": float(cell_lengths[1]),
                "c": float(cell_lengths[2]),
                "alpha": float(cell_angles[0]),
                "beta": float(cell_angles[1]),
                "gamma": float(cell_angles[2]),
            }
        except Exception:
            cell = None

        scaled = atoms.get_scaled_positions(wrap=True)
        syms = atoms.get_chemical_symbols()
        counts: Dict[str, int] = {}
        rows: List[Dict[str, Any]] = []
        for sym, pos in zip(syms, scaled):
            counts[sym] = counts.get(sym, 0) + 1
            rows.append(
                {
                    "symbol": sym,
                    "label": f"{sym}{counts[sym]}",
                    "fx": float(pos[0]),
                    "fy": float(pos[1]),
                    "fz": float(pos[2]),
                    "charge": 0.0,
                }
            )
        return cell, rows

    try:
        from ase.geometry import get_distances
    except Exception:
        return None, []

    try:
        D, _ = get_distances(
            atoms.get_positions(),
            cell=atoms.get_cell(),
            pbc=True,
        )
    except Exception:
        return None, []

    keep = [True] * len(atoms)
    for i in range(len(atoms)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(atoms)):
            if keep[j] and D[i, j] < overlap_tol:
                keep[j] = False

    atoms = atoms[[i for i, k in enumerate(keep) if k]]

    try:
        cell_lengths = atoms.cell.lengths()
        cell_angles = atoms.cell.angles()
        cell = {
            "a": float(cell_lengths[0]),
            "b": float(cell_lengths[1]),
            "c": float(cell_lengths[2]),
            "alpha": float(cell_angles[0]),
            "beta": float(cell_angles[1]),
            "gamma": float(cell_angles[2]),
        }
    except Exception:
        cell = None

    scaled = atoms.get_scaled_positions(wrap=True)
    syms = atoms.get_chemical_symbols()

    counts: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    for sym, pos in zip(syms, scaled):
        counts[sym] = counts.get(sym, 0) + 1
        rows.append(
            {
                "symbol": sym,
                "label": f"{sym}{counts[sym]}",
                "fx": float(pos[0]),
                "fy": float(pos[1]),
                "fz": float(pos[2]),
                "charge": 0.0,
            }
        )

    return cell, rows


# ═══════════════════════════════════════════════════════════════════════
# 5. SANITIZER
# ═══════════════════════════════════════════════════════════════════════

class CIFSanitizer:
    """
    Clean and validate MOF CIF files.
    """

    def __init__(
        self,
        min_atoms: int = 3,
        max_atoms: int = 10_000,
        overlap_tol: float = 0.5,
        min_cell_length: float = 2.0,
        max_cell_length: float = 200.0,
    ):
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.overlap_tol = overlap_tol
        self.min_cell_length = min_cell_length
        self.max_cell_length = max_cell_length

    def sanitize(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Dict[str, Any]:
        inp = Path(input_path)
        out = Path(output_path)

        report: Dict[str, Any] = {
            "input": str(inp),
            "output": str(out),
            "valid": False,
            "warnings": [],
            "n_atoms_raw": 0,
            "n_atoms_clean": 0,
        }

        try:
            text = inp.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            report["warnings"].append(f"Cannot read CIF: {e}")
            return report

        data_name = _extract_data_name(text, inp.stem)

        cell = read_cell_from_cif(inp)
        atom_rows = _extract_atom_rows_from_text(text)

        if cell is None or not atom_rows:
            ase_cell, ase_rows = _extract_atom_rows_with_ase(inp)
            if cell is None:
                cell = ase_cell
            if not atom_rows:
                atom_rows = ase_rows
                if ase_rows:
                    report["warnings"].append("Used ASE fallback to recover atom rows.")

        if cell is None:
            report["warnings"].append("Cannot parse cell parameters.")
            return report

        for key in ("a", "b", "c"):
            L = cell[key]
            if L < self.min_cell_length or L > self.max_cell_length:
                report["warnings"].append(f"Cell {key}={L:.3f} Å out of range.")
                return report

        report["cell"] = cell
        report["n_atoms_raw"] = len(atom_rows)

        if len(atom_rows) < self.min_atoms:
            report["warnings"].append(f"Too few atoms ({len(atom_rows)}).")
            return report

        if len(atom_rows) > self.max_atoms:
            report["warnings"].append(f"Too many atoms ({len(atom_rows)}).")
            return report

        cleaned_rows = atom_rows
        cleaned_cell = cell

        if _ASE_AVAILABLE:
            tmp_cif: Optional[Path] = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
                    tmp_cif = Path(tmp.name)

                _write_clean_p1_cif(tmp_cif, data_name, cell, atom_rows)
                ase_cell2, ase_rows2 = _remove_overlaps_with_ase(
                    tmp_cif,
                    overlap_tol=self.overlap_tol,
                )

                if ase_cell2 is not None and len(ase_rows2) >= self.min_atoms:
                    if len(ase_rows2) == len(atom_rows):
                        for i in range(len(ase_rows2)):
                            ase_rows2[i]["charge"] = atom_rows[i].get("charge", 0.0)

                    cleaned_cell = ase_cell2
                    cleaned_rows = ase_rows2
            except Exception as e:
                report["warnings"].append(f"Overlap removal skipped: {e}")
            finally:
                if tmp_cif is not None and tmp_cif.exists():
                    try:
                        tmp_cif.unlink()
                    except Exception:
                        pass

        if len(cleaned_rows) < self.min_atoms:
            report["warnings"].append(
                f"Too few atoms after cleaning ({len(cleaned_rows)})."
            )
            return report

        out.parent.mkdir(parents=True, exist_ok=True)
        _write_clean_p1_cif(out, data_name, cleaned_cell, cleaned_rows)

        report["n_atoms_clean"] = len(cleaned_rows)
        report["valid"] = True
        return report

    def sanitize_batch(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        pattern: str = "*.cif",
    ) -> List[Dict[str, Any]]:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        reports: List[Dict[str, Any]] = []
        for cif in sorted(input_dir.glob(pattern)):
            reports.append(self.sanitize(cif, output_dir / cif.name))

        n_ok = sum(bool(r.get("valid", False)) for r in reports)
        logger.info("Sanitised %d CIFs: %d valid.", len(reports), n_ok)
        return reports


__all__ = ["CIFSanitizer", "read_cell_from_cif", "count_atoms_in_cif"]