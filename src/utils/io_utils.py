"""
File I/O utilities for the UC-TPNO pipeline.

This module is the single I/O backbone that every other module in the
pipeline imports from.  It centralises *all* file-system interaction so
that path conventions, serialisation formats, error handling, and
compression policies are consistent across the codebase.

Key capabilities
────────────────
1.  **Config management** — load / save / merge hierarchical YAML and
    JSON configs with dot-path overrides (e.g. ``training.lr=3e-4``).
2.  **CIF file reading** — ASE-based reader with automatic supercell
    detection, P1 reduction flag, and lightweight metadata extraction
    (composition, cell parameters, volume, density).
3.  **Graph serialisation** — save / load PyTorch Geometric ``Data``
    objects with optional gzip compression and batch I/O over directories.
4.  **Registry & isotherm data** — read / write Parquet and CSV files
    for the MOF registry, adsorption isotherms, and charge tables.
5.  **RASPA output parsing** — robust parser for GCMC ``*.data`` output
    files (loadings, energies, host–guest energies, convergence).
6.  **NIST ISODB helpers** — parse experimental isotherm JSON into
    tidy DataFrames.
7.  **Safe file operations** — atomic writes (write-to-temp then rename),
    directory creation, glob helpers, and size-aware cleanup.
8.  **Pipeline directory layout** — canonical paths for raw / intermediate
    / processed / graph / checkpoint / results directories.

Design goals
────────────
* Every public function accepts both ``str`` and ``pathlib.Path``.
* All writes use atomic semantics where possible (no partial files on
  crash).
* Heavy optional dependencies (``ase``, ``pandas``, ``torch``,
  ``torch_geometric``) are imported lazily so the module can be loaded
  in lightweight analysis scripts.

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import io
import json
import logging
import os
import re
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


# ═══════════════════════════════════════════════════════════════════════
# 1.  SAFE FILE PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════

def ensure_dir(path: PathLike) -> Path:
    """Create directory (and parents) if it does not exist.  Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: PathLike, content: str, encoding: str = "utf-8") -> None:
    """
    Write text to *path* atomically: write to a temporary file in the
    same directory, then ``os.replace`` (which is atomic on POSIX and
    near-atomic on Windows).  Prevents half-written files on crash.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp, p)
    except BaseException:
        os.unlink(tmp)
        raise


def atomic_write_bytes(path: PathLike, data: bytes) -> None:
    """Write bytes to *path* atomically (see :func:`atomic_write_text`)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, p)
    except BaseException:
        os.unlink(tmp)
        raise


def safe_open(path: PathLike, mode: str = "r", **kwargs):
    """
    Open a file with automatic gzip detection based on suffix.

    Supports ``.gz`` files transparently.  All other files are opened
    with the built-in ``open``.
    """
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, mode, **kwargs)
    return open(p, mode, **kwargs)


def file_sha256(path: PathLike, chunk_size: int = 1 << 20) -> str:
    """Return hex-encoded SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def glob_sorted(
    directory: PathLike,
    pattern: str = "*",
    key=None,
) -> List[Path]:
    """Glob a directory and return sorted results."""
    paths = list(Path(directory).glob(pattern))
    return sorted(paths, key=key or (lambda p: p.name))


def directory_size_mb(path: PathLike) -> float:
    """Total size of a directory tree in MiB."""
    total = 0
    for f in Path(path).rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


# ═══════════════════════════════════════════════════════════════════════
# 2.  CONFIG MANAGEMENT (YAML / JSON)
# ═══════════════════════════════════════════════════════════════════════

def load_yaml(path: PathLike) -> Dict[str, Any]:
    """Load a YAML config file.  Returns an empty dict if file is empty."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML configs: pip install pyyaml")

    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def save_yaml(data: Dict[str, Any], path: PathLike) -> None:
    """Save a dict as YAML (atomic write, sorted keys)."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required: pip install pyyaml")

    content = yaml.dump(data, default_flow_style=False, sort_keys=True)
    atomic_write_text(path, content)


def load_json(path: PathLike) -> Any:
    """Load a JSON file (supports ``.json.gz``)."""
    with safe_open(path, "r") as f:
        return json.load(f)


def save_json(data: Any, path: PathLike, indent: int = 2) -> None:
    """Save data as JSON (atomic write)."""
    content = json.dumps(data, indent=indent, default=str, ensure_ascii=False)
    atomic_write_text(path, content)


def load_config(path: PathLike) -> Dict[str, Any]:
    """
    Load a config from YAML or JSON (detected by extension).

    Supports ``.yaml``, ``.yml``, ``.json``.
    """
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        return load_yaml(p)
    elif p.suffix == ".json":
        return load_json(p)
    else:
        # Try YAML first, then JSON
        try:
            return load_yaml(p)
        except Exception:
            return load_json(p)


def save_config(data: Dict[str, Any], path: PathLike) -> None:
    """Save a config as YAML or JSON (detected by extension)."""
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        save_yaml(data, p)
    else:
        save_json(data, p)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge *override* into *base*.  Nested dicts are merged
    recursively; all other values are replaced.

    >>> merge_configs({'a': {'b': 1, 'c': 2}}, {'a': {'b': 99}})
    {'a': {'b': 99, 'c': 2}}
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(val, dict)
        ):
            result[key] = merge_configs(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def apply_dot_overrides(
    config: Dict[str, Any],
    overrides: Sequence[str],
) -> Dict[str, Any]:
    """
    Apply dot-separated command-line overrides to a config dict.

    Each override is a string ``"key.subkey=value"`` where *value* is
    parsed as YAML (so ``'true'`` → ``True``, ``'3e-4'`` → ``0.0003``,
    ``'[1,2,3]'`` → ``[1, 2, 3]``).

    Example
    -------
    >>> cfg = {'training': {'lr': 0.001}}
    >>> apply_dot_overrides(cfg, ['training.lr=3e-4', 'training.epochs=200'])
    {'training': {'lr': 0.0003, 'epochs': 200}}
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for dot overrides")

    result = copy.deepcopy(config)
    for ov in overrides:
        if "=" not in ov:
            logger.warning("Ignoring malformed override (no '='): %s", ov)
            continue
        key_path, val_str = ov.split("=", 1)
        keys = key_path.strip().split(".")
        val = yaml.safe_load(val_str)

        # yaml.safe_load keeps scientific notation (e.g. "3e-4") as str;
        # attempt numeric conversion when the result is still a string.
        if isinstance(val, str):
            for converter in (int, float):
                try:
                    val = converter(val)
                    break
                except (ValueError, TypeError):
                    continue

        # Walk to the parent dict
        d = result
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = val

    return result


def load_config_with_overrides(
    path: PathLike,
    overrides: Optional[Sequence[str]] = None,
    merge_from: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """
    Convenience: load a base config, optionally merge another config on
    top, then apply CLI dot-overrides.

    This mirrors the common pattern::

        base_cfg → merge(experiment_cfg) → override(cli_args)
    """
    cfg = load_config(path)
    if merge_from is not None:
        overlay = load_config(merge_from)
        cfg = merge_configs(cfg, overlay)
    if overrides:
        cfg = apply_dot_overrides(cfg, overrides)
    return cfg


# ═══════════════════════════════════════════════════════════════════════
# 3.  CIF FILE I/O
# ═══════════════════════════════════════════════════════════════════════

def read_cif_ase(path: PathLike):
    """
    Read a CIF file using ASE and return an ``ase.Atoms`` object.

    Handles common CIF quirks (duplicate labels, missing symmetry)
    by falling back to ``format='cif'`` with ``reader='ase'``.
    """
    from ase.io import read as ase_read

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CIF file not found: {p}")

    try:
        atoms = ase_read(str(p), format="cif")
    except Exception as e:
        logger.warning("Primary CIF read failed for %s: %s — trying fallback", p.name, e)
        # Fallback: try with store_tags
        atoms = ase_read(str(p), format="cif", store_tags=True)

    return atoms


def cif_metadata(path: PathLike) -> Dict[str, Any]:
    """
    Extract lightweight metadata from a CIF file without building a
    full graph.

    Returns
    -------
    Dict with keys: ``mof_id``, ``n_atoms``, ``composition`` (formula),
    ``cell_lengths`` [Å], ``cell_angles`` [°], ``volume`` [ų],
    ``density`` [g/cm³], ``elements`` (sorted unique list).
    """
    atoms = read_cif_ase(path)
    cell = atoms.get_cell()
    volume = cell.volume if hasattr(cell, "volume") else float(np.linalg.det(cell))

    symbols = atoms.get_chemical_symbols()
    masses = atoms.get_masses()
    total_mass = float(np.sum(masses))

    # Density in g/cm³: mass in amu, volume in ų
    # 1 amu = 1.66054e-24 g, 1 ų = 1e-24 cm³
    density = (total_mass * 1.66054e-24) / (volume * 1e-24) if volume > 0 else 0.0

    # Composition as reduced formula
    from collections import Counter
    from math import gcd
    from functools import reduce

    counts = Counter(symbols)
    g = reduce(gcd, counts.values()) if counts else 1
    formula = "".join(
        f"{el}{counts[el] // g}" if counts[el] // g > 1 else el
        for el in sorted(counts)
    )

    return {
        "mof_id": Path(path).stem,
        "n_atoms": len(atoms),
        "composition": formula,
        "elements": sorted(set(symbols)),
        "cell_lengths": cell.lengths().tolist() if hasattr(cell, "lengths") else [0, 0, 0],
        "cell_angles": cell.angles().tolist() if hasattr(cell, "angles") else [0, 0, 0],
        "volume_A3": round(volume, 3),
        "density_g_cm3": round(density, 4),
    }


def cif_to_graph(
    path: PathLike,
    cutoff: float = 5.0,
    self_loops: bool = False,
) -> Any:
    """
    Convert a CIF file to a PyTorch Geometric ``Data`` graph.

    Parameters
    ----------
    path       : Path to the CIF file.
    cutoff     : Neighbour-list cutoff radius [Å].
    self_loops : If *True*, include self-loop edges (i == j).

    Returns
    -------
    ``torch_geometric.data.Data`` with attributes:
        ``z`` (atomic numbers), ``pos`` (Cartesian coords),
        ``cell`` (3×3 lattice vectors), ``edge_index``,
        ``edge_attr`` (distances), ``num_nodes``, ``mof_id``.
    """
    import torch
    from torch_geometric.data import Data

    atoms = read_cif_ase(path)

    z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float32)

    # Neighbour list with periodic boundary conditions
    try:
        from ase.neighborlist import neighbor_list
        i_idx, j_idx, dists = neighbor_list("ijd", atoms, cutoff=cutoff)
    except ImportError:
        # Fallback: brute-force (slow, only for tiny cells)
        logger.warning("ase.neighborlist not available — using brute-force neighbour search")
        i_idx, j_idx, dists = _brute_force_neighbours(atoms, cutoff)

    if not self_loops:
        mask = i_idx != j_idx
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        dists = dists[mask]

    edge_index = torch.tensor(np.stack([i_idx, j_idx]), dtype=torch.long)
    edge_attr = torch.tensor(dists, dtype=torch.float32).unsqueeze(-1)

    data = Data(
        z=z,
        pos=pos,
        cell=cell.unsqueeze(0),  # (1, 3, 3) for batching compatibility
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(z),
    )
    data.mof_id = Path(path).stem
    return data


def _brute_force_neighbours(atoms, cutoff: float):
    """Minimal brute-force PBC neighbour list (fallback)."""
    from itertools import product

    pos = atoms.get_positions()
    cell = np.array(atoms.get_cell())
    n = len(pos)

    i_list, j_list, d_list = [], [], []
    for ia in range(n):
        for ja in range(n):
            for shift in product([-1, 0, 1], repeat=3):
                offset = np.dot(shift, cell)
                d = np.linalg.norm(pos[ja] + offset - pos[ia])
                if 0 < d <= cutoff:
                    i_list.append(ia)
                    j_list.append(ja)
                    d_list.append(d)

    return (
        np.array(i_list, dtype=np.int64),
        np.array(j_list, dtype=np.int64),
        np.array(d_list, dtype=np.float64),
    )


# ═══════════════════════════════════════════════════════════════════════
# 4.  GRAPH SERIALISATION
# ═══════════════════════════════════════════════════════════════════════

def save_graph(data: Any, path: PathLike, compress: bool = False) -> None:
    """
    Save a PyG ``Data`` object.

    Parameters
    ----------
    data     : ``torch_geometric.data.Data``.
    path     : Destination path (``.pt`` or ``.pt.gz``).
    compress : If *True*, gzip the serialised bytes.
    """
    import torch

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if compress or p.suffix == ".gz":
        buf = io.BytesIO()
        torch.save(data, buf)
        with gzip.open(p, "wb") as f:
            f.write(buf.getvalue())
    else:
        torch.save(data, p)


def load_graph(path: PathLike) -> Any:
    """
    Load a PyG ``Data`` object (auto-detects gzip by extension).
    """
    import torch

    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rb") as f:
            buf = io.BytesIO(f.read())
        return torch.load(buf, weights_only=False)
    return torch.load(p, weights_only=False)


def save_graph_batch(
    graphs: Dict[str, Any],
    directory: PathLike,
    compress: bool = False,
) -> int:
    """
    Save a dict of ``{mof_id: Data}`` to a directory.  Returns count.
    """
    d = ensure_dir(directory)
    ext = ".pt.gz" if compress else ".pt"
    for mof_id, graph in graphs.items():
        save_graph(graph, d / f"{mof_id}{ext}", compress=compress)
    return len(graphs)


def load_graph_batch(
    directory: PathLike,
    mof_ids: Optional[List[str]] = None,
    pattern: str = "*.pt*",
) -> Dict[str, Any]:
    """
    Load all graphs from a directory.  If *mof_ids* is given, only load
    those.  Returns ``{mof_id: Data}``.
    """
    d = Path(directory)
    result = {}

    if mof_ids is not None:
        for mid in mof_ids:
            for ext in (".pt", ".pt.gz"):
                p = d / f"{mid}{ext}"
                if p.exists():
                    result[mid] = load_graph(p)
                    break
    else:
        for p in glob_sorted(d, pattern):
            mid = p.stem.replace(".pt", "")  # handles .pt.gz double suffix
            result[mid] = load_graph(p)

    return result


def iter_graphs(
    directory: PathLike,
    pattern: str = "*.pt*",
) -> Iterator[Tuple[str, Any]]:
    """
    Lazily iterate over graphs in a directory.  Yields ``(mof_id, Data)``
    tuples without loading everything into memory.
    """
    for p in glob_sorted(directory, pattern):
        mid = p.stem.replace(".pt", "")
        yield mid, load_graph(p)


# ═══════════════════════════════════════════════════════════════════════
# 5.  TABULAR DATA I/O  (Parquet, CSV)
# ═══════════════════════════════════════════════════════════════════════

def load_parquet(path: PathLike):
    """Load a Parquet file into a pandas DataFrame."""
    import pandas as pd
    return pd.read_parquet(path)


def save_parquet(df, path: PathLike, **kwargs) -> None:
    """Save a pandas DataFrame as Parquet (atomic write)."""
    import pandas as pd

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write to temp file then rename for atomicity
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".parquet.tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, **kwargs)
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_csv(path: PathLike, **kwargs):
    """Load a CSV file into a pandas DataFrame (auto-detects gzip)."""
    import pandas as pd

    p = Path(path)
    compression = "gzip" if p.suffix == ".gz" else "infer"
    return pd.read_csv(p, compression=compression, **kwargs)


def save_csv(df, path: PathLike, **kwargs) -> None:
    """Save a DataFrame as CSV."""
    import pandas as pd

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, **kwargs)


def load_registry(path: PathLike):
    """
    Load the MOF registry.  Accepts Parquet or CSV.

    Returns a DataFrame indexed by ``mof_id``.
    """
    p = Path(path)
    if p.suffix == ".parquet":
        df = load_parquet(p)
    else:
        df = load_csv(p)

    if "mof_id" in df.columns:
        df = df.set_index("mof_id")
    return df


def load_isotherm_data(path: PathLike):
    """
    Load adsorption isotherm data (Parquet or CSV).

    Expected columns: ``mof_id``, ``temperature``, ``pressure``,
    ``y_CO2``, ``y_N2``, ``y_H2O``, ``q_CO2``, ``q_N2``, ``q_H2O``,
    and optionally ``fidelity``, ``force_field``, ``source``.
    """
    p = Path(path)
    if p.suffix == ".parquet":
        return load_parquet(p)
    return load_csv(p)


# ═══════════════════════════════════════════════════════════════════════
# 6.  NIST ISODB PARSING
# ═══════════════════════════════════════════════════════════════════════

def parse_nist_isodb_json(path: PathLike) -> List[Dict[str, Any]]:
    """
    Parse a NIST ISODB isotherm JSON file into a list of point dicts.

    Each dict contains: ``adsorbent``, ``adsorbate``, ``temperature``,
    ``pressure``, ``loading``, ``units_pressure``, ``units_loading``,
    ``DOI``.
    """
    raw = load_json(path)

    # The ISODB format can vary; handle both list and dict-of-isotherms
    isotherms = raw if isinstance(raw, list) else raw.get("isotherms", [raw])

    points = []
    for iso in isotherms:
        meta = {
            "adsorbent": iso.get("adsorbent", {}).get("name", "unknown"),
            "adsorbate": iso.get("adsorbateGas", [{}])[0].get("name", "unknown")
            if isinstance(iso.get("adsorbateGas"), list)
            else iso.get("adsorbate", "unknown"),
            "temperature": iso.get("temperature", None),
            "units_pressure": iso.get("pressureUnits", "bar"),
            "units_loading": iso.get("adsorptionUnits", "mmol/g"),
            "DOI": iso.get("DOI", ""),
        }

        data_points = iso.get("isotherm_data", iso.get("data", []))
        for pt in data_points:
            entry = dict(meta)
            # Support various key names
            entry["pressure"] = pt.get("pressure", pt.get("P", pt.get("x", None)))
            entry["loading"] = pt.get("species_data", [{}])[0].get(
                "adsorption", pt.get("loading", pt.get("q", pt.get("y", None)))
            )
            points.append(entry)

    return points


def nist_isodb_to_dataframe(path: PathLike):
    """Parse NIST ISODB JSON → pandas DataFrame."""
    import pandas as pd

    points = parse_nist_isodb_json(path)
    return pd.DataFrame(points)


# ═══════════════════════════════════════════════════════════════════════
# 7.  RASPA OUTPUT PARSING
# ═══════════════════════════════════════════════════════════════════════

def parse_raspa_output(work_dir: PathLike) -> Dict[str, Any]:
    """
    Parse RASPA GCMC simulation output from a working directory.

    Scans all ``.data`` files for:
    * Average loadings per component [mol/kg and molec/uc].
    * Host–guest interaction energies [kJ/mol].
    * Henry coefficients (if available).

    Returns
    -------
    Dict with ``loadings``, ``energies``, ``henry``, ``converged``, ``raw_files``.
    """
    d = Path(work_dir)
    data_files = list(d.glob("*.data")) + list(d.glob("Output/System_0/*.data"))

    results: Dict[str, Any] = {
        "loadings_mol_kg": {},
        "loadings_molec_uc": {},
        "energies_kJ_mol": {},
        "henry_mol_kg_Pa": {},
        "converged": True,
        "raw_files": [str(f) for f in data_files],
    }

    # Regex patterns for RASPA output
    _re_loading_abs = re.compile(
        r"absolute adsorption:\s*"
        r"([\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)\s*\[mol(?:ecules)?/(?:kg|unit cell)\]"
    )
    _re_loading_mol_kg = re.compile(
        r"([\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)\s*\[mol/kg\s*framework\]"
    )
    _re_loading_molec_uc = re.compile(
        r"([\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)\s*\[molecules/uc\]"
    )
    _re_component = re.compile(r"Component\s+\d+\s*\[([^\]]+)\]")
    _re_energy = re.compile(
        r"Average\s+(.*?)\s*energy:\s*([\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)"
    )
    _re_henry = re.compile(
        r"Henry.*?coefficient.*?:\s*([\d.eE+-]+)"
    )

    for data_file in data_files:
        try:
            text = data_file.read_text(errors="replace")
        except Exception as e:
            logger.warning("Could not read %s: %s", data_file, e)
            continue

        current_component = None
        for line in text.splitlines():
            # Track which component section we are in
            m_comp = _re_component.search(line)
            if m_comp:
                current_component = m_comp.group(1).strip()

            # Loadings in mol/kg
            m_mol = _re_loading_mol_kg.search(line)
            if m_mol and current_component:
                val = float(m_mol.group(1))
                err = float(m_mol.group(2))
                results["loadings_mol_kg"][current_component] = {
                    "value": val, "error": err,
                }

            # Loadings in molecules/uc
            m_uc = _re_loading_molec_uc.search(line)
            if m_uc and current_component:
                val = float(m_uc.group(1))
                err = float(m_uc.group(2))
                results["loadings_molec_uc"][current_component] = {
                    "value": val, "error": err,
                }

            # Energies
            m_en = _re_energy.search(line)
            if m_en:
                label = m_en.group(1).strip()
                val = float(m_en.group(2))
                results["energies_kJ_mol"][label] = val

            # Henry coefficient
            m_h = _re_henry.search(line)
            if m_h and current_component:
                results["henry_mol_kg_Pa"][current_component] = float(m_h.group(1))

            # Convergence check
            if "WARNING" in line and "NOT CONVERGED" in line.upper():
                results["converged"] = False

    return results


# ═══════════════════════════════════════════════════════════════════════
# 8.  SPLITS I/O
# ═══════════════════════════════════════════════════════════════════════

def save_splits(
    splits: Dict[str, List[str]],
    path: PathLike,
) -> None:
    """
    Save train/val/test splits to JSON.

    Expected format::

        {"train": ["MOF_A", ...], "val": [...], "test": [...]}
    """
    save_json(splits, path)


def load_splits(path: PathLike) -> Dict[str, List[str]]:
    """Load train/val/test splits from JSON."""
    data = load_json(path)
    assert isinstance(data, dict), f"Expected dict, got {type(data)}"
    for key in ("train", "val", "test"):
        if key not in data:
            logger.warning("Split key '%s' missing from %s", key, path)
    return data


# ═══════════════════════════════════════════════════════════════════════
# 9.  PIPELINE DIRECTORY LAYOUT
# ═══════════════════════════════════════════════════════════════════════

class PipelineDirectories:
    """
    Canonical directory layout for the UC-TPNO pipeline.

    ::

        project_root/
        ├── data/
        │   ├── raw/                     # original downloads
        │   ├── intermediate/
        │   │   ├── cifs_raw/
        │   │   ├── cifs_sanitized/
        │   │   ├── charges/
        │   │   ├── adsorption/
        │   │   └── experimental/
        │   └── processed/
        │       ├── registry/            # mof_metadata.parquet
        │       ├── adsorption/          # humid_isotherms.parquet
        │       ├── graphs/              # PyG .pt files
        │       │   ├── train/
        │       │   ├── val/
        │       │   └── test/
        │       └── splits.json
        ├── checkpoints/
        ├── results/
        └── configs/
    """

    def __init__(self, root: PathLike = "."):
        self.root = Path(root).resolve()

    # ── data ─────────────────────────────────────────────────────
    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def intermediate(self) -> Path:
        return self.data / "intermediate"

    @property
    def cifs_raw(self) -> Path:
        return self.intermediate / "cifs_raw"

    @property
    def cifs_sanitized(self) -> Path:
        return self.intermediate / "cifs_sanitized"

    @property
    def charges(self) -> Path:
        return self.intermediate / "charges"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def registry_dir(self) -> Path:
        return self.processed / "registry"

    @property
    def adsorption_dir(self) -> Path:
        return self.processed / "adsorption"

    @property
    def graphs_dir(self) -> Path:
        return self.processed / "graphs"

    def graphs_split(self, split: str) -> Path:
        return self.graphs_dir / split

    @property
    def splits_file(self) -> Path:
        return self.processed / "splits.json"

    # ── model artefacts ──────────────────────────────────────────
    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    # ── helpers ──────────────────────────────────────────────────

    def create_all(self) -> None:
        """Create the complete directory tree."""
        for attr in [
            "raw", "cifs_raw", "cifs_sanitized", "charges",
            "registry_dir", "adsorption_dir",
            "checkpoints", "results", "configs",
        ]:
            ensure_dir(getattr(self, attr))
        for split in ("train", "val", "test"):
            ensure_dir(self.graphs_split(split))
        logger.info("Pipeline directories created under %s", self.root)

    def summary(self) -> Dict[str, str]:
        """Return a dict mapping directory names to paths."""
        return {
            attr: str(getattr(self, attr))
            for attr in dir(self)
            if not attr.startswith("_")
            and isinstance(getattr(type(self), attr, None), property)
        }


# ═══════════════════════════════════════════════════════════════════════
# 10.  CHECKPOINT CONVENIENCE  (thin wrappers for trainer)
# ═══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    path: PathLike,
    *,
    epoch: int,
    model_state: Dict,
    optimizer_state: Dict,
    scheduler_state: Optional[Dict] = None,
    metrics: Optional[Dict] = None,
    config: Optional[Dict] = None,
) -> None:
    """
    Save a training checkpoint (thin wrapper used by the trainer module).

    For the enhanced version with RNG states and integrity hashes, see
    :func:`reproducibility.save_reproducible_checkpoint`.
    """
    import torch

    payload = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
    }
    if scheduler_state is not None:
        payload["scheduler_state_dict"] = scheduler_state
    if metrics is not None:
        payload["metrics"] = metrics
    if config is not None:
        payload["config"] = config

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)
    logger.info("Checkpoint saved → %s (epoch %d)", p, epoch)


def load_checkpoint(
    path: PathLike,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a training checkpoint and return the raw dict."""
    import torch

    return torch.load(path, map_location=device, weights_only=False)


# ═══════════════════════════════════════════════════════════════════════
# 11.  MISCELLANEOUS HELPERS
# ═══════════════════════════════════════════════════════════════════════

def count_parameters(model: Any) -> Dict[str, int]:
    """Return ``{'total': …, 'trainable': …, 'frozen': …}``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024  # type: ignore[assignment]
    return f"{size_bytes:.1f} PiB"


def collect_cif_files(
    *directories: PathLike,
    extensions: Tuple[str, ...] = (".cif",),
) -> List[Path]:
    """Recursively collect CIF files from one or more directories."""
    result = []
    for d in directories:
        p = Path(d)
        if not p.exists():
            logger.warning("Directory %s does not exist — skipping", p)
            continue
        for ext in extensions:
            result.extend(p.rglob(f"*{ext}"))
    return sorted(set(result))


# ═══════════════════════════════════════════════════════════════════════
# 12.  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    # Safe primitives
    "ensure_dir",
    "atomic_write_text",
    "atomic_write_bytes",
    "safe_open",
    "file_sha256",
    "glob_sorted",
    "directory_size_mb",
    # Config management
    "load_yaml",
    "save_yaml",
    "load_json",
    "save_json",
    "load_config",
    "save_config",
    "merge_configs",
    "apply_dot_overrides",
    "load_config_with_overrides",
    # CIF I/O
    "read_cif_ase",
    "cif_metadata",
    "cif_to_graph",
    "collect_cif_files",
    # Graph serialisation
    "save_graph",
    "load_graph",
    "save_graph_batch",
    "load_graph_batch",
    "iter_graphs",
    # Tabular data
    "load_parquet",
    "save_parquet",
    "load_csv",
    "save_csv",
    "load_registry",
    "load_isotherm_data",
    # NIST ISODB
    "parse_nist_isodb_json",
    "nist_isodb_to_dataframe",
    # RASPA
    "parse_raspa_output",
    # Splits
    "save_splits",
    "load_splits",
    # Pipeline layout
    "PipelineDirectories",
    # Checkpoint convenience
    "save_checkpoint",
    "load_checkpoint",
    # Helpers
    "count_parameters",
    "human_readable_size",
]