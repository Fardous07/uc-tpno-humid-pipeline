# src/simulation/gctmmc/__init__.py
from .parser import (
    GCTMMCConfig,
    GCTMMCRunner,
    generate_tmmc_input,
    parse_tmmc_output,
    synthetic_collection_matrix,
    collection_matrix_to_ln_pi,
    ln_pi_to_isotherm,
)
from .runner import parse_collection_matrix

__all__ = [
    # runner API (from parser.py — project naming is swapped)
    "GCTMMCConfig",
    "GCTMMCRunner",
    "generate_tmmc_input",
    # parser utilities (from runner.py, re-exported via parser.py)
    "parse_tmmc_output",
    "parse_collection_matrix",
    "synthetic_collection_matrix",
    "collection_matrix_to_ln_pi",
    "ln_pi_to_isotherm",
]