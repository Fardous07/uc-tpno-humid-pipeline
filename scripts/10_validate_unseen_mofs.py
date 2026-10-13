#!/usr/bin/env python3
"""
10_validate_unseen_mofs.py — Held-out-MOF generalization check for UC-TPNO.

Runs fresh RASPA GCMC, at the model's OWN training conditions (humid, low
pressure), on a handful of MOFs the model has never trained on, and compares
the simulated loadings against the model's predictions for those same
MOFs/conditions.

This answers a different question than scripts/07_validate.py's NIST-ISODB
tier. NIST ISODB is dry, pure-component, often higher-pressure experimental
data -- a different regime than this model was trained on, so a mismatch
there says nothing about whether the model generalizes to new MOF
structures. This script instead asks: "does the model correctly predict
fresh simulation results for MOFs it never saw, under the SAME humid,
low-pressure regime it was trained on?" -- the fair test.

Reuses, rather than reimplementing:
  * discover_candidate_cifs(), build_screening_conditions()
        from scripts/08_active_learning_loop.py
  * Condition, canonicalize_cif(), determine_unit_cells(),
    build_multicomponent_input(), run_raspa_job(), ensure_dir(),
    detect_raspa_share(), resolve_forcefield(), prepare_runtime(),
    find_molecule_dir(), check_ff_assets(), check_mol_def(), parse_multi()
        from scripts/04_run_simulations.py
  * build_model_from_config(), load_checkpoint_into_model()
        from scripts/07_validate.py
  * _unpack_model_output()
        from scripts/08_active_learning_loop.py
  * CIFSanitizer                          from src.data.preprocessing.sanitize
  * GraphBuilder, GraphConfig             from src.data.preprocessing.graph_builder
  * compute_regression_metrics            from src.evaluation.metrics
  * ResultVisualizer                      from src.evaluation.visualizer

Getting the fairness of this test right -- the whole reason this script
exists -- depends on two things:

1.  mu_CO2/mu_N2/mu_H2O for BOTH the fresh GCMC run and the model query
    come from the exact same array: build_screening_conditions() (which,
    after the fixes to 04_run_simulations.py / 08_active_learning_loop.py,
    calls src.utils.chemistry's Peng-Robinson mixture EOS -- the same
    formula Stage 3 now uses for the real training labels). If this script
    is ever changed to recompute conditions a different way for the GCMC
    run vs. the model query, it silently reintroduces the exact kind of
    mismatch this whole validation exists to catch.

2.  use_charges is read from pipeline.yaml's simulation.gcmc section --
    the same knob Stage 3 training data generation uses -- with a loud
    warning if it disagrees with model.use_charges. The model is queried
    with RAW (unnormalized) mu: ThermodynamicPotentialNO.forward()
    normalizes internally via its own registered mu_mean/mu_std buffers
    (src/models/operator/tpno.py, TPNOEnsemble.forward -> normalize_mu).
    Pre-normalizing mu before calling the model, the way
    scripts/07_validate.py's NIST-ISODB tier does, double-normalizes and
    corrupts the comparison -- do not copy that pattern here.

Everything is written under --output-dir (default
results/unseen_mof_validation/) -- experiments/run_008 and
results/validation are never touched.

Usage
-----
python scripts/10_validate_unseen_mofs.py \\
    --mof-cif data/famous_mofs/HKUST-1.cif data/famous_mofs/UiO-66.cif \\
    --model-checkpoint experiments/run_001/final_model.pt \\
    --config configs/pipeline.yaml \\
    --raspa-path /path/to/raspa/bin/simulate \\
    --raspa-data-dir /path/to/raspa
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocessing.sanitize import CIFSanitizer
from src.data.preprocessing.graph_builder import GraphBuilder, GraphConfig
from src.evaluation.metrics import compute_regression_metrics
from src.evaluation.visualizer import ResultVisualizer

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _load_script_module(name: str, rel_path: str):
    """
    Load scripts/0N_*.py as an importable module.

    Digit-prefixed filenames can't be `import`-ed with a normal statement,
    so this loads them by file path instead of duplicating their functions.
    """
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module   # required before exec: dataclasses in the
                                  # loaded module need sys.modules[__module__]
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Charge sanity check
# ---------------------------------------------------------------------------

_CHARGE_TAGS = (
    "_atom_site_charge",
    "_atom_type_partial_charge",
    "_atom_site_partial_charge",
    "_atom_site_charge_partial",
)


def _cif_has_nonzero_charges(cif_path: Path) -> bool:
    """
    Best-effort check that a sanitised CIF actually carries partial charges,
    so 'use_charges=True' isn't silently a no-op for this MOF.

    This pipeline has no charge-assignment step of its own (EQeq/DDEC) --
    training MOFs came pre-charged from data/raw/arc_mof_charges /
    core_mof_2019. A "famous" MOF CIF pulled from elsewhere may have no
    charge column at all, which would make RASPA's
    UseChargesFromCIFFile=yes silently use zero framework electrostatics --
    quietly reintroducing the exact use_charges mismatch this whole
    exercise is trying to eliminate.
    """
    try:
        text = cif_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    if not any(tag in text for tag in _CHARGE_TAGS):
        return False
    for line in text.splitlines():
        for tok in line.split():
            try:
                if abs(float(tok)) > 1e-6:
                    return True
            except ValueError:
                continue
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Validate UC-TPNO on held-out MOFs via fresh, training-matching GCMC."
    )
    p.add_argument("--mof-cif", nargs="+", required=True,
                   help="Raw CIF file(s) of MOFs the model has never trained on.")
    p.add_argument("--model-checkpoint", required=True)
    p.add_argument("--config", default="configs/pipeline.yaml")
    p.add_argument("--adsorption-data",
                   default="data/processed/adsorption/adsorption_training.parquet",
                   help="Used only to verify none of --mof-cif were in the training set.")
    p.add_argument("--raspa-path", required=True)
    p.add_argument("--raspa-data-dir", required=True)
    p.add_argument("--output-dir", default="results/unseen_mof_validation",
                   help="Everything is written here -- experiments/run_008 and "
                        "results/validation are never touched.")
    p.add_argument("--n-cycles", type=int, default=0,
                   help="0 = use simulation.gcmc.n_cycles from --config.")
    p.add_argument("--keep-workspaces", action="store_true")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-delay", type=int, default=60)
    p.add_argument("--co2-definition", default="TraPPE")
    p.add_argument("--n2-definition", default="TraPPE")
    p.add_argument("--h2o-definition", default="TraPPE")
    args = p.parse_args()

    configure_logging()

    s04 = _load_script_module("s10dep_04_run_simulations", "scripts/04_run_simulations.py")
    s07 = _load_script_module("s10dep_07_validate", "scripts/07_validate.py")
    s08 = _load_script_module("s10dep_08_active_learning_loop", "scripts/08_active_learning_loop.py")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    sim_cfg = cfg.get("simulation", {})
    model_cfg = cfg.get("model", {})

    # ------------------------------------------------------------------
    # charges must match Stage 3
    # ------------------------------------------------------------------
    use_charges = bool(sim_cfg.get("gcmc", {}).get("use_charges", True))
    if bool(model_cfg.get("use_charges", False)) != use_charges:
        logger.warning(
            "config drift: model.use_charges=%s but simulation.gcmc.use_charges=%s "
            "in %s -- this run uses simulation.gcmc.use_charges=%s (matching how "
            "Stage 3 training GCMC is run). Fix pipeline.yaml before trusting "
            "these numbers.",
            model_cfg.get("use_charges"), use_charges, args.config, use_charges,
        )

    forcefield_requested = sim_cfg.get("forcefield", "GenericMOFs")
    n_cycles = args.n_cycles or int(sim_cfg.get("gcmc", {}).get("n_cycles", 5000))
    cutoff = float(sim_cfg.get("gcmc", {}).get("cutoff", 12.0))
    # Matches what build_model_from_config() (07_validate.py) actually wires
    # into the encoder at inference time -- model.cutoff, not the unused
    # model.encoder_config.cutoff.
    graph_cutoff = float(model_cfg.get("cutoff", 5.0))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = output_dir / "graphs"
    sanitized_dir = output_dir / "cifs_sanitized"
    graph_dir.mkdir(parents=True, exist_ok=True)
    sanitized_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Exclude any MOF that's actually in the training set
    # ------------------------------------------------------------------
    trained_ids: set = set()
    ads_path = Path(args.adsorption_data)
    if ads_path.exists():
        trained_ids = set(pd.read_parquet(ads_path)["mof_id"].astype(str).unique())
    else:
        logger.warning("Adsorption data not found at %s -- cannot verify these "
                        "MOFs are actually unseen.", ads_path)

    raw_cifs = [Path(c) for c in args.mof_cif]
    unseen_cifs = []
    for cif in raw_cifs:
        if cif.stem in trained_ids:
            logger.warning("SKIPPING %s: already present in the training set (%s).",
                            cif.stem, ads_path)
            continue
        unseen_cifs.append(cif)
    if not unseen_cifs:
        raise RuntimeError("No unseen MOFs left after filtering against the training set.")

    # ------------------------------------------------------------------
    # 2. Sanitize + graph-build each unseen MOF
    # ------------------------------------------------------------------
    sanitizer = CIFSanitizer()
    graph_builder = GraphBuilder(GraphConfig(cutoff=graph_cutoff, edge_features=True))

    prepared: List[Dict[str, Any]] = []
    for cif in unseen_cifs:
        mof_id = cif.stem
        clean_cif = sanitized_dir / f"{mof_id}.cif"
        report = sanitizer.sanitize(cif, clean_cif)
        if not report.get("valid"):
            logger.warning("SKIPPING %s: sanitisation failed (%s).",
                            mof_id, report.get("warnings"))
            continue
        if use_charges and not _cif_has_nonzero_charges(clean_cif):
            logger.warning(
                "%s: use_charges=True but no non-zero partial charges were found "
                "in this CIF -- RASPA's UseChargesFromCIFFile will be a no-op "
                "(framework electrostatics = 0) for this MOF. It needs "
                "pre-assigned partial charges (e.g. EQeq/DDEC) to match how the "
                "training data was generated.", mof_id,
            )
        graph_path = graph_dir / f"{mof_id}.pt"
        try:
            graph_builder.build_single(clean_cif, graph_path)
        except Exception as e:
            logger.warning("SKIPPING %s: graph build failed (%s).", mof_id, e)
            continue
        prepared.append({"mof_id": mof_id, "cif": clean_cif, "graph": graph_path})

    if not prepared:
        raise RuntimeError("No MOFs survived sanitisation/graph-building.")

    logger.info("Validating on %d unseen MOF(s): %s",
                len(prepared), [d["mof_id"] for d in prepared])

    # ------------------------------------------------------------------
    # 3. Training-matching screening conditions.
    #    build_screening_conditions() now calls chemistry.py's
    #    Peng-Robinson mixture EOS (same fix applied to Stage 3) --
    #    reused verbatim, not recomputed, so this can't drift from
    #    training's thermodynamics.
    # ------------------------------------------------------------------
    cond_np, cond_meta = s08.build_screening_conditions()
    logger.info("Using %d training-matching (humid, low-pressure) conditions "
                "from build_screening_conditions().", len(cond_meta))

    conditions: List[Any] = []
    for row, meta in zip(cond_np, cond_meta):
        comp = meta["composition"]
        p_total = float(meta["pressure"])
        conditions.append(s04.Condition(
            temperature_k=float(meta["temperature"]),
            pressure_bar=p_total,
            rh=float(meta["relative_humidity"]),
            y_co2_dry=float(comp.get("CO2", 0.0)),
            y_n2_dry=float(comp.get("N2", 0.0)),
            x_co2_total=float(comp.get("CO2", 0.0)),
            x_n2_total=float(comp.get("N2", 0.0)),
            x_h2o_total=float(comp.get("H2O", 0.0)),
            p_h2o_bar=float(comp.get("H2O", 0.0)) * p_total,
            p_co2_bar=float(comp.get("CO2", 0.0)) * p_total,
            p_n2_bar=float(comp.get("N2", 0.0)) * p_total,
            mu_co2=float(row[0]), mu_n2=float(row[1]), mu_h2o=float(row[2]),
        ))

    # ------------------------------------------------------------------
    # 4. RASPA runtime setup (reusing 04_run_simulations.py's helpers)
    # ------------------------------------------------------------------
    raspa_path = Path(args.raspa_path).expanduser().resolve()
    raspa_dd = Path(args.raspa_data_dir).expanduser().resolve()
    if not raspa_path.exists():
        raise FileNotFoundError(f"RASPA binary not found: {raspa_path}")

    raspa_share = s04.detect_raspa_share(raspa_dd)
    ff_name = s04.resolve_forcefield(raspa_share, forcefield_requested)
    runtime_root, runtime_share, fw_cif_dir = s04.prepare_runtime(output_dir, raspa_share)

    co2_dir = s04.find_molecule_dir(raspa_share, "CO2", args.co2_definition)
    n2_dir = s04.find_molecule_dir(raspa_share, "N2", args.n2_definition)
    h2o_dir = s04.find_molecule_dir(raspa_share, "H2O", args.h2o_definition)
    s04.check_ff_assets(runtime_share, ff_name)
    s04.check_mol_def(runtime_share, co2_dir, "CO2")
    s04.check_mol_def(runtime_share, n2_dir, "N2")
    s04.check_mol_def(runtime_share, h2o_dir, "H2O")

    # ------------------------------------------------------------------
    # 5. Fresh GCMC ("ground truth") on each unseen MOF, at the same
    #    conditions and use_charges setting as Stage 3.
    # ------------------------------------------------------------------
    sim_rows: List[Dict[str, Any]] = []
    ws_root = s04.ensure_dir(output_dir / "gcmc_workspace")
    total_jobs = len(prepared) * len(conditions)
    job = 0
    for entry in prepared:
        mof_id, cif_path = entry["mof_id"], entry["cif"]
        uc = s04.determine_unit_cells(cif_path, cutoff)
        for cond in conditions:
            job += 1
            logger.info("[%d/%d] %s | T=%.0fK P=%gbar RH=%.0f%% (fresh GCMC)",
                        job, total_jobs, mof_id, cond.temperature_k,
                        cond.pressure_bar, cond.rh * 100)
            ws = ws_root / (f"{mof_id}__T{cond.temperature_k:.0f}"
                             f"__P{cond.pressure_bar:g}__RH{cond.rh:.3f}")
            inp = s04.build_multicomponent_input(
                mof_id, uc, cond, n_cycles, ff_name, use_charges, cutoff,
                co2_dir, n2_dir, h2o_dir,
            )
            res = s04.run_raspa_job(
                raspa_path, runtime_root, fw_cif_dir, cif_path, ws, inp,
                args.max_retries, args.retry_delay,
            )
            if res.ok:
                parsed = s04.parse_multi(res.combined_text)
                sim_rows.append({
                    "mof_id": mof_id,
                    "temperature_k": cond.temperature_k,
                    "pressure_bar": cond.pressure_bar,
                    "rh": cond.rh,
                    "mu_CO2": cond.mu_co2, "mu_N2": cond.mu_n2, "mu_H2O": cond.mu_h2o,
                    **parsed,
                })
            else:
                logger.warning("  FAILED rc=%s", res.returncode)
            if not args.keep_workspaces and ws.exists():
                shutil.rmtree(ws, ignore_errors=True)

    sim_df = pd.DataFrame(sim_rows)
    sim_out = output_dir / "fresh_gcmc_ground_truth.parquet"
    sim_df.to_parquet(sim_out, index=False)
    logger.info("Ground-truth GCMC: %d/%d points succeeded -> %s",
                len(sim_df), total_jobs, sim_out)

    if sim_df.empty:
        raise RuntimeError("No successful GCMC points -- cannot compare against the model.")

    # ------------------------------------------------------------------
    # 6. Load the trained model
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = s07.build_model_from_config(model_cfg, device=device)
    s07.load_checkpoint_into_model(model, args.model_checkpoint, device=device)
    model.eval()

    normalization: Dict[str, Any] = {}
    ckpt = torch.load(args.model_checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        normalization = ckpt.get("normalization", {})
    if not normalization:
        resolved = Path(args.model_checkpoint).parent.parent / "resolved_config.json"
        if resolved.exists():
            with open(resolved) as f:
                normalization = json.load(f).get("normalization", {})
    if not normalization:
        logger.warning(
            "No normalization stats found in the checkpoint or resolved_config.json "
            "-- predicted loadings below are in the model's raw output space "
            "(possibly log1p'd), not mol/kg. Re-run with a checkpoint that saved "
            "normalization stats for a trustworthy comparison."
        )
        normalization = {"q_mean": [0.0, 0.0, 0.0], "q_std": [1.0, 1.0, 1.0]}
    q_mean = np.array(normalization.get("q_mean", [0.0, 0.0, 0.0]), dtype=np.float64)
    q_std = np.array(normalization.get("q_std", [1.0, 1.0, 1.0]), dtype=np.float64)
    q_std = np.where(q_std > 0, q_std, 1.0)

    # ------------------------------------------------------------------
    # 7. Query the model at the SAME raw mu used for the fresh GCMC above.
    #    NOTE: raw, not normalized -- see module docstring point 2.
    # ------------------------------------------------------------------
    from torch_geometric.data import Batch

    cond_t = torch.tensor(cond_np, dtype=torch.float32, device=device)  # [P, 4]

    pred_rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for entry in prepared:
            graph = torch.load(str(entry["graph"]), weights_only=False)
            graph_batch = Batch.from_data_list([graph]).to(device)
            conditions_batched = cond_t.unsqueeze(0)  # [1, P, 4]
            out = model(graph_batch, conditions_batched, return_hessian=False)
            q_pred, _ = s08._unpack_model_output(out)   # [1, P, 3]
            q_pred = q_pred[0]  # [P, 3]

            for i, meta in enumerate(cond_meta):
                q_denorm = q_pred[i] * q_std + q_mean
                pred_rows.append({
                    "mof_id": entry["mof_id"],
                    "temperature_k": meta["temperature"],
                    "pressure_bar": meta["pressure"],
                    "rh": meta["relative_humidity"],
                    "co2_loading_molkg_pred": float(q_denorm[0]),
                    "n2_loading_molkg_pred": float(q_denorm[1]),
                    "h2o_loading_molkg_pred": float(q_denorm[2]),
                })

    pred_df = pd.DataFrame(pred_rows)

    # ------------------------------------------------------------------
    # 8. Merge + metrics + plots
    # ------------------------------------------------------------------
    merge_keys = ["mof_id", "temperature_k", "pressure_bar", "rh"]
    merged = pd.merge(sim_df, pred_df, on=merge_keys, how="inner")
    if merged.empty:
        raise RuntimeError("Ground-truth and prediction rows did not align on "
                            "(mof_id, T, P, RH) -- check condition construction.")

    components = ["CO2", "N2", "H2O"]
    y_true = merged[["co2_loading_molkg", "n2_loading_molkg", "h2o_loading_molkg"]] \
        .to_numpy(dtype=np.float64)
    y_pred = merged[["co2_loading_molkg_pred", "n2_loading_molkg_pred", "h2o_loading_molkg_pred"]] \
        .to_numpy(dtype=np.float64)

    metrics = compute_regression_metrics(y_true, y_pred, component_names=components, prefix="")

    merged.to_parquet(output_dir / "unseen_mof_comparison.parquet", index=False)
    merged.to_csv(output_dir / "unseen_mof_comparison.csv", index=False)
    with open(output_dir / "unseen_mof_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: (float(v) if isinstance(v, (float, np.floating)) else v)
             for k, v in metrics.items()},
            f, indent=2,
        )

    try:
        viz = ResultVisualizer(save_dir=str(output_dir), component_names=components)
        viz.full_report(y_true=y_true, y_pred=y_pred, y_std=None)
    except Exception as e:
        logger.warning("Plotting failed (metrics/tables were still saved): %s", e)

    logger.info("=" * 65)
    logger.info("UNSEEN-MOF VALIDATION COMPLETE")
    logger.info("  MOFs tested      : %s", [d["mof_id"] for d in prepared])
    logger.info("  Comparison points: %d", len(merged))
    for k, v in sorted(metrics.items()):
        if isinstance(v, (int, float)):
            logger.info("  %s: %.4f", k, v)
    logger.info("  Output dir       : %s", output_dir)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
