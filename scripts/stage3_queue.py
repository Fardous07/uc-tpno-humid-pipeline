#!/usr/bin/env python3
"""
stage3_queue.py -- finish the remaining Stage 3 GCMC work on a shared job queue.

Static chunks (one fixed MOF list per worker) leave cores idle once the light
workers finish, while one worker with heavy MOFs runs for weeks. This script
re-cuts every unfinished worker's remaining work into small units -- one MOF
at ONE temperature (15 pressure x RH jobs) -- and runs them through a pool of
N slots, heaviest units first, so every core stays busy until the end.

Each unit is an ordinary 04_run_simulations.py Stage 3 run in its own
output dir data/processed/adsorption_v2/worker_q<NN>/, so merge_simulation_
results.py picks it up like any other worker. A unit's checkpoint is seeded
with rows the original worker already finished for that MOF+temperature, so
nothing already computed is redone.

The original unfinished workers are "finalized": their chunk is cut down to
only their fully completed MOFs and they are rerun as zero-cost units, which
just reload the checkpoint and write adsorption_training.parquet.

Fresh run (no earlier workers): `prepare-fresh` builds one unit per
(selected MOF, temperature) straight from gcmc_selected_mofs.txt, biggest
simulation boxes first.

Usage:
    python scripts/stage3_queue.py prepare-fresh [--dry-run]
    python scripts/stage3_queue.py prepare [--workers 0 1 2 15 16] [--skip-mofs ID ...]
    python scripts/stage3_queue.py prepare --dry-run      # show the plan, change nothing
    python scripts/stage3_queue.py status
    bash   scripts/stage3_queue.sh start [N_SLOTS]    # runs `run` in tmux
    bash   scripts/stage3_queue.sh kill
Safe to kill and restart at any point: finished units are skipped and
unfinished ones resume from their checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/home/fardo/uc_tpno_humid_pipeline/uc_tpno_humid_pipeline")
# adsorption_v3: first run with framework charges + the UFF_TIP4P2005 force
# field (adsorption_v2 used neither and is invalid).
# The QUEUE_* environment variables point the same queue at another MOF set
# (e.g. the Level 3/4 validation runs, scripts/validation/validation_gcmc.sh)
# with identical simulation settings.
OUT_DIR = Path(os.environ.get("QUEUE_OUT_DIR", PROJECT_ROOT / "data/processed/adsorption_v3"))
LOG_DIR = OUT_DIR / "logs"
CHUNK_DIR = PROJECT_ROOT / "data/intermediate/cif_chunks_stage3"
QUEUE_CIF_DIR = Path(os.environ.get("QUEUE_CIF_DIR", PROJECT_ROOT / "data/intermediate/cif_chunks_queue_v3"))
SELECTED_FILE = Path(os.environ.get("QUEUE_SELECTED", OUT_DIR / "gcmc_selected_mofs.txt"))
CUTOFF_A = 12.0
CIF_DIR = Path(os.environ.get("QUEUE_SRC_CIF_DIR", PROJECT_ROOT / "data/intermediate/cifs_sanitized"))
PLAN_FILE = OUT_DIR / "stage3_queue_plan.json"
PID_FILE = OUT_DIR / "stage3_queue.pid"
PYTHON = PROJECT_ROOT / ".venv/bin/python"
SCRIPT_PY = PROJECT_ROOT / "scripts/04_run_simulations.py"
RASPA_BIN = "/home/fardo/RASPA2-install/bin/simulate"
RASPA_DATA = "/home/fardo/RASPA2-install"

# Must match run_parallel_gcmc.sh so queued jobs use identical settings.
STAGE3_CYCLES = 2000  # production cycles per Stage 3 job
TEMPERATURES = [298.15, 313.15, 333.15]
PRESSURES = "0.1,0.5,1.0,5.0,10.0"
RHS = "0.0,0.05,0.15"
DRY_CO2_FRAC = 0.15
JOBS_PER_TEMP = 15  # 5 pressures x 3 RH
KEY = ["mof_id", "temperature_k", "pressure_bar", "rh"]

JOB_LINE = re.compile(r"^(\d\d:\d\d:\d\d) \| INFO \|   \[\d+/\d+\] (\S+) \|")


def load_checkpoint(worker_dir: Path) -> pd.DataFrame:
    p = worker_dir / "gcmc_adsorption.checkpoint.csv"
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p)
    return pd.DataFrame(columns=KEY)


def minutes_per_job(log_path: Path) -> dict[str, float]:
    """Average wall-clock minutes per job for each MOF seen in a worker log,
    measured between consecutive job-start lines (gaps > 12 h are restarts)."""
    per: dict[str, list[float]] = {}
    prev = None
    for line in log_path.read_text(errors="replace").splitlines():
        m = JOB_LINE.match(line)
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%H:%M:%S")
        if prev is not None:
            dt = (t - prev[0]).total_seconds()
            if dt < 0:
                dt += 86400
            if dt < 12 * 3600:
                per.setdefault(prev[1], []).append(dt / 60)
        prev = (t, m.group(2))
    return {k: sum(v) / len(v) for k, v in per.items() if v}


def unit_cmd(cif_dir: Path, out_dir: Path, temps: list[float]) -> list[str]:
    return [
        str(PYTHON), str(SCRIPT_PY),
        "--cif-dir", str(cif_dir), "--output-dir", str(out_dir),
        "--raspa-path", RASPA_BIN, "--raspa-data-dir", RASPA_DATA,
        "--skip-stage1", "--skip-stage2",
        "--stage3-n-cycles", str(STAGE3_CYCLES),
        "--temperatures", ",".join(str(t) for t in temps),
        "--pressures", PRESSURES, "--rhs", RHS,
        "--dry-co2-frac", str(DRY_CO2_FRAC),
    ]


# ─────────────────────────────────────────────────────────────────────
# prepare
# ─────────────────────────────────────────────────────────────────────
def prepare(workers: list[int], skip_mofs: set[str], dry_run: bool = False) -> None:
    if PLAN_FILE.exists() and not dry_run:
        sys.exit(f"{PLAN_FILE} already exists -- the queue is already prepared. "
                 f"Use `status` / `stage3_queue.sh start`.")

    backup = CHUNK_DIR.with_name(CHUNK_DIR.name + ".before_queue")
    if not backup.exists() and not dry_run:
        shutil.copytree(CHUNK_DIR, backup, symlinks=True)
        print(f"Backed up chunk assignment -> {backup}")

    units, finalize = [], []
    for w in workers:
        wdir = OUT_DIR / f"worker_{w}"
        chunk = CHUNK_DIR / f"worker_{w}"
        if (wdir / "adsorption_training.parquet").exists():
            print(f"worker_{w}: already complete, skipping")
            continue
        mofs = sorted(p.stem for p in chunk.glob("*.cif"))
        ckpt = load_checkpoint(wdir)
        speed = minutes_per_job(LOG_DIR / f"worker_{w}.log")
        worker_avg = sum(speed.values()) / len(speed) if speed else 30.0

        complete = []
        for mof in mofs:
            if mof in skip_mofs:
                continue
            rows = ckpt[ckpt.mof_id == mof]
            if len(rows.drop_duplicates(KEY)) >= len(TEMPERATURES) * JOBS_PER_TEMP:
                complete.append(mof)
                continue
            for t in TEMPERATURES:
                done_t = rows[(rows.temperature_k - t).abs() < 1e-3]
                left = JOBS_PER_TEMP - len(done_t.drop_duplicates(KEY))
                if left <= 0:
                    continue
                units.append({
                    "mof_id": mof, "temperature_k": t, "from_worker": w,
                    "jobs_left": left,
                    "est_min": left * speed.get(mof, worker_avg),
                    "seed_rows": len(done_t),
                })

        # Cut the original chunk down to its completed MOFs so rerunning it
        # only reloads the checkpoint and writes the final parquet.
        for p in chunk.glob("*.cif"):
            if p.stem not in complete and not dry_run:
                p.unlink()
        finalize.append({"worker": w, "complete_mofs": len(complete)})
        print(f"worker_{w}: {len(complete)} MOFs complete, "
              f"{sum(u['from_worker'] == w for u in units)} units queued")

    # Heaviest first (longest-processing-time scheduling).
    units.sort(key=lambda u: -u["est_min"])
    if dry_run:
        for i, u in enumerate(units):
            print(f"  q{i:02d} ~{u['est_min']/60:5.1f} h  {u['jobs_left']:2d} jobs  T={u['temperature_k']:.0f}  w{u['from_worker']}  seed={u['seed_rows']}  {u['mof_id'][:45]}")
        print(f"\n{len(units)} units, {sum(u['jobs_left'] for u in units)} jobs, ~{sum(u['est_min'] for u in units)/60:.0f} core-hours")
        return
    for i, u in enumerate(units):
        name = f"worker_q{i:02d}"
        u["name"] = name
        cdir = QUEUE_CIF_DIR / name
        cdir.mkdir(parents=True, exist_ok=True)
        link = cdir / f"{u['mof_id']}.cif"
        if not link.exists():
            link.symlink_to(CIF_DIR / f"{u['mof_id']}.cif")
        odir = OUT_DIR / name
        odir.mkdir(parents=True, exist_ok=True)
        src = load_checkpoint(OUT_DIR / f"worker_{u['from_worker']}")
        seed = src[(src.mof_id == u["mof_id"]) &
                   ((src.temperature_k - u["temperature_k"]).abs() < 1e-3)]
        if len(seed):
            seed.to_csv(odir / "gcmc_adsorption.checkpoint.csv", index=False)

    PLAN_FILE.write_text(json.dumps({"finalize": finalize, "units": units}, indent=2))
    total_h = sum(u["est_min"] for u in units) / 60
    print(f"\n{len(units)} units, {sum(u['jobs_left'] for u in units)} jobs, "
          f"~{total_h:.0f} core-hours estimated. Plan -> {PLAN_FILE}")
    print(f"Heaviest unit: ~{units[0]['est_min'] / 60:.0f} h ({units[0]['mof_id'][:50]}...)"
          if units else "")


def box_atoms(cif: Path) -> int:
    """Atoms in the simulation box (cell atoms x unit cells) -- cost proxy."""
    import math
    import warnings
    from ase.io import read
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = read(str(cif))
    reps = [max(1, math.ceil(2 * CUTOFF_A / L)) for L in a.cell.lengths()]
    return len(a) * reps[0] * reps[1] * reps[2]


def prepare_fresh(dry_run: bool = False) -> None:
    if PLAN_FILE.exists() and not dry_run:
        sys.exit(f"{PLAN_FILE} already exists -- the queue is already prepared.")
    if not SELECTED_FILE.exists():
        sys.exit(f"{SELECTED_FILE} missing -- run Stage 1+2 first: "
                 f"bash scripts/run_parallel_gcmc.sh --selection-only")
    mofs = [m for m in SELECTED_FILE.read_text().split() if m]
    cost = {m: box_atoms(CIF_DIR / f"{m}.cif") for m in mofs}
    units = [{"mof_id": m, "temperature_k": t, "from_worker": None,
              "jobs_left": JOBS_PER_TEMP, "est_min": float(cost[m]), "seed_rows": 0}
             for m in mofs for t in TEMPERATURES]
    # Biggest boxes first (longest-processing-time scheduling); est_min holds
    # the box-atom proxy here, not minutes.
    units.sort(key=lambda u: -u["est_min"])
    if dry_run:
        for i, u in enumerate(units[:20]):
            print(f"  q{i:03d} {int(u['est_min']):7d} box atoms  T={u['temperature_k']:.0f}  {u['mof_id'][:55]}")
        print(f"  ...\n{len(units)} units ({len(mofs)} MOFs x {len(TEMPERATURES)} T), "
              f"{len(units) * JOBS_PER_TEMP} jobs")
        return
    for i, u in enumerate(units):
        name = f"worker_q{i:03d}"
        u["name"] = name
        cdir = QUEUE_CIF_DIR / name
        cdir.mkdir(parents=True, exist_ok=True)
        link = cdir / f"{u['mof_id']}.cif"
        if not link.exists():
            link.symlink_to(CIF_DIR / f"{u['mof_id']}.cif")
        (OUT_DIR / name).mkdir(parents=True, exist_ok=True)
    PLAN_FILE.write_text(json.dumps({"finalize": [], "units": units}, indent=2))
    print(f"{len(units)} units, {len(units) * JOBS_PER_TEMP} jobs. Plan -> {PLAN_FILE}")


# ─────────────────────────────────────────────────────────────────────
# run
# ─────────────────────────────────────────────────────────────────────
def is_done(name: str) -> bool:
    return (OUT_DIR / name / "adsorption_training.parquet").exists()


def run_unit(name: str, cif_dir: Path, temps: list[float]) -> None:
    if is_done(name):
        return
    log = LOG_DIR / f"{name}.log"
    with open(log, "a") as fh:
        fh.write(f"===== START {datetime.now():%F %T} =====\n")
        fh.flush()
        rc = subprocess.call(unit_cmd(cif_dir, OUT_DIR / name, temps),
                             stdout=fh, stderr=subprocess.STDOUT,
                             cwd=PROJECT_ROOT)
    print(f"{datetime.now():%F %T}  {name}: exit {rc}", flush=True)


def run(n_slots: int) -> None:
    plan = json.loads(PLAN_FILE.read_text())
    PID_FILE.write_text(str(os.getpid()))
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Finalize the original workers first -- instant, no simulations.
    for f in plan["finalize"]:
        name = f"worker_{f['worker']}"
        if f["complete_mofs"] and not is_done(name):
            run_unit(name, CHUNK_DIR / name, TEMPERATURES)

    todo = [u for u in plan["units"] if not is_done(u["name"])]
    print(f"{datetime.now():%F %T}  {len(todo)} units to run on {n_slots} slots", flush=True)
    with ThreadPoolExecutor(max_workers=n_slots) as pool:
        for u in todo:
            pool.submit(run_unit, u["name"], QUEUE_CIF_DIR / u["name"], [u["temperature_k"]])
    print(f"{datetime.now():%F %T}  queue finished", flush=True)
    PID_FILE.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────
# status
# ─────────────────────────────────────────────────────────────────────
def status() -> None:
    plan = json.loads(PLAN_FILE.read_text())
    running = False
    if PID_FILE.exists():
        try:
            os.kill(int(PID_FILE.read_text()), 0)
            running = True
        except (ProcessLookupError, ValueError):
            pass
    units = plan["units"]
    done = [u for u in units if is_done(u["name"])]
    jobs_left, active = 0, []
    for u in units:
        if is_done(u["name"]):
            continue
        n = len(load_checkpoint(OUT_DIR / u["name"]).drop_duplicates(KEY))
        jobs_left += JOBS_PER_TEMP - n
        # a unit's log is created when it starts, so started-but-unfinished ==
        # active (even during one long high-pressure job).
        log = LOG_DIR / f"{u['name']}.log"
        if log.exists():
            idle_h = (time.time() - log.stat().st_mtime) / 3600
            active.append((u["name"], n, u["mof_id"], u["temperature_k"], idle_h))
    fin = [f"worker_{f['worker']}" for f in plan["finalize"]]
    print("=" * 60)
    print(f"STAGE 3 QUEUE  ({'RUNNING' if running else 'NOT RUNNING'})")
    print("=" * 60)
    print(f"  Original workers finalized: "
          f"{sum(is_done(n) for n in fin)}/{len(fin)}")
    print(f"  Units done: {len(done)}/{len(units)} | jobs left: {jobs_left}")
    print(f"  Units started, not finished: {len(active)}")
    for name, n, mof, t, idle_h in active:
        print(f"    {name}: {n:2d}/{JOBS_PER_TEMP} jobs  T={t:.0f}K  current job {idle_h:4.1f} h  {mof[:48]}")
    n_final = len(list(OUT_DIR.glob("worker_*/adsorption_training.parquet")))
    print(f"  All worker dirs with final output: {n_final}")
    if not running and len(done) == len(units):
        print("  ALL DONE -> bash scripts/run_parallel_gcmc.sh --merge")
    print("=" * 60)


def kill() -> None:
    if not PID_FILE.exists():
        print("Queue not running (no pid file).")
    else:
        pid = int(PID_FILE.read_text())
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            print(f"Stopped queue process group {pid}.")
        except ProcessLookupError:
            print("Queue process already gone.")
        PID_FILE.unlink(missing_ok=True)
    # Unit runs and their RASPA children, in case any were orphaned.
    subprocess.call(["pkill", "-f", f"{SCRIPT_PY} --cif-dir {QUEUE_CIF_DIR}"])
    subprocess.call(["pkill", "-f", f"{SCRIPT_PY} --cif-dir {CHUNK_DIR}"])
    subprocess.call(["pkill", "-f", f"{RASPA_BIN}"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2, 15, 16])
    p.add_argument("--skip-mofs", nargs="*", default=[])
    p.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    pf = sub.add_parser("prepare-fresh")
    pf.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    r = sub.add_parser("run")
    r.add_argument("n_slots", type=int, nargs="?", default=max(1, (os.cpu_count() or 2) - 1))
    sub.add_parser("status")
    sub.add_parser("kill")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.workers, set(a.skip_mofs), a.dry_run)
    elif a.cmd == "prepare-fresh":
        prepare_fresh(a.dry_run)
    elif a.cmd == "run":
        run(a.n_slots)
    elif a.cmd == "status":
        status()
    else:
        kill()


if __name__ == "__main__":
    main()
