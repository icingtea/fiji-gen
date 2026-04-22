#!/usr/bin/env python3
"""
benchmark_rlga.py
-----------------
Runs `rlga` and `par_rlga` repeatedly until Ctrl-C.
Each run is capped at TIMEOUT seconds. All statistics are appended to a
JSON Lines file (one JSON object per line) for easy downstream plotting.

Usage:
    python3 benchmark_rlga.py
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent
BIN          = REPO_ROOT / "bin" / "deceptive_royal_road_problem_rl"
PARAMS_SEQ   = REPO_ROOT / "params_rlga.txt"
PARAMS_PAR   = REPO_ROOT / "params_rlga_par.txt"
RESULTS_FILE = REPO_ROOT / "benchmark_results.jsonl"
TIMEOUT      = 60          # seconds per run
# ─────────────────────────────────────────────────────────────────────────────


def load_params(path: Path) -> dict:
    """Parse a key-value params file into a dict (comments stripped)."""
    params = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                params[parts[0]] = parts[1].strip()
    return params


# ── Output parsers ────────────────────────────────────────────────────────────

def parse_seq_output(stdout: str) -> dict:
    """
    Extract per-episode rows and summary from rlga stdout.

    Episode line format:
        Ep   0 | gens=  50 | reward= -12.00 | best=  48.0 | eps=0.4850
    Final line (optional):
        Solved at episode 12 after 50 gens this episode (650 total cumulative gens).
    """
    episodes = []
    ep_re = re.compile(
        r"Ep\s+(\d+)\s+\|\s+gens=\s*(\d+)\s+\|\s+reward=\s*([\d.\-]+)"
        r"\s+\|\s+best=\s*([\d.]+)\s+\|\s+eps=([\d.]+)"
    )
    solved_re = re.compile(
        r"Solved at episode\s+(\d+)\s+after\s+(\d+)\s+gens.*?\((\d+)\s+total"
    )

    solved       = False
    solved_ep    = None
    solved_gens_ep = None
    total_gens   = None

    for line in stdout.splitlines():
        m = ep_re.search(line)
        if m:
            episodes.append({
                "episode":      int(m.group(1)),
                "gens_this_ep": int(m.group(2)),
                "total_reward": float(m.group(3)),
                "best_fitness": float(m.group(4)),
                "epsilon":      float(m.group(5)),
            })
            continue
        m = solved_re.search(line)
        if m:
            solved          = True
            solved_ep       = int(m.group(1))
            solved_gens_ep  = int(m.group(2))
            total_gens      = int(m.group(3))

    return {
        "episodes":         episodes,
        "solved":           solved,
        "solved_episode":   solved_ep,
        "solved_gens_this_ep": solved_gens_ep,
        "total_gens":       total_gens,
        "n_episodes_run":   len(episodes),
        "final_best":       episodes[-1]["best_fitness"] if episodes else None,
        "final_epsilon":    episodes[-1]["epsilon"]      if episodes else None,
    }


def parse_par_output(stdout: str) -> dict:
    """
    Extract per-island log lines and the summary block from par_rlga stdout.

    Island log line:
        [ Island 3 | ep 2 | gen 412 ] Best: 256.0 | Avg: 201.3 | eps: 0.4372
    Results lines:
        Island 0 best fitness: 256 | generations: 412 [SOLVED]
        Island 1 best fitness: 241 | generations: 600 [did not solve]
    Summary lines:
        Islands solved: 7 / 20
        Minimum generations to solution: 388
    """
    island_logs = []
    island_log_re = re.compile(
        r"\[\s*Island\s+(\d+)\s+\|\s+ep\s+(\d+)\s+\|\s+gen\s+(\d+)\s*\]"
        r"\s+Best:\s*([\d.]+)\s+\|\s+Avg:\s*([\d.]+)\s+\|\s+eps:\s*([\d.]+)"
    )

    island_results = []
    result_re = re.compile(
        r"Island\s+(\d+)\s+best fitness:\s*([\d.]+)\s+\|\s+generations:\s+(\d+)"
        r"\s+\[(SOLVED|did not solve)\]"
    )

    solved_count_re   = re.compile(r"Islands solved:\s*(\d+)\s*/\s*(\d+)")
    min_gens_re       = re.compile(r"Minimum generations to solution:\s*(\d+)")
    no_solve_re       = re.compile(r"No island reached the optimal solution")

    solved_count     = 0
    total_islands    = None
    min_gens_solved  = None
    any_solved       = False

    for line in stdout.splitlines():
        m = island_log_re.search(line)
        if m:
            island_logs.append({
                "island":       int(m.group(1)),
                "episode":      int(m.group(2)),
                "generation":   int(m.group(3)),
                "best_fitness": float(m.group(4)),
                "avg_fitness":  float(m.group(5)),
                "epsilon":      float(m.group(6)),
            })
            continue

        m = result_re.search(line)
        if m:
            island_results.append({
                "island":       int(m.group(1)),
                "best_fitness": float(m.group(2)),
                "generations":  int(m.group(3)),
                "solved":       m.group(4) == "SOLVED",
            })
            continue

        m = solved_count_re.search(line)
        if m:
            solved_count  = int(m.group(1))
            total_islands = int(m.group(2))
            any_solved    = solved_count > 0
            continue

        m = min_gens_re.search(line)
        if m:
            min_gens_solved = int(m.group(1))
            continue

        if no_solve_re.search(line):
            any_solved = False

    return {
        "island_logs":       island_logs,
        "island_results":    island_results,
        "solved":            any_solved,
        "solved_count":      solved_count,
        "total_islands":     total_islands,
        "min_gens_solved":   min_gens_solved,
        "per_island": {
            r["island"]: r for r in island_results
        },
    }


# ── Runner ────────────────────────────────────────────────────────────────────

def run_once(mode: str, params_file: Path) -> dict:
    """
    Launch one run of `mode` (rlga | par_rlga), wait up to TIMEOUT seconds.
    Returns a rich statistics dict.
    """
    cmd    = [str(BIN), mode, str(params_file)]
    t0     = time.monotonic()
    timed_out = False

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
        stdout   = proc.stdout
        stderr   = proc.stderr
        ret_code = proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout    = e.stdout or ""
        stderr    = e.stderr or ""
        ret_code  = -1
        timed_out = True

    elapsed = time.monotonic() - t0

    if mode == "rlga":
        parsed = parse_seq_output(stdout)
    else:
        parsed = parse_par_output(stdout)

    record = {
        "mode":       mode,
        "params":     str(params_file.name),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "elapsed_s":  round(elapsed, 3),
        "timed_out":  timed_out,
        "return_code": ret_code,
        "stderr":     stderr.strip()[-500:] if stderr.strip() else "",
        **parsed,
    }
    return record


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print(f"Binary  : {BIN}")
    print(f"Results : {RESULTS_FILE}")
    print(f"Timeout : {TIMEOUT}s per run")
    print(f"Press Ctrl-C to stop.\n")

    run_number = 0

    # Count existing records to resume numbering
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            run_number = sum(1 for line in f if line.strip())
        print(f"Resuming from run #{run_number} (appending to existing file).\n")

    experiments = [
        ("rlga",     PARAMS_SEQ),
        ("par_rlga", PARAMS_PAR),
    ]

    try:
        while True:
            for mode, params_file in experiments:
                run_number += 1
                print(f"[Run #{run_number:04d}] {mode:10s} ", end="", flush=True)

                record = run_once(mode, params_file)
                record["run_number"] = run_number

                # ── Print one-line summary ────────────────────────────────
                status  = "TIMEOUT" if record["timed_out"] else (
                          "SOLVED"  if record["solved"]    else "not solved")
                elapsed = record["elapsed_s"]

                if mode == "rlga":
                    ep_run  = record.get("n_episodes_run", "?")
                    best    = record.get("final_best", "?")
                    tg      = record.get("total_gens", "?")
                    print(f"| {status:10s} | {elapsed:6.1f}s "
                          f"| ep={ep_run} | best={best} | total_gens={tg}")
                else:
                    sc      = record.get("solved_count", 0)
                    ti      = record.get("total_islands", "?")
                    mg      = record.get("min_gens_solved", "N/A")
                    print(f"| {status:10s} | {elapsed:6.1f}s "
                          f"| islands_solved={sc}/{ti} | min_gens={mg}")

                # ── Append to JSONL file ──────────────────────────────────
                with open(RESULTS_FILE, "a") as f:
                    f.write(json.dumps(record) + "\n")

    except KeyboardInterrupt:
        print(f"\n\nStopped after {run_number} total runs.")
        print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
