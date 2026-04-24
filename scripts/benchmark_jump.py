#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

import itertools
import subprocess
import json
import os
import sys

# =============================================================================
# Sweep axes — edit these to change what gets benchmarked
# =============================================================================
TOTAL_POPS        = [10000, 40000]
GENOME_SIZES      = [50, 100]
JUMP_SIZES        = [2, 4]
MIGRANT_PCTS      = [0.05, 0.10, 0.25, 0.50]  # fraction of island pop
N_THREADS_LIST    = [4, 20]
PARALLEL_VARIANTS = ["sum", "per"]  # sum: total_pop / threads per island | per: full pop per island

N_RUNS      = 10   # repetitions per configuration
TIMEOUT_SEC = 60   # hard kill per binary invocation (seconds)
OUTPUT_DIR  = "data"
TMP_FILE    = "jump_param.tmp.json"

# =============================================================================
# GA defaults (non-swept)
# =============================================================================
GA_MUT_RATE        = 0.01
GA_SELECTION_RATE  = 0.5
PAR_MIGRATION_PROB = 0.5
PAR_QUORUM         = 2

# =============================================================================
# RL defaults (non-swept)
# =============================================================================
RL_GEN_PER_EP    = 100
RL_N_EP          = 200
RL_GAMMA         = 0.9
RL_EPSILON       = 0.5
RL_EPSILON_DECAY = 0.97
RL_EPSILON_MIN   = 0.05
RL_LR            = 0.001
RL_BATCH_SIZE    = 128
RL_MEMORY_SIZE   = 10000
RL_MUT_RATES     = [0.01, 0.05, 0.2, 0.4]
RL_SEL_RATES     = [0.3, 0.6, 0.9]
# =============================================================================


def build_config(mode, pop, genome, jump, threads, migrants_pct, par_var):
    """
    Assemble the complete param JSON dict for this run from scratch.

    sum variant: island_pop = total_pop // n_threads   (islands share the budget)
    per variant: island_pop = total_pop                (every island gets full pop)
    """
    is_parallel = mode in ("par", "par_rlga")
    is_rl       = mode in ("rlga", "par_rlga")

    if is_parallel:
        island_pop = pop // threads if par_var == "sum" else pop
        n_migrants = max(1, int(island_pop * migrants_pct))
    else:
        island_pop = pop
        n_migrants = 0

    config = {
        "problem": {
            "GENOME_LENGTH": genome,
            "JUMP_SIZE":     jump,
        },
        "genetic_algorithm": {
            "pop_size":       island_pop,
            "mut_rate":       GA_MUT_RATE,
            "selection_rate": GA_SELECTION_RATE,
        },
    }

    if is_parallel:
        config["parallel"] = {
            "n_threads":             threads,
            "n_migrants":            n_migrants,
            "migration_probability": PAR_MIGRATION_PROB,
            "quorum":                PAR_QUORUM,
        }

    if is_rl:
        config["reinforcement_learning"] = {
            "gen_per_ep":    RL_GEN_PER_EP,
            "n_ep":          RL_N_EP,
            "gamma":         RL_GAMMA,
            "epsilon":       RL_EPSILON,
            "epsilon_decay": RL_EPSILON_DECAY,
            "epsilon_min":   RL_EPSILON_MIN,
            "lr":            RL_LR,
            "batch_size":    RL_BATCH_SIZE,
            "memory_size":   RL_MEMORY_SIZE,
            "mut_rates":     RL_MUT_RATES,
            "sel_rates":     RL_SEL_RATES,
        }

    return config, island_pop, n_migrants


def run_once(binary, mode, output_file, timeout_sec):
    """Run the binary once. The binary appends one JSONL line to output_file itself."""
    timed_out = False
    try:
        subprocess.run(
            ["./bin/" + binary, mode, TMP_FILE, output_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        timed_out = True
    return timed_out


def describe_variant(mode, pop, genome, jump, threads, migrants_pct, par_var, island_pop, n_migrants):
    if mode in ("seq", "rlga"):
        return f"{mode:<10}  pop={island_pop:<6}"
    else:
        return (f"{mode}/{par_var:<3}  total_pop={pop:<6}  island_pop={island_pop:<6}  "
                f"th={threads:<3}  migrants={n_migrants:<5}  ({int(migrants_pct*100)}%)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pre-compute total number of individual binary invocations for progress tracking
    n_seq_modes = 2  # seq + rlga
    n_par_combos = len(N_THREADS_LIST) * len(MIGRANT_PCTS) * len(PARALLEL_VARIANTS) * 2  # par + par_rlga
    variants_per_group = n_seq_modes + n_par_combos
    total_groups = len(GENOME_SIZES) * len(JUMP_SIZES) * len(TOTAL_POPS)
    total_invocations = total_groups * variants_per_group * N_RUNS
    invocation_count = 0

    print("=" * 70)
    print(f"  Jump Problem Benchmark")
    print(f"  Groups (genome x jump x pop): {total_groups}")
    print(f"  Variants per group:           {variants_per_group}")
    print(f"  Runs per variant:             {N_RUNS}")
    print(f"  Total invocations:            {total_invocations}")
    print(f"  Timeout per run:              {TIMEOUT_SEC}s")
    print(f"  Output dir:                   {OUTPUT_DIR}/")
    print("=" * 70)

    for genome, jump, pop in itertools.product(GENOME_SIZES, JUMP_SIZES, TOTAL_POPS):
        output_file = os.path.join(OUTPUT_DIR, f"genome{genome}_jump{jump}_pop{pop}.jsonl")

        print(f"\n{'─'*70}")
        print(f"  genome={genome}  jump={jump}  total_pop={pop}")
        print(f"  Output: {output_file}")
        print(f"{'─'*70}")

        # Build list of all variants for this group
        variants = []
        variants.append(("jump_problem",    "seq",     1,  0,    "N/A"))
        variants.append(("jump_problem_rl", "rlga",    1,  0,    "N/A"))
        for th, pct, var in itertools.product(N_THREADS_LIST, MIGRANT_PCTS, PARALLEL_VARIANTS):
            variants.append(("jump_problem",    "par",      th, pct,  var))
            variants.append(("jump_problem_rl", "par_rlga", th, pct,  var))

        for v_idx, (binary, mode, threads, migrants_pct, par_var) in enumerate(variants, 1):
            config, island_pop, n_migrants = build_config(
                mode, pop, genome, jump, threads, migrants_pct, par_var
            )
            with open(TMP_FILE, "w") as f:
                json.dump(config, f, indent=2)

            label = describe_variant(mode, pop, genome, jump, threads, migrants_pct, par_var, island_pop, n_migrants)
            print(f"\n  [{v_idx}/{len(variants)}] {label}")
            print(f"  ", end="", flush=True)

            timeouts = 0
            for run_i in range(1, N_RUNS + 1):
                invocation_count += 1
                pct_done = invocation_count / total_invocations * 100
                timed_out = run_once(binary, mode, output_file, TIMEOUT_SEC)
                if timed_out:
                    timeouts += 1
                    print(f"T", end="", flush=True)
                else:
                    print(f".", end="", flush=True)

            timeout_note = f"  ({timeouts} timeout{'s' if timeouts != 1 else ''})" if timeouts else ""
            print(f"  [{invocation_count}/{total_invocations} done | {pct_done:.1f}%]{timeout_note}")

    if os.path.exists(TMP_FILE):
        os.remove(TMP_FILE)

    print(f"\n{'='*70}")
    print(f"  Benchmark complete. Results in {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
