#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Usage:
    python scripts/plot_jump_benchmark.py data/genome50_jump2_pop10000.jsonl

One PNG per distinct n_threads value found in the data.
Layout per PNG:
    rows  = migrant percentages (one row per unique mig%)
    cols  = Runtime (candlestick) | Generations (candlestick)

Output: data/plots/<stem>/threads_<N>.png
Output: data/plots/<stem>/threads_<N>_par_only.png
"""

import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                records.append(json.loads(s))
    return records


def parse_total_pop(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    for part in stem.split("_"):
        if part.startswith("pop"):
            try:
                return int(part[3:])
            except ValueError:
                pass
    return None


def get_par_variant(r, total_pop):
    island_pop = r.get("island_pop", r.get("pop_size"))
    if island_pop is None or total_pop is None:
        return "?"
    return "per" if island_pop >= total_pop else "sum"


def get_mig_pct(r):
    island_pop = r.get("island_pop", r.get("pop_size", 0))
    n_mig = r.get("n_migrants", 0)
    return round(n_mig / island_pop, 2) if island_pop > 0 else 0.0


def get_generations(r):
    for key in ("generations", "total_generations", "min_gens_solved", "min_generations"):
        if key in r:
            return r[key]
    return None


def record_is_solved(r):
    mode = r["mode"]
    if mode == "seq":
        return r.get("best_fitness", 0) >= r.get("genome_length", float("inf"))
    if mode == "rlga":
        return bool(r.get("solved", False))
    if mode == "par":
        return r.get("overall_best_fitness", 0) >= r.get("genome_length", float("inf"))
    if mode == "par_rlga":
        return r.get("solved_count", 0) > 0
    return False


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_candlesticks(ax, groups, ylabel, x_labels):
    """
    groups: list of [values]  — one per x slot
    Candlestick: whisker = min/max, box = IQR, line = median
    """
    for i, vals in enumerate(groups):
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        q25, med, q75 = np.percentile(arr, [25, 50, 75])
        lo, hi = arr.min(), arr.max()

        # Whisker
        ax.plot([i, i], [lo, hi], color="black", linewidth=1.5)
        ax.plot([i - 0.15, i + 0.15], [lo, lo], color="black", linewidth=1.2)
        ax.plot([i - 0.15, i + 0.15], [hi, hi], color="black", linewidth=1.2)
        # IQR box
        ax.bar(i, q75 - q25, bottom=q25, width=0.5,
               color="steelblue", edgecolor="black", linewidth=0.8, alpha=0.85)
        # Median
        ax.plot([i - 0.25, i + 0.25], [med, med], color="black", linewidth=2)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)


# ── per-threads figure ────────────────────────────────────────────────────────

def make_plot_for_threads(records, n_threads, total_pop, mig_pcts, out_dir, stem, par_only=False):
    seq_recs      = [r for r in records if r["mode"] == "seq"]
    rlga_recs     = [r for r in records if r["mode"] == "rlga"]
    par_all       = [r for r in records if r["mode"] == "par"      and r.get("n_threads") == n_threads]
    par_rlga_all  = [r for r in records if r["mode"] == "par_rlga" and r.get("n_threads") == n_threads]

    n_rows = len(mig_pcts)
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(10, n_rows * 3.5),
                             squeeze=False)
    
    if par_only:
        fig.suptitle(f"{stem}  —  {n_threads} island{'s' if n_threads > 1 else ''} (Parallel Only)", fontsize=13)
        x_labels = ["par\n(sum)", "par\n(per)", "par_rlga\n(sum)", "par_rlga\n(per)"]
    else:
        fig.suptitle(f"{stem}  —  {n_threads} island{'s' if n_threads > 1 else ''}", fontsize=13)
        x_labels = ["seq", "par\n(sum)", "par\n(per)", "rlga", "par_rlga\n(sum)", "par_rlga\n(per)"]

    # Column headers on first row only
    axes[0][0].set_title("Runtime (ms)", fontsize=11)
    axes[0][1].set_title("Generations to Best", fontsize=11)

    for row_i, mig_pct in enumerate(mig_pcts):
        pct_label = f"migrants = {int(round(mig_pct * 100))}%"

        def filt(recs):
            return [r for r in recs if abs(get_mig_pct(r) - mig_pct) < 0.001]

        par_sum  = [r for r in filt(par_all)      if get_par_variant(r, total_pop) == "sum"]
        par_per  = [r for r in filt(par_all)      if get_par_variant(r, total_pop) == "per"]
        prl_sum  = [r for r in filt(par_rlga_all) if get_par_variant(r, total_pop) == "sum"]
        prl_per  = [r for r in filt(par_rlga_all) if get_par_variant(r, total_pop) == "per"]

        if par_only:
            slot_recs = [par_sum, par_per, prl_sum, prl_per]
        else:
            slot_recs = [seq_recs, par_sum, par_per, rlga_recs, prl_sum, prl_per]

        # ── Runtime ─────────────────────────────────────────────────────────
        rt_groups = [[r["time_ms"] for r in recs] for recs in slot_recs]
        draw_candlesticks(axes[row_i][0], rt_groups, "ms", x_labels)

        # ── Generations ──────────────────────────────────────────────────────
        gen_groups = [
            [v for r in recs if (v := get_generations(r)) is not None]
            for recs in slot_recs
        ]
        draw_candlesticks(axes[row_i][1], gen_groups, "generations", x_labels)

        # Annotate the row on the left edge
        axes[row_i][0].set_ylabel(f"{pct_label}\n\nms", fontsize=9)

        # Row annotation box on the far left
        axes[row_i][0].annotate(
            pct_label,
            xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-0.38, 0.5), textcoords="axes fraction",
            fontsize=10, va="center", ha="center",
            rotation=90,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9),
        )

    fig.tight_layout(rect=[0.05, 0, 1, 0.97])
    suffix = "_par_only" if par_only else ""
    out_path = os.path.join(out_dir, f"threads_{n_threads}{suffix}.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")

# ── stats table ───────────────────────────────────────────────────────────────

def print_stats_table(records, total_pop):
    print("\n" + "="*84)
    print(f"{'Mode':<15} | {'Threads':<7} | {'MigPct':<6} | {'Var':<3} | {'Runtime (ms) (Median)':<21} | {'Generations (Median)':<20} | {'Solve %':<7}")
    print("-" * 84)
    
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        mode = r["mode"]
        n_threads = r.get("n_threads", 0)
        mig_pct = get_mig_pct(r) if mode.startswith("par") else 0.0
        var = get_par_variant(r, total_pop) if mode.startswith("par") else "N/A"
        groups[(mode, n_threads, mig_pct, var)].append(r)
        
    for k in sorted(groups.keys()):
        mode, n_threads, mig_pct, var = k
        recs = groups[k]
        
        rts = [r["time_ms"] for r in recs]
        gens = [v for r in recs if (v := get_generations(r)) is not None]
        solves = [record_is_solved(r) for r in recs]
        
        med_rt = np.median(rts) if rts else 0.0
        med_gen = np.median(gens) if gens else 0.0
        solve_pct = sum(solves) / len(solves) * 100 if solves else 0.0
        
        th_str = str(n_threads) if mode.startswith("par") else "N/A"
        mig_str = f"{mig_pct*100:.0f}%" if mode.startswith("par") else "N/A"
        
        print(f"{mode:<15} | {th_str:<7} | {mig_str:<6} | {var:<3} | {med_rt:>21.2f} | {med_gen:>20.1f} | {solve_pct:>6.1f}%")
        
    print("="*84 + "\n")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: plot_jump_benchmark.py <path/to/file.jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    stem = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join("data", "plots", stem)
    total_pop = parse_total_pop(path)

    print(f"Loading: {path}")
    records = load_jsonl(path)
    print(f"  {len(records)} records  |  total_pop from filename: {total_pop}")

    thread_counts = sorted({r["n_threads"] for r in records if "n_threads" in r})
    mig_pcts = sorted({
        get_mig_pct(r)
        for r in records
        if r["mode"] in ("par", "par_rlga") and "n_migrants" in r
    })

    print(f"  Thread counts : {thread_counts}")
    print(f"  Migrant pcts  : {[f'{p*100:.0f}%' for p in mig_pcts]}")
    print(f"  Output dir    : {out_dir}/\n")
    
    print_stats_table(records, total_pop)

    for th in thread_counts:
        print(f"Generating: threads_{th}.png ...")
        make_plot_for_threads(records, th, total_pop, mig_pcts, out_dir, stem)
        print(f"Generating: threads_{th}_par_only.png ...")
        make_plot_for_threads(records, th, total_pop, mig_pcts, out_dir, stem, par_only=True)

    print("Done.")


if __name__ == "__main__":
    main()
