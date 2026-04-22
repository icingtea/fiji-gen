#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.10"]
# ///
"""
plot_benchmark.py
-----------------
Reads benchmark_results.jsonl and produces a comprehensive PDF of graphs
covering: runtime, solve rate, generations-to-solve, timeout rate, and
per-episode fitness curves (rlga) / per-island fitness (par_rlga).

Run with:
    uv run plot_benchmark.py
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

# ── Config ───────────────────────────────────────────────────────────────────
RESULTS  = Path("benchmark_results.jsonl")
OUT_PDF  = Path("benchmark_plots.pdf")
MAX_FIT  = 256.0          # known optimal fitness for 128-bit / block-8 problem

SEQ_COL  = "#4C9BE8"      # blue  – rlga
PAR_COL  = "#E8824C"      # orange – par_rlga
ALPHA    = 0.35
# ─────────────────────────────────────────────────────────────────────────────

# ── Load & split data ─────────────────────────────────────────────────────────
records = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
seq = [r for r in records if r["mode"] == "rlga"]
par = [r for r in records if r["mode"] == "par_rlga"]

print(f"Loaded {len(seq)} rlga runs,  {len(par)} par_rlga runs.")
if not seq and not par:
    raise SystemExit("No data found in benchmark_results.jsonl")

# ── Helpers ───────────────────────────────────────────────────────────────────
def style(ax, title, xlabel, ylabel, grid=True):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    if grid:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(axis="y", which="major", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.3, alpha=0.4)


def safe(lst, default=float("nan")):
    return lst if lst else [default]


def jitter(n, scale=0.15):
    import random
    return [random.gauss(0, scale) for _ in range(n)]

# ── Per-run summary stats ──────────────────────────────────────────────────────
def seq_stats(r):
    return {
        "run":          r["run_number"],
        "elapsed":      r["elapsed_s"],
        "timed_out":    r["timed_out"],
        "solved":       r["solved"],
        "total_gens":   r.get("total_gens") or float("nan"),
        "n_ep":         r.get("n_episodes_run", 0),
        "final_best":   r.get("final_best") or float("nan"),
        "final_eps":    r.get("final_epsilon") or float("nan"),
    }

def par_stats(r):
    return {
        "run":          r["run_number"],
        "elapsed":      r["elapsed_s"],
        "timed_out":    r["timed_out"],
        "solved":       r["solved"],
        "min_gens":     r.get("min_gens_solved") or float("nan"),
        "solved_count": r.get("solved_count", 0),
        "total_islands":r.get("total_islands") or float("nan"),
        "best_fitness": max(
            (ir["best_fitness"] for ir in r.get("island_results", [])),
            default=float("nan"),
        ),
    }

ss = [seq_stats(r) for r in seq]
ps = [par_stats(r) for r in par]

# ── Build PDF ─────────────────────────────────────────────────────────────────
with PdfPages(OUT_PDF) as pdf:

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 – Overview dashboard (2×3)
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("RLGA vs Par-RLGA  ·  Benchmark Overview", fontsize=14, fontweight="bold")
    plt.subplots_adjust(hspace=0.38, wspace=0.32)

    # ── 1. Runtime distribution (box) ────────────────────────────────────────
    ax = axes[0, 0]
    data_box = []
    labels_box = []
    if ss:
        data_box.append([s["elapsed"] for s in ss])
        labels_box.append("rlga\n(seq)")
    if ps:
        data_box.append([s["elapsed"] for s in ps])
        labels_box.append("par_rlga\n(parallel)")
    bp = ax.boxplot(data_box, patch_artist=True, notch=False, widths=0.4,
                    medianprops=dict(color="white", linewidth=2))
    colors = [SEQ_COL, PAR_COL]
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax.set_xticklabels(labels_box, fontsize=9)
    style(ax, "Runtime Distribution", "", "seconds")

    # ── 2. Solve rate bar ─────────────────────────────────────────────────────
    ax = axes[0, 1]
    def solve_rate(lst): return (sum(1 for s in lst if s["solved"]) / len(lst) * 100) if lst else 0
    rates = []
    blabels = []
    cols_sr = []
    if ss:
        rates.append(solve_rate(ss)); blabels.append("rlga"); cols_sr.append(SEQ_COL)
    if ps:
        rates.append(solve_rate(ps)); blabels.append("par_rlga"); cols_sr.append(PAR_COL)
    bars = ax.bar(blabels, rates, color=cols_sr, alpha=0.8, width=0.4)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 115)
    style(ax, "Solve Rate", "", "% of runs solved")

    # ── 3. Timeout rate bar ───────────────────────────────────────────────────
    ax = axes[0, 2]
    def timeout_rate(lst): return (sum(1 for s in lst if s["timed_out"]) / len(lst) * 100) if lst else 0
    t_rates = []
    t_labels = []
    t_cols = []
    if ss:
        t_rates.append(timeout_rate(ss)); t_labels.append("rlga"); t_cols.append(SEQ_COL)
    if ps:
        t_rates.append(timeout_rate(ps)); t_labels.append("par_rlga"); t_cols.append(PAR_COL)
    tbars = ax.bar(t_labels, t_rates, color=t_cols, alpha=0.8, width=0.4)
    for bar, rate in zip(tbars, t_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 115)
    style(ax, "Timeout Rate", "", "% of runs timed out")

    # ── 4. Generations to solve (box, solved-only) ────────────────────────────
    ax = axes[1, 0]
    gen_box = []
    gen_labels = []
    gen_cols = []
    seq_gen_solved = [s["total_gens"] for s in ss if s["solved"] and not math.isnan(s["total_gens"])]
    par_gen_solved = [s["min_gens"] for s in ps if s["solved"] and not math.isnan(s["min_gens"])]
    if seq_gen_solved:
        gen_box.append(seq_gen_solved); gen_labels.append("rlga\n(seq)"); gen_cols.append(SEQ_COL)
    if par_gen_solved:
        gen_box.append(par_gen_solved); gen_labels.append("par_rlga\nmin gens"); gen_cols.append(PAR_COL)
    if gen_box:
        gbp = ax.boxplot(gen_box, patch_artist=True, widths=0.4,
                         medianprops=dict(color="white", linewidth=2))
        for patch, col in zip(gbp["boxes"], gen_cols):
            patch.set_facecolor(col); patch.set_alpha(0.75)
        ax.set_xticklabels(gen_labels, fontsize=9)
    else:
        ax.text(0.5, 0.5, "No solved runs yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
    style(ax, "Generations to Solve\n(solved runs only)", "", "generations")

    # ── 5. Best fitness per run (scatter) ─────────────────────────────────────
    ax = axes[1, 1]
    if ss:
        bf_seq = [s["final_best"] for s in ss]
        ax.scatter(range(len(bf_seq)), bf_seq, color=SEQ_COL, alpha=0.6, s=20, label="rlga")
    if ps:
        bf_par = [s["best_fitness"] for s in ps]
        ax.scatter(range(len(bf_par)), bf_par, color=PAR_COL, alpha=0.6, s=20, label="par_rlga")
    ax.axhline(MAX_FIT, color="red", linestyle="--", linewidth=0.8, label=f"optimal ({MAX_FIT:.0f})")
    ax.legend(fontsize=7)
    style(ax, "Best Fitness Per Run", "run index", "best fitness")

    # ── 6. Runtime over time (trend) ──────────────────────────────────────────
    ax = axes[1, 2]
    if ss:
        ax.plot([s["run"] for s in ss], [s["elapsed"] for s in ss],
                color=SEQ_COL, linewidth=0.8, alpha=0.7, label="rlga")
    if ps:
        ax.plot([s["run"] for s in ps], [s["elapsed"] for s in ps],
                color=PAR_COL, linewidth=0.8, alpha=0.7, label="par_rlga")
    ax.axhline(60, color="red", linestyle="--", linewidth=0.7, label="timeout (60s)")
    ax.legend(fontsize=7)
    style(ax, "Elapsed Time per Run", "run number", "seconds")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 – Sequential RLGA deep-dive
    # ══════════════════════════════════════════════════════════════════════════
    if seq:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("Sequential RLGA  ·  Deep Dive", fontsize=14, fontweight="bold")
        plt.subplots_adjust(hspace=0.38, wspace=0.32)

        # collect episode traces (best_fitness per episode, flattened per run)
        all_ep_best   = []   # list of lists: one list per run
        all_ep_reward = []
        all_ep_eps    = []
        for r in seq:
            eps_data = r.get("episodes", [])
            if eps_data:
                all_ep_best.append([e["best_fitness"] for e in eps_data])
                all_ep_reward.append([e["total_reward"] for e in eps_data])
                all_ep_eps.append([e["epsilon"] for e in eps_data])

        # a) Best fitness trajectory (all runs overlaid)
        ax = axes[0, 0]
        for trace in all_ep_best:
            ax.plot(range(len(trace)), trace, color=SEQ_COL, alpha=0.25, linewidth=0.7)
        if all_ep_best:
            max_len = max(len(t) for t in all_ep_best)
            # mean across runs (pad with NaN)
            import statistics
            mean_trace = []
            for i in range(max_len):
                vals = [t[i] for t in all_ep_best if i < len(t)]
                mean_trace.append(statistics.mean(vals))
            ax.plot(range(len(mean_trace)), mean_trace, color="white", linewidth=1.8,
                    label="mean", zorder=5)
        ax.axhline(MAX_FIT, color="red", linestyle="--", linewidth=0.8, label="optimal")
        ax.legend(fontsize=7)
        style(ax, "Best Fitness per Episode\n(all runs)", "episode", "best fitness")
        ax.set_facecolor("#111111")

        # b) Total reward per episode
        ax = axes[0, 1]
        for trace in all_ep_reward:
            ax.plot(range(len(trace)), trace, color="#7EC8E3", alpha=0.25, linewidth=0.7)
        style(ax, "Total Reward per Episode\n(all runs)", "episode", "total reward")
        ax.set_facecolor("#111111")

        # c) Epsilon decay per episode
        ax = axes[0, 2]
        for trace in all_ep_eps:
            ax.plot(range(len(trace)), trace, color="#A0E87C", alpha=0.3, linewidth=0.7)
        style(ax, "Epsilon Decay per Episode\n(all runs)", "episode", "ε")
        ax.set_facecolor("#111111")

        # d) total_gens histogram
        ax = axes[1, 0]
        tg_all  = [s["total_gens"] for s in ss if not math.isnan(s["total_gens"])]
        tg_sol  = [s["total_gens"] for s in ss if s["solved"] and not math.isnan(s["total_gens"])]
        if tg_all:
            ax.hist(tg_all, bins=20, color=SEQ_COL, alpha=0.6, label="all")
        if tg_sol:
            ax.hist(tg_sol, bins=20, color="#2ECC71", alpha=0.7, label="solved")
        ax.legend(fontsize=7)
        style(ax, "Total Generations Histogram", "total gens", "count")

        # e) n_episodes_run histogram
        ax = axes[1, 1]
        ep_run_vals = [s["n_ep"] for s in ss]
        if ep_run_vals:
            ax.hist(ep_run_vals, bins=max(1, max(ep_run_vals)), color=SEQ_COL, alpha=0.7)
        style(ax, "Episodes Run per Experiment", "episodes run", "count")

        # f) Final best fitness histogram
        ax = axes[1, 2]
        fb_vals = [s["final_best"] for s in ss if not math.isnan(s["final_best"])]
        if fb_vals:
            ax.hist(fb_vals, bins=20, color=SEQ_COL, alpha=0.7)
        ax.axvline(MAX_FIT, color="red", linestyle="--", linewidth=0.8, label="optimal")
        ax.legend(fontsize=7)
        style(ax, "Final Best Fitness Distribution", "fitness", "count")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 – Parallel RLGA deep-dive
    # ══════════════════════════════════════════════════════════════════════════
    if par:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("Parallel RLGA (Island Model)  ·  Deep Dive", fontsize=14, fontweight="bold")
        plt.subplots_adjust(hspace=0.38, wspace=0.32)

        # a) min_gens_solved histogram
        ax = axes[0, 0]
        mg_vals = [s["min_gens"] for s in ps if s["solved"] and not math.isnan(s["min_gens"])]
        if mg_vals:
            ax.hist(mg_vals, bins=max(5, len(mg_vals)//3), color=PAR_COL, alpha=0.8)
            ax.axvline(min(mg_vals), color="lime", linestyle="--", linewidth=1,
                       label=f"min={min(mg_vals):.0f}")
            ax.axvline(sum(mg_vals)/len(mg_vals), color="white", linestyle="--", linewidth=1,
                       label=f"mean={sum(mg_vals)/len(mg_vals):.0f}")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "No solved runs", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
        style(ax, "Min Gens to Solve\n(solved runs)", "generations", "count")

        # b) Islands solved per run
        ax = axes[0, 1]
        sc_vals  = [s["solved_count"] for s in ps]
        tot_isl  = ps[0]["total_islands"] if ps else 1
        ax.bar(range(len(sc_vals)), sc_vals, color=PAR_COL, alpha=0.75, width=1.0)
        ax.axhline(tot_isl, color="red", linestyle="--", linewidth=0.8, label=f"total ({tot_isl})")
        ax.legend(fontsize=7)
        style(ax, "Islands Solved per Run", "run index", "# islands solved")

        # c) Best fitness per run (par)
        ax = axes[0, 2]
        bfp = [s["best_fitness"] for s in ps if not math.isnan(s["best_fitness"])]
        if bfp:
            ax.plot(bfp, color=PAR_COL, linewidth=0.8, alpha=0.8)
            ax.axhline(MAX_FIT, color="red", linestyle="--", linewidth=0.8, label="optimal")
            ax.legend(fontsize=7)
        style(ax, "Best Fitness per Run\n(best island)", "run index", "fitness")

        # d) Per-island best-fitness trace (from island_logs, last run only)
        ax = axes[1, 0]
        last_par = par[-1]
        island_traces = {}
        for log in last_par.get("island_logs", []):
            iid = log["island"]
            island_traces.setdefault(iid, {"gen": [], "best": [], "avg": []})
            island_traces[iid]["gen"].append(log["generation"])
            island_traces[iid]["best"].append(log["best_fitness"])
            island_traces[iid]["avg"].append(log["avg_fitness"])

        cmap = plt.cm.plasma
        n_isl = max(1, len(island_traces))
        for idx, (iid, tr) in enumerate(sorted(island_traces.items())):
            col = cmap(idx / n_isl)
            ax.plot(tr["gen"], tr["best"], color=col, linewidth=0.7, alpha=0.7)
        ax.axhline(MAX_FIT, color="red", linestyle="--", linewidth=0.8)
        ax.set_title("Per-Island Best Fitness\n(last run)", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("generation (logged every 100)", fontsize=9)
        ax.set_ylabel("best fitness", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
        ax.set_facecolor("#111111")

        # e) Per-island avg-fitness trace (last run)
        ax = axes[1, 1]
        for idx, (iid, tr) in enumerate(sorted(island_traces.items())):
            col = cmap(idx / n_isl)
            ax.plot(tr["gen"], tr["avg"], color=col, linewidth=0.7, alpha=0.7)
        ax.set_title("Per-Island Avg Fitness\n(last run)", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("generation (logged every 100)", fontsize=9)
        ax.set_ylabel("avg fitness", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
        ax.set_facecolor("#111111")

        # f) Elapsed time histogram
        ax = axes[1, 2]
        el_par = [s["elapsed"] for s in ps]
        ax.hist(el_par, bins=20, color=PAR_COL, alpha=0.8)
        ax.axvline(60, color="red", linestyle="--", linewidth=0.8, label="timeout")
        ax.legend(fontsize=7)
        style(ax, "Runtime Distribution (par_rlga)", "seconds", "count")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 4 – Head-to-head comparison
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Head-to-Head Comparison", fontsize=14, fontweight="bold")
    plt.subplots_adjust(wspace=0.35)

    # a) CDF of generations to solve
    ax = axes[0]
    def plot_cdf(ax, data, color, label):
        if not data:
            return
        sorted_d = sorted(data)
        n = len(sorted_d)
        ax.plot(sorted_d, [(i+1)/n for i in range(n)],
                color=color, linewidth=1.5, label=label)
    plot_cdf(ax,
             [s["total_gens"] for s in ss if s["solved"] and not math.isnan(s["total_gens"])],
             SEQ_COL, "rlga (total_gens)")
    plot_cdf(ax,
             [s["min_gens"] for s in ps if s["solved"] and not math.isnan(s["min_gens"])],
             PAR_COL, "par_rlga (min_gens)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    style(ax, "CDF – Generations to Solve", "generations", "cumulative probability")

    # b) CDF of runtime
    ax = axes[1]
    plot_cdf(ax, [s["elapsed"] for s in ss], SEQ_COL, "rlga")
    plot_cdf(ax, [s["elapsed"] for s in ps], PAR_COL, "par_rlga")
    ax.axvline(60, color="red", linestyle="--", linewidth=0.8, label="timeout")
    ax.legend(fontsize=8)
    style(ax, "CDF – Runtime", "seconds", "cumulative probability")

    # c) Solve rate summary table as a text plot
    ax = axes[2]
    ax.axis("off")
    def fmt(lst, key, fmt_str="{:.1f}"):
        vals = [s[key] for s in lst if not (isinstance(s[key], float) and math.isnan(s[key]))]
        if not vals:
            return "N/A"
        return fmt_str.format(sum(vals) / len(vals))

    rows = [
        ["Metric", "rlga (seq)", "par_rlga"],
        ["# runs", str(len(ss)), str(len(ps))],
        ["Solve rate", f"{(sum(1 for s in ss if s['solved'])/max(1,len(ss))*100):.1f}%",
                       f"{(sum(1 for s in ps if s['solved'])/max(1,len(ps))*100):.1f}%"],
        ["Timeout rate", f"{(sum(1 for s in ss if s['timed_out'])/max(1,len(ss))*100):.1f}%",
                         f"{(sum(1 for s in ps if s['timed_out'])/max(1,len(ps))*100):.1f}%"],
        ["Mean runtime (s)", fmt(ss, "elapsed"), fmt(ps, "elapsed")],
        ["Mean gens (solved)",
         fmt([s for s in ss if s["solved"]], "total_gens", "{:.0f}"),
         fmt([s for s in ps if s["solved"]], "min_gens", "{:.0f}")],
        ["Min gens (solved)",
         str(int(min((s["total_gens"] for s in ss if s["solved"] and not math.isnan(s["total_gens"])), default=float("nan")))) if any(s["solved"] for s in ss) else "N/A",
         str(int(min((s["min_gens"]   for s in ps if s["solved"] and not math.isnan(s["min_gens"])),   default=float("nan")))) if any(s["solved"] for s in ps) else "N/A"],
        ["Mean best fitness", fmt(ss, "final_best"), fmt(ps, "best_fitness")],
    ]
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.6)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#1A252F")
            cell.set_text_props(color="#DDDDDD")
        else:
            cell.set_facecolor("#243342")
            cell.set_text_props(color="#DDDDDD")
        cell.set_edgecolor("#4A5568")
    ax.set_title("Summary Statistics", fontsize=11, fontweight="bold", pad=12)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(f"\n✓ Saved {OUT_PDF}  ({OUT_PDF.stat().st_size // 1024} KB)")
