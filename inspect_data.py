#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
import json
from pathlib import Path

lines = Path("benchmark_results.jsonl").read_text().splitlines()
modes = {}
for l in lines:
    if not l.strip():
        continue
    r = json.loads(l)
    modes.setdefault(r["mode"], []).append(r)

for m, recs in modes.items():
    print(f"\n=== {m}: {len(recs)} runs ===")
    sample = recs[-1]
    print("  top-level keys:", list(sample.keys()))
    if m == "par_rlga":
        print("  island_results sample:", sample.get("island_results", [])[:2])
        print("  island_logs count:", len(sample.get("island_logs", [])))
        print("  min_gens_solved:", sample.get("min_gens_solved"))
    if m == "rlga":
        print("  episodes sample:", sample.get("episodes", [])[:2])
        print("  total_gens:", sample.get("total_gens"))
