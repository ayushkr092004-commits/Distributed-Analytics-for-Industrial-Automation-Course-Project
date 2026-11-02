#!/usr/bin/env python3
"""
run_experiments.py — reproducible evaluation sweeps.

Three experiments, each averaged over several random seeds:

  scalability   : 4 → 16 UAVs on progressively larger / denser maps
  robustness    : fixed team, packet loss 0 % → 60 %, radio range 8 → 4
  backend       : mock (rule-based) vs claude on identical scenarios
                  (only runs if ANTHROPIC_API_KEY is set)

Outputs: results/experiments.csv and results/experiments.png

    python scripts/run_experiments.py --seeds 5
    python scripts/run_experiments.py --seeds 3 --with-claude
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from uavplan.environment import Environment  # noqa: E402
from uavplan.llm.backend import make_backend  # noqa: E402
from uavplan.simulation import Simulation  # noqa: E402

FIELDS = ["experiment", "label", "backend", "uavs", "targets", "size", "loss", "comm_range",
          "seeds", "success_rate", "collisions", "near_misses", "makespan", "total_distance",
          "wait_steps", "messages_sent", "messages_delivered", "llm_calls", "wall_time_s"]


def run_config(experiment, label, backend, uavs, targets, size, density, loss, comm_range, seeds,
               shared_backend=None, max_steps=250):
    rows = []
    for seed in range(seeds):
        env = Environment.random(size, size, n_targets=targets, obstacle_density=density, seed=seed)
        sim = Simulation(env, n_uavs=uavs, backend=backend, comm_range=comm_range, loss_prob=loss,
                         max_steps=max_steps, seed=seed, shared_backend=shared_backend)
        rows.append(sim.run().as_dict())
        print(f"  [{experiment}/{label}] seed {seed}: targets {rows[-1]['targets_done']}/{targets} "
              f"steps {rows[-1]['steps']} coll {rows[-1]['collisions']} llm {rows[-1]['llm_calls']}",
              flush=True)

    def mean(k):
        vals = [r[k] for r in rows if r[k] is not None]
        return round(statistics.mean(vals), 2) if vals else None

    return {
        "experiment": experiment, "label": label, "backend": backend, "uavs": uavs,
        "targets": targets, "size": size, "loss": loss, "comm_range": comm_range, "seeds": seeds,
        "success_rate": round(sum(r["success"] for r in rows) / len(rows), 2),
        "collisions": sum(r["collisions"] for r in rows),
        "near_misses": mean("near_misses"), "makespan": mean("makespan"),
        "total_distance": mean("total_distance"), "wait_steps": mean("wait_steps"),
        "messages_sent": mean("messages_sent"), "messages_delivered": mean("messages_delivered"),
        "llm_calls": mean("llm_calls"), "wall_time_s": mean("wall_time_s"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--with-claude", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    results = []
    t0 = time.time()

    print("== Experiment 1: scalability ==")
    for uavs, targets, size, density in [(4, 6, 30, 0.12), (8, 12, 40, 0.16), (12, 16, 45, 0.18), (16, 20, 50, 0.2)]:
        results.append(run_config("scalability", f"{uavs} UAVs", "mock", uavs, targets, size,
                                  density, 0.0, 8.0, a.seeds))

    print("== Experiment 2: communication robustness ==")
    for loss, rng in [(0.0, 8.0), (0.2, 6.0), (0.4, 5.0), (0.6, 4.0)]:
        results.append(run_config("robustness", f"loss {int(loss*100)}% / range {rng}", "mock",
                                  8, 12, 40, 0.16, loss, rng, a.seeds))

    if a.with_claude:
        print("== Experiment 3: backend comparison (mock vs claude) ==")
        results.append(run_config("backend", "mock", "mock", 4, 6, 30, 0.12, 0.1, 7.0, a.seeds))
        try:
            claude = make_backend("claude")
            results.append(run_config("backend", "claude", "claude", 4, 6, 30, 0.12, 0.1, 7.0,
                                      a.seeds, shared_backend=claude))
            print(f"  claude tokens: in={claude.stats.input_tokens} out={claude.stats.output_tokens} "
                  f"cache hits={claude.stats.cache_hits}")
        except RuntimeError as e:
            print("  skipped:", e)

    path = os.path.join(a.out, "experiments.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(results)
    print(f"\nsaved {path}  ({time.time()-t0:.0f}s)")

    # ---- summary table
    print(f"\n{'experiment':<12}{'label':<24}{'succ':>6}{'coll':>6}{'near':>7}{'mkspan':>8}{'dist':>9}{'waits':>7}{'msgs':>8}{'llm':>7}")
    for r in results:
        print(f"{r['experiment']:<12}{r['label']:<24}{r['success_rate']:>6}{r['collisions']:>6}"
              f"{r['near_misses']:>7}{r['makespan']:>8}{r['total_distance']:>9}{r['wait_steps']:>7}"
              f"{r['messages_sent']:>8}{r['llm_calls']:>7}")

    # ---- chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, title in zip(axes, ["makespan", "near_misses", "llm_calls"],
                              ["Makespan (steps)", "Near misses per run", "LLM decisions per run"]):
        labels = [f"{r['experiment'][:5]}\n{r['label']}" for r in results]
        ax.bar(range(len(results)), [r[key] or 0 for r in results],
               color=["tab:blue" if r["experiment"] == "scalability" else
                      "tab:orange" if r["experiment"] == "robustness" else "tab:green" for r in results])
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"Distributed LLM-assisted multi-UAV planning — {a.seeds} seeds per config, "
                 f"total collisions = {sum(r['collisions'] for r in results)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "experiments.png"), dpi=130)
    print("chart:", os.path.join(a.out, "experiments.png"))


if __name__ == "__main__":
    main()
