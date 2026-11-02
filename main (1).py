#!/usr/bin/env python3
"""
main.py — run one distributed multi-UAV mission from the command line.

Examples
--------
# fully offline, deterministic (rule-based stand-in for the LLM)
python main.py --uavs 6 --targets 10 --backend mock --seed 3 --plot --gif

# real LLM: each UAV's high-level decisions are made by Claude
export ANTHROPIC_API_KEY=sk-ant-...
python main.py --uavs 4 --targets 6 --backend claude --seed 1 --plot --explain

# harsh conditions: short radio range and 30 % packet loss
python main.py --uavs 10 --targets 14 --comm-range 5 --loss 0.3 --plot
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from uavplan.environment import Environment
from uavplan.simulation import Simulation


def parse():
    p = argparse.ArgumentParser(description="Distributed LLM-assisted multi-UAV path planning")
    p.add_argument("--uavs", type=int, default=4)
    p.add_argument("--targets", type=int, default=6)
    p.add_argument("--size", type=int, default=30, help="map width = height")
    p.add_argument("--density", type=float, default=0.12, help="obstacle density")
    p.add_argument("--no-fly", type=int, default=2, help="number of no-fly zones")
    p.add_argument("--backend", choices=["mock", "claude"], default="mock")
    p.add_argument("--comm-range", type=float, default=8.0)
    p.add_argument("--loss", type=float, default=0.0, help="packet loss probability")
    p.add_argument("--steps", type=int, default=300, help="max simulation steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    p.add_argument("--plot", action="store_true", help="save trajectory PNG")
    p.add_argument("--gif", action="store_true", help="save animated GIF")
    p.add_argument("--explain", action="store_true", help="print per-UAV decision narrative")
    p.add_argument("--transcript", action="store_true", help="save all LLM prompts/answers to JSON")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    env = Environment.random(a.size, a.size, n_targets=a.targets, obstacle_density=a.density,
                             n_no_fly=a.no_fly, seed=a.seed)
    print(env.ascii())
    print(f"\n{a.uavs} UAVs, {a.targets} targets, backend={a.backend}, comm range={a.comm_range}, loss={a.loss}\n")
    try:
        sim = Simulation(env, n_uavs=a.uavs, backend=a.backend, comm_range=a.comm_range,
                         loss_prob=a.loss, max_steps=a.steps, seed=a.seed, verbose=a.verbose)
    except RuntimeError as e:
        sys.exit(f"error: {e}")
    m = sim.run()

    print("\n=== Metrics ===")
    print(json.dumps(m.as_dict(), indent=2))
    tag = f"{a.backend}_u{a.uavs}_t{a.targets}_s{a.seed}"
    with open(os.path.join(a.out, f"metrics_{tag}.json"), "w") as f:
        json.dump(m.as_dict(), f, indent=2)

    if a.explain:
        print()
        print(sim.explain())
    if a.transcript:
        with open(os.path.join(a.out, f"llm_transcript_{tag}.json"), "w") as f:
            json.dump(sim.llm.stats.transcript, f, indent=2)
        print(f"LLM transcript saved ({len(sim.llm.stats.transcript)} decisions)")
    if a.plot or a.gif:
        from uavplan.visualize import animate_mission, plot_mission
        if a.plot:
            print("plot:", plot_mission(sim, os.path.join(a.out, f"mission_{tag}.png")))
        if a.gif:
            print("gif: ", animate_mission(sim, os.path.join(a.out, f"mission_{tag}.gif")))


if __name__ == "__main__":
    main()
