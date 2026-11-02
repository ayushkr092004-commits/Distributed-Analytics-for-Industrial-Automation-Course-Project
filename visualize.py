"""
visualize.py
============

Plots for understanding and presenting a mission.

* ``plot_mission``  – static PNG: map, obstacles, no-fly zones, targets and
                      the full trajectory of every UAV (start = circle,
                      end = star). Collision / near-miss sites are marked.
* ``animate_mission`` – GIF: the UAVs moving step by step with their
                        communication links drawn when in radio range.

Both take a finished ``Simulation`` object so they never influence planning.
"""

from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402

from .environment import NO_FLY, OBSTACLE, euclid  # noqa: E402

COLORS = plt.cm.tab20(np.linspace(0, 1, 20))


def _draw_map(ax, env):
    grid = np.array(env.grid)
    img = np.ones((env.height, env.width, 3))
    img[grid == OBSTACLE] = (0.25, 0.25, 0.25)
    img[grid == NO_FLY] = (1.0, 0.85, 0.85)
    ax.imshow(img, origin="lower", extent=(-0.5, env.width - 0.5, -0.5, env.height - 0.5))
    for t in env.targets:
        ax.plot(*t.cell, marker="X", color="green" if t.is_done else "red", ms=11, mec="k", zorder=5)
        ax.annotate(f"T{t.tid}(p{t.priority})", t.cell, xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_mission(sim, path: str, title: Optional[str] = None) -> str:
    fig, ax = plt.subplots(figsize=(9, 9))
    _draw_map(ax, sim.env)
    for uid, traj in sim.trajectory.items():
        xs, ys = zip(*traj)
        c = COLORS[uid % 20]
        ax.plot(xs, ys, "-", color=c, lw=2, alpha=0.9, label=f"UAV {uid}")
        ax.plot(xs[0], ys[0], "o", color=c, ms=9, mec="k", zorder=6)
        ax.plot(xs[-1], ys[-1], "*", color=c, ms=15, mec="k", zorder=6)
    for ev in sim.collision_events:
        ax.plot(*ev["cell"], "s", color="none", mec="red", ms=18, mew=2, zorder=7)
    m = sim.metrics
    ax.set_title(title or (f"{sim.env.name} | {len(sim.agents)} UAVs | backend={sim.llm.name} | "
                           f"targets {m.targets_done}/{m.targets_total} | steps {m.steps} | "
                           f"collisions {m.collisions} | LLM calls {m.llm_calls}"), fontsize=10)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def animate_mission(sim, path: str, fps: int = 6) -> str:
    env = sim.env
    n_steps = max(len(t) for t in sim.trajectory.values())
    fig, ax = plt.subplots(figsize=(8, 8))

    def frame(i):
        ax.clear()
        _draw_map(ax, env)
        pos = {}
        for uid, traj in sim.trajectory.items():
            k = min(i, len(traj) - 1)
            pos[uid] = traj[k]
            c = COLORS[uid % 20]
            xs, ys = zip(*traj[: k + 1])
            ax.plot(xs, ys, "-", color=c, lw=1.5, alpha=0.6)
            ax.plot(xs[-1], ys[-1], "o", color=c, ms=11, mec="k", zorder=6)
            ax.annotate(str(uid), traj[k], ha="center", va="center", fontsize=7, zorder=7)
        # communication links
        ids = list(pos)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                if euclid(pos[ids[a]], pos[ids[b]]) <= sim.comm_range:
                    ax.plot([pos[ids[a]][0], pos[ids[b]][0]], [pos[ids[a]][1], pos[ids[b]][1]],
                            ":", color="blue", lw=0.8, alpha=0.5)
        ax.set_title(f"t = {i}   (dotted = radio link, range {sim.comm_range})", fontsize=10)

    anim = animation.FuncAnimation(fig, frame, frames=n_steps, interval=1000 // fps)
    anim.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    return path
