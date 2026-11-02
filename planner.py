"""
planner.py
==========

The **low-level, geometric planner**: space-time A*.

Research background
-------------------
The literature is clear that LLMs alone are poor at the *geometric* part of
multi-agent path finding (they mis-count cells, forget obstacles, and produce
paths 3x longer than optimal — Chen et al., "Why Solving MAPF with LLMs has not
Succeeded Yet", 2024). The successful pattern is *hierarchical*: let the LLM do
semantic / strategic reasoning and let a classical search algorithm guarantee
collision-free, near-optimal geometry (LLM-A*, LLMDR, SkyAgent, ...).

This module is therefore deliberately **LLM-free**. It receives

* the static environment,
* a start cell and goal cell,
* a *reservation table* — the (cell, time) pairs that higher-priority
  neighbours have already announced they will occupy,
* optionally a list of *waypoints* suggested by the LLM (LLM-A* style
  "target states" that bias the search),

and returns a time-indexed path ``[c_0, c_1, ..., c_T]`` where ``c_t`` is the
cell occupied at time step ``t``. Waiting in place is a legal move.

Reservation semantics
---------------------
``reserved`` is a set of ``(x, y, t)`` tuples. A move ``a -> b`` from time
``t`` to ``t+1`` is illegal if

1. ``(b, t+1)`` is reserved           (vertex conflict), or
2. ``(b, t)`` and ``(a, t+1)`` are reserved (edge / swap conflict), or
3. ``b`` is within ``safety_radius`` (Chebyshev) of any reserved cell at ``t+1``
   — this is what gives the UAVs a physical separation distance rather than
   mere "not the same cell".

Beyond the horizon of a neighbour's announced path we assume it *stays at its
last cell* (a common conservative assumption in decentralised MAPF).
"""

from __future__ import annotations

import heapq
import itertools
from typing import Dict, List, Optional, Set, Tuple

from .environment import Cell, Environment, chebyshev

Reservation = Set[Tuple[int, int, int]]  # (x, y, t)

SQRT2 = 2 ** 0.5


def build_reservations(
    neighbour_paths: Dict[int, List[Cell]],
    horizon: int,
    safety_radius: int = 1,
) -> Reservation:
    """Turn the announced paths of neighbours into a reservation set.

    Each neighbour reserves its cell and every cell within ``safety_radius`` at
    every time step up to ``horizon``; after its path ends it is assumed to
    hover at its final cell.
    """
    res: Reservation = set()
    for path in neighbour_paths.values():
        if not path:
            continue
        for t in range(horizon + 1):
            c = path[min(t, len(path) - 1)]
            for dx in range(-safety_radius, safety_radius + 1):
                for dy in range(-safety_radius, safety_radius + 1):
                    res.add((c[0] + dx, c[1] + dy, t))
    return res


def space_time_astar(
    env: Environment,
    start: Cell,
    goal: Cell,
    reserved: Optional[Reservation] = None,
    waypoints: Optional[List[Cell]] = None,
    horizon: int = 200,
    max_expansions: int = 60_000,
    waypoint_bonus: float = 0.0,
) -> Optional[List[Cell]]:
    """Space-time A* from ``start`` to ``goal`` honouring ``reserved``.

    ``waypoints`` (optional) are LLM-suggested intermediate cells. We implement
    the LLM-A* idea in its simplest, provably-safe form: the search is run
    *segment by segment* through the waypoints. Because every segment is still
    a complete A* search, obstacle avoidance and reservation safety are never
    compromised by a bad LLM suggestion — a bad waypoint only costs path length.
    """
    reserved = reserved or set()
    waypoints = [w for w in (waypoints or []) if env.is_free(w)]

    full_path: List[Cell] = [start]
    t0 = 0
    for sub_goal in waypoints + [goal]:
        seg = _astar_segment(env, full_path[-1], sub_goal, reserved, t0, horizon, max_expansions)
        if seg is None:
            # If a waypoint is unreachable, skip it instead of failing outright.
            if sub_goal != goal:
                continue
            return None
        full_path.extend(seg[1:])
        t0 = len(full_path) - 1
    return full_path


def _astar_segment(
    env: Environment,
    start: Cell,
    goal: Cell,
    reserved: Reservation,
    t_start: int,
    horizon: int,
    max_expansions: int,
) -> Optional[List[Cell]]:
    """Plain space-time A* between two cells starting at time ``t_start``."""
    if start == goal:
        return [start]

    counter = itertools.count()
    open_heap: List[Tuple[float, int, int, Cell]] = []  # (f, tie, t, cell)
    g: Dict[Tuple[Cell, int], float] = {(start, t_start): 0.0}
    parent: Dict[Tuple[Cell, int], Optional[Tuple[Cell, int]]] = {(start, t_start): None}
    heapq.heappush(open_heap, (chebyshev(start, goal), next(counter), t_start, start))
    expansions = 0

    while open_heap:
        f, _, t, cell = heapq.heappop(open_heap)
        expansions += 1
        if expansions > max_expansions:
            return None
        if cell == goal:
            return _reconstruct(parent, (cell, t))
        if t >= horizon:
            continue
        cur_g = g[(cell, t)]
        for nxt in env.neighbors(cell):
            nt = t + 1
            if (nxt[0], nxt[1], nt) in reserved:
                continue
            # swap conflict: someone moves nxt -> cell while we move cell -> nxt
            if (nxt[0], nxt[1], t) in reserved and (cell[0], cell[1], nt) in reserved:
                continue
            step = 0.0 if nxt == cell else (SQRT2 if nxt[0] != cell[0] and nxt[1] != cell[1] else 1.0)
            # Waiting is cheap but not free so the UAV doesn't dawdle.
            if nxt == cell:
                step = 0.5
            ng = cur_g + step + env.traversal_cost(nxt)
            key = (nxt, nt)
            if ng < g.get(key, float("inf")):
                g[key] = ng
                parent[key] = (cell, t)
                heapq.heappush(open_heap, (ng + chebyshev(nxt, goal), next(counter), nt, nxt))
    return None


def _reconstruct(parent, node) -> List[Cell]:
    path = []
    while node is not None:
        path.append(node[0])
        node = parent[node]
    path.reverse()
    return path


def path_length(path: List[Cell]) -> float:
    """Euclidean length of a grid path (diagonals count sqrt(2))."""
    total = 0.0
    for a, b in zip(path, path[1:]):
        if a == b:
            continue
        total += SQRT2 if (a[0] != b[0] and a[1] != b[1]) else 1.0
    return total
