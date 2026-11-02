"""
environment.py
==============

The *world* in which the UAVs fly.

We model the airspace as a 2-D occupancy grid (altitude is assumed to be
de-conflicted by a fixed cruise level, which is the standard simplification in
grid-based multi-UAV planning research). Each cell is either

* FREE      – the UAV may fly through it,
* OBSTACLE  – a building / terrain feature that can never be entered,
* NO_FLY    – a restricted zone (e.g. airport, military area). It is *not*
              physically blocking but entering it costs a heavy penalty, so the
              low-level planner avoids it unless there is no other way.

The environment also owns the list of *targets* (mission points that must be
visited by *some* UAV) and the random generator used to build scenarios, so
that every experiment is reproducible from a single seed.

Nothing in this file knows about LLMs — it is pure, deterministic geometry.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set, Tuple

Cell = Tuple[int, int]  # (x, y) grid coordinate

FREE, OBSTACLE, NO_FLY = 0, 1, 2

# 8-connected movement (diagonals allowed). Staying still is also a legal
# action (needed for "wait" behaviour during conflict resolution).
MOVES: List[Cell] = [
    (0, 0),
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
]


@dataclass
class Target:
    """A mission point that must be visited by exactly one UAV."""
    tid: int
    cell: Cell
    priority: int = 1              # higher = more important (used by LLM task allocation)
    claimed_by: Optional[int] = None
    visited_by: Optional[int] = None

    @property
    def is_done(self) -> bool:
        return self.visited_by is not None


@dataclass
class Environment:
    width: int
    height: int
    grid: List[List[int]] = field(default_factory=list)
    targets: List[Target] = field(default_factory=list)
    seed: int = 0
    name: str = "custom"

    # ------------------------------------------------------------------ build
    @classmethod
    def random(
        cls,
        width: int = 30,
        height: int = 30,
        n_targets: int = 6,
        obstacle_density: float = 0.12,
        n_no_fly: int = 2,
        seed: int = 0,
        n_blocks: int = 6,
    ) -> "Environment":
        """Procedurally generate a reproducible scenario.

        Obstacles are placed as rectangular 'buildings' (more realistic than
        single random cells, and they create the corridors / bottlenecks that
        make multi-agent coordination hard).
        """
        rng = random.Random(seed)
        env = cls(width, height, seed=seed, name=f"random-{seed}")
        env.grid = [[FREE for _ in range(width)] for _ in range(height)]

        # Rectangular buildings
        target_cells = int(width * height * obstacle_density)
        placed = 0
        attempts = 0
        while placed < target_cells and attempts < 500:
            attempts += 1
            w = rng.randint(2, max(2, width // 6))
            h = rng.randint(2, max(2, height // 6))
            x0 = rng.randint(1, width - w - 2)
            y0 = rng.randint(1, height - h - 2)
            for x in range(x0, x0 + w):
                for y in range(y0, y0 + h):
                    if env.grid[y][x] == FREE:
                        env.grid[y][x] = OBSTACLE
                        placed += 1

        # No-fly zones (soft obstacles)
        for _ in range(n_no_fly):
            w = rng.randint(3, max(3, width // 5))
            h = rng.randint(3, max(3, height // 5))
            x0 = rng.randint(0, width - w - 1)
            y0 = rng.randint(0, height - h - 1)
            for x in range(x0, x0 + w):
                for y in range(y0, y0 + h):
                    if env.grid[y][x] == FREE:
                        env.grid[y][x] = NO_FLY

        # Targets on free cells
        free = env.free_cells()
        rng.shuffle(free)
        for i in range(n_targets):
            env.targets.append(Target(tid=i, cell=free[i], priority=rng.randint(1, 3)))
        return env

    # --------------------------------------------------------------- queries
    def in_bounds(self, c: Cell) -> bool:
        x, y = c
        return 0 <= x < self.width and 0 <= y < self.height

    def cell_type(self, c: Cell) -> int:
        return self.grid[c[1]][c[0]]

    def is_free(self, c: Cell) -> bool:
        """Physically enterable (FREE or NO_FLY)."""
        return self.in_bounds(c) and self.cell_type(c) != OBSTACLE

    def traversal_cost(self, c: Cell) -> float:
        """Cost of *being* in a cell; the A* planner adds this to step cost."""
        return 8.0 if self.cell_type(c) == NO_FLY else 0.0

    def free_cells(self) -> List[Cell]:
        return [(x, y) for y in range(self.height) for x in range(self.width)
                if self.grid[y][x] == FREE]

    def neighbors(self, c: Cell) -> Iterable[Cell]:
        """Legal successor cells (8-connected, no corner cutting through obstacles)."""
        x, y = c
        for dx, dy in MOVES:
            n = (x + dx, y + dy)
            if not self.is_free(n):
                continue
            # Forbid squeezing diagonally between two obstacles.
            if dx != 0 and dy != 0:
                if not self.is_free((x + dx, y)) or not self.is_free((x, y + dy)):
                    continue
            yield n

    def pending_targets(self) -> List[Target]:
        return [t for t in self.targets if not t.is_done]

    def all_targets_done(self) -> bool:
        return all(t.is_done for t in self.targets)

    # ---------------------------------------------------------- description
    def ascii(self, uav_positions: Optional[dict] = None) -> str:
        """Human readable map — also what we send to the LLM (compact text)."""
        uav_positions = uav_positions or {}
        pos_to_uav = {v: k for k, v in uav_positions.items()}
        tgt = {t.cell: t for t in self.targets}
        rows = []
        for y in range(self.height - 1, -1, -1):
            row = []
            for x in range(self.width):
                c = (x, y)
                if c in pos_to_uav:
                    row.append(str(pos_to_uav[c] % 10))
                elif c in tgt:
                    row.append("x" if tgt[c].is_done else "T")
                elif self.grid[y][x] == OBSTACLE:
                    row.append("#")
                elif self.grid[y][x] == NO_FLY:
                    row.append("~")
                else:
                    row.append(".")
            rows.append("".join(row))
        return "\n".join(rows)


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def chebyshev(a: Cell, b: Cell) -> int:
    """Admissible heuristic for 8-connected grids with unit diagonal cost."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def euclid(a: Cell, b: Cell) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
