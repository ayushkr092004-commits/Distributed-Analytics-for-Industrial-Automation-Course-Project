"""
simulation.py
=============

The **simulator** advances discrete time, delivers radio messages, feeds each
UAV its on-board sensor readings, and records metrics. It does *not* plan for
anyone — that is the whole point of a distributed system. Think of it as the
physics engine plus the radio channel plus the referee.

Metrics recorded
----------------
* ``targets_done``        – how many mission points were serviced
* ``makespan``            – steps until the last target was reached
* ``total_distance``      – sum of distances flown by all UAVs
* ``collisions``          – two UAVs in the same cell, or swapping cells
* ``near_misses``         – two UAVs within SAFETY_RADIUS (adjacent cells)
* ``messages_*``          – radio statistics (sent / delivered / dropped)
* ``llm_calls``           – number of high-level decisions requested
* ``wait_steps``          – total hover steps (coordination overhead)
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .agent import SAFETY_RADIUS, SENSE_RANGE, UAVAgent
from .comms import MessageBus
from .environment import Cell, Environment, chebyshev, euclid
from .llm.backend import LLMBackend, make_backend


@dataclass
class Metrics:
    steps: int = 0
    targets_done: int = 0
    targets_total: int = 0
    makespan: Optional[int] = None
    total_distance: float = 0.0
    collisions: int = 0
    near_misses: int = 0
    wait_steps: int = 0
    messages_sent: int = 0
    messages_delivered: int = 0
    messages_dropped: int = 0
    llm_calls: int = 0
    llm_failures: int = 0
    llm_cache_hits: int = 0
    wall_time_s: float = 0.0
    success: bool = False

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Simulation:
    env: Environment
    n_uavs: int = 4
    backend: str = "mock"
    comm_range: float = 8.0
    loss_prob: float = 0.0
    max_steps: int = 300
    seed: int = 0
    verbose: bool = False
    shared_backend: Optional[LLMBackend] = None   # one backend object is fine: it is stateless per call

    agents: List[UAVAgent] = field(default_factory=list)
    bus: MessageBus = field(init=False)
    trajectory: Dict[int, List[Cell]] = field(default_factory=dict)  # for visualisation
    metrics: Metrics = field(default_factory=Metrics)
    collision_events: List[dict] = field(default_factory=list)

    def __post_init__(self):
        rng = random.Random(self.seed + 7)
        self.bus = MessageBus(comm_range=self.comm_range, loss_prob=self.loss_prob, seed=self.seed)
        llm = self.shared_backend or make_backend(self.backend)
        free = self.env.free_cells()
        tcells = {t.cell for t in self.env.targets}
        rng.shuffle(free)
        starts: List[Cell] = []
        for c in free:
            if c in tcells or any(chebyshev(c, s) <= 2 for s in starts):
                continue
            starts.append(c)
            if len(starts) == self.n_uavs:
                break
        for uid, c in enumerate(starts):
            self.agents.append(UAVAgent(uid=uid, env=self.env, llm=llm, position=c,
                                        battery=self.max_steps + 50))
            self.trajectory[uid] = [c]
        self.metrics.targets_total = len(self.env.targets)
        self.llm = llm

    # ------------------------------------------------------------------
    def positions(self) -> Dict[int, Cell]:
        return {a.uid: a.position for a in self.agents if a.alive}

    def run(self) -> Metrics:
        t0 = time.time()
        inboxes: Dict[int, list] = {a.uid: [] for a in self.agents}
        for t in range(self.max_steps):
            self.metrics.steps = t + 1
            positions = self.positions()
            prev = dict(positions)

            # 1. each agent acts using ONLY its inbox + short-range sensor
            outgoing = []
            for a in self.agents:
                a.receive(inboxes.get(a.uid, []))
                sensed = {u: p for u, p in positions.items()
                          if u != a.uid and euclid(p, a.position) <= SENSE_RANGE}
                outgoing.extend(a.step(t, sensed))

            # 2. radio channel
            for m in outgoing:
                self.bus.send(m)
            inboxes = self.bus.deliver(self.positions())

            # 3. referee: physics / collisions / metrics
            self._check_collisions(t, prev)
            for a in self.agents:
                self.trajectory[a.uid].append(a.position)

            done = sum(1 for tg in self.env.targets if tg.is_done)
            if done > self.metrics.targets_done:
                self.metrics.targets_done = done
                self.metrics.makespan = t + 1
            if self.verbose and t % 10 == 0:
                print(f"[t={t:3d}] done {done}/{len(self.env.targets)}  "
                      f"pos={[a.position for a in self.agents]}")
            if self.env.all_targets_done():
                break

        self._finalise(time.time() - t0)
        return self.metrics

    # ------------------------------------------------------------------
    def _check_collisions(self, t: int, prev: Dict[int, Cell]) -> None:
        agents = [a for a in self.agents if a.alive]
        for i, a in enumerate(agents):
            for b in agents[i + 1:]:
                same = a.position == b.position
                swap = (a.position == prev[b.uid] and b.position == prev[a.uid]
                        and a.position != prev[a.uid])
                if same or swap:
                    self.metrics.collisions += 1
                    self.collision_events.append({"t": t, "uavs": [a.uid, b.uid], "cell": a.position})
                elif chebyshev(a.position, b.position) <= SAFETY_RADIUS:
                    self.metrics.near_misses += 1

    def _finalise(self, wall: float) -> None:
        m = self.metrics
        m.total_distance = round(sum(a.distance_flown for a in self.agents), 2)
        m.wait_steps = sum(a.waited_total for a in self.agents)
        m.messages_sent = self.bus.stats.sent
        m.messages_delivered = self.bus.stats.delivered
        m.messages_dropped = self.bus.stats.dropped
        m.llm_calls = self.llm.stats.calls
        m.llm_failures = self.llm.stats.failures
        m.llm_cache_hits = self.llm.stats.cache_hits
        m.wall_time_s = round(wall, 2)
        m.success = self.env.all_targets_done() and m.collisions == 0

    # ------------------------------------------------------------------
    def explain(self, max_events: int = 60) -> str:
        """Human-readable narrative of what each UAV decided and why."""
        lines = [f"=== Mission narrative ({self.env.name}, backend={self.llm.name}) ==="]
        for a in self.agents:
            lines.append(f"\nUAV {a.uid}: start {self.trajectory[a.uid][0]}, visited {a.visited_targets}, "
                         f"flew {a.distance_flown:.1f} cells, waited {a.waited_total} steps")
            for ev in a.log.events[:max_events]:
                extra = {k: v for k, v in ev.items() if k not in ("t", "event")}
                lines.append(f"  t={ev['t']:3d}  {ev['event']:<18} {extra}")
        return "\n".join(lines)
