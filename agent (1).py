"""
agent.py
========

The **UAV agent** — one instance per aircraft, each running the full planning
stack independently. This is the heart of the *distributed* design: there is no
object anywhere in the code base that plans for more than one UAV.

Per-step decision cycle (``UAVAgent.step``)
--------------------------------------------
::

    inbox ──► 1. update_beliefs()      merge neighbour PLAN/CLAIM/DONE msgs
              2. allocate_task()       [LLM]  pick / confirm a mission target
              3. plan_path()           [A*]   space-time path honouring higher-
                                              priority neighbours' reservations
              4. detect_conflicts()           lookahead against ALL known plans
              5. negotiate()           [LLM]  proceed / yield / reroute
              6. check_deadlock()      [LLM]  escape if no progress for K steps
              7. sense_and_avoid()            last-resort local safety guard
              8. execute_move()               advance one cell, consume battery
              9. broadcast()                  event-triggered PLAN / CLAIM / DONE
    outbox ◄──

Why this split?
---------------
* Steps 2, 5, 6 are *semantic* decisions where an LLM adds value (reasoning
  about priorities, intentions, and messages in natural language).
* Steps 3, 4, 7 are *geometric* and are done by exact algorithms, because
  research shows LLMs are unreliable at cell-level spatial reasoning.
* Every LLM answer passes through a validator (``_validate_*``); anything
  infeasible is replaced by a safe default. The LLM can therefore make the
  system smarter but never less safe.

Priority rule (used both by the planner and inside the LLM prompt so that
independently reasoning UAVs stay compatible):

    fewer remaining steps  >  more steps already waited  >  lower id
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from .comms import Message
from .environment import Cell, Environment, chebyshev, euclid
from .llm.backend import LLMBackend
from .planner import build_reservations, space_time_astar

PLAN_HORIZON = 25         # how many future steps we announce / reserve
CONFLICT_LOOKAHEAD = 6    # steps ahead we check for conflicts
SAFETY_RADIUS = 1         # Chebyshev separation (cells) required between UAVs
DEADLOCK_WINDOW = 8       # steps without progress before deadlock handling
HEARTBEAT = 5             # re-broadcast plan at least this often
SENSE_RANGE = 3.0         # on-board sensor range; must exceed 2*sqrt(2) so any two UAVs
                          # that could meet in one step are mutually visible


@dataclass
class NeighbourBelief:
    """What this UAV currently believes about one neighbour."""
    uid: int
    position: Cell
    path: List[Cell]
    target: Optional[int]
    remaining: int
    waited: int
    t_heard: int

    def offset(self, t_now: int) -> int:
        """How many steps the neighbour has advanced since it announced the plan.
        A PLAN sent at the end of step t_heard-1 has path[0] = position at the
        start of step t_heard."""
        return max(0, t_now - self.t_heard)

    def cell_at(self, t_now: int, k: int) -> Cell:
        """Predicted cell k steps ahead of *now* (hover at the end of the path)."""
        if not self.path:
            return self.position
        return self.path[min(len(self.path) - 1, self.offset(t_now) + k)]

    def shifted_path(self, t_now: int) -> List[Cell]:
        """Path re-indexed so that element 0 is the neighbour's current cell."""
        if not self.path:
            return [self.position]
        off = min(self.offset(t_now), len(self.path) - 1)
        return self.path[off:]

    def remaining_now(self, t_now: int) -> int:
        return max(0, self.remaining - self.offset(t_now))


@dataclass
class AgentLog:
    """Everything the agent decided, for explainability / debugging."""
    events: List[dict] = field(default_factory=list)

    def add(self, t: int, kind: str, **kw):
        self.events.append({"t": t, "event": kind, **kw})


@dataclass
class UAVAgent:
    uid: int
    env: Environment
    llm: LLMBackend
    position: Cell
    battery: int = 400
    sensing_radius: int = 6

    # -- planning state ----------------------------------------------------
    target: Optional[int] = None
    path: List[Cell] = field(default_factory=list)      # path[0] == position
    wait_steps: int = 0
    waited_total: int = 0
    plan_dirty: bool = True                              # event-trigger flag
    last_broadcast: int = -100
    _replan_around: Optional[int] = None                 # set by the sense-and-avoid guard

    # -- beliefs -------------------------------------------------------------
    neighbours: Dict[int, NeighbourBelief] = field(default_factory=dict)
    known_claims: Dict[int, int] = field(default_factory=dict)   # tid -> uid
    known_done: set = field(default_factory=set)
    inbox: List[Message] = field(default_factory=list)
    outbox: List[Message] = field(default_factory=list)

    # -- bookkeeping ----------------------------------------------------------
    history: Deque[Cell] = field(default_factory=lambda: deque(maxlen=DEADLOCK_WINDOW + 2))
    dist_history: Deque[int] = field(default_factory=lambda: deque(maxlen=DEADLOCK_WINDOW + 2))
    visited_targets: List[int] = field(default_factory=list)
    distance_flown: float = 0.0
    log: AgentLog = field(default_factory=AgentLog)
    alive: bool = True

    # ====================================================================
    # Public API used by the simulator
    # ====================================================================
    def receive(self, msgs: List[Message]) -> None:
        self.inbox.extend(msgs)

    def step(self, t: int, sensed_positions: Dict[int, Cell]) -> List[Message]:
        """Run one full decision cycle. ``sensed_positions`` are the true
        positions of UAVs inside the short on-board *sensor* range (not the
        radio range) — this is the only non-message information an agent gets,
        and it mirrors a real sense-and-avoid sensor."""
        self.outbox = []
        if not self.alive or self.battery <= 0:
            self.alive = False
            return []

        self._update_beliefs(t)
        self._allocate_task(t)
        if self.target is not None:
            self._plan_path(t)
        if self.target is not None:          # target may be dropped by a failed replan
            self._negotiate_conflicts(t)
        if self.target is not None:
            self._check_deadlock(t)
        self._sense_and_avoid(t, sensed_positions)
        self._execute_move(t)
        self._broadcast(t)
        self.inbox = []
        return self.outbox

    # ====================================================================
    # 1. Beliefs
    # ====================================================================
    def _update_beliefs(self, t: int) -> None:
        for m in self.inbox:
            p = m.payload
            if m.mtype == "PLAN":
                self.neighbours[m.sender] = NeighbourBelief(
                    uid=m.sender, position=tuple(p["position"]), path=[tuple(c) for c in p["path"]],
                    target=p.get("target"), remaining=p.get("remaining", 999),
                    waited=p.get("waited", 0), t_heard=t)
                if p.get("target") is not None:
                    self.known_claims[p["target"]] = m.sender
            elif m.mtype == "CLAIM":
                tid, uid = p["target"], m.sender
                self.known_claims[tid] = uid
                # Conflict on the same target: the one with fewer remaining
                # steps keeps it (same tie-break as everywhere else).
                if tid == self.target and uid != self.uid:
                    # Both sides use the same estimate (straight-line distance)
                    # and the same tie-break, so exactly one of them backs off.
                    mine = (self._claim_cost(tid), self.uid)
                    theirs = (p.get("dist", 999), uid)
                    if theirs < mine:
                        self.log.add(t, "release_target", target=tid, reason=f"UAV {uid} closer")
                        self._set_target(None)
            elif m.mtype == "RELEASE":
                if self.known_claims.get(p["target"]) == m.sender:
                    self.known_claims.pop(p["target"], None)
            elif m.mtype == "DONE":
                self.known_done.add(p["target"])
                self.known_claims.pop(p["target"], None)
                if p["target"] == self.target:
                    self._set_target(None)
        # Forget neighbours we have not heard from for a long time.
        stale = [u for u, nb in self.neighbours.items() if t - nb.t_heard > 3 * HEARTBEAT]
        for u in stale:
            self.neighbours.pop(u)

    # ====================================================================
    # 2. Task allocation  [LLM]
    # ====================================================================
    def _allocate_task(self, t: int) -> None:
        if self.target is not None:
            tgt = self.env.targets[self.target]
            if tgt.is_done or self.target in self.known_done:
                self._set_target(None)
            else:
                return
        candidates = [tg for tg in self.env.targets
                      if not tg.is_done and tg.tid not in self.known_done]
        if not candidates:
            return
        ctx = self._base_ctx(t)
        ctx["targets"] = [{"tid": tg.tid, "cell": list(tg.cell), "priority": tg.priority,
                           "claimed_by": self.known_claims.get(tg.tid)} for tg in candidates]
        ctx["battery"] = self.battery
        ans = self.llm.decide_task(ctx)
        tid = self._validate_task(ans, candidates)
        if tid is None:
            self.log.add(t, "no_task", llm=ans)
            return
        self._set_target(tid)
        self.known_claims[tid] = self.uid
        self.outbox.append(Message(self.uid, "CLAIM", {"target": tid,
                                                       "dist": self._claim_cost(tid)}, t))
        self.log.add(t, "claim_target", target=tid, reason=ans.get("reason"))

    def _validate_task(self, ans: dict, candidates) -> Optional[int]:
        """Feasibility projection: accept the LLM's choice only if it is a real,
        unfinished target; otherwise fall back to nearest unclaimed."""
        valid = {tg.tid for tg in candidates}
        tid = ans.get("target") if isinstance(ans, dict) else None
        if isinstance(tid, int) and tid in valid:
            return tid
        free = [tg for tg in candidates if self.known_claims.get(tg.tid) in (None, self.uid)]
        pool = free or list(candidates)
        return min(pool, key=lambda tg: chebyshev(tg.cell, self.position)).tid

    # ====================================================================
    # 3. Low-level planning  [A*]
    # ====================================================================
    def _plan_path(self, t: int, waypoint: Optional[Cell] = None, force: bool = False,
                   around: Optional[int] = None) -> None:
        """(Re)plan if needed. ``around`` forces the given neighbour to be
        treated as higher priority (we route around it)."""
        if self._replan_around is not None and around is None:
            around, self._replan_around = self._replan_around, None
            force = True
        goal = self.env.targets[self.target].cell
        need = force or not self.path or self.path[-1] != goal or self.path[0] != self.position
        if not need:
            return
        higher = {nb.uid: nb.shifted_path(t) for nb in self.neighbours.values()
                  if nb.uid == around or self._has_priority_over_me(nb, t)}
        reserved = build_reservations(higher, PLAN_HORIZON, SAFETY_RADIUS)
        new = space_time_astar(self.env, self.position, goal, reserved,
                               waypoints=[waypoint] if waypoint else None, horizon=PLAN_HORIZON * 4)
        if new is None:
            # Ignore reservations as last resort (sense-and-avoid still protects us)
            new = space_time_astar(self.env, self.position, goal, set(), horizon=PLAN_HORIZON * 4)
        if new is None:
            self.log.add(t, "unreachable", target=self.target)
            self._set_target(None)
            return
        self.path = new
        self.plan_dirty = True
        self.log.add(t, "replan", target=self.target, length=len(new) - 1,
                     avoided=list(higher.keys()), waypoint=waypoint)

    def _has_priority_over_me(self, nb: NeighbourBelief, t: int) -> bool:
        mine = (self._eta(), -self.waited_total, self.uid)
        theirs = (nb.remaining_now(t), -nb.waited, nb.uid)
        return theirs < mine

    # ====================================================================
    # 4 + 5. Conflict detection and negotiation  [LLM]
    # ====================================================================
    def _first_conflict(self, nb: NeighbourBelief, t: int) -> Optional[Tuple[int, Cell]]:
        """Return (steps_ahead, cell) of the first safety violation with nb."""
        horizon = min(CONFLICT_LOOKAHEAD, len(self.path) - 1)
        for k in range(1, horizon + 1):
            mine = self.path[k]
            theirs = nb.cell_at(t, k)
            if chebyshev(mine, theirs) <= SAFETY_RADIUS:
                return k, mine
        return None

    def _negotiate_conflicts(self, t: int) -> None:
        if self.wait_steps > 0:
            return
        for nb in list(self.neighbours.values()):
            if nb.uid not in self.neighbours:
                continue
            conflict = self._first_conflict(nb, t)
            if conflict is None:
                continue
            k, cell = conflict
            proposals = [m.to_dict() for m in self.inbox
                         if m.mtype == "PROPOSAL" and m.sender == nb.uid]
            ctx = self._base_ctx(t)
            ctx.update({
                "me": {"id": self.uid, "position": list(self.position), "target": self.target,
                       "remaining": self._eta(), "waited": self.waited_total},
                "other": {"id": nb.uid, "position": list(nb.cell_at(t, 0)), "target": nb.target,
                          "remaining": nb.remaining_now(t), "waited": nb.waited},
                "conflict": {"in_steps": k, "cell": list(cell)},
                "messages": proposals,
            })
            ans = self.llm.negotiate(ctx)
            action, wait, wp = self._validate_negotiation(ans, nb, proposals, t)
            self.log.add(t, "negotiate", with_uav=nb.uid, in_steps=k, action=action,
                         llm=ans.get("action"), reason=ans.get("reason"))
            self.outbox.append(Message(self.uid, "PROPOSAL", {"with": nb.uid, "action": action,
                                                              "wait_steps": wait}, t,
                                       broadcast=False, recipient=nb.uid))
            if action == "yield":
                self.wait_steps = wait
                self.path = [self.position] * (wait + 1) + self.path[1:]
                self.plan_dirty = True
            elif action == "reroute":
                self._plan_path(t, waypoint=wp, force=True, around=nb.uid)
            # "proceed": nothing to do.
            return  # one negotiation per step keeps behaviour stable

    def _validate_negotiation(self, ans: dict, nb: NeighbourBelief, proposals: List[dict], t: int):
        """Safety net around the LLM's answer.

        The LLM may be creative *within* the rules, but the deterministic
        tie-break is enforced when two UAVs would both 'proceed' — otherwise a
        single hallucination could cause a collision."""
        action = ans.get("action") if isinstance(ans, dict) else None
        wait = ans.get("wait_steps", 2) if isinstance(ans, dict) else 2
        wp = ans.get("waypoint") if isinstance(ans, dict) else None
        other_yielded = any(p.get("action") == "yield" for p in proposals)
        i_have_priority = not self._has_priority_over_me(nb, t)
        if action not in ("proceed", "yield", "reroute"):
            action = "proceed" if i_have_priority else "yield"
        if action == "proceed" and not i_have_priority and not other_yielded:
            action = "yield"  # rule enforcement
        if not isinstance(wait, int) or not (1 <= wait <= 4):
            wait = 2
        if action == "reroute":
            if not (isinstance(wp, list) and len(wp) == 2 and all(isinstance(v, int) for v in wp)
                    and self.env.is_free(tuple(wp))):
                wp = None
            else:
                wp = tuple(wp)
        return action, wait, wp

    # ====================================================================
    # 6. Deadlock handling  [LLM]
    # ====================================================================
    def _check_deadlock(self, t: int) -> None:
        goal = self.env.targets[self.target].cell
        d = chebyshev(self.position, goal)
        self.dist_history.append(d)
        if len(self.dist_history) < DEADLOCK_WINDOW:
            return
        if min(self.dist_history) < self.dist_history[0]:
            return  # some progress was made
        ctx = self._base_ctx(t)
        ctx.update({"target": self.target, "goal": list(goal), "stuck_steps": len(self.dist_history),
                    "recent_cells": [list(c) for c in self.history]})
        ans = self.llm.resolve_deadlock(ctx)
        action = ans.get("action") if isinstance(ans, dict) else None
        wp = ans.get("waypoint") if isinstance(ans, dict) else None
        self.log.add(t, "deadlock", action=action, reason=ans.get("reason"))
        self.dist_history.clear()
        if action == "release" or (isinstance(ans, dict) and ans.get("release_target")):
            old = self.target
            self._set_target(None)
            self.outbox.append(Message(self.uid, "RELEASE", {"target": old}, t))
            return
        if action == "wait":
            self.wait_steps = 2
            self.path = [self.position] * 3 + self.path[1:]
            return
        # reroute (default)
        if not (isinstance(wp, list) and len(wp) == 2 and self.env.is_free(tuple(wp))):
            wp = self._heuristic_escape(goal)
        else:
            wp = tuple(wp)
        self._plan_path(t, waypoint=wp, force=True)

    def _heuristic_escape(self, goal: Cell) -> Optional[Cell]:
        """Pick the free cell 3-4 steps away that is farthest from all neighbours."""
        best, best_score = None, -1.0
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                c = (self.position[0] + dx, self.position[1] + dy)
                if not self.env.is_free(c) or chebyshev(c, self.position) < 3:
                    continue
                score = min([euclid(c, nb.position) for nb in self.neighbours.values()] or [10.0])
                score -= 0.1 * chebyshev(c, goal)
                if score > best_score:
                    best, best_score = c, score
        return best

    # ====================================================================
    # 7. Sense-and-avoid guard (geometric, local, no messages needed)
    # ====================================================================
    def _sense_and_avoid(self, t: int, sensed: Dict[int, Cell]) -> None:
        """Last line of defence. Uses ONLY the on-board sensor - no radio
        beliefs at all - so it keeps working under total message loss.

        Two UAVs can collide only if both move into the same cell C, or swap.
        Any cell a neighbour can reach next step is within Chebyshev distance 1
        of its *current* position, so:

        1. I never move into a cell a sensed UAV currently occupies (swap-proof).
        2. If my next cell is within distance 1 of a sensed UAV's current cell,
           exactly one of us must hover. The tie-break is STATIC (lower id
           proceeds) because a rule that depends on mutable state can be seen
           differently by the two sides when messages are lost.
        3. Whoever hovers asks the planner to route around the other next step,
           so a head-on encounter resolves in a couple of steps instead of
           waiting for the deadlock timer.

        This guard is deliberately dumb: it can only ever cost a hover step,
        never a collision. Smart behaviour lives in the LLM layer above it.
        """
        if len(self.path) < 2:
            return
        nxt = self.path[1]
        for uid, pos in sorted(sensed.items()):
            if uid == self.uid:
                continue
            blocked = nxt == pos                                      # rule 1
            if not blocked and chebyshev(nxt, pos) <= SAFETY_RADIUS:  # rule 2
                blocked = uid < self.uid
            if blocked:
                self.path.insert(1, self.position)
                self.waited_total += 1
                self.plan_dirty = True
                self._replan_around = uid                             # rule 3
                self.log.add(t, "sense_avoid_hover", near=uid)
                return

    # ====================================================================
    # 8. Execution
    # ====================================================================
    def _execute_move(self, t: int) -> None:
        self.history.append(self.position)
        if len(self.path) >= 2:
            nxt = self.path[1]
            if nxt != self.position:
                self.distance_flown += euclid(self.position, nxt)
            else:
                self.waited_total += 1
            self.position = nxt
            self.path = self.path[1:]
        if self.wait_steps > 0:
            self.wait_steps -= 1
        self.battery -= 1
        if self.target is not None and self.position == self.env.targets[self.target].cell:
            tgt = self.env.targets[self.target]
            if not tgt.is_done:
                tgt.visited_by = self.uid
                self.visited_targets.append(tgt.tid)
                self.known_done.add(tgt.tid)
                self.outbox.append(Message(self.uid, "DONE", {"target": tgt.tid}, t))
                self.log.add(t, "target_reached", target=tgt.tid)
            self._set_target(None)

    # ====================================================================
    # 9. Event-triggered broadcast
    # ====================================================================
    def _broadcast(self, t: int) -> None:
        if self.plan_dirty or t - self.last_broadcast >= HEARTBEAT:
            self.outbox.append(Message(self.uid, "PLAN", {
                "position": list(self.position),
                "path": [list(c) for c in self.path[:PLAN_HORIZON + 1]],
                "target": self.target, "remaining": self._eta(), "waited": self.waited_total,
            }, t))
            self.plan_dirty = False
            self.last_broadcast = t

    # ====================================================================
    # helpers
    # ====================================================================
    def _set_target(self, tid: Optional[int]) -> None:
        if self.target is not None and tid is None and self.known_claims.get(self.target) == self.uid:
            self.known_claims.pop(self.target, None)
        self.target = tid
        self.path = [self.position]
        self.plan_dirty = True
        self.dist_history.clear()

    def _claim_cost(self, tid: int) -> int:
        """Cheap, deterministic estimate used to settle duplicate claims."""
        return chebyshev(self.position, self.env.targets[tid].cell)

    def _eta(self) -> int:
        return max(0, len(self.path) - 1) if self.target is not None else 999

    def _base_ctx(self, t: int) -> dict:
        return {
            "uav_id": self.uid, "t": t, "position": list(self.position),
            "map_size": [self.env.width, self.env.height],
            "map_excerpt": self._map_excerpt(),
            "neighbours": [{"id": nb.uid, "position": list(nb.position), "target": nb.target,
                            "remaining": nb.remaining} for nb in self.neighbours.values()],
            "messages": [m.to_dict() for m in self.inbox],
            "battery": self.battery,
        }

    def _map_excerpt(self) -> str:
        """Local window of the map around the UAV (what the LLM 'sees')."""
        r = self.sensing_radius
        x0, y0 = self.position
        pos = {self.uid: self.position, **{nb.uid: nb.position for nb in self.neighbours.values()}}
        full = self.env.ascii(pos).split("\n")  # row 0 is y = height-1
        rows = []
        for y in range(min(self.env.height - 1, y0 + r), max(-1, y0 - r - 1), -1):
            line = full[self.env.height - 1 - y]
            rows.append(line[max(0, x0 - r): x0 + r + 1])
        return "\n".join(rows)
