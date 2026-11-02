"""
llm/prompts.py
==============

Prompt templates. Design principles (drawn from the decentralised LLM
goal-assignment and LLMDR papers):

* **Local information only.** The prompt contains exactly what a real UAV
  would know: its own state, the mission targets it has heard about, and the
  messages received from neighbours in range. Nothing global.
* **Structured input.** Lists of dicts, not prose, because LLMs reason more
  reliably over structured facts.
* **Explicit rules & checklist.** Conflict-resolution rules are spelled out so
  two UAVs reasoning independently reach *compatible* decisions.
* **Strict JSON output.** Keeps parsing trivial and lets us validate the answer
  before it touches the flight controller.
"""

from __future__ import annotations

import json
from typing import Any, Dict

SYSTEM_PROMPT = """You are the onboard mission-planning assistant of ONE autonomous UAV that is
part of a distributed multi-UAV team. There is no ground station: every UAV runs its own
copy of you and coordinates only through short radio messages with neighbours in range.

You make HIGH-LEVEL decisions only (which target to claim, whether to yield in a conflict,
how to escape a deadlock). A separate on-board A* planner handles exact obstacle avoidance,
so never try to output full cell-by-cell paths.

Rules you must respect:
1. Safety first: never propose an action that moves the UAV into an obstacle.
2. Be cooperative: the team objective is to visit ALL targets with minimum total time
   and no collisions - not to maximise your own score.
3. Be consistent: two UAVs facing the same conflict must reach compatible decisions, so
   always apply the tie-break rules exactly as written.
4. Reply with ONE JSON object and nothing else. No markdown, no prose outside the JSON.
"""


def _j(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


def task_prompt(ctx: Dict[str, Any]) -> str:
    return f"""TASK ALLOCATION DECISION  (time step {ctx['t']})

You are UAV {ctx['uav_id']} at cell {ctx['position']}. Battery/steps remaining: {ctx['battery']}.
Map is {ctx['map_size'][0]}x{ctx['map_size'][1]}; '#'=obstacle, '~'=no-fly (avoid), 'T'=pending target, 'x'=done.

Local map excerpt around you (north is up):
{ctx['map_excerpt']}

Targets you know about (cell, priority 1-3 where 3 is most important, who has claimed it):
{_j(ctx['targets'])}

Neighbours in radio range (id, position, their current target, distance to it):
{_j(ctx['neighbours'])}

Recent messages received:
{_j(ctx['messages'][-8:])}

Decide which target YOU should claim. Checklist:
- Prefer unclaimed targets. Only contest a claimed one if you are MUCH closer (>5 cells).
- Prefer high-priority targets, then nearby ones.
- If a neighbour is obviously better placed for a target, leave it to them.
- If no suitable target exists, return null.

Answer JSON: {{"target": <tid or null>, "reason": "<one short sentence>"}}"""


def negotiation_prompt(ctx: Dict[str, Any]) -> str:
    return f"""CONFLICT NEGOTIATION  (time step {ctx['t']})

Your planned path and UAV {ctx['other']['id']}'s announced path come within the safety
distance in {ctx['conflict']['in_steps']} step(s) near cell {ctx['conflict']['cell']}.

ME:    {_j(ctx['me'])}
OTHER: {_j(ctx['other'])}
('remaining' = steps left to reach own goal; 'waited' = steps already spent waiting)

Local map excerpt (north is up):
{ctx['map_excerpt']}

Messages from the other UAV about this conflict (may be empty):
{_j(ctx['messages'])}

Choose exactly one action for YOURSELF:
- "proceed": keep my path, expect the other to yield.
- "yield":   hover in place for wait_steps (1-4) and then re-plan around the other.
- "reroute": immediately re-plan via the given waypoint to get out of the other's way.

Tie-break rules (apply in order, both UAVs use the same rules so decisions stay compatible):
1. The UAV with the FEWER 'remaining' steps proceeds; the other yields or reroutes.
2. If equal, the UAV that has already 'waited' more proceeds.
3. If still equal, the UAV with the LOWER id proceeds.
Exception: if the other UAV already sent "yield", you should "proceed".

Answer JSON: {{"action": "proceed"|"yield"|"reroute", "wait_steps": <int>,
"waypoint": [x,y] or null, "reason": "<one short sentence>"}}"""


def deadlock_prompt(ctx: Dict[str, Any]) -> str:
    return f"""DEADLOCK RESOLUTION  (time step {ctx['t']})

You are UAV {ctx['uav_id']} at {ctx['position']} heading to target {ctx['target']} at {ctx['goal']}.
You have made NO progress for {ctx['stuck_steps']} steps. Likely cause: a narrow corridor
shared with neighbours, or repeated mutual yielding.

Local map excerpt (north is up, you are '{ctx['uav_id'] % 10}'):
{ctx['map_excerpt']}

Neighbours: {_j(ctx['neighbours'])}
Cells you tried recently: {_j(ctx['recent_cells'])}

Options:
- "reroute": give a waypoint 2-5 cells away in open space ('.') that detours around the jam.
- "release": give up this target so a better-placed neighbour can take it, and pick another.
- "wait": hold position (only if you believe the jam is clearing).

Answer JSON: {{"action": "reroute"|"release"|"wait", "waypoint": [x,y] or null,
"release_target": true|false, "reason": "<one short sentence>"}}"""
