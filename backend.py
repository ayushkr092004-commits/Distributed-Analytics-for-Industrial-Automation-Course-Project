"""
llm/backend.py
==============

Pluggable **LLM backends**. Every UAV agent owns a backend and asks it three
kinds of *high-level* questions (see ``prompts.py`` for the exact wording):

1. ``decide_task``      – which mission target should I claim?
2. ``negotiate``        – my planned path conflicts with a neighbour's; what
                          should I do (yield / proceed / reroute)?
3. ``resolve_deadlock`` – I have made no progress for a while; how do I escape?

Each question is answered with a small JSON object. The agent **never trusts
the JSON blindly**: it is validated against the true world state (a technique
the literature calls *feasibility projection*) and if it is invalid or the
backend fails, the agent falls back to a safe heuristic. This is what makes an
LLM usable in a safety-relevant control loop.

Two backends are provided:

``ClaudeBackend``
    Calls Anthropic's Messages API. Needs ``ANTHROPIC_API_KEY`` in the
    environment. The model name is read from ``UAV_LLM_MODEL`` so it can be
    changed without touching code. Responses are cached per prompt so repeated
    identical situations don't cost extra tokens.

``MockBackend``
    A fully deterministic rule-based "LLM stand-in" that answers in exactly the
    same JSON schema. It lets the whole pipeline run offline, makes unit tests
    reproducible, and provides the *baseline* that the real LLM is compared
    against in experiments.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import prompts


@dataclass
class LLMStats:
    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    transcript: List[dict] = field(default_factory=list)  # for explainability


class LLMBackend(ABC):
    """Interface every backend must implement."""

    name: str = "abstract"

    def __init__(self):
        self.stats = LLMStats()

    # -- the three high-level decisions ----------------------------------
    def decide_task(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return self._ask("task", prompts.task_prompt(ctx), ctx)

    def negotiate(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return self._ask("negotiate", prompts.negotiation_prompt(ctx), ctx)

    def resolve_deadlock(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return self._ask("deadlock", prompts.deadlock_prompt(ctx), ctx)

    # -- plumbing ----------------------------------------------------------
    def _ask(self, kind: str, prompt: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self.stats.calls += 1
        try:
            raw = self.complete(kind, prompt, ctx)
            parsed = extract_json(raw)
            if parsed is None:
                raise ValueError(f"no JSON in response: {raw[:200]!r}")
        except Exception as e:  # noqa: BLE001 - we really want to catch everything here
            self.stats.failures += 1
            parsed = {"error": str(e)}
        self.stats.transcript.append({
            "uav": ctx.get("uav_id"), "t": ctx.get("t"), "kind": kind,
            "prompt": prompt, "response": parsed,
        })
        return parsed

    @abstractmethod
    def complete(self, kind: str, prompt: str, ctx: Dict[str, Any]) -> str:
        """Return the raw text answer for ``prompt``."""


# --------------------------------------------------------------------------
def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of a (possibly chatty) LLM reply."""
    if not text:
        return None
    text = text.strip()
    # strip ```json fences
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


# ==========================================================================
class ClaudeBackend(LLMBackend):
    """Anthropic Claude via the official Python SDK."""

    name = "claude"

    def __init__(self, model: Optional[str] = None, max_tokens: int = 400, cache: bool = True):
        super().__init__()
        try:
            import anthropic  # imported lazily so MockBackend users don't need it
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install anthropic") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set; use --backend mock or export the key")
        self._client = anthropic.Anthropic()
        self.model = model or os.environ.get("UAV_LLM_MODEL", "claude-sonnet-4-6")
        self.max_tokens = max_tokens
        self._cache: Dict[str, str] = {} if cache else None  # type: ignore[assignment]

    def complete(self, kind: str, prompt: str, ctx: Dict[str, Any]) -> str:
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if self._cache is not None and key in self._cache:
            self.stats.cache_hits += 1
            return self._cache[key]
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=prompts.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        self.stats.input_tokens += getattr(resp.usage, "input_tokens", 0)
        self.stats.output_tokens += getattr(resp.usage, "output_tokens", 0)
        if self._cache is not None:
            self._cache[key] = text
        return text


# ==========================================================================
class MockBackend(LLMBackend):
    """Deterministic heuristic stand-in that speaks the same JSON protocol.

    The rules are intentionally simple and *explainable*:

    * task       – pick the nearest unclaimed target, weighted by priority,
                   and avoid targets a neighbour is clearly closer to.
    * negotiate  – the UAV with lower priority (farther from its goal, or
                   higher id on ties) yields; the other proceeds.
    * deadlock   – release the current target and pick an escape waypoint
                   perpendicular to the blocked direction.
    """

    name = "mock"

    def complete(self, kind: str, prompt: str, ctx: Dict[str, Any]) -> str:
        if kind == "task":
            return json.dumps(self._task(ctx))
        if kind == "negotiate":
            return json.dumps(self._negotiate(ctx))
        if kind == "deadlock":
            return json.dumps(self._deadlock(ctx))
        return "{}"

    # -------------------------------------------------------------------
    @staticmethod
    def _task(ctx):
        me = ctx["position"]
        best, best_score = None, float("inf")
        for tgt in ctx["targets"]:
            if tgt["claimed_by"] not in (None, ctx["uav_id"]):
                continue
            d = max(abs(tgt["cell"][0] - me[0]), abs(tgt["cell"][1] - me[1]))
            # a neighbour that is much closer should take it
            crowd = 0
            for nb in ctx["neighbours"]:
                nd = max(abs(tgt["cell"][0] - nb["position"][0]), abs(tgt["cell"][1] - nb["position"][1]))
                if nd + 3 < d:
                    crowd += 1
            score = d / max(1, tgt["priority"]) + 6 * crowd
            if score < best_score:
                best, best_score = tgt["tid"], score
        return {"target": best, "reason": "nearest unclaimed target weighted by priority and neighbour proximity"}

    @staticmethod
    def _negotiate(ctx):
        me, other = ctx["me"], ctx["other"]
        # Priority: closer-to-goal UAV proceeds; ties broken by lower id.
        if (me["remaining"], me["id"]) < (other["remaining"], other["id"]):
            return {"action": "proceed", "wait_steps": 0, "reason": "I am closer to my goal (higher priority)"}
        return {"action": "yield", "wait_steps": 2, "reason": "neighbour is closer to its goal; I wait"}

    @staticmethod
    def _deadlock(ctx):
        x, y = ctx["position"]
        gx, gy = ctx["goal"] if ctx["goal"] else (x, y)
        # perpendicular escape: step sideways relative to goal direction
        dx, dy = (gx - x), (gy - y)
        side = (-dy, dx) if (x + y) % 2 == 0 else (dy, -dx)
        norm = max(1, max(abs(side[0]), abs(side[1])))
        wp = [x + 3 * side[0] // norm, y + 3 * side[1] // norm]
        return {"action": "reroute", "waypoint": wp, "release_target": False,
                "reason": "sidestep perpendicular to the blocked direction"}


def make_backend(name: str, **kw) -> LLMBackend:
    name = name.lower()
    if name == "mock":
        return MockBackend()
    if name == "claude":
        return ClaudeBackend(**kw)
    raise ValueError(f"unknown backend {name!r} (use 'mock' or 'claude')")
