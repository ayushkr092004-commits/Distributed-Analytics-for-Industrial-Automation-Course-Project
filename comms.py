"""
comms.py
========

Simulated **inter-UAV communication**. This is what makes the system
*distributed* rather than centralised:

* There is **no ground station** and no global planner. Every UAV only ever
  sees messages from UAVs inside its ``comm_range`` (a disc around it).
* Messages can be **lost** with probability ``loss_prob`` (wireless links are
  unreliable) — so agents must be robust to stale / missing information.
* Communication is **event-triggered**: an agent only broadcasts when its plan
  changes or a neighbour requests it, which is the bandwidth-saving pattern
  recommended in the decentralised multi-UAV literature.

Message types
-------------
``PLAN``     – "here is the path I intend to fly for the next H steps"
``CLAIM``    – "I am going to service target T"            (task allocation)
``RELEASE``  – "I give up target T"                        (task allocation)
``PROPOSAL`` – "to resolve our conflict I propose ..."     (LLM negotiation)
``DONE``     – "target T has been serviced"                (gossip)

All messages are plain dictionaries so they can be shown to the LLM verbatim
and logged for debugging.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .environment import Cell, euclid


@dataclass
class Message:
    sender: int
    mtype: str
    payload: dict
    t_sent: int
    broadcast: bool = True
    recipient: int = -1  # only used when broadcast=False

    def to_dict(self) -> dict:
        return {"from": self.sender, "type": self.mtype, "t": self.t_sent, **self.payload}


@dataclass
class CommStats:
    sent: int = 0
    delivered: int = 0
    dropped: int = 0
    out_of_range: int = 0


@dataclass
class MessageBus:
    comm_range: float = 8.0
    loss_prob: float = 0.0
    seed: int = 0
    _rng: random.Random = field(init=False)
    _pending: List[Message] = field(default_factory=list)
    stats: CommStats = field(default_factory=CommStats)
    log: List[dict] = field(default_factory=list)

    def __post_init__(self):
        self._rng = random.Random(self.seed + 991)

    # ------------------------------------------------------------- sending
    def send(self, msg: Message) -> None:
        self.stats.sent += 1
        self._pending.append(msg)
        self.log.append({"t": msg.t_sent, "from": msg.sender, "type": msg.mtype,
                         "to": "all" if msg.broadcast else msg.recipient})

    # ----------------------------------------------------------- delivery
    def deliver(self, positions: Dict[int, Cell]) -> Dict[int, List[Message]]:
        """Deliver all pending messages, respecting range and packet loss.

        Called once per simulation step. Returns ``{uav_id: [messages]}``.
        """
        inboxes: Dict[int, List[Message]] = {uid: [] for uid in positions}
        for msg in self._pending:
            src = positions.get(msg.sender)
            if src is None:
                continue
            targets = positions.keys() if msg.broadcast else [msg.recipient]
            for uid in targets:
                if uid == msg.sender or uid not in positions:
                    continue
                if euclid(src, positions[uid]) > self.comm_range:
                    self.stats.out_of_range += 1
                    continue
                if self._rng.random() < self.loss_prob:
                    self.stats.dropped += 1
                    continue
                inboxes[uid].append(msg)
                self.stats.delivered += 1
        self._pending.clear()
        return inboxes

    def neighbours_of(self, uid: int, positions: Dict[int, Cell]) -> List[int]:
        """IDs of UAVs currently inside ``uid``'s communication range."""
        me = positions[uid]
        return [o for o, p in positions.items() if o != uid and euclid(me, p) <= self.comm_range]
