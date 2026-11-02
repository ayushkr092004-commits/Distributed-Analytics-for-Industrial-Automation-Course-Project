"""
uavplan — Distributed, LLM-assisted path planning for multi-UAV systems.

Module map
----------
environment.py   grid world, obstacles, no-fly zones, mission targets
planner.py       space-time A* (low-level geometric planner)
comms.py         range-limited, lossy, event-triggered radio
llm/prompts.py   prompt templates for the three high-level decisions
llm/backend.py   Claude backend + deterministic mock backend
agent.py         the per-UAV decision cycle (the distributed part)
simulation.py    time-stepping, collision checking, metrics
visualize.py     plots and animated GIF
"""

__version__ = "1.0.0"
