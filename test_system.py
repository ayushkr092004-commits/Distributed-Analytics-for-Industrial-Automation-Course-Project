"""
Unit + integration tests. Run with:  python -m pytest -q
All tests use the deterministic MockBackend, so no API key is needed.
"""

import json

from uavplan.comms import Message, MessageBus
from uavplan.environment import FREE, OBSTACLE, Environment, chebyshev
from uavplan.llm.backend import MockBackend, extract_json
from uavplan.planner import build_reservations, path_length, space_time_astar
from uavplan.simulation import Simulation


# ---------------------------------------------------------------- planner
def _empty_env(w=10, h=10):
    env = Environment(w, h)
    env.grid = [[FREE] * w for _ in range(h)]
    return env


def test_astar_straight_line():
    env = _empty_env()
    p = space_time_astar(env, (0, 0), (5, 0))
    assert p[0] == (0, 0) and p[-1] == (5, 0)
    assert len(p) == 6  # 5 unit moves


def test_astar_avoids_obstacle_wall():
    env = _empty_env()
    for y in range(0, 9):           # wall with a gap at the top
        env.grid[y][5] = OBSTACLE
    p = space_time_astar(env, (0, 0), (9, 0))
    assert p is not None
    assert all(env.is_free(c) for c in p)
    assert any(c[1] >= 9 for c in p)  # went through the gap


def test_astar_respects_reservation_and_waits():
    env = _empty_env()
    # neighbour sits at (3,0) forever -> we must go around it (radius 1)
    reserved = build_reservations({9: [(3, 0)]}, horizon=50, safety_radius=1)
    p = space_time_astar(env, (0, 0), (6, 0), reserved)
    assert p is not None
    for t, c in enumerate(p):
        assert (c[0], c[1], t) not in reserved


def test_astar_waypoint_is_safe_even_if_bad():
    env = _empty_env()
    env.grid[5][5] = OBSTACLE
    p = space_time_astar(env, (0, 0), (9, 9), waypoints=[(5, 5), (2, 8)])
    assert p is not None and p[-1] == (9, 9)
    assert (5, 5) not in p


def test_path_length():
    assert path_length([(0, 0), (1, 1), (2, 1), (2, 1)]) == 2 ** 0.5 + 1


# ----------------------------------------------------------------- comms
def test_messages_respect_range():
    bus = MessageBus(comm_range=3.0)
    bus.send(Message(0, "PLAN", {"x": 1}, 0))
    inbox = bus.deliver({0: (0, 0), 1: (2, 0), 2: (10, 10)})
    assert len(inbox[1]) == 1 and len(inbox[2]) == 0
    assert bus.stats.out_of_range == 1


def test_packet_loss_is_applied():
    bus = MessageBus(comm_range=100, loss_prob=1.0)
    bus.send(Message(0, "PLAN", {}, 0))
    inbox = bus.deliver({0: (0, 0), 1: (1, 1)})
    assert inbox[1] == [] and bus.stats.dropped == 1


# -------------------------------------------------------------------- llm
def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here you go: {"action": "yield", "wait_steps": 2} ok') == \
        {"action": "yield", "wait_steps": 2}
    assert extract_json("no json here") is None


def test_mock_backend_task_choice_is_nearest_unclaimed():
    b = MockBackend()
    ctx = {"uav_id": 0, "t": 0, "position": [0, 0], "neighbours": [],
           "targets": [{"tid": 0, "cell": [9, 9], "priority": 1, "claimed_by": None},
                       {"tid": 1, "cell": [2, 2], "priority": 1, "claimed_by": 5},
                       {"tid": 2, "cell": [3, 0], "priority": 1, "claimed_by": None}]}
    assert b._task(ctx)["target"] == 2


# ------------------------------------------------------------- integration
def test_full_mission_completes_without_collision():
    env = Environment.random(30, 30, n_targets=6, seed=1)
    sim = Simulation(env, n_uavs=4, backend="mock", seed=1, max_steps=300)
    m = sim.run()
    assert m.targets_done == 6
    assert m.collisions == 0
    assert m.success


def test_harsh_conditions_no_collisions():
    """Short range + heavy packet loss: the sensor-only guard must hold."""
    for seed in range(6):
        env = Environment.random(40, 40, n_targets=12, obstacle_density=0.18, seed=seed)
        sim = Simulation(env, n_uavs=10, backend="mock", seed=seed, max_steps=250,
                         comm_range=5, loss_prob=0.4)
        m = sim.run()
        assert m.collisions == 0, f"seed {seed}: {sim.collision_events}"
        assert m.targets_done == 12


def test_llm_answers_are_validated():
    """An LLM that returns garbage must not break or endanger the mission."""
    class GarbageBackend(MockBackend):
        name = "garbage"
        def complete(self, kind, prompt, ctx):
            return '{"target": "banana", "action": "fly_into_wall", "waypoint": [-99, -99]}'

    env = Environment.random(30, 30, n_targets=6, seed=2)
    sim = Simulation(env, n_uavs=4, backend="mock", seed=2, max_steps=300,
                     shared_backend=GarbageBackend())
    m = sim.run()
    assert m.collisions == 0
    assert m.targets_done == 6


def test_every_target_visited_exactly_once():
    env = Environment.random(30, 30, n_targets=8, seed=4)
    sim = Simulation(env, n_uavs=5, backend="mock", seed=4, max_steps=300)
    sim.run()
    visited = [tid for a in sim.agents for tid in a.visited_targets]
    assert sorted(visited) == list(range(8))


def test_agents_use_only_local_information():
    """A UAV out of radio range must not know about the other's claim."""
    env = _empty_env(60, 60)
    env.targets = []
    from uavplan.environment import Target
    env.targets = [Target(0, (58, 58)), Target(1, (1, 1))]
    sim = Simulation(env, n_uavs=2, backend="mock", seed=0, max_steps=3, comm_range=2.0)
    sim.agents[0].position = (0, 0)
    sim.agents[1].position = (59, 59)
    sim.run()
    a, b = sim.agents
    assert b.uid not in a.neighbours and a.uid not in b.neighbours
