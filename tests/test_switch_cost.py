"""Switch cost: rate decay on the slot a link is (re)activated."""
from __future__ import annotations

import pytest

from qkd_rl.env.env import QKDEnv
from tests.helpers import build_test_env


@pytest.fixture
def env() -> QKDEnv:
    return build_test_env(".")


def _windows(env):
    return env.rate_provider.get_all_edge_windows(env.t)


def test_first_activation_is_a_switch(env):
    edge_id = env.scenario.edges[0].edge_id
    windows = _windows(env)
    env.last_activated_edges = []
    generated = env._generate_keys([edge_id], windows)
    assert generated[edge_id] == pytest.approx(windows[edge_id].rates[0] * 60 * 0.5)


def test_kept_active_link_generates_full_rate(env):
    edge_id = env.scenario.edges[0].edge_id
    windows = _windows(env)
    env.last_activated_edges = [edge_id]
    generated = env._generate_keys([edge_id], windows)
    assert generated[edge_id] == pytest.approx(windows[edge_id].rates[0] * 60)


def test_switching_to_another_link_decays(env):
    e1, e2 = env.scenario.edges[0].edge_id, env.scenario.edges[1].edge_id
    windows = _windows(env)
    env.last_activated_edges = [e1]
    generated = env._generate_keys([e2], windows)
    assert generated[e2] == pytest.approx(windows[e2].rates[0] * 60 * 0.5)


def test_switch_cost_can_be_disabled(env):
    edge_id = env.scenario.edges[0].edge_id
    env.config["env"]["switch_cost"]["enabled"] = False
    windows = _windows(env)
    env.last_activated_edges = []
    generated = env._generate_keys([edge_id], windows)
    assert generated[edge_id] == pytest.approx(windows[edge_id].rates[0] * 60)
