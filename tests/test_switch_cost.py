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


def test_realized_cloud_gate_is_applied_after_action(env, monkeypatch):
    edge_id = env.scenario.edges[0].edge_id
    windows = _windows(env)
    env.last_activated_edges = [edge_id]
    monkeypatch.setattr(env.rate_provider, "get_cloud_gate", lambda edge, t: 0.0, raising=False)
    generated = env._generate_keys([edge_id], windows)
    assert generated[edge_id] == 0.0


def test_precomputed_finite_key_first_slot_rate_overrides_multiplier(env, monkeypatch):
    edge_id = env.scenario.edges[0].edge_id
    windows = _windows(env)
    env.last_activated_edges = []
    monkeypatch.setattr(env.rate_provider, "get_activation_rate", lambda edge, t: 1.25, raising=False)
    monkeypatch.setattr(env.rate_provider, "capture_gamma", 0.5, raising=False)
    generated = env._generate_keys([edge_id], windows)
    assert generated[edge_id] == pytest.approx(1.25 * env.scenario.slot_seconds)




def test_v75_qkd_block_releases_only_when_block_closes(env, monkeypatch):
    edge_id = env.scenario.edges[0].edge_id
    windows = _windows(env)
    env.config["env"]["qkd_block"] = {
        "enabled": True,
        "release_mode": "asymptotic_proxy",
        "max_duration_s": 120.0,
    }
    env._qkd_blocks = {}
    env._qkd_fk = None

    gains = [2.0e-3, 9.0e-4, 5.0e-7]
    y0 = gains[-1]
    errors = [0.01 * q + 0.49 * y0 for q in gains]

    def stats(edge, t, activation=False):
        return {
            "gains": gains,
            "errors": errors,
            "y1": 2.2e-3,
            "n_pulses": 5.4e9 if activation else 6.0e9,
        }

    monkeypatch.setattr(env.rate_provider, "get_qkd_slot_stats", stats, raising=False)
    monkeypatch.setattr(env.rate_provider, "get_cloud_gate", lambda edge, t: 1.0, raising=False)
    monkeypatch.setattr(
        env.rate_provider,
        "qkd_params",
        {"f_rep": 100e6, "slot_s": 60.0, "e_d": 0.01},
        raising=False,
    )

    env.last_activated_edges = []
    first = env._generate_keys([edge_id], windows)
    assert first == {}
    assert edge_id in env._qkd_blocks

    env.last_activated_edges = [edge_id]
    second = env._generate_keys([edge_id], windows)
    assert second[edge_id] > 0.0
    assert edge_id not in env._qkd_blocks


def test_v75_blocked_cloud_slot_advances_block_without_key_counts(env, monkeypatch):
    edge_id = env.scenario.edges[0].edge_id
    windows = _windows(env)
    env.config["env"]["qkd_block"] = {
        "enabled": True,
        "release_mode": "asymptotic_proxy",
        "max_duration_s": 60.0,
    }
    env._qkd_blocks = {}
    env._qkd_fk = None
    monkeypatch.setattr(
        env.rate_provider,
        "get_qkd_slot_stats",
        lambda edge, t, activation=False: {
            "gains": [2.0e-3, 9.0e-4, 5.0e-7],
            "errors": [2.0245e-5, 9.245e-6, 2.5e-7],
            "y1": 2.2e-3,
            "n_pulses": 5.4e9,
        },
        raising=False,
    )
    monkeypatch.setattr(env.rate_provider, "get_cloud_gate", lambda edge, t: 0.0, raising=False)
    monkeypatch.setattr(
        env.rate_provider,
        "qkd_params",
        {"f_rep": 100e6, "slot_s": 60.0, "e_d": 0.01},
        raising=False,
    )
    env.last_activated_edges = []
    generated = env._generate_keys([edge_id], windows)
    assert generated == {}
    assert edge_id not in env._qkd_blocks


def test_v75_continuous_session_block_closes_only_on_deactivation(env, monkeypatch):
    edge_id = env.scenario.edges[0].edge_id
    windows = _windows(env)
    env.config["env"]["qkd_block"] = {
        "enabled": True,
        "release_mode": "asymptotic_proxy",
        "max_duration_s": None,
    }
    env._qkd_blocks = {}
    env._qkd_fk = None

    gains = [2.0e-3, 9.0e-4, 5.0e-7]
    y0 = gains[-1]
    errors = [0.01 * q + 0.49 * y0 for q in gains]

    monkeypatch.setattr(
        env.rate_provider,
        "get_qkd_slot_stats",
        lambda edge, t, activation=False: {
            "gains": gains,
            "errors": errors,
            "y1": 2.2e-3,
            "n_pulses": 5.4e9 if activation else 6.0e9,
        },
        raising=False,
    )
    monkeypatch.setattr(env.rate_provider, "get_cloud_gate", lambda edge, t: 1.0, raising=False)
    monkeypatch.setattr(
        env.rate_provider,
        "qkd_params",
        {"f_rep": 100e6, "slot_s": 60.0, "e_d": 0.01},
        raising=False,
    )

    env.last_activated_edges = []
    assert env._generate_keys([edge_id], windows) == {}
    assert edge_id in env._qkd_blocks

    for _ in range(3):
        env.last_activated_edges = [edge_id]
        assert env._generate_keys([edge_id], windows) == {}
        assert edge_id in env._qkd_blocks

    env.last_activated_edges = [edge_id]
    released = env._generate_keys([], windows)
    assert released[edge_id] > 0.0
    assert edge_id not in env._qkd_blocks


def test_v75_physical_capture_is_independent_of_switch_cost_control(env, monkeypatch):
    edge_id = env.scenario.edges[0].edge_id
    windows = _windows(env)
    env.config["env"]["qkd_block"] = {
        "enabled": True,
        "release_mode": "asymptotic_proxy",
        "max_duration_s": 60.0,
    }
    env.config["env"]["switch_cost"]["enabled"] = False
    env._qkd_blocks = {}
    env._qkd_fk = None
    activation_flags = []

    gains = [2.0e-3, 9.0e-4, 5.0e-7]
    y0 = gains[-1]
    errors = [0.01 * q + 0.49 * y0 for q in gains]

    def stats(edge, t, activation=False):
        activation_flags.append(bool(activation))
        return {
            "gains": gains,
            "errors": errors,
            "y1": 2.2e-3,
            "n_pulses": 6.0e9,
        }

    monkeypatch.setattr(env.rate_provider, "get_qkd_slot_stats", stats, raising=False)
    monkeypatch.setattr(env.rate_provider, "get_cloud_gate", lambda edge, t: 1.0, raising=False)
    monkeypatch.setattr(
        env.rate_provider,
        "qkd_params",
        {"f_rep": 100e6, "slot_s": 60.0, "e_d": 0.01},
        raising=False,
    )
    env.last_activated_edges = []
    released = env._generate_keys([edge_id], windows)
    assert activation_flags == [True]
    assert released[edge_id] > 0.0
