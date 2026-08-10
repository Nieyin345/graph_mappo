"""Physical-edge QKP remaining-capacity feature (and switch-feature removal)."""
from __future__ import annotations

import pytest

from qkd_rl.env.factory import load_default_config
from qkd_rl.env.state import EnvState
from tests.helpers import build_test_env


def _config():
    return load_default_config(".")


def test_vectorized_path_tracks_pool_level():
    """The last physical-edge feature column is qkp_capacity_left and it
    tracks (capacity - level) / capacity through the vectorized path."""
    config = _config()
    env = build_test_env(".")
    env.reset()
    obs0 = env._build_observation()
    assert obs0.physical_edge_ids
    edge_id = obs0.physical_edge_ids[0]
    physical_dim = config["features"]["dims"]["physical_edge_dim_resolved"]
    cap = env.qkp.get_capacity(edge_id)

    row = obs0.edge_ids.index(edge_id)
    level0 = env.qkp.get_level(edge_id)
    assert obs0.edge_features[row, physical_dim - 1] == pytest.approx((cap - level0) / cap)

    env.qkp.add_keys(edge_id, 0.5 * cap, env.t)
    obs1 = env._build_observation()
    row1 = obs1.edge_ids.index(edge_id)
    level1 = env.qkp.get_level(edge_id)
    assert obs1.edge_features[row1, physical_dim - 1] == pytest.approx((cap - level1) / cap)

    env.qkp.add_keys(edge_id, cap, env.t)
    obs2 = env._build_observation()
    row2 = obs2.edge_ids.index(edge_id)
    level2 = env.qkp.get_level(edge_id)
    assert obs2.edge_features[row2, physical_dim - 1] == pytest.approx((cap - level2) / cap)
    assert obs2.edge_features[row2, physical_dim - 1] == pytest.approx(0.0)


def test_loop_path_qkp_capacity_left():
    """Legacy plain-dict-window path computes the same feature."""
    config = _config()
    env = build_test_env(".")
    env.reset()
    edge = env.scenario.edges[0]
    windows = {e.edge_id: env.rate_provider.get_all_edge_windows(env.t)[e.edge_id] for e in env.scenario.edges}
    state = EnvState(
        t=env.t,
        qkp_snapshot=env.qkp.snapshot(),
        pending_requests=[],
        edge_windows=windows,
        last_activated_edges=[],
        prev_activated_edges=[],
    )
    cfg = config["features"]
    physical_dim = cfg["dims"]["physical_edge_dim_resolved"]
    demand_dim = cfg["dims"]["demand_edge_dim_resolved"]
    rows = env.graph_builder._build_physical_edge_rows_loop(
        env.scenario.edges, state, cfg["edge"], physical_dim, demand_dim
    )
    capacity = env.qkp.get_capacity(edge.edge_id)
    level = env.qkp.get_level(edge.edge_id)
    assert rows[0, physical_dim - 1] == pytest.approx((capacity - level) / capacity)


def test_switch_flag_feature_removed_from_config():
    """include_switch_cost is no longer a physical-edge feature (the env rate
    decay still exists; only the redundant feature was removed)."""
    config = _config()
    edge_cfg = config["features"]["edge"]
    assert "include_switch_cost" not in edge_cfg
    assert edge_cfg.get("include_qkp_capacity_left") is True
