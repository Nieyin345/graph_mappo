"""Config-code contract tests.

Guards against the YAML/code mismatches that a manual audit found (dead
config blocks, option keys never read, unimplemented option values accepted
silently). These tests are intentionally static-cheap plus a few behaviour
checks for the switches that were wired up to the code.
"""

from __future__ import annotations

import yaml

import pytest
import torch

from qkd_rl.core.config import ConfigValidator, deep_merge, load_config
from qkd_rl.core.types import LinkType
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.env.graph_builder import GraphObservation
from qkd_rl.env.state import EnvState
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic
from qkd_rl.models.history_encoder import HistoryEncoder
from tests.helpers import ROOT, point_config_to_h5

def _yaml_leaf_keys() -> set[str]:
    keys: set[str] = set()
    for f in sorted((ROOT / "configs").glob("*.yaml")):
        data = yaml.safe_load(f.read_text(encoding="utf-8"))

        def walk(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    keys.add(str(k))
                    walk(v)

        walk(data)
    return keys


# ---------------------------------------------------------------------------
# Static contract: every YAML key must be referenced by the code, and every
# config file must merge + validate.
# ---------------------------------------------------------------------------

def test_all_yaml_leaf_keys_referenced_in_code() -> None:
    """No YAML setting may be dead: each leaf key must appear in qkd_rl/scripts."""
    code_files = list((ROOT / "qkd_rl").glob("**/*.py")) + list((ROOT / "scripts").glob("*.py"))
    code = "\n".join(p.read_text(encoding="utf-8") for p in code_files)
    missing = sorted(k for k in _yaml_leaf_keys() if k not in code)
    assert not missing, f"YAML keys never referenced in code: {missing}"


def test_all_config_files_merge_and_validate() -> None:
    """All shipped YAML files must merge cleanly and pass validation."""
    names = [
        "default.yaml",
        "rate_provider.yaml",
        "features.yaml",
        "env_small.yaml",
        "graph_mappo.yaml",
        "train_mappo.yaml",
        "env_full.yaml",
        "baselines.yaml",
        "train_profiles.yaml",
    ]
    config = load_config([ROOT / "configs" / name for name in names])
    ConfigValidator().validate(config)
    assert config["env"]["name"] == "qkd_full"  # env_full overrides env_small


# ---------------------------------------------------------------------------
# Validation: unimplemented option values must be rejected, not ignored.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "patch",
    [
        {"model": {"encoder": {"gnn_type": "gat"}}},
        {"model": {"encoder": {"share_actor_critic_encoder": False}}},
        {"model": {"actor": {"share_actor_across_node_types": False}}},
        {"model": {"actor": {"temperature": 0.0}}},
        {"model": {"mode": "bogus"}},
        {"qkp": {"type": "node"}},
        {"qkp": {"overflow_policy": "reject"}},
        {"routing": {"mode": "multi_path"}},
        {"routing": {"path_selection": "random"}},
        {"routing": {"serve_order": "fifo"}},
        {"rate_provider": {"rate": {"negative_rate_policy": "bogus"}}},
        {"runtime": {"dtype": "float64"}},
        {"features": {"history_encoder": {"enabled": True, "type": "transformer"}}},
    ],
)
def test_unimplemented_option_values_raise(patch: dict) -> None:
    config = deep_merge(load_default_config(ROOT), patch)
    with pytest.raises(ValueError):
        ConfigValidator().validate(config)


def test_supported_option_values_pass() -> None:
    config = deep_merge(
        load_default_config(ROOT),
        {
            "rate_provider": {"rate": {"negative_rate_policy": "raise"}},
            "features": {"history_encoder": {"enabled": True, "type": "lstm"}},
        },
    )
    ConfigValidator().validate(config)  # must not raise


# ---------------------------------------------------------------------------
# Behaviour: switches that were wired into the code actually change behaviour.
# ---------------------------------------------------------------------------

def test_include_idle_action_switch() -> None:
    config = point_config_to_h5(load_default_config(ROOT))
    config["features"]["action"]["include_idle_action"] = False
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    obs = env.reset()
    for node in obs.node_ids:
        assert "idle" not in obs.action_candidates[node]
        assert len(obs.action_masks[node]) == len(obs.action_candidates[node])
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    output = model(obs)
    assert set(output.logits) == set(obs.node_ids)


def test_actor_temperature_scales_logits() -> None:
    config = point_config_to_h5(load_default_config(ROOT))
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    obs = env.reset()
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    out1 = model(obs)
    model.actor.temperature = 2.0
    out2 = model(obs)
    for node, logits1 in out1.logits.items():
        mask = torch.tensor(obs.action_masks[node], dtype=torch.bool)
        keep = mask.nonzero(as_tuple=True)[0]
        assert torch.allclose(out2.logits[node][keep], logits1[keep] / 2.0)


def test_terminate_on_year_end_switch() -> None:
    config = point_config_to_h5(load_default_config(ROOT))
    ConfigValidator().validate(config)

    env = build_env_from_config(config)
    env.reset(seed=1)
    env.scenario.end_t = env.t + 2
    obs = env._build_observation()
    actions = {n: c[0] for n, c in obs.action_candidates.items()}
    scores = {n: {a: 0.0 for a in c} for n, c in obs.action_candidates.items()}
    for _ in range(3):
        obs, _, _, truncated, _ = env.step(actions, scores)
    assert truncated  # reached end_t with terminate_on_year_end=True

    env2 = build_env_from_config(config)
    env2.config["env"]["terminate_on_year_end"] = False
    env2.reset(seed=1)
    env2.scenario.end_t = env2.t + 2
    obs = env2._build_observation()
    actions = {n: c[0] for n, c in obs.action_candidates.items()}
    for _ in range(3):
        obs, _, _, truncated, _ = env2.step(actions, scores)
    assert not truncated


def test_random_day_aligns_to_day_steps() -> None:
    config = point_config_to_h5(load_default_config(ROOT))
    config["env"]["episode_start_mode"] = "random_day"
    config["env"]["day_steps"] = 100
    config["env"]["episode_steps"] = 300
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    for seed in range(5):
        env.reset(seed=seed)
        assert (env.t - env.scenario.start_t) % 100 == 0


def _make_obs(node_ids, edge_pairs, rates_list, levels, requests=None) -> GraphObservation:
    from qkd_rl.link.rate_provider import EdgeWindow

    edge_ids = [f"E_{s}__{d}" for s, d in edge_pairs]
    windows = {
        edge_id: EdgeWindow(
            edge_id=edge_id,
            rates=rates,
            available=[True] * len(rates),
            link_type=LinkType.GS_HAP,
        )
        for edge_id, rates in zip(edge_ids, rates_list)
    }
    candidates = {n: ["idle"] + sorted({d for s, d in edge_pairs if s == n} | {s for s, d in edge_pairs if d == n}) for n in node_ids}
    state = EnvState(
        t=0,
        qkp_snapshot=dict(levels),
        pending_requests=list(requests) if requests else [],
        edge_windows=windows,
        last_activated_edges=[],
        prev_activated_edges=[],
    )
    return GraphObservation(
        node_features=[[0.0] for _ in node_ids],
        edge_index=[],
        edge_features=[[0.0] for _ in edge_ids],
        node_ids=node_ids,
        edge_ids=edge_ids,
        physical_edge_ids=edge_ids,
        demand_edge_ids=[],
        action_candidates=candidates,
        action_masks={n: [True] * len(candidates[n]) for n in node_ids},
        state=state,
    )


def test_greedy_rate_future_mean_switch() -> None:
    from qkd_rl.baselines.greedy_rate import GreedyRatePolicy

    obs = _make_obs(
        node_ids=["GS1", "GS2", "GS3"],
        edge_pairs=[("GS1", "GS2"), ("GS1", "GS3")],
        rates_list=[[10.0, 100.0, 100.0], [50.0, 1.0, 1.0]],
        levels={},
    )
    now = GreedyRatePolicy(use_future_mean_rate=False).act(obs)[0]
    future = GreedyRatePolicy(use_future_mean_rate=True).act(obs)[0]
    assert now["GS1"] == "GS3"  # current slot rate 50 > 10
    assert future["GS1"] == "GS2"  # future-window mean 100 > 1


def test_greedy_qkp_inventory_weight_switch() -> None:
    from qkd_rl.baselines.greedy_qkp import GreedyQKPPolicy

    obs = _make_obs(
        node_ids=["GS1", "GS2", "GS3"],
        edge_pairs=[("GS1", "GS2"), ("GS1", "GS3")],
        rates_list=[[1.0], [1.0]],
        levels={"E_GS1__GS2": 9.0, "E_GS1__GS3": 3.0},
    )
    level_only = GreedyQKPPolicy(low_inventory_weight=0.0).act(obs)[0]
    low_inventory = GreedyQKPPolicy(low_inventory_weight=10.0).act(obs)[0]
    assert level_only["GS1"] == "GS2"  # higher level wins
    assert low_inventory["GS1"] == "GS3"  # much lower inventory wins


def test_greedy_demand_rate_weight_switch() -> None:
    from qkd_rl.baselines.greedy_demand import GreedyDemandPolicy

    # Demand favours GS2 (50 > 20); a large rate_weight flips the pick to the
    # much faster GS3 edge.
    from qkd_rl.core.types import KeyRequest

    requests = [
        KeyRequest(request_id="r1", src_gs="GS1", dst_gs="GS2", amount=50.0, arrival_t=0, deadline_t=100),
        KeyRequest(request_id="r2", src_gs="GS1", dst_gs="GS3", amount=20.0, arrival_t=0, deadline_t=100),
    ]
    obs = _make_obs(
        node_ids=["GS1", "GS2", "GS3"],
        edge_pairs=[("GS1", "GS2"), ("GS1", "GS3")],
        rates_list=[[1.0], [100.0]],
        levels={},
        requests=requests,
    )
    demand_only = GreedyDemandPolicy(rate_weight=0.0).act(obs)[0]
    rate_heavy = GreedyDemandPolicy(rate_weight=1.0).act(obs)[0]
    assert demand_only["GS1"] == "GS2"
    assert rate_heavy["GS1"] == "GS3"


def test_baselines_yaml_policies_constructible() -> None:
    """Every YAML baseline section must map 1:1 onto a constructible policy,
    using the same wiring as scripts/run_baselines.py."""
    from qkd_rl.baselines.greedy_demand import GreedyDemandPolicy
    from qkd_rl.baselines.greedy_matching import GreedyMatchingPolicy
    from qkd_rl.baselines.greedy_qkp import GreedyQKPPolicy
    from qkd_rl.baselines.greedy_rate import GreedyRatePolicy
    from qkd_rl.baselines.greedy_relay import GreedyRelayPolicy
    from qkd_rl.baselines.greedy_relay_diffusion import GreedyRelayDiffusionPolicyV3
    from qkd_rl.baselines.random_policy import RandomPolicy

    base_cfg = load_config([ROOT / "configs" / "baselines.yaml"]).get("baselines", {})
    assert set(base_cfg) == {
        "random",
        "greedy_rate",
        "greedy_qkp",
        "greedy_demand",
        "greedy_matching",
        "greedy_relay",
        "greedy_relay_diffusion_v3",
    }
    # Construct every enabled policy exactly like run_baselines.py does.
    if base_cfg["random"]["enabled"]:
        RandomPolicy(seed=0)
    GreedyRatePolicy(use_future_mean_rate=base_cfg["greedy_rate"]["use_future_mean_rate"])
    GreedyQKPPolicy(
        low_inventory_weight=base_cfg["greedy_qkp"]["low_inventory_weight"],
        rate_weight=base_cfg["greedy_qkp"]["rate_weight"],
    )
    GreedyDemandPolicy(rate_weight=base_cfg["greedy_demand"]["rate_weight"])
    GreedyMatchingPolicy(
        rate_weight=base_cfg["greedy_matching"]["rate_weight"],
        level_weight=base_cfg["greedy_matching"]["level_weight"],
        demand_weight=base_cfg["greedy_matching"]["demand_weight"],
        keep_weight=base_cfg["greedy_matching"]["keep_weight"],
    )
    GreedyRelayPolicy(
        rate_weight=base_cfg["greedy_relay"]["rate_weight"],
        demand_weight=base_cfg["greedy_relay"]["demand_weight"],
        completion_multiplier=base_cfg["greedy_relay"]["completion_multiplier"],
        keep_weight=base_cfg["greedy_relay"]["keep_weight"],
        deadline_window=base_cfg["greedy_relay"]["deadline_window"],
    )
    GreedyRelayDiffusionPolicyV3(
        rate_weight=base_cfg["greedy_relay_diffusion_v3"]["rate_weight"],
        importance_weight=base_cfg["greedy_relay_diffusion_v3"]["importance_weight"],
        completion_weight=base_cfg["greedy_relay_diffusion_v3"]["completion_weight"],
        keep_weight=base_cfg["greedy_relay_diffusion_v3"]["keep_weight"],
        switch_weight=base_cfg["greedy_relay_diffusion_v3"]["switch_weight"],
        hop_decay_factor=base_cfg["greedy_relay_diffusion_v3"]["hop_decay_factor"],
        max_path_links=base_cfg["greedy_relay_diffusion_v3"]["max_path_links"],
        wait_urgency_tau_ratio=base_cfg["greedy_relay_diffusion_v3"]["wait_urgency_tau_ratio"],
        ignore_consumption=base_cfg["greedy_relay_diffusion_v3"]["ignore_consumption"],
        include_stocked_unavailable=base_cfg["greedy_relay_diffusion_v3"]["include_stocked_unavailable"],
    )


def test_history_encoder_channels_follow_yaml() -> None:
    """Demand-only history channels must follow the YAML include_* keys."""
    config = point_config_to_h5(load_default_config(ROOT))
    config["features"]["history_encoder"]["enabled"] = True
    ConfigValidator().validate(config)
    enc = HistoryEncoder(config)
    assert enc.phys_channels == 0
    assert enc.node_channels == 0
    assert enc.demand_channels == 10
    cfg2 = deep_merge(
        config,
        {"features": {"history_encoder": {"demand_edge": {"include_pending_wait_buckets": False}}}},
    )
    ConfigValidator().validate(cfg2)
    enc2 = HistoryEncoder(cfg2)
    assert enc2.demand_channels == 0
