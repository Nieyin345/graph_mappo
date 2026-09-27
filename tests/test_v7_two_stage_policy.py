"""v7 exposes one two-stage matching through the normal policy interface."""

from types import SimpleNamespace

import pytest
import torch

from qkd_rl.rl.algos.policy import MAPPOPolicy


class _TwoStageModel:
    def __init__(self, *, flat_scores=False):
        stop_logit = 0.0 if flat_scores else -20.0
        self.actor = SimpleNamespace(
            temperature=1.0,
            stop_logit=torch.tensor(stop_logit),
            two_stage_enabled=True,
            prepare=SimpleNamespace(stop_logit=torch.tensor(stop_logit)),
        )
        self.action_space = SimpleNamespace(IDLE="IDLE", node_ids=list("abcd"))
        self.flat_scores = flat_scores

    def to(self, _device):
        return self

    def __call__(self, _obs, _device, build_logits_dict=False):
        del build_logits_dict
        service_scores = (0.0, 0.0) if self.flat_scores else (5.0, 10.0)
        preparation_scores = (0.0, 0.0) if self.flat_scores else (-5.0, 5.0)
        return SimpleNamespace(
            logits={},
            logits_node_order=list("abcd"),
            value=torch.tensor(0.0),
            edge_scores={
                ("a", "b"): torch.tensor(service_scores[0], requires_grad=True),
                # A one-stage sampler would choose this first. The stage-one
                # eligibility contract must still force the service arc first.
                ("c", "d"): torch.tensor(service_scores[1], requires_grad=True),
            },
            preparation_scores={
                ("a", "b"): torch.tensor(preparation_scores[0], requires_grad=True),
                ("c", "d"): torch.tensor(preparation_scores[1], requires_grad=True),
            },
            service_arcs={("a", "b")},
        )


def test_policy_runs_service_then_preparation_and_replays_probability():
    policy = MAPPOPolicy(_TwoStageModel(), "cpu")
    obs = SimpleNamespace(node_ids=list("abcd"))

    step = policy.act(obs, deterministic=True)

    assert step.matched_edges == [("a", "b"), ("c", "d")]
    replay_lp = next(iter(policy.evaluate_actions(obs, step.actions, step.matched_edges)[0].values()))
    assert replay_lp.item() == pytest.approx(step.mean_log_prob.item(), abs=1e-6)
    assert replay_lp.requires_grad


def test_two_stage_sampling_uses_the_policy_seed():
    obs = SimpleNamespace(node_ids=list("abcd"))
    policies = [MAPPOPolicy(_TwoStageModel(flat_scores=True), "cpu") for _ in range(2)]

    sequences = []
    for global_seed, policy in zip((1, 999), policies):
        torch.manual_seed(global_seed)
        policy.set_sample_seed(123)
        sequences.append([
            policy.act(obs, deterministic=False).matched_edges
            for _ in range(20)
        ])

    assert sequences[0] == sequences[1]


def test_v7_real_model_builds_and_replays_its_action():
    from qkd_rl.core.config import ConfigValidator, deep_merge, load_config
    from qkd_rl.env.factory import build_env_from_config, load_default_config
    from qkd_rl.model_zoo import build_model
    from tests.helpers import ROOT, point_config_to_h5

    config = point_config_to_h5(load_default_config(ROOT))
    config = deep_merge(config, load_config([ROOT / "qkd_rl/model_zoo/v7/config.yaml"]))
    config = deep_merge(config, load_config([ROOT / "configs/env_full_demand_v2.yaml"]))
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    obs = env.reset(seed=7)
    policy = MAPPOPolicy(build_model(env.action_resolver.action_space, config), "cpu")

    step = policy.act(obs, deterministic=True)
    replay = policy.evaluate_actions(obs, step.actions, list(step.matched_edges or []))

    assert step.matched_edges is not None
    assert replay[2].ndim == 0
    if replay[0]:
        replay_lp = next(iter(replay[0].values()))
        assert replay_lp.item() == pytest.approx(step.mean_log_prob.item(), abs=1e-5)
        assert replay_lp.requires_grad
        (-replay_lp).backward()
        assert any(
            parameter.grad is not None
            for parameter in policy.model.actor.prepare.parameters()
        )

    batched_step = policy.act_batched([obs], deterministic=True)[0]
    batched_replay = policy.evaluate_actions_batched(
        [obs], [batched_step.actions], [list(batched_step.matched_edges or [])]
    )[0]
    assert batched_replay[2].ndim == 0
