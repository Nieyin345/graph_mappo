"""Opt-in: train the v8 forecast head as a read-out of the encoder.

v2c diagnosis (2026-10-01): with the head on and bias_weight=0 (arm rec_head)
the policy still collapsed (3x: -0.156 vs expert), so it is the BCE step on the
shared encoder -- not the bias -- that damages the representation. The flag
``ppo.usage_head_train_encoder: false`` restricts the BCE optimiser to the head
and must leave every encoder tensor (values AND .grad) untouched. Default stays
True so all existing arms are bit-identical.
"""
from __future__ import annotations

import copy

import pytest
import torch

from qkd_rl.env.factory import build_env_from_config
from qkd_rl.model_zoo import build_model
from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer
from qkd_rl.rl.algos.policy import MAPPOPolicy
from scripts.experiment import resolve_experiment


def _trainer(tmp_path, train_encoder):
    cfg = copy.deepcopy(resolve_experiment("v8", 42, 1, ["env_full_demand_v2.yaml"], "t_detach"))
    cfg["env"]["episode_steps"] = 24
    cfg["train"]["rollout_steps"] = 24
    cfg["train"]["episodes_per_update"] = 1
    cfg["train"].pop("curriculum", None)
    cfg["train"]["episode_days_min"] = cfg["train"]["episode_days_max"] = 1
    uf = cfg["model"]["actor"]["usage_forecast"]
    uf["enabled"], uf["horizon"], uf["loss_weight"] = True, 4, 1.0
    if train_encoder is not None:
        cfg["train"]["ppo"]["usage_head_train_encoder"] = train_encoder
    env = build_env_from_config(cfg)
    model = build_model(env.action_resolver.action_space, cfg)
    return MAPPOTrainer(env, MAPPOPolicy(model), cfg, tmp_path, device="cpu"), cfg


@pytest.mark.parametrize("flag,enc_moves", [(None, True), (True, True), (False, False)])
def test_head_bce_touches_encoder_only_when_allowed(tmp_path, flag, enc_moves):
    torch.manual_seed(0)
    tr, cfg = _trainer(tmp_path, flag)
    head = tr.model.actor.usage_forecast
    assert head is not None
    buf = tr.collect_rollout()
    assert any(s.usage_label is not None for s in buf.steps)
    enc0 = [p.detach().clone() for p in tr.model.encoder.parameters()]
    head0 = [p.detach().clone() for p in head.parameters()]
    for p in tr.model.parameters():
        p.grad = None
    bce, _ = tr._train_usage_head(buf, cfg["train"]["ppo"])
    assert bce > 0.0
    assert any(not torch.equal(a, b) for a, b in zip(head0, head.parameters()))
    enc_changed = any(not torch.equal(a, b) for a, b in zip(enc0, tr.model.encoder.parameters()))
    assert enc_changed == enc_moves
    if not enc_moves:
        leaked = [p for p in tr.model.encoder.parameters()
                  if p.grad is not None and float(p.grad.abs().sum()) > 0]
        assert not leaked, "BCE gradient leaked into encoder .grad"
