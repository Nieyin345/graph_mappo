#!/usr/bin/env python
"""决定性探针：PPO 一次 update 内，相邻 minibatch 的 actor 梯度方向一致吗？

不动被追踪的代码 —— 用 monkeypatch 钩住 `torch.nn.utils.clip_grad_norm_`。
`clip_per_role: true` 时它每个 minibatch 被调用两次（actor / critic），
钩子里读 `p.grad` 就能拿到该 minibatch 的完整梯度向量。

判读：
  cos(g_i, g_{i+1}) ≈ 1   → 梯度一致累加，Adam 正常
  cos ≈ 0                  → 方向随机，一个 update 内互相抵消
  cos < 0                  → 系统性反号，比随机更糟（与 Adam SNR 互印证）

背景：实测 Adam SNR = |m|/√v，actor 0.110 / critic 0.033，
       而纯 i.i.d. 噪声的基线是 0.297。低于基线 = 方向反号。
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.utils as tu

_tgm_spec = importlib.util.spec_from_file_location(
    "_tgm", ROOT / "scripts" / "train" / "train_graph_mappo.py")
tgm = importlib.util.module_from_spec(_tgm_spec)
_tgm_spec.loader.exec_module(tgm)

from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer
from qkd_rl.rl.algos.policy import MAPPOPolicy
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic
from qkd_rl.rl.algos.checkpoint import load_checkpoint
from qkd_rl.env.factory import build_env_from_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=3)
    ap.add_argument("--ckpt", default="outputs/supervised_pg_phased/supervised_pg_phased_latest.pt")
    ap.add_argument("--rl", default="", help="可选：先载入已训 checkpoint")
    args = ap.parse_args()

    tgm_args = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"],
        mode="random_episode", run_name=None, num_updates=args.updates,
        seed=42, checkpoint=None, device="cpu")
    cfg = tgm.build_config(tgm_args)
    cfg["runtime"]["device"] = "cpu"
    cfg["train"]["n_rollout_workers"] = 1     # 探针：单进程，便于确定性

    torch.manual_seed(42)
    device = "cpu"

    env = build_env_from_config(cfg)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
    policy = MAPPOPolicy(model, device)
    data = load_checkpoint(str(ROOT / args.ckpt), device)
    model.load_state_dict(data.model_state)
    if args.rl:
        d2 = load_checkpoint(str(ROOT / args.rl), device)
        model.load_state_dict(d2.model_state)
        print(f"已载入 RL checkpoint {args.rl} (update={d2.update})")

    out_dir = Path("/tmp/grad_probe_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, cfg, out_dir, device=device)

    # ---- 钩住 clip_grad_norm_，记录每个 minibatch 的梯度 ----
    recorded = []          # [(kind, flat_grad)]
    orig_clip = tu.clip_grad_norm_
    policy_ids = {id(p) for p in trainer._policy_params}
    value_ids = {id(p) for p in trainer._value_params}

    def spy_clip(parameters, max_norm, *a, **kw):
        params = list(parameters)
        flat = []
        for p in params:
            if p.grad is not None:
                flat.append(p.grad.detach().reshape(-1).clone())
        if flat:
            g = torch.cat(flat)
            ids = {id(p) for p in params}
            if ids & policy_ids:
                kind = "actor"
            elif ids & value_ids:
                kind = "critic"
            else:
                kind = "other"
            recorded.append((kind, g))
        return orig_clip(parameters, max_norm, *a, **kw)

    tu.clip_grad_norm_ = spy_clip
    # mappo_trainer 里是 `torch.nn.utils.clip_grad_norm_(...)`，走模块属性，能钩到

    try:
        for u in range(args.updates):
            buf = trainer.collect_rollout()
            n_before = len(recorded)
            stats = trainer.update(buf)
            print(f"update {u+1}: actor_grad={stats.actor_grad_norm:.4f} "
                  f"critic_grad={stats.critic_grad_norm:.4f} kl={stats.kl:.5f} "
                  f"|A|={stats.mean_abs_advantage:.4f} nb={stats.n_minibatches}")
            batch_rec = recorded[n_before:]
            for kind in ("actor", "critic"):
                gs = [g for k, g in batch_rec if k == kind]
                if len(gs) < 2:
                    continue
                cos_in = []
                for i in range(len(gs) - 1):
                    a, b = gs[i], gs[i + 1]
                    c = torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12)
                    cos_in.append(float(c))
                mean_cos = sum(cos_in) / len(cos_in)
                # 相对第一个梯度的余弦（整体漂移）
                first = gs[0]
                drift = [float(torch.dot(g, first) / (g.norm() * first.norm()).clamp_min(1e-12))
                         for g in gs]
                print(f"    {kind:<7} n={len(gs):<3} "
                      f"相邻cos 均值={mean_cos:+.4f} 中位={sorted(cos_in)[len(cos_in)//2]:+.4f} "
                      f"最小={min(cos_in):+.4f} 最大={max(cos_in):+.4f}")
                print(f"            vs 首个: {[round(d,3) for d in drift[:12]]}")
    finally:
        tu.clip_grad_norm_ = orig_clip


if __name__ == "__main__":
    main()
