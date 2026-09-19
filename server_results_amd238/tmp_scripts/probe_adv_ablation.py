#!/usr/bin/env python
"""因果消融：优势归一化到底是不是梯度反号的原因？

已有证据（probe_grad_coherence.py，默认配置）：
  一次 update 内 45 个 minibatch，actor 相邻梯度余弦中位 +0.63、最小 −0.92，
  相对首个 minibatch 的余弦 +0.53 → −0.46（**符号翻转**）。
  Adam SNR 0.110 < 纯噪声基线 0.297。

机制假设：优势已趋零（|A| 1.06→0.56），而 `_loss_for_batch` 对**每个
minibatch 各自**做 (adv-mean)/std，把已经趋零的优势重新放大回单位尺度。
若那个 minibatch 的真实优势本就 ≈0，归一化后的就是纯噪声。

本探针：在同一份 rollout 上跑三次 update，只改归一化开关：
  (a) 原样        normalize_advantages: true
  (b) 关掉        normalize_advantages: false
  (c) 全局标准化   用整个 rollout 的 mean/std，而不是逐 minibatch

(c) 是为区分两种解释：
  若 (b) 好而 (c) 差 → 问题在**逐 minibatch**这个粒度（尺度不稳定）
  若 (b)(c) 都好     → 问题在**归一化本身**（把噪声放大）

用法：
  python .tmp/probe_adv_ablation.py --updates 2
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


def grad_stats(recorded, n_before):
    """返回 (actor 相邻cos均值, actor vs首个cos, critic 相邻cos均值)。"""
    out = {}
    for kind in ("actor", "critic"):
        gs = [g for k, g in recorded[n_before:] if k == kind]
        if len(gs) < 2:
            out[kind] = (float("nan"), [])
            continue
        cos_in = [float(torch.dot(gs[i], gs[i + 1]) /
                        (gs[i].norm() * gs[i + 1].norm()).clamp_min(1e-12))
                  for i in range(len(gs) - 1)]
        first = gs[0]
        drift = [float(torch.dot(g, first) / (g.norm() * first.norm()).clamp_min(1e-12))
                 for g in gs]
        out[kind] = (sum(cos_in) / len(cos_in), drift)
    return out


def run_arm(label, cfg, device, mode, updates, seed=42):
    """mode: 'per_batch' | 'off' | 'global'"""
    import qkd_rl.rl.algos.mappo_trainer as MT

    torch.manual_seed(seed)
    env = build_env_from_config(cfg)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
    policy = MAPPOPolicy(model, device)
    data = load_checkpoint(str(ROOT / "outputs/supervised_pg_phased/"
                                  "supervised_pg_phased_latest.pt"), device)
    model.load_state_dict(data.model_state)
    out_dir = Path(f"/tmp/adv_ablation_{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, cfg, out_dir, device=device)

    # --- (c) 全局标准化：包一层 _loss_for_batch ---
    orig_loss = trainer._loss_for_batch
    if mode == "global":
        state = {"m": None, "s": None}

        def wrapper(batch, clip_eps, entropy_coef, value_coef, normalize_adv):
            return orig_loss(batch, clip_eps=clip_eps, entropy_coef=entropy_coef,
                             value_coef=value_coef, normalize_adv=False)

        trainer._loss_for_batch = wrapper
        # 直接在 collect->update 之间改 step.advantages
        orig_update = trainer.update

        def update_global(buffer):
            alladv = torch.stack([s.advantages.to(device) for s in buffer.steps])
            m, s = alladv.mean(), alladv.std().clamp_min(1e-6)
            for st in buffer.steps:
                st.advantages = (st.advantages.to(device) - m) / s
            return orig_update(buffer)

        trainer.update = update_global
    elif mode == "off":
        orig_update = trainer.update

        def update_off(buffer):
            cfg["train"]["ppo"]["normalize_advantages"] = False
            return orig_update(buffer)

        trainer.update = update_off

    recorded = []
    orig_clip = tu.clip_grad_norm_
    policy_ids = {id(p) for p in trainer._policy_params}
    value_ids = {id(p) for p in trainer._value_params}

    def spy_clip(parameters, max_norm, *a, **kw):
        params = list(parameters)
        flat = [p.grad.detach().reshape(-1).clone() for p in params if p.grad is not None]
        if flat:
            ids = {id(p) for p in params}
            kind = "actor" if ids & policy_ids else ("critic" if ids & value_ids else "other")
            recorded.append((kind, torch.cat(flat)))
        return orig_clip(parameters, max_norm, *a, **kw)

    tu.clip_grad_norm_ = spy_clip
    try:
        print(f"\n===== 臂：{label} =====")
        for u in range(updates):
            buf = trainer.collect_rollout()
            nb = len(recorded)
            # 记录真实优势尺度（归一化前）
            alladv = torch.stack([s.advantages.to(device).reshape(())
                                  for s in buf.steps])
            stats = trainer.update(buf)
            gs = grad_stats(recorded, nb)
            a_in, a_drift = gs["actor"]
            c_in, _ = gs["critic"]
            drift_head = " ".join(f"{v:+.2f}" for v in a_drift[:8])
            print(f"  u{u+1}: |A|={alladv.abs().mean():.4f} 归一化后std={alladv.std():.4f} "
                  f"kl={stats.kl:.5f} nb={stats.n_minibatches}")
            print(f"        actor  相邻cos={a_in:+.4f}   vs首个: {drift_head}")
            print(f"        critic 相邻cos={c_in:+.4f}")
    finally:
        tu.clip_grad_norm_ = orig_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=2)
    ap.add_argument("--arms", default="per_batch,off,global")
    args = ap.parse_args()

    tgm_args = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"],
        mode="random_episode", run_name=None, num_updates=1, seed=42,
        checkpoint=None, device="cpu")
    cfg = tgm.build_config(tgm_args)
    cfg["runtime"]["device"] = "cpu"
    cfg["train"]["n_rollout_workers"] = 1
    cfg["train"]["ppo"]["batch_chunk"] = 64

    LABELS = {"per_batch": "原样（逐 minibatch 归一化）",
              "off": "关掉归一化",
              "global": "全局标准化（整个 rollout 的 mean/std）"}
    for mode in args.arms.split(","):
        run_arm(LABELS.get(mode, mode), cfg, "cpu", mode, args.updates)


if __name__ == "__main__":
    main()
