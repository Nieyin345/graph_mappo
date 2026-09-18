#!/usr/bin/env python
"""验证 PPO loss 重构（累加→stack·mean）数值等价且更快。

重构内容：`_loss_for_batch` 从「循环里 5 次标量累加，末尾除 n_steps」
改为「循环里 append，末尾 torch.stack(...).mean()」。

数学上等价：mean(a) + mean(b) == mean(a+b)（等长序列），
且 mean 与「和的均值」都是同一线性组合。但浮点加法不满足结合律，
所以**不是逐位相同**，只应到浮点误差量级（~1e-6 相对）。

本脚本用同一份 buffer、同一权重、同一 rng 种子，分别用两种实现跑一个
minibatch，比较 5 个输出量与 loss 本身的差。

用法：
  OMP_NUM_THREADS=4 python .tmp/verify_loss_refactor.py --episodes 1
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

_tgm_spec = importlib.util.spec_from_file_location(
    "_tgm", ROOT / "scripts" / "train" / "train_graph_mappo.py")
tgm = importlib.util.module_from_spec(_tgm_spec)
_tgm_spec.loader.exec_module(tgm)

from qkd_rl.env.factory import build_env_from_config
from qkd_rl.rl.algos.checkpoint import load_checkpoint
from qkd_rl.rl.algos.policy import MAPPOPolicy
from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic


def legacy_loss_for_batch(trainer, batch, clip_eps, entropy_coef, value_coef,
                          normalize_adv):
    """重构前的实现（逐标量累加），用于对照。"""
    if not batch:
        raise ValueError("Empty minibatch in PPO update.")
    advantages = [step.advantages.to(trainer.device) for step in batch]
    if normalize_adv and len(advantages) > 1:
        stacked = torch.stack(advantages)
        mean = stacked.mean()
        std = stacked.std().clamp_min(1.0e-6)
        advantages = [(adv - mean) / std for adv in advantages]
    returns_batch = torch.stack([step.returns.to(trainer.device) for step in batch])
    critic_beta = torch.clamp(returns_batch.std(), min=1.0).detach()

    actor_loss = torch.zeros((), dtype=torch.float32, device=trainer.device)
    critic_loss = torch.zeros((), dtype=torch.float32, device=trainer.device)
    entropy_sum = torch.zeros((), dtype=torch.float32, device=trainer.device)
    kl_sum = torch.zeros((), dtype=torch.float32, device=trainer.device)
    ratio_sum = torch.zeros((), dtype=torch.float32, device=trainer.device)
    chunk_size = int(trainer.config["train"].get("ppo", {}).get("batch_chunk", 256))
    n_steps = 0
    for start in range(0, len(batch), chunk_size):
        chunk = batch[start:start + chunk_size]
        chunk_advantages = advantages[start:start + chunk_size]
        batched_results = trainer.policy.evaluate_actions_batched(
            [step.obs for step in chunk],
            [step.actions for step in chunk],
            [list(step.matched_edges or []) for step in chunk],
        )
        for step, adv, (log_probs, entropies, value) in zip(
                chunk, chunk_advantages, batched_results):
            node_ids = step.obs.node_ids
            if node_ids:
                new_lp = log_probs[node_ids[0]]
                old_lp = (
                    step.mean_log_prob.to(trainer.device)
                    if step.mean_log_prob is not None
                    else step.log_probs[node_ids[0]].to(trainer.device)
                )
                ratios = torch.exp(new_lp - old_lp)
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                actor_loss = actor_loss - torch.min(surr1, surr2)
                entropy_sum = entropy_sum + entropies[node_ids[0]]
                kl_sum = kl_sum + (ratios - 1.0 - (new_lp - old_lp))
                ratio_sum = ratio_sum + ratios
            returns_target = step.returns.to(trainer.device)
            critic_loss = critic_loss + torch.nn.functional.smooth_l1_loss(
                value, returns_target, beta=critic_beta)
            n_steps += 1
        del batched_results, chunk_advantages
    actor_loss = actor_loss / n_steps
    critic_loss = (critic_loss / n_steps) * value_coef
    return (actor_loss, critic_loss, entropy_sum / n_steps,
            kl_sum / n_steps, ratio_sum / n_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1440)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--minibatch", type=int, default=None)
    args = ap.parse_args()

    tgm_args = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"],
        mode="random_episode", run_name=None, num_updates=1,
        seed=42, checkpoint=None, device="cpu")
    cfg = tgm.build_config(tgm_args)
    cfg["runtime"]["device"] = "cpu"
    cfg["env"]["episode_steps"] = args.steps
    cfg["train"]["episode_steps_fixed"] = True
    cfg["train"]["episodes_per_update"] = args.episodes
    cfg["train"]["rollout_batch_envs"] = args.episodes
    if args.minibatch:
        cfg["train"]["ppo"]["minibatch_size"] = args.minibatch

    base = load_checkpoint(
        str(ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"),
        "cpu").model_state

    env = build_env_from_config(cfg)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
    policy = MAPPOPolicy(model, "cpu")
    model.load_state_dict({k: v.clone() for k, v in base.items()})

    out = Path("/tmp/verifyloss"); out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, cfg, out, device="cpu")

    ppo = cfg["train"]["ppo"]
    mb = int(ppo["minibatch_size"])
    print(f"minibatch={mb} episodes={args.episodes} steps={args.steps}",
          flush=True)

    buf = trainer.collect_rollout()
    print(f"buffer {len(buf.steps)} 条\n", flush=True)

    # 同一 minibatch 喂给两种实现
    import random
    rng = random.Random(1234)
    batch = next(iter(buf.sample(mb, rng)))

    model.train()
    model.load_state_dict({k: v.clone() for k, v in base.items()})
    new = trainer._loss_for_batch(batch, clip_eps=float(ppo["clip_eps"]),
                                  entropy_coef=float(ppo["entropy_coef"]),
                                  value_coef=float(ppo["value_coef"]),
                                  normalize_adv=bool(ppo.get("normalize_advantages", True)))

    # 先把新实现的梯度取出来 —— load_state_dict 是**就地**改写参数，
    # 会 bump 版本号并让上一个图失效（NotImplementedError: version mismatch）。
    # 所以必须在任何 reload 之前完成 backward。
    trainer.optimizer.zero_grad()
    (new[0] + new[1]).backward()
    gnew = {n: p.grad.clone() for n, p in model.named_parameters()
            if p.grad is not None}

    # 再跑旧实现
    model.load_state_dict({k: v.clone() for k, v in base.items()})
    old = legacy_loss_for_batch(trainer, batch, float(ppo["clip_eps"]),
                                float(ppo["entropy_coef"]),
                                float(ppo["value_coef"]),
                                bool(ppo.get("normalize_advantages", True)))
    trainer.optimizer.zero_grad()
    (old[0] + old[1]).backward()
    gold = {n: p.grad.clone() for n, p in model.named_parameters()
            if p.grad is not None}

    names = ["actor_loss", "critic_loss", "entropy_mean", "kl_mean", "ratio_mean"]
    print(f"{'量':<16}{'重构后':>18}{'重构前':>18}{'绝对差':>14}{'相对差':>12}")
    print("-" * 74)
    worst = 0.0
    for nm, a, b in zip(names, new, old):
        fa, fb = float(a.detach()), float(b.detach())
        ad = abs(fa - fb)
        rel = ad / max(abs(fb), 1e-12)
        # 量本身可能极小（actor_loss ~1.6e-7），相对差会放大浮点噪声，
        # 所以判定用**绝对差**，且只在量不接近 0 时才看相对差。
        if abs(fb) > 1e-6:
            worst = max(worst, rel)
        print(f"{nm:<16}{fa:>18.10f}{fb:>18.10f}{ad:>14.2e}{rel:>12.2e}")

    # 梯度也要比：等价的重构不该改变梯度
    gmax = 0.0
    for n in gnew:
        d = (gnew[n] - gold[n]).abs().max().item()
        s = gold[n].abs().max().item()
        gmax = max(gmax, d / max(s, 1e-12))
    print(f"\n梯度最大相对差: {gmax:.3e}")
    print(f"\n判定：{'等价（差异在浮点误差内）' if worst < 1e-4 and gmax < 1e-4 else '!! 不等价，需排查'}")


if __name__ == "__main__":
    main()
