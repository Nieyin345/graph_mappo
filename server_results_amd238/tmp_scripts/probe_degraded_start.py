#!/usr/bin/env python
"""决定性实验：把 BC 起点人为退化，看 PPO 能不能爬回来。

已知：
  BC 起点   0.8483（训练 regime，10 种子）
  专家      0.8692
  RL 训20轮 0.8549   ← 只比起点高 0.0066
  一次 update 内 45 个 minibatch 的 actor 梯度**整段反号**

两种解释要分开，它们的处方完全不同：

  (甲) 算法是好的，只是 BC 起点已近最优 → 没有可爬的坡，训练"无效"是**对的**。
       处方：换目标（成功率已到顶）、换评测 regime、或在验证集上挑权重。
  (乙) 算法坏了 → 梯度无法把参数推向更高回报。
       处方：改更新信号（归一化粒度 / 信任域 / 基线）。

**区分办法**：把 BC 起点的 actor 参数加噪声，造出一个明确更差的起点，
再训几轮。若回报**回升**，说明算法能把参数往正确方向推（甲）；
若原地不动甚至继续跌，说明更新信号是坏的（乙）。

这是"是否能学"的判据，与"当前是否有坡"无关 —— 退化起点保证有坡。

用法：
  python .tmp/probe_degraded_start.py --sigmas 0.02,0.05,0.10 --updates 6
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
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


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def evaluate(policy, cfg, seeds, max_steps=None):
    """在训练 regime 上跑确定性评测，返回平均成功率。"""
    srs = []
    with torch.no_grad():
        for seed in seeds:
            env = build_env_from_config(cfg)
            obs = env.reset(seed=seed)
            served = 0.0
            done = False
            n = 0
            while not done:
                st = policy.act(obs, deterministic=True)
                obs, _r, term, trunc, info = env.step(
                    st.actions, st.action_scores,
                    edge_scores=st.edge_scores,
                    expected_matched_edges=list(st.matched_edges or []))
                served += float(info.get("served_keys", 0.0))
                done = term or trunc
                n += 1
                if max_steps and n >= max_steps:
                    break
            s = env.metrics.episode_summary()
            arr = float(s.get("arrived_keys", 0.0))
            srs.append(served / arr if arr else 0.0)
    return sum(srs) / len(srs), srs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", default="0.02,0.05,0.10")
    ap.add_argument("--updates", type=int, default=6)
    ap.add_argument("--eval-seeds", default="7,8,9,10,11,12")
    ap.add_argument("--train-seeds", default="7,8,9,10,11,12,13,14,15,16")
    args = ap.parse_args()

    sigmas = [float(s) for s in args.sigmas.split(",")]
    eval_seeds = parse_seeds(args.eval_seeds)

    tgm_args = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"],
        mode="random_episode", run_name=None, num_updates=args.updates,
        seed=42, checkpoint=None, device="cpu")
    cfg = tgm.build_config(tgm_args)
    cfg["runtime"]["device"] = "cpu"
    cfg["train"]["n_rollout_workers"] = 1
    cfg["train"]["ppo"]["batch_chunk"] = 64
    cfg["train"]["num_updates"] = args.updates

    CKPT = ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
    base_state = load_checkpoint(str(CKPT), "cpu").model_state
    device = "cpu"

    def build_policy(sigma, seed):
        torch.manual_seed(seed)
        env = build_env_from_config(cfg)
        model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
        state = {k: v.clone() for k, v in base_state.items()}
        if sigma > 0:
            # 只扰动 actor（含 encoder+actor 两段），critic 保持原样：
            # 这样"退化"退化的是策略，而不是把基线也打乱。
            #
            # 尺度用**全体 actor 参数的 RMS**，而不是各自的 std()：
            # 标量参数（如 actor.stop_logit）的 std() 是 NaN（自由度 0），
            # 用它会把参数污染成 NaN。RMS 对任何形状都有定义。
            actor_keys = [k for k in state
                          if not k.startswith("critic") and state[k].is_floating_point()]
            total_sq = sum(float(state[k].pow(2).sum()) for k in actor_keys)
            total_n = sum(state[k].numel() for k in actor_keys)
            rms = (total_sq / max(1, total_n)) ** 0.5
            print(f"  [扰动尺度] actor 参数 RMS = {rms:.5f}，"
                  f"σ={sigma} → 逐元素噪声 std = {sigma*rms:.5f}")
            for k in actor_keys:
                state[k] = state[k] + torch.randn_like(state[k]) * sigma * rms
        model.load_state_dict(state)
        return model, MAPPOPolicy(model, device)

    print("=" * 74)
    print("退化起点实验：BC 起点 0.8483 / 专家 0.8692（训练 regime，10 种子）")
    print(f"评测种子 {eval_seeds}（{len(eval_seeds)} 个），训练 {args.updates} 轮")
    print("=" * 74)
    print()

    summary = []
    for sigma in sigmas:
        print(f"----- 噪声 σ={sigma} -----")
        model, policy = build_policy(sigma, seed=42)
        sr0, _ = evaluate(policy, cfg, eval_seeds)
        print(f"  退化后（训练前）  success = {sr0:.4f}")

        # 训练
        env = build_env_from_config(cfg)
        out_dir = Path(f"/tmp/degraded_s{sigma}")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(42)
        trainer = MAPPOTrainer(env, policy, cfg, out_dir, device=device)
        curve = [sr0]
        for u in range(args.updates):
            buf = trainer.collect_rollout()
            stats = trainer.update(buf)
            sr, _ = evaluate(policy, cfg, eval_seeds)
            curve.append(sr)
            print(f"  u{u+1}: success={sr:.4f}  kl={stats.kl:.5f} "
                  f"|A|={stats.mean_abs_advantage:.4f} entropy={stats.entropy:.3f}")
        print(f"  → 曲线 {[round(c,4) for c in curve]}")
        delta = curve[-1] - curve[0]
        recovered = curve[-1] - sr0
        print(f"  → 退化后 {curve[0]:.4f} → 训练后 {curve[-1]:.4f}  "
              f"（回升 {recovered:+.4f}）")
        best = max(curve)
        print(f"  → 峰值 {best:.4f}（相对退化起点 {best-curve[0]:+.4f}）")
        summary.append((sigma, curve[0], curve[-1], best))
        print()

    print("=" * 74)
    print("判读")
    print("=" * 74)
    print(f"  {'σ':>6} {'退化后':>9} {'训练后':>9} {'回升':>9} {'峰值':>9}")
    for sigma, c0, c1, best in summary:
        print(f"  {sigma:>6} {c0:>9.4f} {c1:>9.4f} {c1-c0:>+9.4f} {best:>9.4f}")
    print()
    any_recovers = any(c1 - c0 > 0.02 or best - c0 > 0.02 for _, c0, c1, best in summary)
    if any_recovers:
        print("  有回升 → 算法能把参数推向更高回报（解释甲）")
        print("  即：训练'无效'是因为 BC 起点已近最优，不是更新信号坏了。")
    else:
        print("  无回升 → 更新信号无法把参数推向更高回报（解释乙）")
        print("  即：算法本身是坏的，必须改更新信号。")


if __name__ == "__main__":
    main()
