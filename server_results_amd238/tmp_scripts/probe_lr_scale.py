#!/usr/bin/env python
"""检验「学习率小了一个数量级」这个假设。

量级依据（都是训练 regime，相对 L2 位移）：
  BC起点 → r6_base u20   actor 累计位移 = 4.2e-02
  BC起点 + σ=0.3×RMS 噪声  → 行为几乎不变（0.8483 → 0.8551）
     该扰动的相对 L2 就是 0.3

即：**要改变行为需要动 ~30%，而 20 轮训练只动了 4.2%，差 7 倍。**

若假设成立，把 actor_lr 放大 10 倍应当在少数几轮内就产生**可测的行为变化**
（无论变好变坏——先看它是否动，再看方向）。

对照组用同一份 rollout 数据的分支做不到，所以用**同种子、同轮数**跑不同 lr：
  lr ×1（=3e-4，基线）/ ×3 / ×10 / ×30
每个臂评测同一批种子，看成功率是否离开 BC 起点的 0.8483。

判读：
  - 若高 lr 让成功率明显偏离 0.848 且方向向上 → 假设成立，该调 lr
  - 若高 lr 只是让成功率下降 → 步长不是问题，方向才是
  - 若高 lr 也不动 → 策略对该方向真的不敏感，得改目标/表示

用法：
  python .tmp/probe_lr_scale.py --mult 1,3,10,30 --updates 4
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


def rel_l2(model, base_state, prefix_filter):
    """相对 L2 位移，只看 actor（含 encoder）或只看 critic。"""
    num = 0.0
    den = 0.0
    for k, v in model.state_dict().items():
        if k not in base_state or not v.is_floating_point():
            continue
        is_critic = k.startswith("critic")
        if prefix_filter == "actor" and is_critic:
            continue
        if prefix_filter == "critic" and not is_critic:
            continue
        num += float((v.float() - base_state[k].float()).pow(2).sum())
        den += float(base_state[k].float().pow(2).sum())
    return (num / den) ** 0.5 if den > 0 else 0.0


def evaluate(policy, cfg, seeds):
    srs = []
    with torch.no_grad():
        for seed in seeds:
            env = build_env_from_config(cfg)
            obs = env.reset(seed=seed)
            served = 0.0
            done = False
            while not done:
                st = policy.act(obs, deterministic=True)
                obs, _r, term, trunc, info = env.step(
                    st.actions, st.action_scores,
                    edge_scores=st.edge_scores,
                    expected_matched_edges=list(st.matched_edges or []))
                served += float(info.get("served_keys", 0.0))
                done = term or trunc
            s = env.metrics.episode_summary()
            arr = float(s.get("arrived_keys", 0.0))
            srs.append(served / arr if arr else 0.0)
    return sum(srs) / len(srs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mult", default="1,3,10,30")
    ap.add_argument("--updates", type=int, default=4)
    ap.add_argument("--eval-seeds", default="7,8,9,10,11,12")
    args = ap.parse_args()

    mults = [float(m) for m in args.mult.split(",")]
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

    BASE_ACTOR_LR = float(cfg["train"]["optimizer"]["actor_lr"])
    BASE_CRITIC_LR = float(cfg["train"]["optimizer"]["critic_lr"])
    print(f"基线 actor_lr={BASE_ACTOR_LR}  critic_lr={BASE_CRITIC_LR}")
    print(f"每臂 {args.updates} 轮，评测 {len(eval_seeds)} 个种子")
    print(f"参照：BC 起点 0.8483 / 专家 0.8692（训练 regime）\n")

    base_state = load_checkpoint(
        str(ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"),
        "cpu").model_state

    rows = []
    for mult in mults:
        torch.manual_seed(42)
        env = build_env_from_config(cfg)
        model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
        policy = MAPPOPolicy(model, "cpu")
        model.load_state_dict({k: v.clone() for k, v in base_state.items()})
        cfg["train"]["optimizer"]["actor_lr"] = BASE_ACTOR_LR * mult
        cfg["train"]["optimizer"]["critic_lr"] = BASE_CRITIC_LR * mult

        out_dir = Path(f"/tmp/lrscale_{mult}")
        out_dir.mkdir(parents=True, exist_ok=True)
        trainer = MAPPOTrainer(env, policy, cfg, out_dir, device="cpu")

        sr0 = evaluate(policy, cfg, eval_seeds)
        last_kl = 0.0
        for u in range(args.updates):
            buf = trainer.collect_rollout()
            st = trainer.update(buf)
            last_kl = st.kl
        sr1 = evaluate(policy, cfg, eval_seeds)
        d_actor = rel_l2(model, base_state, "actor")
        d_crit = rel_l2(model, base_state, "critic")
        print(f"  ×{mult:<5} lr={BASE_ACTOR_LR*mult:<9.2e}  "
              f"{sr0:.4f} → {sr1:.4f}  (Δ{sr1-sr0:+.4f})  "
              f"未轮kl={last_kl:.5f}  ‖Δactor‖/‖w‖={d_actor:.4f} "
              f"critic={d_crit:.4f}")
        rows.append((mult, sr0, sr1, d_actor))

    print("\n=== 汇总 ===")
    print(f"  {'倍数':>6} {'位移(actor)':>12} {'起点':>8} {'终点':>8} {'Δ':>9}")
    for mult, s0, s1, da in rows:
        print(f"  {mult:>6.0f} {da:>12.4f} {s0:>8.4f} {s1:>8.4f} {s1-s0:>+9.4f}")
    print()
    print("  参照：BC起点 0.8483，专家 0.8692")
    moved = [r for r in rows if abs(r[2] - r[1]) > 0.02]
    if moved:
        best = max(rows, key=lambda r: r[2])
        print(f"  有臂产生可测变化（|Δ|>0.02）→ 步长确实是瓶颈之一")
        print(f"  最佳臂 ×{best[0]:.0f} → {best[2]:.4f}")
    else:
        print("  所有臂的变化都在噪声内 → 步长不是瓶颈，方向/目标才是")


if __name__ == "__main__":
    main()
