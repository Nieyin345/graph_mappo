#!/usr/bin/env python
"""把一次 update 拆成 forward / loss 构图 / backward 三段，定位图节点从哪来。

动机：cProfile 显示 run_backward 占 63%（34.7s/55s），而前向 linear 只有 9.7s。
backward 是 forward 的 3.5 倍，典型是 autograd 图节点过多（每步几百个小算子），
CPU 上遍历开销压过算术本身。

怀疑点：_loss_for_batch 的内层循环对**每个 step** 都做 5 次标量张量累加
（actor/critic/entropy/kl/ratio），2880 步 × 5 = 14400 个 add 节点。
若属实，改成收集到 list 再一次性 stack().mean() 能把节点数砍掉一个量级，
且数学等价（和的均值 = 均值的和）。

本脚本只测量、不改代码。

用法：
  python .tmp/probe_loss_breakdown.py --steps 1440 --episodes 2
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


def count_graph_nodes(t):
    """数一个张量的 autograd 图里有多少个节点。"""
    seen, stack, n = set(), [t.grad_fn], 0
    while stack:
        fn = stack.pop()
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        n += 1
        for nxt, _ in fn.next_functions:
            if nxt is not None:
                stack.append(nxt)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1440)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=None)
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
    if args.chunk:
        cfg["train"]["ppo"]["batch_chunk"] = args.chunk

    base = load_checkpoint(
        str(ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"),
        "cpu").model_state

    env = build_env_from_config(cfg)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
    policy = MAPPOPolicy(model, "cpu")
    model.load_state_dict({k: v.clone() for k, v in base.items()})

    out = Path("/tmp/lossbd"); out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, cfg, out, device="cpu")

    mb = int(cfg["train"]["ppo"]["minibatch_size"])
    ck = int(cfg["train"]["ppo"].get("batch_chunk", 64))
    print(f"episodes={args.episodes} steps={args.steps} minibatch={mb} chunk={ck}",
          flush=True)
    print("收集 rollout...", flush=True)
    buf = trainer.collect_rollout()
    n = len(buf.steps)
    print(f"  {n} 条\n", flush=True)

    ppo = cfg["train"]["ppo"]
    clip_eps = float(ppo["clip_eps"]); ent_c = float(ppo["entropy_coef"])
    val_c = float(ppo["value_coef"])
    normalize_adv = bool(ppo.get("normalize_advantages", True))

    # 取第一个 minibatch 做分解
    gen = buf.sample(mb, trainer.rng)
    batch = next(iter(gen))
    print(f"取一个 minibatch：{len(batch)} 步\n")

    model.train()

    # --- 1) 纯前向（batched_forward + 匹配概率），不建 loss 图 ---
    t0 = time.perf_counter()
    res = policy.evaluate_actions_batched(
        [s.obs for s in batch],
        [s.actions for s in batch],
        [list(s.matched_edges or []) for s in batch],
    )
    t_fwd = time.perf_counter() - t0
    print(f"1) 前向（batched_forward + 匹配概率）: {t_fwd:6.2f}s")

    # --- 2) 加上 loss 构图 ---
    t0 = time.perf_counter()
    a_loss, c_loss, e_m, kl_m, r_m = trainer._loss_for_batch(
        batch, clip_eps=clip_eps, entropy_coef=ent_c,
        value_coef=val_c, normalize_adv=normalize_adv,
    )
    loss = a_loss + c_loss - ent_c * e_m
    t_loss = time.perf_counter() - t0
    print(f"2) + loss 构图:                        {t_loss:6.2f}s")

    nodes = count_graph_nodes(loss)
    print(f"   loss 的 autograd 图节点数: {nodes}  "
          f"（{nodes/len(batch):.1f} 节点/步）")

    # --- 3) backward ---
    trainer.optimizer.zero_grad()
    t0 = time.perf_counter()
    loss.backward()
    t_bwd = time.perf_counter() - t0
    print(f"3) backward:                           {t_bwd:6.2f}s")

    tot = t_fwd + t_loss + t_bwd
    print(f"\n合计 {tot:.2f}s  →  前向 {t_fwd/tot*100:.0f}%  "
          f"构图 {t_loss/tot*100:.0f}%  backward {t_bwd/tot*100:.0f}%")

    # 每步节点数拆解：单独测一个 step 的 loss 图
    one = batch[:1]
    a1, c1, e1, k1, r1 = trainer._loss_for_batch(
        one, clip_eps=clip_eps, entropy_coef=ent_c,
        value_coef=val_c, normalize_adv=False,
    )
    print(f"\n单步 loss 图节点数: {count_graph_nodes(a1 + c1)}")
    print(f"单步 actor 项节点数: {count_graph_nodes(a1)}")
    print(f"单步 critic 项节点数: {count_graph_nodes(c1)}")


if __name__ == "__main__":
    main()
