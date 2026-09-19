#!/usr/bin/env python
"""测 PPO loss 重构（累加→stack·mean）的实际加速。

先做等价性验证（.tmp/verify_loss_refactor.py，已确认梯度逐位相同），
再在这里量速。用交替测量抵消外部负载漂移（当时有 5 个训练在跑，
一次性扫描的结果会被污染 —— chunk 扫描就是这么误判的）。

因为重构已经把新实现装进 mappo_trainer.py，本脚本测不了"旧实现"；
改为测**总 update 时间**并与历史同配置的 update_s 对照：
  r8_base_s42 首轮 update_s=143（OMP=4，机器空闲，ent=0.001）
  ent01_* 首轮 update_s=150~157（OMP=4，5 并发）
若重构后单轮 update 明显低于这些，即为收益。

用法：
  OMP_NUM_THREADS=4 python .tmp/probe_after_refactor.py --episodes 2 --reps 2
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1440)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--reps", type=int, default=2)
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
    cfg["train"]["n_rollout_workers"] = min(8, args.episodes)

    base = load_checkpoint(
        str(ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"),
        "cpu").model_state

    env = build_env_from_config(cfg)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
    policy = MAPPOPolicy(model, "cpu")
    model.load_state_dict({k: v.clone() for k, v in base.items()})

    out = Path("/tmp/afterref"); out.mkdir(parents=True, exist_ok=True)

    mb = int(cfg["train"]["ppo"]["minibatch_size"])
    n_expect = args.episodes * args.steps
    print(f"OMP={os.environ.get('OMP_NUM_THREADS')} torch={torch.get_num_threads()}"
          f"  minibatch={mb}  episodes={args.episodes} → {n_expect} 步/轮")
    print(f"minibatch 数/轮 = {n_expect/mb:.1f}\n", flush=True)

    trainer = MAPPOTrainer(env, policy, cfg, out, device="cpu")
    buf = trainer.collect_rollout()
    n = len(buf.steps)
    print(f"buffer {n} 条\n", flush=True)

    ts = []
    for r in range(args.reps):
        model.load_state_dict({k: v.clone() for k, v in base.items()})
        t0 = time.perf_counter()
        trainer.update(buf)
        dt = time.perf_counter() - t0
        ts.append(dt)
        print(f"  rep{r} update {dt:6.1f}s   {dt/n*1000:.2f} ms/步", flush=True)

    mn, md = min(ts), statistics.median(ts)
    print(f"\n重构后 update: min {mn:.1f}s  median {md:.1f}s")
    print(f"\n历史对照（同 OMP=4、同 8 局 1440 步，机器近似空闲）：")
    print(f"  r8_base_s42  首轮 update_s=143  ({143/n*1000:.2f} ms/步)")
    print(f"  r8_base_s43  首轮 update_s=145")
    print(f"  r8_base_s44  首轮 update_s=143")
    if md < 143 * 0.95:
        print(f"\n→ 重构有效：{md:.0f}s vs 143s，省 {143-md:.0f}s/轮 "
              f"({(1-md/143)*100:.0f}%)")
    else:
        print(f"\n→ 与历史 143s 无显著差别（{md:.0f}s）。可能原因：本次机器上"
              f"有其它负载；或在当前规模下 Python 循环开销不是瓶颈。")


if __name__ == "__main__":
    main()
