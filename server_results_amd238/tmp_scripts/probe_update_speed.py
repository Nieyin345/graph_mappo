#!/usr/bin/env python
"""实测 update 阶段的耗时分布，定位热点，并扫描 batch_chunk。

为什么值得测：每轮 rollout=46s 但 update=145s（占 75%）。若 chunk 拐点
已经移动（代码从写注释那版改过多次），把 64 换掉能直接省时间。

只测**更新阶段**（不含 rollout），因为 rollout 已经并行到 8 worker。
同一份 buffer 反复用不同 chunk 跑 update，比的是纯算子吞吐。

用法：
  python .tmp/probe_update_speed.py --chunks 32,64,96,128,256
"""
from __future__ import annotations

import argparse
import copy
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="32,64,96,128,256")
    ap.add_argument("--steps", type=int, default=1440)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    chunks = [int(c) for c in args.chunks.split(",")]

    tgm_args = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"],
        mode="random_episode", run_name=None, num_updates=1,
        seed=42, checkpoint=None, device="cpu")
    cfg = tgm.build_config(tgm_args)
    cfg["runtime"]["device"] = "cpu"
    cfg["env"]["episode_steps"] = args.steps
    cfg["train"]["episode_steps_fixed"] = True
    cfg["train"]["n_rollout_workers"] = args.workers

    base = load_checkpoint(
        str(ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"),
        "cpu").model_state

    print(f"rollout {args.steps} 步 × {cfg['train']['episodes_per_update']} 局，"
          f"{args.workers} workers")
    print(f"扫描 batch_chunk: {chunks}\n")

    env = build_env_from_config(cfg)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
    policy = MAPPOPolicy(model, "cpu")
    model.load_state_dict({k: v.clone() for k, v in base.items()})

    out = Path("/tmp/upspeed"); out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, cfg, out, device="cpu")

    print("收集一份 rollout 作为固定输入...")
    t0 = time.perf_counter()
    buf = trainer.collect_rollout()
    t_roll = time.perf_counter() - t0
    n = len(buf.steps)
    print(f"  rollout 耗时 {t_roll:.1f}s，buffer 内 {n} 条\n")

    rows = []
    for c in chunks:
        # 每档都从同一权重出发（否则比的是不同的模型）
        model.load_state_dict({k: v.clone() for k, v in base.items()})
        cfg2 = copy.deepcopy(cfg)
        cfg2["train"]["ppo"]["batch_chunk"] = c
        tr = MAPPOTrainer(env, policy, cfg2, out, device="cpu")
        # 用同一份 buffer，绕开重新采样
        t0 = time.perf_counter()
        tr.update(buf)
        dt = time.perf_counter() - t0
        rows.append((c, dt))
        print(f"  chunk={c:<4} update {dt:6.1f}s   "
              f"({dt/max(1,n):.4f} s/step)")

    best = min(rows, key=lambda r: r[1])
    base_chunk = int(cfg["train"]["ppo"].get("batch_chunk", 64))
    base_t = next((t for c, t in rows if c == base_chunk), None)
    print(f"\n=== 汇总 ===")
    for c, t in sorted(rows, key=lambda r: r[1]):
        mark = " ← 当前" if c == base_chunk else ""
        gain = ""
        if base_t and c != base_chunk:
            gain = f"  {base_t/t:.2f}x"
        print(f"  chunk={c:<4} {t:6.1f}s{gain}{mark}")
    print(f"\n最快 chunk={best[0]} ({best[1]:.1f}s)；"
          f"当前 {base_chunk} ({base_t:.1f}s)" if base_t else "")
    if base_t and best[1] < base_t * 0.95:
        print(f"  → 换 chunk={best[0]} 可省 {base_t-best[1]:.1f}s/轮 "
              f"({(1-best[1]/base_t)*100:.0f}%)")
    else:
        print("  → 当前 chunk 已在拐点附近，换它没有意义")


if __name__ == "__main__":
    main()
