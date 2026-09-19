#!/usr/bin/env python
"""扫描 batch_chunk：测「一次完整 update」的墙钟，找到图的算子规模拐点。

背景（2026-09-19 实测）：
  update() 里 batch_chunk=64 是 minibatch_size=256 的**内层**切分，
  每个 minibatch 被切成 4 块独立做 block-diagonal 前向。cProfile 显示
  backward 的 tottime 是 forward linear 的 3.5 倍 —— autograd 引擎在
  遍历海量小算子。

若把 chunk 提到 256（=minibatch），算子数量减到 1/4 而单算子变大，
CPU 上通常显著更快。数学不变：block-diagonal 前向是逐图独立的，
切几刀不改变每图的 logits。

用法（务必固定线程数，与训练一致）：
  OMP_NUM_THREADS=4 python .tmp/probe_chunk_sweep.py --chunks 64,128,256
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
    ap.add_argument("--chunks", default="64,128,256")
    ap.add_argument("--steps", type=int, default=1440)
    ap.add_argument("--episodes", type=int, default=2,
                    help="环境局数；2 局 = 12 个 minibatch，够分辨相对快慢")
    ap.add_argument("--reps", type=int, default=1)
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
    cfg["train"]["episodes_per_update"] = args.episodes
    cfg["train"]["rollout_batch_envs"] = args.episodes

    base = load_checkpoint(
        str(ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"),
        "cpu").model_state

    env = build_env_from_config(cfg)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
    policy = MAPPOPolicy(model, "cpu")
    model.load_state_dict({k: v.clone() for k, v in base.items()})

    out = Path("/tmp/chunksweep"); out.mkdir(parents=True, exist_ok=True)

    mb = int(cfg["train"]["ppo"]["minibatch_size"])
    print(f"minibatch_size={mb}  episodes={args.episodes}  steps={args.steps}")
    print(f"线程 OMP={os.environ.get('OMP_NUM_THREADS')} "
          f"torch={torch.get_num_threads()}")
    print(f"扫描 chunk: {chunks}（max 有效值 = minibatch_size = {mb}）\n",
          flush=True)

    print("收集一份固定 rollout 作为输入...", flush=True)
    tr0 = MAPPOTrainer(env, policy, cfg, out, device="cpu")
    t0 = time.perf_counter()
    buf = tr0.collect_rollout()
    n = len(buf.steps)
    print(f"  {n} 条，rollout {time.perf_counter()-t0:.1f}s\n", flush=True)

    rows = []
    for c in chunks:
        if c > mb:
            print(f"  chunk={c} 超过 minibatch_size={mb}，无效（内层循环"
                  f"不会被触发），跳过")
            continue
        ts = []
        for _ in range(args.reps):
            # 每档从同一权重出发，避免比的是不同模型
            model.load_state_dict({k: v.clone() for k, v in base.items()})
            cfg2 = copy.deepcopy(cfg)
            cfg2["train"]["ppo"]["batch_chunk"] = c
            tr = MAPPOTrainer(env, policy, cfg2, out, device="cpu")
            t0 = time.perf_counter()
            tr.update(buf)
            ts.append(time.perf_counter() - t0)
        best = min(ts)
        rows.append((c, best))
        print(f"  chunk={c:<4} update {best:7.1f}s   "
              f"{best/max(1,n)*1000:.2f} ms/步", flush=True)

    if not rows:
        return
    base_chunk = int(cfg["train"]["ppo"].get("batch_chunk", 64))
    base_t = next((t for cc, t in rows if cc == base_chunk), None)
    print("\n=== 汇总（按耗时升序）===")
    fastest = min(t for _, t in rows)
    for c, t in sorted(rows, key=lambda r: r[1]):
        mark = "  ← 当前" if c == base_chunk else ""
        gain = f"  {base_t/t:.2f}x vs 当前" if base_t and c != base_chunk else ""
        print(f"  chunk={c:<4} {t:7.1f}s{gain}{mark}")
    if base_t:
        bestc = min(rows, key=lambda r: r[1])
        if bestc[1] < base_t * 0.95:
            print(f"\n→ 换 chunk={bestc[0]} 可省 {base_t-bestc[1]:.1f}s/轮 "
                  f"({(1-bestc[1]/base_t)*100:.0f}%)")
        else:
            print(f"\n→ 当前 chunk={base_chunk} 已在拐点附近，换它意义不大")


if __name__ == "__main__":
    main()
