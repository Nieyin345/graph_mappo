#!/usr/bin/env python
"""剖面：一次 update() 的 175s 花在哪。

为什么要它：训练每轮 = rollout ~60s + update ~175s。update 占 3/4，
但此前只优化过 `_loss_for_batch`（且实测无提速）。**先测再改**——
上次正是没测就宣布"快 3.7x"，被同负载 A/B 打掉。

用 cProfile 按**累积时间**排序，并单独报出 autograd 引擎
（`torch/autograd`）与 GNN 前向的占比。

用法：
  OMP_NUM_THREADS=4 python .tmp/probe_update_profile.py --episodes 1 --steps 1440
"""
from __future__ import annotations

import argparse
import cProfile
import importlib.util
import io
import os
import pstats
import sys
import time
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402

from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.rl.algos.checkpoint import load_checkpoint  # noqa: E402
from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer  # noqa: E402
from qkd_rl.rl.algos.policy import MAPPOPolicy  # noqa: E402
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1440)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--top", type=int, default=28)
    args = ap.parse_args()

    _spec = importlib.util.spec_from_file_location(
        "_tgm", ROOT / "scripts" / "train" / "train_graph_mappo.py")
    tgm = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(tgm)

    tgm_args = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"],
        mode="random_episode", run_name=None, num_updates=1,
        seed=42, checkpoint=None, device="cpu")
    cfg = tgm.build_config(tgm_args)
    cfg["runtime"]["device"] = "cpu"
    cfg["env"]["episode_steps"] = args.steps
    cfg["train"]["episode_steps_fixed"] = True
    cfg["train"]["episodes_per_update"] = args.episodes
    cfg["train"]["rollout_batch"] = False
    cfg["train"]["n_rollout_workers"] = 1

    base = load_checkpoint(
        str(ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"),
        "cpu").model_state
    env = build_env_from_config(cfg)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
    policy = MAPPOPolicy(model, "cpu")
    model.load_state_dict({k: v.clone() for k, v in base.items()})

    out = Path("/tmp/updprof")
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, cfg, out, device="cpu")

    print(f"OMP={os.environ.get('OMP_NUM_THREADS')} torch={torch.get_num_threads()}"
          f"  minibatch={cfg['train']['ppo']['minibatch_size']}"
          f"  chunk={cfg['train']['ppo'].get('batch_chunk')}", flush=True)

    buf = trainer.collect_rollout()
    n = len(buf.steps)
    print(f"buffer {n} 条，minibatch={cfg['train']['ppo']['minibatch_size']}"
          f" → {n // cfg['train']['ppo']['minibatch_size']} 个 minibatch\n", flush=True)

    # 先热身一次（避免首次的 lazy alloc / 缓存构建混进剖面）
    t0 = time.perf_counter()
    trainer.update(buf)
    warm = time.perf_counter() - t0
    print(f"热身 update: {warm:.1f}s  ({warm/n*1000:.2f} ms/步)\n", flush=True)

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    trainer.update(buf)
    pr.disable()
    dt = time.perf_counter() - t0
    print(f"剖面 update: {dt:.1f}s  ({dt/n*1000:.2f} ms/步)\n", flush=True)

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(args.top)
    print("=== 按 tottime（自身耗时，不含子调用）===")
    print(s.getvalue())

    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("cumulative")
    ps2.print_stats(args.top)
    print("=== 按 cumulative（含子调用）===")
    print(s2.getvalue())

    # 汇总几大类
    print("=== 分类汇总（tottime）===")
    buckets = {"autograd引擎": 0.0, "GNN前向/线性": 0.0, "matching评估": 0.0,
               "其它": 0.0}
    for (fname, _lin, func), (cc, nc, tt, ct, _callers) in ps.stats.items():
        key = (fname, func)
        if "autograd" in fname:
            buckets["autograd引擎"] += tt
        elif "graph_mappo" in fname or "functional" in fname or "linear" in func:
            buckets["GNN前向/线性"] += tt
        elif "_matching" in func or "matching" in func:
            buckets["matching评估"] += tt
        else:
            buckets["其它"] += tt
    tot = sum(buckets.values()) or 1.0
    for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<16} {v:8.2f}s  {100*v/tot:5.1f}%")


if __name__ == "__main__":
    main()
