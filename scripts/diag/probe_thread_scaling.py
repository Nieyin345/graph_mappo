#!/usr/bin/env python
"""线程数扫描：update 段（一轮的 75%）到底吃不吃线程？

### 为什么要问这个

CLAUDE.md 的并行判据是 **内存**，不是核数——这条是对的，每个 run 23.3 GB，
125 GB 机器只能并发 3。但**"并发数受内存限制"推不出"单 run 该用几个线程"**：

    3 个 run × OMP_NUM_THREADS=4 = **12 线程**，而机器是 **32 核**
    ⟹ 20 个核完全空闲（实测 load average 7.4，nproc 32）

而日志 3740 行的 update 剖面说：`run_backward` 57% + 前向 41% = **98% 是模型计算**，
没有框架开销可挤。**矩阵乘是线程扩展性最好的负载**，而这里恰好是纯 matmul + backward。

CLAUDE.md 已经把"加线程数"从直觉里排除掉了，但排除它的论据是
**"并发本身不吃速度"（12.41 ms/步，5 并发与空闲完全相同）**——
那个测量说的是**同时跑多个 run 时每个 run 不被拖慢**，
**不是**"单个 run 增加线程没用"。这是两条不同的命题，前者已被证实，
后者**从未测过**。

### 但线程数会改变训练结果，这是硬约束

记忆 `thread-count-changes-training`：`OMP_NUM_THREADS` 对训练结果有**确定性**
影响（差 0.018），所以 A/B 必须固定线程数。因此本探针**只量耗时，不下"该改"的
结论**——真要改，必须整批重跑，或用"同线程数对照"重新建立基线。

### 做法

同一份 buffer、同一份权重，只改线程数，**交错**跑（4,8,16,32,4,8,16,32...），
让机器漂移均摊到各线程数上。判据是每步毫秒数的**组内极差**，不是绝对值。

用法（服务器上）：
    cd /opt/qkd/graph_mappo && /opt/qkd/venv/bin/python scripts/diag/probe_thread_scaling.py
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("QKD_ROOT", "/opt/qkd/graph_mappo"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402

_tgm_spec = importlib.util.spec_from_file_location(
    "_tgm", ROOT / "scripts" / "train" / "train_graph_mappo.py")
tgm = importlib.util.module_from_spec(_tgm_spec)
_tgm_spec.loader.exec_module(tgm)

from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.rl.algos.checkpoint import load_checkpoint  # noqa: E402
from qkd_rl.rl.algos.policy import MAPPOPolicy  # noqa: E402
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic  # noqa: E402
import qkd_rl.rl.algos.mappo_trainer as trainer_mod  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1440)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--threads", default="4,8,16,32")
    args = ap.parse_args()

    thread_list = [int(x) for x in args.threads.split(",")]

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
    # 单进程收集：本探针只量 update 段，不需要 8 个 worker（那会多占 9 GB）
    cfg["train"]["n_rollout_workers"] = 1

    base = load_checkpoint(
        str(ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"),
        "cpu").model_state

    print("=" * 78)
    print(f"机器 nproc={os.cpu_count()}  收集 {args.episodes} 集 × {args.steps} 步")
    print(f"线程扫描 {thread_list}  × {args.reps} 轮（交错，漂移均摊）")
    print("=" * 78, flush=True)

    torch.set_num_threads(thread_list[0])
    env = build_env_from_config(cfg)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
    policy = MAPPOPolicy(model, "cpu")
    model.load_state_dict({k: v.clone() for k, v in base.items()})

    out = Path("/tmp/thr_scan")
    out.mkdir(parents=True, exist_ok=True)
    trainer = trainer_mod.MAPPOTrainer(env, policy, cfg, out, device="cpu")

    print("收集 buffer（单进程，可能要几分钟）...", flush=True)
    t0 = time.perf_counter()
    buf = trainer.collect_rollout()
    print(f"  {len(buf.steps)} 步，耗时 {time.perf_counter()-t0:.1f}s\n", flush=True)

    n = len(buf.steps)
    n_mb = (n + int(cfg["train"]["ppo"]["minibatch_size"]) - 1) // int(
        cfg["train"]["ppo"]["minibatch_size"])
    print(f"minibatch {cfg['train']['ppo']['minibatch_size']} ⟹ {n_mb} 个/轮\n",
          flush=True)

    results: dict[int, list[float]] = {t: [] for t in thread_list}

    # 热身：让惰性分配、内存池、BLAS 初始化都发生一次，不混进第一组读数
    print("热身（4 线程）...", flush=True)
    torch.set_num_threads(4)
    model.load_state_dict({k: v.clone() for k, v in base.items()})
    trainer.update(buf)
    print("  完成\n", flush=True)

    for r in range(args.reps):
        for nt in thread_list:
            torch.set_num_threads(nt)
            model.load_state_dict({k: v.clone() for k, v in base.items()})
            t0 = time.perf_counter()
            trainer.update(buf)
            dt = time.perf_counter() - t0
            results[nt].append(dt)
            print(f"  rep{r}  threads={nt:<3} {dt:7.1f}s   "
                  f"{dt/n*1000:6.2f} ms/步", flush=True)

    print()
    print("=" * 78)
    print("汇总（取每组的 min：min 最接近无干扰的那一次）")
    print("=" * 78)
    print(f"  {'线程':>5}{'min(s)':>10}{'ms/步':>10}{'相对4线程':>12}{'组内极差':>12}")
    ref = None
    for nt in thread_list:
        v = results[nt]
        mn = min(v)
        if ref is None:
            ref = mn
        spread = (max(v) - min(v)) / mn * 100
        print(f"  {nt:>5}{mn:>10.1f}{mn/n*1000:>10.2f}{ref/mn:>11.2f}x{spread:>11.1f}%")

    print()
    base_t = min(results[thread_list[0]])
    best_nt = min(thread_list, key=lambda t: min(results[t]))
    best_t = min(results[best_nt])
    print(f"  最快 = {best_nt} 线程，比 4 线程快 {base_t/best_t:.2f}x")
    # 一轮 = rollout + update；update 占 75%（日志 3740 行的剖面）
    print(f"  若 update 占一轮 75%：整轮提速 "
          f"{1/(0.25 + 0.75*best_t/base_t):.2f}x")
    print()
    print("  ⚠ 本探针只量耗时。记忆 thread-count-changes-training 已证")
    print("     OMP_NUM_THREADS 会**确定性改变训练结果**（差 0.018），")
    print("     所以任何线程数的改动都需要整批重跑或重建同线程数基线。")
    print("=" * 78)


if __name__ == "__main__":
    main()
