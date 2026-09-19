#!/usr/bin/env python
"""A/B：update 路径上的 batch_chunk / minibatch_size。

**为什么值得测**：剖面显示一次 update 里 backward 13.3s(57%) + forward 9.7s(41%)
= 98% 是模型计算，而 forward 的 9.7s 里 5.4s 花在 `torch._C._nn.linear`。
`batch_chunk=64` 把每个 256 步的 minibatch 拆成 4 次更小的 matmul
（实测 23 次 `evaluate_actions_batched` = 5×4+3，与 ceil(256/64) 吻合）。

而 `configs/rl_algorithm.yaml` 里 chunk=64 的调优记录是在**rollout**路径上测的
（无 autograd）。**更新路径从未测过** —— 那里的算术强度完全不同
（forward+backward、有 autograd 图），最优 chunk 不一定是 64。

这是**纯配置**改动（改 yaml 即可，不动代码），符合"先找现成的、只改配置"。

**纪律**（`scripts/diag/README.md`）：A/B 必须**同负载交替**测，取 min，
不能拿"现在"比"历史"——上次 chunk=128 的"快 1.23x"就是这么被骗出来的。

用法：
  OMP_NUM_THREADS=4 python .tmp/probe_chunk_ab.py --episodes 1 --reps 3
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
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--chunks", nargs="+", type=int,
                    default=[64, 128, 256, 512])
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

    mb = int(cfg["train"]["ppo"]["minibatch_size"])
    base = load_checkpoint(
        str(ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"),
        "cpu").model_state
    env = build_env_from_config(cfg)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
    policy = MAPPOPolicy(model, "cpu")
    model.load_state_dict({k: v.clone() for k, v in base.items()})

    out = Path("/tmp/chunkab")
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, cfg, out, device="cpu")
    buf = trainer.collect_rollout()
    n = len(buf.steps)
    n_mb = -(-n // mb)
    print(f"OMP={os.environ.get('OMP_NUM_THREADS')} torch={torch.get_num_threads()}"
          f"  buffer {n} 步  minibatch={mb} → {n_mb} 个 minibatch")
    print(f"chunk 候选 {args.chunks}；每个 minibatch 的 forward 次数 = "
          + ", ".join(f"{c}:{-(-mb//c)}" for c in args.chunks))
    print("同负载**交替**测，取 min。\n", flush=True)

    def run(chunk: int) -> float:
        cfg["train"]["ppo"]["batch_chunk"] = chunk
        model.load_state_dict({k: v.clone() for k, v in base.items()})
        t0 = time.perf_counter()
        trainer.update(buf)
        return time.perf_counter() - t0

    # 热身：每个 chunk 各一次，避免首次的惰性分配混进测量
    for c in args.chunks:
        run(c)
    print("热身完成\n", flush=True)

    times: dict[int, list[float]] = {c: [] for c in args.chunks}
    for r in range(args.reps):
        for c in args.chunks:            # 每轮内交替，负载波动对各方等同
            dt = run(c)
            times[c].append(dt)
            print(f"  rep{r} chunk={c:<4} {dt:7.1f}s  {dt/n*1000:6.2f} ms/步",
                  flush=True)

    print("\n=== 汇总（min / median）===")
    best = min(times, key=lambda c: min(times[c]))
    ref = 64
    for c in args.chunks:
        mn, md = min(times[c]), statistics.median(times[c])
        tag = ""
        if c == best:
            tag = "  ← 最快"
        if c != ref and ref in times:
            tag += f"   vs chunk=64: {min(times[ref])/mn:.3f}x"
        print(f"  chunk={c:<5} min {mn:7.1f}s ({mn/n*1000:5.2f} ms/步)"
              f"   median {md:7.1f}s{tag}")

    if best != ref:
        gain = min(times[ref]) - min(times[best])
        print(f"\n→ chunk={best} 比 {ref} 每轮省 {gain:.1f}s"
              f"（全天 45 minibatch 的真实训练里会放大 ~7.5x）")
    else:
        print(f"\n→ chunk={ref} 已是最优，无需改动")


if __name__ == "__main__":
    main()
