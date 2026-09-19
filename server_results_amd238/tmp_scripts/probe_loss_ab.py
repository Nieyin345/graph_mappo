#!/usr/bin/env python
"""严格 A/B：旧版 vs 新版 _loss_for_batch 的 update 耗时。

为什么必须 A/B 而不是比历史：历史 143s（r8_base，机器空闲）与现在 39s
（5 个训练并发）不可比 —— 负载不同。chunk 扫描就是被这个坑误导过。

做法：用 importlib 把 git HEAD 版的 mappo_trainer.py 作为**独立模块**加载
（模块名不同，避免覆盖），与当前工作区版并存，同一份 buffer、同一权重、
**交替**跑（A,B,A,B...），偏移被均摊。

前置：先把 HEAD 版导出到 /tmp/mappo_trainer_head.py
  cd /opt/qkd/graph_mappo && git show HEAD:qkd_rl/rl/algos/mappo_trainer.py > /tmp/mappo_trainer_head.py

用法：
  OMP_NUM_THREADS=4 python .tmp/probe_loss_ab.py --episodes 2 --reps 3
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
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic

import qkd_rl.rl.algos.mappo_trainer as new_mod


def load_head(path: Path):
    spec = importlib.util.spec_from_file_location("mappo_head", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mappo_head"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="/tmp/mappo_trainer_head.py")
    ap.add_argument("--steps", type=int, default=1440)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    head_path = Path(args.head)
    if not head_path.exists():
        print(f"缺 {head_path}，先执行：\n"
              f"  cd /opt/qkd/graph_mappo && git show HEAD:qkd_rl/rl/algos/"
              f"mappo_trainer.py > {head_path}")
        sys.exit(1)

    head_mod = load_head(head_path)

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

    out = Path("/tmp/lossab"); out.mkdir(parents=True, exist_ok=True)
    n_expect = args.episodes * args.steps
    print(f"OMP={os.environ.get('OMP_NUM_THREADS')} torch={torch.get_num_threads()}"
          f"  {n_expect} 步/轮  minibatch={cfg['train']['ppo']['minibatch_size']}")
    print("A = HEAD(旧，逐标量累加)   B = 工作区(新，stack·mean)\n", flush=True)

    collector = new_mod.MAPPOTrainer(env, policy, cfg, out, device="cpu")
    buf = collector.collect_rollout()
    n = len(buf.steps)
    print(f"buffer {n} 条\n", flush=True)

    def make(mod):
        return mod.MAPPOTrainer(env, policy, cfg, out, device="cpu")

    tA, tB = [], []
    # 热身
    for mod in (head_mod, new_mod):
        model.load_state_dict({k: v.clone() for k, v in base.items()})
        make(mod).update(buf)
    print("热身完成\n", flush=True)

    for r in range(args.reps):
        for tag, mod, acc in (("A_old", head_mod, tA), ("B_new", new_mod, tB)):
            model.load_state_dict({k: v.clone() for k, v in base.items()})
            t0 = time.perf_counter()
            make(mod).update(buf)
            dt = time.perf_counter() - t0
            acc.append(dt)
            print(f"  rep{r} {tag} {dt:7.1f}s  {dt/n*1000:6.2f} ms/步",
                  flush=True)

    ma, mb = min(tA), min(tB)
    print(f"\n=== 汇总 ===")
    print(f"  旧(min) {ma:6.1f}s   median {statistics.median(tA):6.1f}s")
    print(f"  新(min) {mb:6.1f}s   median {statistics.median(tB):6.1f}s")
    if mb < ma * 0.97:
        print(f"\n→ 重构快 {ma/mb:.3f}x，省 {ma-mb:.1f}s/轮"
              f"（每轮约 {250:.0f}s 的 {(ma-mb)/250*100:.1f}%）")
    elif mb > ma * 1.03:
        print(f"\n→ 重构反而慢 {mb/ma:.3f}x！应回滚。")
    else:
        print(f"\n→ 差异在 ±3% 内，测不出收益；重构只是等价改写，不宣称提速。")


if __name__ == "__main__":
    main()
