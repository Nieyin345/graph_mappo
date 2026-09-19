#!/usr/bin/env python
"""决定性探针：在**训练 regime**（days 0-295，1440 步）上跑专家 vs RL。

背景：24 个 run 的训练成功率全部冻结在 0.855-0.866（r=1.0，25 轮以上）。
权重在动（actor 每轮相对位移折合元素 RMS ≈ lr）、KL≈0.001、行为不变。
两种解释要分开：

  (a) 训练 regime 已饱和 —— 0.857 就是天花板，专家也在顶上，
      策略不动是**对的**，该去别处找收益。
  (b) RL 钉在 BC 起点 —— 专家显著更高，只是 RL 学不上去。

做法：直接调用 `train_graph_mappo.build_config()`（与训练逐字段一致），
把专家 `PathScoreGreedy(phased=True)` 放进这个 env 跑。

用法：
  python .tmp/probe_train_regime.py --seeds 7,8,9,10,11,12,13,14,15,16
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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
from qkd_rl.baselines.path_greedy import PathScoreGreedy
from qkd_rl.baselines.serve_probe import ServeProbe
from qkd_rl.rl.algos.checkpoint import load_checkpoint
from qkd_rl.rl.algos.policy import MAPPOPolicy
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,8,9,10,11,12,13,14,15,16")
    ap.add_argument("--rl", default="outputs/r6_base/checkpoint_final.pt")
    ap.add_argument("--no-rl", action="store_true")
    ap.add_argument("--out", default="/tmp/train_regime_probe.json")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)

    # 复用训练入口的配置装配，保证 env 逐字段一致
    tgm_args = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"],
        mode="random_episode", run_name=None, num_updates=1, seed=42,
        checkpoint=None, device="cpu")
    cfg = tgm.build_config(tgm_args)

    print("=== 训练 regime（来自 train_graph_mappo.build_config）===")
    env_cfg = cfg["env"]
    print(f"  activation_window = {env_cfg.get('activation_window_start_day')}"
          f" .. {env_cfg.get('activation_window_end_day')}")
    print(f"  episode_steps     = {env_cfg.get('episode_steps')}")
    print(f"  start_mode        = {env_cfg.get('episode_start_mode')}"
          f"  start_day={env_cfg.get('episode_start_day')}")
    print(f"  time_limit.days   = {cfg.get('scenario', {}).get('time_limit', {}).get('days')}")
    print(f"  env_seed          = {cfg['seed'].get('env_seed')}")
    print()

    expert_rows = []
    print("=== 专家 PathScoreGreedy(phased=True) 在训练 regime ===")
    t0 = time.perf_counter()
    for seed in seeds:
        env = build_env_from_config(cfg)
        obs = env.reset(seed=seed)
        expert = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True,
                                 principles=False, router=ServeProbe(env))
        served = 0.0
        n_step = 0
        done = False
        while not done:
            actions, scores = expert.act(obs)
            obs, _r, terminated, truncated, info = env.step(actions, scores)
            served += float(info.get("served_keys", 0.0))
            n_step += 1
            done = terminated or truncated
        s = env.metrics.episode_summary()
        arr = float(s.get("arrived_keys", 0.0))
        sr = served / arr if arr else 0.0
        expert_rows.append((seed, sr, served, arr, n_step))
        print(f"  seed {seed:>3}  专家 success {sr:.4f}  served {served:>14,.0f}"
              f"  arrived {arr:>14,.0f}  步数 {n_step}")
    n = len(expert_rows)
    e_mean = sum(r[1] for r in expert_rows) / n
    print(f"\n  专家 mean = {e_mean:.4f}   "
          f"min {min(r[1] for r in expert_rows):.4f}  "
          f"max {max(r[1] for r in expert_rows):.4f}   "
          f"耗时 {time.perf_counter()-t0:.0f}s")
    print()

    rl_mean = None
    if not args.no_rl and Path(ROOT / args.rl).exists():
        print(f"=== RL {args.rl} 在训练 regime ===")
        device = "cpu"
        tmpl = build_env_from_config(cfg)
        model = GraphMAPPOActorCritic(tmpl.action_resolver.action_space, cfg)
        rl = MAPPOPolicy(model, device)
        data = load_checkpoint(str(ROOT / args.rl), device)
        model.load_state_dict(data.model_state)
        model.eval()
        print(f"  checkpoint update={data.update}")
        rl_rows = []
        with torch.no_grad():
            for seed in seeds:
                env = build_env_from_config(cfg)
                obs = env.reset(seed=seed)
                served = 0.0
                done = False
                while not done:
                    step = rl.act(obs, deterministic=True)
                    obs, _r, terminated, truncated, info = env.step(
                        step.actions, step.action_scores,
                        edge_scores=step.edge_scores,
                        expected_matched_edges=list(step.matched_edges or []))
                    served += float(info.get("served_keys", 0.0))
                    done = terminated or truncated
                s = env.metrics.episode_summary()
                arr = float(s.get("arrived_keys", 0.0))
                sr = served / arr if arr else 0.0
                rl_rows.append((seed, sr))
                print(f"  seed {seed:>3}  RL success {sr:.4f}  served {served:>14,.0f}")
        rl_mean = sum(r[1] for r in rl_rows) / len(rl_rows)
        print(f"\n  RL mean = {rl_mean:.4f}")
        print(f"  Δ(专家 - RL) = {e_mean - rl_mean:+.4f}")
        print()
        print("  判读：")
        if rl_mean is not None and e_mean - rl_mean < 0.02:
            print("    专家 ≈ RL → 训练 regime 已饱和（解释 a）。")
            print("    收益要去别处找：评测 regime / 更大规模 / 别的指标。")
        else:
            print(f"    专家高出 {e_mean - rl_mean:+.4f} → RL 有上升空间（解释 b），")
            print("    RL 钉在 BC 起点附近，问题在更新信号。")

    Path(args.out).write_text(json.dumps(
        {"expert": expert_rows, "rl_mean": rl_mean, "expert_mean": e_mean}),
        encoding="utf-8")


if __name__ == "__main__":
    main()
