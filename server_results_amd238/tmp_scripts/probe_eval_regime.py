#!/usr/bin/env python
"""验证 regime（days 330-365，240 步，15 个种子）上跑专家 vs RL。

与 probe_train_regime.py 配对：那一头测出训练 regime 已饱和
（专家 0.8692 vs RL 0.8549，差 0.014）。这一头看验证 regime 是否不同 ——
训练日志里 eval_validation 一直钉在 0.648，而专家在同样条件下的数是多少？

若专家在验证 regime 上明显更高 → 验证集上确有可学空间，0.648 是 RL 的
真实短板；若专家也就 0.65 左右 → 0.648 ≈ 天花板，评测 regime 同样饱和，
"提升成功率"这条路整体没有空间，目标函数该换。

用法：
  python .tmp/probe_eval_regime.py --seeds 100-114
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


def build_eval_env_config(cfg, episode_steps):
    """从训练合并配置里取出 validation 段，构造评测 env —— 与 trainer.evaluate 同源。"""
    val = cfg.get("validation", {}) or {}
    win = val.get("window", {}) or {}
    start_day = int(win.get("start_day", 330))
    end_day = int(win.get("end_day", 365))
    env_cfg = dict(cfg)
    env_cfg["env"] = dict(cfg["env"])
    env_cfg["env"]["episode_start_mode"] = str(val.get("start_mode", "random_day"))
    env_cfg["env"]["episode_steps"] = episode_steps
    env_cfg["env"]["activation_window_start_day"] = start_day
    env_cfg["env"]["activation_window_end_day"] = end_day
    env_cfg["env"]["activation_window_days"] = max(0, end_day - start_day)
    env_cfg["env"]["continuous"] = False
    import math
    env_cfg["scenario"] = json.loads(json.dumps(cfg.get("scenario", {})))
    env_cfg["scenario"].setdefault("time_limit", {})
    env_cfg["scenario"]["time_limit"]["days"] = end_day + max(
        1, math.ceil(episode_steps / 1440))
    return env_cfg, start_day, end_day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100,101,102,103,104,105,106,107,108,109,110,111,112,113,114")
    ap.add_argument("--rl", default="outputs/r6_base/checkpoint_final.pt")
    ap.add_argument("--out", default="/tmp/eval_regime_probe.json")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)

    tgm_args = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"],
        mode="random_episode", run_name=None, num_updates=1, seed=42,
        checkpoint=None, device="cpu")
    cfg = tgm.build_config(tgm_args)
    val = cfg.get("validation", {}) or {}
    steps = int(val.get("episode_steps", 240))
    env_cfg, sd, ed = build_eval_env_config(cfg, steps)

    print("=== 验证 regime（来自 train_full_rl.yaml 的 validation 段）===")
    print(f"  window        = {sd} .. {ed}")
    print(f"  episode_steps = {steps}")
    print(f"  start_mode    = {env_cfg['env']['episode_start_mode']}")
    print(f"  seeds         = {seeds[0]}..{seeds[-1]} (n={len(seeds)})")
    print()

    expert_rows = []
    print("=== 专家 PathScoreGreedy(phased=True) ===")
    t0 = time.perf_counter()
    for seed in seeds:
        env = build_env_from_config(env_cfg)
        obs = env.reset(seed=seed)
        expert = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True,
                                 principles=False, router=ServeProbe(env))
        served = 0.0
        done = False
        while not done:
            actions, scores = expert.act(obs)
            obs, _r, term, trunc, info = env.step(actions, scores)
            served += float(info.get("served_keys", 0.0))
            done = term or trunc
        s = env.metrics.episode_summary()
        arr = float(s.get("arrived_keys", 0.0))
        sr = served / arr if arr else 0.0
        expert_rows.append((seed, sr))
        print(f"  seed {seed:>3}  专家 {sr:.4f}")
    e_mean = sum(r[1] for r in expert_rows) / len(expert_rows)
    print(f"\n  专家 mean = {e_mean:.4f}   "
          f"min {min(r[1] for r in expert_rows):.4f}  "
          f"max {max(r[1] for r in expert_rows):.4f}   耗时 {time.perf_counter()-t0:.0f}s")
    print()

    print(f"=== RL {args.rl} ===")
    device = "cpu"
    tmpl = build_env_from_config(env_cfg)
    model = GraphMAPPOActorCritic(tmpl.action_resolver.action_space, env_cfg)
    rl = MAPPOPolicy(model, device)
    data = load_checkpoint(str(ROOT / args.rl), device)
    model.load_state_dict(data.model_state)
    model.eval()
    print(f"  checkpoint update={data.update}")
    rl_rows = []
    with torch.no_grad():
        for seed in seeds:
            env = build_env_from_config(env_cfg)
            obs = env.reset(seed=seed)
            served = 0.0
            done = False
            while not done:
                step = rl.act(obs, deterministic=True)
                obs, _r, term, trunc, info = env.step(
                    step.actions, step.action_scores,
                    edge_scores=step.edge_scores,
                    expected_matched_edges=list(step.matched_edges or []))
                served += float(info.get("served_keys", 0.0))
                done = term or trunc
            s = env.metrics.episode_summary()
            arr = float(s.get("arrived_keys", 0.0))
            sr = served / arr if arr else 0.0
            rl_rows.append((seed, sr))
            print(f"  seed {seed:>3}  RL   {sr:.4f}")

    r_mean = sum(r[1] for r in rl_rows) / len(rl_rows)
    # 配对差（同种子）
    diffs = [e - r for (_, e), (_, r) in zip(expert_rows, rl_rows)]
    mean_d = sum(diffs) / len(diffs)
    var = sum((d - mean_d) ** 2 for d in diffs) / max(1, len(diffs) - 1)
    se = (var / len(diffs)) ** 0.5
    print(f"\n  RL mean = {r_mean:.4f}")
    print(f"  Δ(专家 - RL) = {mean_d:+.4f}   配对 SE = {se:.4f}   t = {mean_d/se if se else 0:.2f}")
    print()
    print("  判读：")
    if mean_d > 2 * se:
        print(f"    专家显著更高（t={mean_d/se:.1f}）→ 验证 regime 上有真实可学空间，")
        print("    RL 卡在 0.648 是真实短板，方向对。")
    else:
        print("    专家 ≈ RL → 验证 regime 也饱和，成功率这条路整体没空间。")

    Path(args.out).write_text(json.dumps(
        {"expert": expert_rows, "rl": rl_rows, "expert_mean": e_mean,
         "rl_mean": r_mean, "paired_delta": mean_d, "paired_se": se}),
        encoding="utf-8")


if __name__ == "__main__":
    main()
