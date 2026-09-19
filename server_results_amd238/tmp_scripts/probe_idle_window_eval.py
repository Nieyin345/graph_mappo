#!/usr/bin/env python
"""在闲置窗口 296-329 上评测 BC 起点与专家，判断它是否值得加进训练。

逻辑：
  - 若 BC 在这段上已经很高（≈0.85）→ 与训练窗口同质，加进去只是多 12% 数据，
    收益有限；但也没有坏处。
  - 若 BC 在这段上明显低（接近验证窗口的 0.65）→ 这段更接近验证难度，
    加进训练才是真正对症的改动。

同时给出「专家 - BC」的差距，与训练窗口（+0.021）和验证窗口（+0.050）对比。

用法：
  python .tmp/probe_idle_window_eval.py --seeds 7-16
"""
from __future__ import annotations

import argparse
import importlib.util
import math
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
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic
from qkd_rl.baselines.path_greedy import PathScoreGreedy
from qkd_rl.baselines.serve_probe import ServeProbe


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


def make_cfg(base, start, end, steps):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    cfg["env"] = dict(base["env"])
    cfg["scenario"] = {k: (dict(v) if isinstance(v, dict) else v)
                       for k, v in base["scenario"].items()}
    cfg["env"]["episode_start_mode"] = "random_day"
    cfg["env"]["episode_steps"] = steps
    cfg["env"]["activation_window_start_day"] = start
    cfg["env"]["activation_window_end_day"] = end
    cfg["env"]["activation_window_days"] = max(0, end - start)
    need = end + max(1, math.ceil(steps / 1440))
    cfg["scenario"].setdefault("time_limit", {})
    cfg["scenario"]["time_limit"]["days"] = need
    return cfg


def run_bc(policy, cfg, seeds):
    rows = []
    with torch.no_grad():
        for seed in seeds:
            env = build_env_from_config(cfg)
            obs = env.reset(seed=seed)
            served = 0.0
            done = False
            while not done:
                st = policy.act(obs, deterministic=True)
                obs, _r, term, trunc, info = env.step(
                    st.actions, st.action_scores, edge_scores=st.edge_scores,
                    expected_matched_edges=list(st.matched_edges or []))
                served += float(info.get("served_keys", 0.0))
                done = term or trunc
            s = env.metrics.episode_summary()
            a = float(s.get("arrived_keys", 0.0))
            rows.append(served / a if a else 0.0)
    return rows


def run_expert(cfg, seeds):
    rows = []
    for seed in seeds:
        env = build_env_from_config(cfg)
        obs = env.reset(seed=seed)
        ex = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True,
                             principles=False, router=ServeProbe(env))
        served = 0.0
        done = False
        while not done:
            a, sc = ex.act(obs)
            obs, _r, term, trunc, info = env.step(a, sc)
            served += float(info.get("served_keys", 0.0))
            done = term or trunc
        s = env.metrics.episode_summary()
        arr = float(s.get("arrived_keys", 0.0))
        rows.append(served / arr if arr else 0.0)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7-16")
    ap.add_argument("--steps", type=int, default=1440)
    ap.add_argument("--anchor", action="store_true",
                    help="同时跑训练窗口锚点，用于同批次对比")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    tgm_args = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"],
        mode="random_episode", run_name=None, num_updates=1,
        seed=42, checkpoint=None, device="cpu")
    base = tgm.build_config(tgm_args)
    base["runtime"]["device"] = "cpu"

    st = load_checkpoint(
        str(ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"),
        "cpu").model_state

    cells = [
        ("训练窗口 0-295", 0, 295),
        ("**闲置 296-329**", 296, 330),
        ("**验证 330-365**", 330, 365),
    ]
    if args.anchor:
        cells = [cells[0], cells[1]]

    print(f"BC 起点 vs 专家，{len(seeds)} 种子，{args.steps} 步\n")
    print(f"{'窗口':<22}{'BC':>10}{'专家':>10}{'专家-BC':>12}")
    res = {}
    for lab, a, b in cells:
        cfg = make_cfg(base, a, b, args.steps)
        env = build_env_from_config(cfg)
        m = GraphMAPPOActorCritic(env.action_resolver.action_space, base)
        pol = MAPPOPolicy(m, "cpu")
        m.load_state_dict({k: v.clone() for k, v in st.items()})
        t0 = time.perf_counter()
        bc = run_bc(pol, cfg, seeds)
        ex = run_expert(cfg, seeds)
        res[lab] = (bc, ex)
        mb, me = sum(bc) / len(bc), sum(ex) / len(ex)
        print(f"{lab:<22}{mb:>10.4f}{me:>10.4f}{me-mb:>+12.4f}"
              f"   ({time.perf_counter()-t0:.0f}s)", flush=True)

    # 配对检验：闲置 vs 训练
    if len(res) >= 2:
        keys = list(res)
        b1, e1 = res[keys[0]]
        b2, e2 = res[keys[1]]
        d = [x - y for x, y in zip(b2, b1)]
        md = sum(d) / len(d)
        var = sum((x - md) ** 2 for x in d) / max(1, len(d) - 1)
        se = (var / len(d)) ** 0.5
        print(f"\n配对差（{keys[1]} − {keys[0]}）BC: {md:+.4f} ± {se:.4f}"
              f"  t={md/se if se else 0:.2f}")


if __name__ == "__main__":
    main()
