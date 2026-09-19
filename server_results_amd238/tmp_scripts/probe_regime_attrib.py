#!/usr/bin/env python
"""归因：验证 regime 比训练 regime 差多少，是「天数窗口」造成的还是「回合长度」造成的？

已知（BC 起点）：
  训练 regime（窗口 0-295，1440 步） 0.8483
  验证 regime（窗口 330-365，240 步） 0.6483
两者差两样东西。本脚本拆成 2x2：

                 | 1440 步          | 240 步
  窗口 0-295     | A（=训练 regime） | B
  窗口 330-365   | D                | C（=验证 regime）

判读：
  - 差距主要来自窗口（A≈B 高、C≈D 低）→ 训练数据覆盖不到难场景 → 该扩窗口/加难度
  - 差距主要来自回合长度（A≈D 高、B≈C 低）→ 240 步里启动瞬态占比大 → 该调训练回合长度
  - 两者交互 → 都得改

同时跑 BC 起点与专家，看「专家 - BC」的 4.5 点差距落在哪些格子里。

用法：
  python .tmp/probe_regime_attrib.py --seeds 7-16
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

_tp_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_tp_spec)
_tp_spec.loader.exec_module(_tp)

from qkd_rl.env.factory import build_env_from_config
from qkd_rl.rl.algos.checkpoint import load_checkpoint
from qkd_rl.rl.algos.policy import MAPPOPolicy
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic
from qkd_rl.baselines.path_greedy import PathScoreGreedy
from qkd_rl.baselines.serve_probe import ServeProbe

DAY_STEPS = 1440


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


def make_cell(base, start, end, steps):
    """从同一份训练配置派生出 2x2 里的一个格子，只动窗口与回合长度。"""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    cfg["env"] = dict(base["env"])
    cfg["scenario"] = {k: (dict(v) if isinstance(v, dict) else v)
                       for k, v in base["scenario"].items()}
    cfg["env"]["episode_start_mode"] = "random_day"
    cfg["env"]["episode_steps"] = steps
    cfg["env"]["activation_window_start_day"] = start
    cfg["env"]["activation_window_end_day"] = end
    if "activation_window_days" in cfg["env"]:
        cfg["env"]["activation_window_days"] = max(0, end - start)
    need = end + max(1, math.ceil(steps / DAY_STEPS))
    old = int(cfg["scenario"].get("time_limit", {}).get("days", 0) or 0)
    cfg["scenario"].setdefault("time_limit", {})
    cfg["scenario"]["time_limit"]["days"] = max(old, need)
    return cfg


def eval_bc(policy, cfg, seeds):
    """逐个种子评测，返回 [(seed, success)]，便于配对。"""
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
                    st.actions, st.action_scores,
                    edge_scores=st.edge_scores,
                    expected_matched_edges=list(st.matched_edges or []))
                served += float(info.get("served_keys", 0.0))
                done = term or trunc
            s = env.metrics.episode_summary()
            arr = float(s.get("arrived_keys", 0.0))
            rows.append((seed, served / arr if arr else 0.0))
    return rows


def eval_expert(cfg, seeds):
    rows = []
    for seed in seeds:
        env = build_env_from_config(cfg)
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
        rows.append((seed, served / arr if arr else 0.0))
    return rows


def mean(rows):
    return sum(r[1] for r in rows) / max(1, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7-16")
    ap.add_argument("--skip-expert", action="store_true")
    ap.add_argument("--cells", default="A,B,C,D")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)

    tgm_args = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"],
        mode="random_episode", run_name=None, num_updates=1,
        seed=42, checkpoint=None, device="cpu")
    base = tgm.build_config(tgm_args)
    base["runtime"]["device"] = "cpu"

    # 确认派生的 C 格与评测器用的规范验证配置一致（只比 env/scenario 关键项）
    prof = _tp.load_validation_profile(ROOT / "configs" / "global.yaml")
    canon = _tp.build_validation_env_config(prof, include_baselines=True,
                                            episode_steps=240)
    print(f"验证 profile：窗口 {prof['window_start_day']}-{prof['window_end_day']}，"
          f"steps {prof['episode_steps']}，mode {prof['start_mode']}")
    print("训练配置 vs 规范验证配置（env/scenario 差异）：")
    diffs = 0
    for sect in ("env", "scenario", "rate_provider"):
        a, b = base.get(sect, {}), canon.get(sect, {})
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                print(f"  [{sect}].{k}: 训练={a.get(k)!r}  验证={b.get(k)!r}")
                diffs += 1
    print(f"  共 {diffs} 处差异\n")

    cells = {
        "A": ("窗口0-295 / 1440步", make_cell(base, 0, 295, 1440)),
        "B": ("窗口0-295 / 240步 ", make_cell(base, 0, 295, 240)),
        "C": ("窗口330-365 / 240步", make_cell(base, 330, 365, 240)),
        "D": ("窗口330-365 / 1440步", make_cell(base, 330, 365, 1440)),
    }
    want = [c.strip() for c in args.cells.split(",") if c.strip()]

    st = load_checkpoint(
        str(ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"),
        "cpu").model_state

    print(f"BC 起点，{len(seeds)} 个种子，逐格评测\n")
    out = {}
    for key in want:
        label, cfg = cells[key]
        env = build_env_from_config(cfg)
        model = GraphMAPPOActorCritic(env.action_resolver.action_space, base)
        policy = MAPPOPolicy(model, "cpu")
        model.load_state_dict({k: v.clone() for k, v in st.items()})

        t0 = time.perf_counter()
        bc = eval_bc(policy, cfg, seeds)
        dt = time.perf_counter() - t0
        ex = None
        if not args.skip_expert:
            ex = eval_expert(cfg, seeds)
        out[key] = (bc, ex)
        line = f"  {key}  {label}  BC={mean(bc):.4f}"
        if ex is not None:
            line += f"  专家={mean(ex):.4f}  差={mean(ex)-mean(bc):+.4f}"
        line += f"   ({dt:.0f}s)"
        print(line, flush=True)

    if not args.skip_expert and all(out[k][1] for k in want):
        print("\n=== BC 起点 2x2 ===")
        print(f"  {'':<16}{'1440步':>10}{'240步':>10}")
        for row, (s, e) in (("窗口 0-295", ("A", "B")), ("窗口 330-365", ("C", "D"))):
            if s in out and e in out:
                print(f"  {row:<16}{mean(out[s][0]):>10.4f}{mean(out[e][0]):>10.4f}")
        print("\n=== 专家 2x2 ===")
        print(f"  {'':<16}{'1440步':>10}{'240步':>10}")
        for row, (s, e) in (("窗口 0-295", ("A", "B")), ("窗口 330-365", ("C", "D"))):
            if s in out and e in out:
                print(f"  {row:<16}{mean(out[s][1]):>10.4f}{mean(out[e][1]):>10.4f}")
        print("\n=== 专家 - BC ===")
        print(f"  {'':<16}{'1440步':>10}{'240步':>10}")
        for row, (s, e) in (("窗口 0-295", ("A", "B")), ("窗口 330-365", ("C", "D"))):
            if s in out and e in out:
                print(f"  {row:<16}{mean(out[s][1])-mean(out[s][0]):>10.4f}"
                      f"{mean(out[e][1])-mean(out[e][0]):>10.4f}")

    print("\n=== 配对明细（BC） ===")
    for k in want:
        print(f"  {k}: " + " ".join(f"{s}:{v:.3f}" for s, v in out[k][0]))


if __name__ == "__main__":
    main()
