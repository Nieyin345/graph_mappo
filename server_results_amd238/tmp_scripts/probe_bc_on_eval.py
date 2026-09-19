#!/usr/bin/env python
"""补齐缺口：BC 起点在**验证 regime** 上是多少？

已知：
  验证 regime（days 330-365，240 步，seed 100-114）
    专家 0.698
    RL（r6_base 训 20 轮）0.6532
    Δ = +0.0447, t=4.07

但缺一个数：**BC 起点在验证 regime 上是多少？** 这决定怎么解释那 4.5 点：
  (i) BC 起点 ≈ 0.653 → RL 没动，差距是 BC 模仿不到专家；
  (ii) BC 起点 > 0.653 → **训练把策略从好的起点训练坏了**，这是更严重的问题。

同时测 r6 各轮次，看验证侧的轨迹。

用法：python .tmp/probe_bc_on_eval.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
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


CKPTS = [
    ("BC起点", "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"),
    ("r6_base_u5", "outputs/r6_base/checkpoint_update_000005.pt"),
    ("r6_base_u10", "outputs/r6_base/checkpoint_update_000010.pt"),
    ("r6_base_u15", "outputs/r6_base/checkpoint_update_000015.pt"),
    ("r6_base_u20", "outputs/r6_base/checkpoint_final.pt"),
    ("r7_ent_u20", "outputs/r7_fix_ent/checkpoint_final.pt"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100,101,102,103,104,105,106,107,108,109,110,111,112,113,114")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    tgm_args = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"],
        mode="random_episode", run_name=None, num_updates=1, seed=42,
        checkpoint=None, device="cpu")
    cfg = tgm.build_config(tgm_args)
    val = cfg.get("validation", {}) or {}
    win = val.get("window", {}) or {}
    steps = int(val.get("episode_steps", 240))
    start_day, end_day = int(win.get("start_day", 330)), int(win.get("end_day", 365))

    env_cfg = dict(cfg)
    env_cfg["env"] = dict(cfg["env"])
    env_cfg["env"].update({
        "episode_start_mode": str(val.get("start_mode", "random_day")),
        "episode_steps": steps,
        "activation_window_start_day": start_day,
        "activation_window_end_day": end_day,
        "activation_window_days": max(0, end_day - start_day),
        "continuous": False,
    })
    env_cfg["scenario"] = json.loads(json.dumps(cfg.get("scenario", {})))
    env_cfg["scenario"].setdefault("time_limit", {})
    env_cfg["scenario"]["time_limit"]["days"] = end_day + max(1, math.ceil(steps / 1440))

    print(f"验证 regime: days {start_day}-{end_day}, {steps} 步, "
          f"{len(seeds)} 种子\n")
    device = "cpu"
    tmpl = build_env_from_config(env_cfg)
    results = {}
    for label, rel in CKPTS:
        p = ROOT / rel
        if not p.exists():
            print(f"!! 缺 {rel}")
            continue
        model = GraphMAPPOActorCritic(tmpl.action_resolver.action_space, env_cfg)
        rl = MAPPOPolicy(model, device)
        d = load_checkpoint(str(p), device)
        model.load_state_dict(d.model_state)
        model.eval()
        srs = []
        with torch.no_grad():
            for seed in seeds:
                env = build_env_from_config(env_cfg)
                obs = env.reset(seed=seed)
                served = 0.0
                done = False
                while not done:
                    st = rl.act(obs, deterministic=True)
                    obs, _r, term, trunc, info = env.step(
                        st.actions, st.action_scores,
                        edge_scores=st.edge_scores,
                        expected_matched_edges=list(st.matched_edges or []))
                    served += float(info.get("served_keys", 0.0))
                    done = term or trunc
                s = env.metrics.episode_summary()
                arr = float(s.get("arrived_keys", 0.0))
                srs.append(served / arr if arr else 0.0)
        m = sum(srs) / len(srs)
        results[label] = {"mean": m, "per_seed": srs}
        print(f"  {label:<12} update={d.update:<3} mean={m:.4f}")

    if "BC起点" in results:
        base = results["BC起点"]["per_seed"]
        print("\n=== 相对 BC 起点的配对差（同种子，验证 regime）===")
        for label, r in results.items():
            if label == "BC起点":
                continue
            dd = [a - b for a, b in zip(r["per_seed"], base)]
            md = sum(dd) / len(dd)
            var = sum((x - md) ** 2 for x in dd) / max(1, len(dd) - 1)
            se = (var / len(dd)) ** 0.5
            print(f"  {label:<12} Δ={md:+.4f}  SE={se:.4f}  t={md/se if se else 0:+.2f}")

    print("\n参考: 专家 = 0.698（同样 15 个种子）")
    Path("/tmp/bc_on_eval.json").write_text(json.dumps(results), encoding="utf-8")


if __name__ == "__main__":
    main()
