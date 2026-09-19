#!/usr/bin/env python
"""训练 regime 上扫多个 checkpoint：训练到底把 BC 起点推动了没有？

已知（probe_train_regime.py）：
  专家 0.8692 / r6_base(20轮) 0.8549 —— 10/10 种子专家都高，配对 Δ=+0.0143, t≈8.5
  统计显著但幅度小。问题是：**这个 1.4 点差距，训练有没有在缩小？**

测法：同一批种子（训练 regime，days 0-295，1440 步）上依次跑
  BC 起点 → 若干训练 checkpoint，
看曲线动没动。若 BC 起点 = 训练 20 轮 = 训练 25 轮，则**训练零效果**，
剩下的 1.4 点要靠改算法去够。

用法：
  python .tmp/probe_checkpoint_scan.py --seeds 7-16
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
    ("r6_base_u20", "outputs/r6_base/checkpoint_final.pt"),
    ("r7_ent_u20", "outputs/r7_fix_ent/checkpoint_final.pt"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,8,9,10,11,12,13,14,15,16")
    ap.add_argument("--out", default="/tmp/ckpt_scan.json")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    tgm_args = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"],
        mode="random_episode", run_name=None, num_updates=1, seed=42,
        checkpoint=None, device="cpu")
    cfg = tgm.build_config(tgm_args)
    print(f"训练 regime: days {cfg['env']['activation_window_start_day']}"
          f"-{cfg['env']['activation_window_end_day']}, "
          f"{cfg['env']['episode_steps']} 步, {len(seeds)} 个种子\n")

    device = "cpu"
    tmpl = build_env_from_config(cfg)
    results = {}
    for label, rel in CKPTS:
        p = ROOT / rel
        if not p.exists():
            print(f"!! 缺 {rel}")
            continue
        model = GraphMAPPOActorCritic(tmpl.action_resolver.action_space, cfg)
        rl = MAPPOPolicy(model, device)
        data = load_checkpoint(str(p), device)
        model.load_state_dict(data.model_state)
        model.eval()
        t0 = time.perf_counter()
        srs = []
        with torch.no_grad():
            for seed in seeds:
                env = build_env_from_config(cfg)
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
        results[label] = {"mean": m, "per_seed": srs, "update": int(getattr(data, "update", 0))}
        print(f"  {label:<12} update={data.update:<3} mean={m:.4f}  "
              f"[{' '.join(f'{v:.4f}' for v in srs)}]  {time.perf_counter()-t0:.0f}s")

    # 与 BC 起点做配对差
    if "BC起点" in results:
        base = results["BC起点"]["per_seed"]
        print("\n=== 相对 BC 起点的配对差（同种子）===")
        for label, r in results.items():
            if label == "BC起点":
                continue
            d = [a - b for a, b in zip(r["per_seed"], base)]
            md = sum(d) / len(d)
            var = sum((x - md) ** 2 for x in d) / max(1, len(d) - 1)
            se = (var / len(d)) ** 0.5
            print(f"  {label:<12} Δ={md:+.4f}  SE={se:.4f}  t={md/se if se else 0:+.2f}")

    Path(args.out).write_text(json.dumps(results), encoding="utf-8")


if __name__ == "__main__":
    main()
