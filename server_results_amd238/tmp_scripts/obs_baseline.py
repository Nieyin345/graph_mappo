"""graph_builder 数值基准：固定种子+固定动作序列下，记录观测输出的逐位指纹。

用途：改造 graph_builder 期间，任何数值偏差都必须让这个脚本变红。
指纹取 sha256 of bytes，所以哪怕最后一位浮点变了也会被发现。

用法：
    python .tmp/obs_baseline.py --save     # 保存基准
    python .tmp/obs_baseline.py --check    # 对比（改动后跑）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qkd_rl.core.config import ConfigValidator, deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config

BASELINE = Path(__file__).resolve().parent / "obs_baseline.json"


def build(seed_day: int):
    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    config = deep_merge(
        config,
        {
            "env": {
                "episode_steps": 30,
                "episode_start_mode": "fixed",
                "episode_start_day": seed_day,
            },
            "runtime": {"device": "cpu"},
        },
    )
    ConfigValidator().validate(config)
    return build_env_from_config(config)


def digest(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes() + str(a.shape).encode()).hexdigest()[:16]


def collect(day: int, n_steps: int = 12) -> dict:
    """固定动作序列（确定性、不依赖策略）下逐步记录观测指纹。"""
    env = build(day)
    obs = env.reset(seed=day)
    rows = []
    for _ in range(n_steps):
        rec = {
            "node_features": digest(obs.node_features),
            "edge_features": digest(obs.edge_features),
            "edge_index": digest(obs.edge_index),
            # 这些是"哪条边被选中"的身份信息，必须一起锁住
            "physical_edge_ids": hashlib.sha256(
                "|".join(obs.physical_edge_ids).encode()
            ).hexdigest()[:16],
            "demand_edge_ids": hashlib.sha256(
                "|".join(obs.demand_edge_ids).encode()
            ).hexdigest()[:16],
            # 浮点摘要：和、均值、极值，捕捉"哈希变了但差值极小"的情况
            "node_sum": float(np.asarray(obs.node_features, dtype=np.float64).sum()),
            "edge_sum": float(np.asarray(obs.edge_features, dtype=np.float64).sum()),
        }
        rows.append(rec)
        actions = {nid: (None, None) for nid in obs.node_ids}
        obs, *_ = env.step(actions, {}, edge_scores=None)  # type: ignore[arg-type]
    return {"day": day, "steps": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    days = [0, 5]
    got = {str(d): collect(d) for d in days}

    if args.save:
        BASELINE.write_text(json.dumps(got, indent=1), encoding="utf-8")
        print(f"基准已保存 -> {BASELINE}")
        for d in days:
            r = got[str(d)]["steps"][-1]
            print(f"  day={d} 末步 node={r['node_features']} edge={r['edge_features']} "
                  f"phys={r['physical_edge_ids'][:8]} dem={r['demand_edge_ids'][:8]}")
        return 0

    if not BASELINE.exists():
        print("没有基准文件，先跑 --save")
        return 2
    ref = json.loads(BASELINE.read_text(encoding="utf-8"))

    bad = 0
    for d, block in got.items():
        rb = ref.get(d)
        if rb is None:
            print(f"day={d}: 基准里没有")
            bad += 1
            continue
        for i, (a, b) in enumerate(zip(block["steps"], rb["steps"])):
            if a != b:
                bad += 1
                diffs = [k for k in a if a[k] != b[k]]
                print(f"  ❌ day={d} step={i} 差异字段: {diffs}")
                for k in diffs:
                    if k.endswith("_sum"):
                        print(f"      {k}: now={a[k]!r} ref={b[k]!r}")
                break
    if bad:
        print(f"\n❌ 数值已改变（{bad} 处）")
        return 1
    print("✅ 数值逐位一致（全部 step 的 node/edge/ids 指纹相同）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
