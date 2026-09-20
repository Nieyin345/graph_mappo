"""量「每条边当前存了多少密钥」—— 决定新增特征的**归一化尺度**。

### 为什么必须先量再写代码

已确认的结构缺口（`graph_builder.py:678-686` + `:741`）：
物理边的动态列里**只有 `qkp_capacity_left = (cap - level) / cap`**，
而 serve 的判据是 `routing.py:238` `serve_now = min(hop_levels + [remaining])`
—— **一条路径能服务多少，完全由最弱一跳的存量决定**。
策略却看不到任何一列的**绝对存量**。

但「加一列 level」有两种写法，差别很大：

- `level / capacity` —— 这在数学上**恰好等于 `1 - capacity_left`**，
  与已有列**完全共线**，加进去是**零信息**（白改一版）
- `level / amount_mean`（或 `/amount_max`）—— 直接对标"这一跳够不够服务一条请求"，
  就是 `min(hop_levels + [remaining])` 里的那个比较

所以本探针量**真实分布**，用数据选尺度，而不是猜。

### 量什么（只读，不改任何东西）

在真实验证协议上跑专家，每步对**活跃物理边**取：

| 量 | 为什么要 |
|---|---|
| `level / capacity` 的分位数 | 若恒在 0.99 附近 ⟹ 现有列**确实是常数**，缺口坐实 |
| `level / amount_max` 的分位数 | 决定新列该用什么尺度；若大量 <1 ⟹ 存量**经常不够**服务一条请求 |
| `level / amount_mean` 的分位数 | 同上，另一把尺子 |
| `min(hop level) / remaining` | **最弱一跳**的相对余量 —— 服务量真正的天花板 |

用法（服务器上）：
    python -u /tmp/probe_qkp_level_stats.py --seeds 100-109 --steps 240
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

from qkd_rl.env.factory import build_env_from_config     # noqa: E402
from qkd_rl.baselines.path_greedy import PathScoreGreedy  # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe      # noqa: E402


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def protocol() -> dict:
    p = _tp.load_validation_profile(ROOT / "configs" / "train_full_rl.yaml")
    v = (yaml.safe_load((ROOT / "configs" / "train_full_rl.yaml").read_text(
        encoding="utf-8")) or {}).get("validation", {}) or {}
    if v:
        p["window_start_day"] = int(v["window"]["start_day"])
        p["window_end_day"] = int(v["window"]["end_day"])
        p["episode_steps"] = int(v["episode_steps"])
        p["start_mode"] = str(v.get("start_mode", "random_day"))
    if p["window_end_day"] <= 30:
        raise SystemExit("★ 协议没读到（window_end_day=%d 是默认值）" % p["window_end_day"])
    return p


def pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-109")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--out", default="/tmp/probe_qkp_level_stats.json")
    a = ap.parse_args()

    profile = protocol()
    seeds = parse_seeds(a.seeds)
    cfg = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=a.steps,
        start_mode=profile["start_mode"])
    cfg["runtime"]["device"] = "cpu"

    amount_mean = float(cfg["requests"]["amount_mean"])
    amount_max = float(cfg["requests"]["amount_max"])
    print("QKP 存量分布（只读）")
    print("  协议：天 %d–%d，%s，%d 步，%d 个种子"
          % (profile["window_start_day"], profile["window_end_day"],
             profile["start_mode"], a.steps, len(seeds)))
    print("  请求量尺度：mean %.0f  max %.0f" % (amount_mean, amount_max))
    print()

    over_cap, over_max, over_mean, hop_ratio = [], [], [], []
    n_pos_frac = []

    for seed in seeds:
        env = build_env_from_config(cfg)
        obs = env.reset(seed=seed, start_seed=int(profile.get("start_seed", 0)) + seed)
        expert = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True,
                                 principles=False, router=ServeProbe(env))
        start_day = int(env.t) // 1440
        done = False
        while not done:
            actions, scores = expert.act(obs)
            obs, _r, term, trunc, _i = env.step(actions, scores)
            done = term or trunc

            ids = obs.physical_edge_ids
            if not ids:
                continue
            levels = np.fromiter((env.qkp.levels.get(e, 0.0) for e in ids),
                                 dtype=np.float64, count=len(ids))
            caps = np.fromiter((env.qkp.get_capacity(e) for e in ids),
                               dtype=np.float64, count=len(ids))
            ok = caps > 0
            over_cap.append(levels[ok] / caps[ok])
            over_max.append(levels[ok] / amount_max)
            over_mean.append(levels[ok] / amount_mean)
            n_pos_frac.append(float((levels > 0).mean()))

            # 最弱一跳的相对余量：对每条 pending 请求，取它最短路上最小存量 / 剩余量
            pending = env.requests.get_pending()
            for req in pending:
                path = env.routing.shortest_path(req.src_gs, req.dst_gs)
                if not path:
                    continue
                hl = [env.qkp.get_level(e) for e in path]
                if not hl:
                    continue
                rem = max(1.0, req.amount - req.served_amount)
                hop_ratio.append(min(hl) / rem)

    cap_a = np.concatenate(over_cap) if over_cap else np.zeros(0)
    max_a = np.concatenate(over_max) if over_max else np.zeros(0)
    mean_a = np.concatenate(over_mean) if over_mean else np.zeros(0)
    hop_a = np.asarray(hop_ratio, dtype=np.float64)

    print("=== 活跃物理边：level/capacity（= 1 − 现有 capacity_left 列）===")
    for q in (1, 5, 25, 50, 75, 95, 99):
        print("   p%-3d %10.6f" % (q, pct(cap_a, q)))
    print("   均值 %.6f  最小 %.6f  最大 %.6f" % (cap_a.mean(), cap_a.min(), cap_a.max()))
    print()
    print("=== level / amount_max（决定新列尺度的关键）===")
    for q in (1, 5, 25, 50, 75, 95, 99):
        print("   p%-3d %10.4f" % (q, pct(max_a, q)))
    print("   <1 的比例 %.4f  （<1 = 这一跳单独撑不起一条最大请求）"
          % float((max_a < 1).mean()))
    print()
    print("=== level / amount_mean ===")
    for q in (1, 5, 25, 50, 75, 95, 99):
        print("   p%-3d %10.4f" % (q, pct(mean_a, q)))
    print("   <1 的比例 %.4f" % float((mean_a < 1).mean()))
    print()
    print("=== 最弱一跳：min(hop level) / remaining ===")
    print("   样本 %d 条（请求·步）" % hop_a.size)
    for q in (1, 5, 25, 50, 75, 95):
        print("   p%-3d %10.4f" % (q, pct(hop_a, q)))
    print("   <1 的比例 %.4f  （<1 = 服务被最弱一跳卡住，只能部分服务）"
          % float((hop_a < 1).mean()) if hop_a.size else "   （无样本）")
    print()
    print("=== 结论判据 ===")
    print("  若 level/capacity 恒 ≈1（p1 也 >0.95）⟹ 现有列是**常数**，")
    print("     新增绝对存量列不是冗余，是补一个**完全缺失**的输入")
    print("  若 level/amount_max 有可观比例 <1 ⟹ 存量**经常**成为瓶颈，")
    print("     该列的尺度就取 amount_max（语义 =「这一跳能不能单独服务一条最大请求」）")

    Path(a.out).write_text(json.dumps({
        "seeds": seeds, "steps": a.steps, "amount_mean": amount_mean,
        "amount_max": amount_max,
        "level_over_capacity_p": {str(q): pct(cap_a, q) for q in (1, 5, 25, 50, 75, 95, 99)},
        "level_over_amount_max_p": {str(q): pct(max_a, q) for q in (1, 5, 25, 50, 75, 95, 99)},
        "level_over_amount_mean_p": {str(q): pct(mean_a, q) for q in (1, 5, 25, 50, 75, 95, 99)},
        "frac_max_lt1": float((max_a < 1).mean()),
        "frac_mean_lt1": float((mean_a < 1).mean()),
        "weakest_hop_ratio_p": {str(q): pct(hop_a, q) for q in (1, 5, 25, 50, 75, 95)},
        "frac_weakest_lt1": float((hop_a < 1).mean()) if hop_a.size else None,
        "n_hop_samples": int(hop_a.size),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写 %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
