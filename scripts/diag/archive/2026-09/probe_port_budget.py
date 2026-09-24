"""机制体检 ②：**端口预算** vs **待服务请求要占的跳数** —— 有没有一个谁都无法突破的上限。

### 为什么问这个（这是「机制是否合理」的直接判据）

已经量到的（`probe_match_load.py` / `probe_qkp_level_stats.py`，真实验证协议）：

| 读数 | 值 |
|---|---|
| 每步配成弧 | **55.97 条**（上限 = 节点数 90，端口利用率 0.62） |
| 每步新到请求 | 0.77 条 |
| 每步服务请求 | 0.56 条；**每步作废 0.20 条** |
| 活跃边上零存量的比例 | **52%** |
| `min(hop level)/remaining` | **p95 都还是 0.0000** ⟹ 挡路的是**空跳**，不是存量不够 |

`routing.py:238` `serve_now = min(hop_levels + [remaining])` ⟹ 一条请求要被服务，
它的**整条路径每一跳都得有密钥**。而路径是 `_find_positive_path` 在
「有存量的边」子图上找的 —— 所以能不能服务，等于**这条路径上的跳是不是都被
近期激活过**。

于是一个纯算术问题：设端口预算 = 节点数 90（每节点一个 Tx 口 + 一个 Rx 口），
平均路径 L 跳，同时挂着的请求 N 条。要在同一时刻覆盖所有请求的路径，需要
**N × L 个不同的跳**。若 `N × L > 90`，则**没有任何策略**能同时服务所有在挂请求
—— 上限是机制给的，不是学出来的。

### 本探针量什么（只读）

逐步记录：待挂请求数 `N`、它们各自的最短路跳数 `L_i`、**并集**跳数 `U`，
以及端口预算 90。报告 `U / 90` 的分位数与 `N*L` 与 `U` 的差
（差 = 多少跳被多条请求共用 ⟹ 共享度越高越省）。

**判据（先写死）**

- `U` 的 p50 **持续 > 90** ⟹ **覆盖需求超过端口预算** ⟹ 成功率有一个
  **机制性上限**，任何策略改动都突破不了；该改的是机制
  （多端口 / 缩路径 / 增 TTL / 合并请求），不是奖励或结构
- `U` 的 p50 **< 90 但明显大于实际激活的 56** ⟹ 端口**够用但没提满** ⟹
  是**决策/观测**问题，去改模型输入
- `N × L ≫ U` ⟹ 跳的**共享度高**，路径成网，覆盖成本远低于朴素估计

用法（服务器上）：
    python -u /tmp/probe_port_budget.py --seeds 100-109 --steps 240
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

from qkd_rl.env.factory import build_env_from_config      # noqa: E402
from qkd_rl.baselines.path_greedy import PathScoreGreedy   # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe       # noqa: E402


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
        raise SystemExit("★ 协议没读到" )
    return p


def q(a: np.ndarray, x: float) -> float:
    return float(np.percentile(a, x)) if a.size else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-109")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--out", default="/tmp/probe_port_budget.json")
    a = ap.parse_args()

    profile = protocol()
    seeds = parse_seeds(a.seeds)
    cfg = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=a.steps,
        start_mode=profile["start_mode"])
    cfg["runtime"]["device"] = "cpu"

    rows = []
    for seed in seeds:
        env = build_env_from_config(cfg)
        obs = env.reset(seed=seed, start_seed=int(profile.get("start_seed", 0)) + seed)
        asp = env.action_resolver.action_space
        n_nodes = len(asp.node_ids)
        expert = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True,
                                 principles=False, router=ServeProbe(env))
        done = False
        while not done:
            actions, scores = expert.act(obs)
            obs, _r, term, trunc, _i = env.step(actions, scores)
            done = term or trunc

            pending = env.requests.get_pending()
            n_pending = len(pending)
            lens, hop_union, n_conflict_hops = [], set(), 0
            covered = 0
            for req in pending:
                path = env.routing.shortest_path(req.src_gs, req.dst_gs)
                if not path:
                    continue
                lens.append(len(path))
                hop_union.update(path)
                # 其中**空跳**有多少、**空跳里已被本步激活**的有多少
                for e in path:
                    if env.qkp.get_level(e) <= 0.0:
                        n_conflict_hops += 1
                        if e in env.last_activated_edges:
                            covered += 1
            rows.append({
                "n_pending": n_pending,
                "mean_len": st.mean(lens) if lens else 0.0,
                "union": len(hop_union),
                "n_needed": int(sum(lens)),
                "n_empty_hops": n_conflict_hops,
                "n_empty_covered": covered,
                "n_activated": len(env.last_activated_edges),
                "n_nodes": n_nodes,
            })

    def col(k):
        return np.asarray([r[k] for r in rows], dtype=np.float64)

    n_nodes = rows[0]["n_nodes"] if rows else 90
    print("端口预算体检（只读）—— 协议：天 %d–%d，%s，%d 步 × %d 种子 = %d 步"
          % (profile["window_start_day"], profile["window_end_day"],
             profile["start_mode"], a.steps, len(seeds), len(rows)))
    print("  端口预算 = 节点数 = %d（每节点一个 Tx 口 + 一个 Rx 口）" % n_nodes)
    print()
    for k, label in (("n_pending", "在挂请求数 N"),
                     ("mean_len", "平均路径长 L"),
                     ("n_needed", "朴素需求 N×L（跳）"),
                     ("union", "并集跳数 U（真实需要的不同跳）"),
                     ("n_activated", "上一步实际激活弧数"),
                     ("n_empty_hops", "其中空跳数"),
                     ("n_empty_covered", "空跳里上一步已激活的")):
        arr = col(k)
        print("  %-26s p5 %8.2f  p25 %8.2f  p50 %8.2f  p75 %8.2f  p95 %8.2f  均值 %8.2f"
              % (label, q(arr, 5), q(arr, 25), q(arr, 50), q(arr, 75), q(arr, 95), arr.mean()))
    print()

    U = col("union")
    need = col("n_needed")
    act = col("n_activated")
    ne = col("n_empty_hops")
    nc = col("n_empty_covered")

    print("=== 关键比值 ===")
    print("  并集 U / 端口预算 %d ： p50 %.3f   p95 %.3f   >1 的比例 %.4f"
          % (n_nodes, q(U / n_nodes, 50), q(U / n_nodes, 95),
             float((U > n_nodes).mean())))
    print("  朴素 N×L / 并集 U      ： p50 %.3f   （跳的共享度，越大越省）"
          % q(need / np.maximum(U, 1.0), 50))
    print("  实际激活 / 端口预算    ： p50 %.3f" % q(act / n_nodes, 50))
    print("  空跳覆盖率 = 已激活/空跳： p50 %.3f"
          % (q(nc / np.maximum(ne, 1.0), 50) if ne.sum() else float("nan")))
    print()

    print("=== 判据 ===")
    frac_over = float((U > n_nodes).mean())
    if frac_over > 0.5:
        print("  ⟹ **覆盖需求超过端口预算**（U > %d 的步占 %.1f%%）"
              % (n_nodes, 100 * frac_over))
        print("     成功率有一个**机制性上限**：同时挂着的请求要的跳数比端口多，")
        print("     没有任何策略能同时覆盖 ⟹ 该改**机制**（多端口/缩路径/增 TTL），")
        print("     改奖励或模型结构都突破不了")
    elif q(U / n_nodes, 50) < 0.7:
        print("  ⟹ 端口**够用但没提满**（p50 U/预算 = %.3f）⟹ 是**决策/观测**问题"
              % q(U / n_nodes, 50))
    else:
        print("  ⟹ 居中：p50 U/预算 = %.3f，接近但不越界" % q(U / n_nodes, 50))
    print("  （空跳覆盖率低 ⟹ 空跳大多**没被激活过**，是覆盖问题；")
    print("    空跳覆盖率高但仍是空 ⟹ 激活了却没变成存量，是**别的**问题）")

    Path(a.out).write_text(json.dumps(
        {"n_nodes": n_nodes, "steps_total": len(rows),
         "union_p": {str(x): q(U, x) for x in (5, 25, 50, 75, 95)},
         "frac_union_over_budget": frac_over},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写 %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
