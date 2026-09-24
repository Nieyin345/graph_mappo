"""机制体检：**每步能配几条边** vs **每步来多少请求** —— 匹配是不是那道墙。

### 为什么问这个

`action_space.py` 里每个节点只有 **一个 Tx 口 + 一个 Rx 口**，且同一对
`u,v` 不许双向同时成立（`_resolve_mutual_choice` 里 `pair_key` 去重）。
这等价于**一般图匹配**：90 个节点、1980 条注册链路，每步最多配
**45 条边**（一半节点配成对，另一半必然空闲）。

而服务一条请求要沿路径**逐跳**消耗：`routing.py:238`
`serve_now = min(hop_levels + [remaining])` —— 一条路径能服务多少，由
**最弱一跳的存量**决定。

如果每步来的请求数远超 45，那么**再多生成密钥也没用**：没有边被激活，
路径上的跳就拿不到新密钥。这正是已有读数暗示的方向（生成/服务 = 104×，
QKP 利用率 0.088），但那些是**训练侧聚合量**，不是这套协议上的因果读数。

### 本探针量什么（**不改任何东西**，只读 env 内部状态）

在**真实验证协议**上跑专家（确定性、无 BLAS ⟹ 与节点无关），逐步记录：

| 量 | 含义 |
|---|---|
| `n_arrived` | 本步新到的**请求条数** |
| `n_matched` | 本步实际配成的**有向弧数**（= 激活的物理边数） |
| `port_util` | `n_matched / 90` —— 端口利用率，上限 0.5 |
| `n_pending` | 步末仍挂着的**请求条数** |
| `served_keys` / `n_served` | 本步服务的密钥量 / 请求条数 |
| `n_expired` | 本步超时作废的请求条数 |

**判据（先写死）**

- `n_arrived ≫ n_matched` 且 `port_util` 贴着 0.5 ⟹ **端口/匹配是约束**，
  改动应该加"每节点多端口"或"一步多次匹配"，而不是调奖励
- `port_util` 明显低于 0.5 ⟹ 匹配**不是**满的，策略自己没提满 ⟹ 那是打分问题
- `n_served / n_matched` 很低 ⟹ 配出来的边大量没在服务请求上 ⟹ 打分问题

★ 三条判据互斥且都指向**不同的下一步改动**，所以这个探针的价值就在于
先把它们分开，而不是先动代码。

用法（服务器上）：
    python -u /tmp/probe_match_load.py --seeds 100-114 --steps 240
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics as st
import sys
import time
from pathlib import Path

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
    """★ 显式读 `train_full_rl.yaml` 的**顶层** validation 段并自检。

    教训（已刻在 `test_protocol.load_validation_profile` 的文档里）：
    原先的 loader 只读 `raw["global"]["validation"]`，而该文件的 `validation:`
    在顶层 ⟹ **静默**落回默认窗口 0–30 天（训练窗口内），判读全错而不报错。
    """
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
    if p["start_mode"] != "random_day":
        raise SystemExit("★ 期望 random_day（真实协议），实得 %s" % p["start_mode"])
    return p


def run_one(cfg, seed: int, start_seed: int, n_nodes: int):
    env = build_env_from_config(cfg)
    obs = env.reset(seed=seed, start_seed=start_seed)
    start_day = int(env.t) // 1440

    # 捕获 ServeResult / expired —— 只读，不改行为。
    box: dict = {"serve": None, "expired": 0}
    orig_serve = env.requests.serve
    orig_expire = env.requests.expire

    def serve_capture(*a, **kw):
        r = orig_serve(*a, **kw)
        box["serve"] = r
        return r

    def expire_capture(*a, **kw):
        r = orig_expire(*a, **kw)
        box["expired"] = len(r)
        return r

    env.requests.serve = serve_capture
    env.requests.expire = expire_capture

    expert = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True,
                             principles=False, router=ServeProbe(env))
    rows = []
    done = False
    while not done:
        n_arr_before = env.metrics.arrived_requests
        actions, scores = expert.act(obs)
        obs, _r, term, trunc, info = env.step(actions, scores)
        done = term or trunc
        sr = box["serve"]
        rows.append({
            "n_arrived": env.metrics.arrived_requests - n_arr_before,
            "n_matched": len(env.last_matched_arcs),
            "n_pending": len(env.requests.pending),
            "served_keys": float(sr.served_keys) if sr else 0.0,
            "n_served": len(sr.served_requests) if sr else 0,
            "n_waiting": len(sr.waiting_requests) if sr else 0,
            "n_expired": box["expired"],
        })
    s = env.metrics.episode_summary()
    arr = float(s.get("arrived_keys", 0.0))
    out = {
        "seed": seed, "start_day": start_day, "steps": len(rows),
        "sr": float(s.get("served_keys", 0.0)) / arr if arr else 0.0,
        "arrived_keys": arr, "served_keys": float(s.get("served_keys", 0.0)),
        "generated_keys": float(s.get("generated_keys", 0.0)),
        "rows": rows,
    }
    env.requests.serve = orig_serve
    env.requests.expire = orig_expire
    return out


def summarize(rows: list[dict], n_nodes: int) -> dict:
    n = len(rows)
    m = lambda k: st.mean([r[k] for r in rows])                       # noqa: E731
    matched = m("n_matched")
    served = m("served_keys")
    arrived_k = sum(r["served_keys"] + 0 for r in rows)  # placeholder, unused
    del arrived_k
    return {
        "steps": n,
        "n_arrived": m("n_arrived"),
        "n_matched": matched,
        "port_util": matched / n_nodes,
        "n_pending": m("n_pending"),
        "served_keys": served,
        "n_served": m("n_served"),
        "n_waiting": m("n_waiting"),
        "n_expired": m("n_expired"),
        "keys_per_match": served / matched if matched else float("nan"),
        "served_per_match": m("n_served") / matched if matched else float("nan"),
    }


def cap_arcs(n_nodes: int) -> int:
    """每条弧吃 **一个 Tx 口（源）+ 一个 Rx 口（宿）**，全网各有 ``n_nodes`` 个口。

    ⟹ 弧数上限 = ``n_nodes``，**不是** ``n_nodes // 2``。

    这条我一开始写错了（以为是"匹配"，上限 45），实测每步能配到 60.8 条 ⟹
    立刻暴露。错因：**一个节点可以同时"发一个、收一个"**（两个不同的口），
    所以可行结构是**出度≤1、入度≤1 的有向图**（paths + cycles），不是匹配。
    与 `_resolve_mutual_choice` 的代码一致：`tx_of`/`rx_of` 各是单值字典，
    `u` 能既是某条弧的源、又是另一条弧的宿。唯一额外限制是同一对
    `pair_key` 不许双向同时成立（去重），但那有 1978 个名额，远不紧。
    """
    return n_nodes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-114")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--out", default="/tmp/probe_match_load.json")
    a = ap.parse_args()

    profile = protocol()
    seeds = parse_seeds(a.seeds)
    cfg = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=a.steps,
        start_mode=profile["start_mode"])
    cfg["runtime"]["device"] = "cpu"

    probe_env = build_env_from_config(cfg)
    # `action_space` 挂在 resolver 上（env 本身没有这个属性）—— 实测踩过。
    asp = probe_env.action_resolver.action_space
    n_nodes = len(asp.node_ids)
    n_edges = len(asp.edges)
    print("机制体检：匹配容量 vs 请求负载（**只读，不改任何东西**）")
    print("  协议：天 %d–%d，%s，%d 步，%d 个种子"
          % (profile["window_start_day"], profile["window_end_day"],
             profile["start_mode"], a.steps, len(seeds)))
    print("  拓扑：%d 个节点，%d 条注册链路" % (n_nodes, n_edges))
    print("  ★ 每条弧吃一个 Tx 口 + 一个 Rx 口 ⟹ 每步弧数上限 = %d（= 节点数，"
          "**不是** 45：一个节点可以同时发一个、收一个）" % cap_arcs(n_nodes))
    print()

    per_seed, all_rows = [], []
    t0 = time.perf_counter()
    for seed in seeds:
        r = run_one(cfg, seed, int(profile.get("start_seed", 0)) + seed, n_nodes)
        per_seed.append(r)
        all_rows.extend(r["rows"])
        print("  seed %3d  起日 %3d  sr %.4f  平均每步：到 %5.1f 条 / 配 %5.1f 条 / 挂 %6.1f 条"
              % (seed, r["start_day"], r["sr"],
                 st.mean([x["n_arrived"] for x in r["rows"]]),
                 st.mean([x["n_matched"] for x in r["rows"]]),
                 st.mean([x["n_pending"] for x in r["rows"]])))
    print()

    s = summarize(all_rows, n_nodes)
    cap = cap_arcs(n_nodes)
    print("=== 全样本均值（%d 步 × %d 种子 = %d 步）===" % (a.steps, len(seeds), s["steps"]))
    print("  每步新到请求        %8.2f 条" % s["n_arrived"])
    print("  每步配成弧          %8.2f 条   （上限 %d）" % (s["n_matched"], cap))
    print("  端口利用率          %8.4f      （上限 1.0000）" % s["port_util"])
    print("  步末挂在队列        %8.2f 条" % s["n_pending"])
    print("  每步服务            %8.2f 条请求 / %12.1f 密钥" % (s["n_served"], s["served_keys"]))
    print("  每步超时作废        %8.2f 条" % s["n_expired"])
    print("  ★ 每条配成的边：服务 %.3f 条请求 / %.0f 密钥"
          % (s["served_per_match"], s["keys_per_match"]))
    print()

    print("=== 判据（先写死）===")
    print("  (a) 若 端口利用率 ≥ ~0.9 且 每步到 ≫ 每步配 ⟹ **匹配容量是墙**，")
    print("      该做的是加端口/加匹配容量，调奖励没用")
    print("  (b) 若 端口利用率 明显 < 0.9 ⟹ 匹配没提满，是**打法分**的问题")
    print("  (c) 若 每条边的服务量很低 ⟹ 配出来的边大量不在请求通路上，也是打分问题")
    print()
    if s["port_util"] >= 0.90:
        print("  ⟹ 实测 (a)：端口利用率 %.4f 贴着实测上限" % s["port_util"])
    elif s["port_util"] < 0.60:
        print("  ⟹ 实测 (b)：端口利用率 %.4f 远低于上限 ⟹ **匹配没提满**" % s["port_util"])
    else:
        print("  ⟹ 居中：端口利用率 %.4f，两头都不满" % s["port_util"])
    if s["n_arrived"] > s["n_matched"]:
        print("  ⟹ 每步新到请求 %.2f 条 > 每步配成弧 %.2f 条 ⟹ 一条边要覆盖多于一条请求"
              % (s["n_arrived"], s["n_matched"]))
    else:
        print("  ⟹ 每步配成弧 %.2f 条 ≫ 每步新到请求 %.2f 条 ⟹ **条数上完全不缺边**"
              % (s["n_matched"], s["n_arrived"]))
        print("     请求是**点对点**的（每次 0.8 条），而边是按**逐跳存量**服务的 ⟹")
        print("     真正的约束是「路径上每一跳都得有密钥」，不是「配得出多少条边」")

    Path(a.out).write_text(json.dumps(
        {"summary": s, "per_seed": [{k: v for k, v in r.items() if k != "rows"}
                                    for r in per_seed]},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n  用时 %.1fs，已写 %s" % (time.perf_counter() - t0, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
