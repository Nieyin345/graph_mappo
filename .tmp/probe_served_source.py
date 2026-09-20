"""机制体检 ③：**服务到底靠本步新密钥还是靠存量** —— 以及请求为什么没被服务。

### 为什么这一条最要命

`docs/训练诊断记录.md:12224` 记着 `mean_history_utilized` **恒为 0**，并特意说明
它不是"缺生产者"（`reward.py:318` 在算 `served - from_new`），是**生产者工作但结果为零**
⟹ **服务量全部由本步新生成的密钥满足，存量从未被动用**。

若这条在**真实协议**上也成立，那么服务一条请求的充要条件变成：

> 这条请求的**整条路径每一跳**都必须在**同一步**被激活

而不是"路径上攒够了密钥"。这两件事对策略的要求天差地别：
后者是慢变量（可以分步攒），前者是**一步内的联合动作** —— L 跳要同时指向同一条路。
`mutual_choice` 下每个节点独立采样，L 个节点同时选对是一件事
（虽然我实测过 argmax 下 `mutual_choice ≡ priority_matching` 逐位相同，
说明**解码**时不缺协调，但**训练**时按随机匹配拿奖励，协调仍可能是学习难点）。

### 本探针量什么（只读）

逐步，对每条在挂请求，用 env 自己的路由判据分类：

| 类 | 判据 |
|---|---|
| `served` | 本步 `served_amount` 增加 |
| `path_active_full` | 存在一条路径，**每一跳都是本步激活的**，但没服务成 |
| `path_positive_full` | 存在一条路径，**每一跳都有存量**，但没服务成 |
| `no_path` | 既没有全激活路径，也没有全存量路径 |

再单独记录每条 `serve_event` 的 `(served, from_new)`，直接验证
"服务全来自本步新密钥"这条。

**判据（先写死）**

- `from_new / served` ≈ 1.0 ⟹ 存量确实不参与 ⟹ 服务 = **一步内的整条路径联合激活**
  ⟹ 该往**协调/相位**方向改（或让存量能用），往奖励里加量纲没用
- `from_new / served` 明显 < 1 ⟹ 旧记录在**这套协议上不成立**，存量在起作用，
  则瓶颈回到"哪条跳没存量"，是**覆盖**问题
- `path_active_full` 占比高 ⟹ 路径激活了却没服务 ⟹ 问题在**服务阶段**
  （deadline / 路由选择 / 端口冲突），不在策略

用法（服务器上）：
    python -u /tmp/probe_served_source.py --seeds 100-109 --steps 240
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
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
        raise SystemExit("★ 协议没读到")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-109")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--out", default="/tmp/probe_served_source.json")
    a = ap.parse_args()

    profile = protocol()
    seeds = parse_seeds(a.seeds)
    cfg = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=a.steps,
        start_mode=profile["start_mode"])
    cfg["runtime"]["device"] = "cpu"

    tot = {"req_step": 0, "served": 0, "path_active_full": 0,
           "path_positive_full": 0, "no_path": 0}
    from_new_sum = 0.0
    served_sum = 0.0
    n_events = 0
    n_partial = 0
    per_seed = []

    for seed in seeds:
        env = build_env_from_config(cfg)
        obs = env.reset(seed=seed, start_seed=int(profile.get("start_seed", 0)) + seed)

        box = {"serve": None}
        orig_serve = env.requests.serve

        def serve_capture(*ar, **kw):
            r = orig_serve(*ar, **kw)
            box["serve"] = r
            return r

        env.requests.serve = serve_capture

        expert = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True,
                                 principles=False, router=ServeProbe(env))
        loc = dict(tot)
        loc_from_new = 0.0
        loc_served = 0.0
        loc_events = 0
        loc_partial = 0

        done = False
        while not done:
            # 服务前先记下每条请求的已服务量，用来判本步谁被服务了
            before = {id(r): r.served_amount for r in env.requests.pending}
            pending_before = list(env.requests.pending)
            actions, scores = expert.act(obs)
            obs, _r, term, trunc, _i = env.step(actions, scores)
            done = term or trunc

            sr = box["serve"]
            activated = set(env.last_activated_edges)
            positive = set(env.qkp.positive)

            # 每条在挂请求：本步服务了多少 / 有没有全激活路径 / 有没有全存量路径
            for req in pending_before:
                loc["req_step"] += 1
                d = req.served_amount - before.get(id(req), 0.0)
                if d > 1.0e-9:
                    loc["served"] += 1
                else:
                    path = env.routing.shortest_path(req.src_gs, req.dst_gs)
                    if path and all(e in activated for e in path):
                        loc["path_active_full"] += 1
                    elif path and all(e in positive for e in path):
                        loc["path_positive_full"] += 1
                    else:
                        loc["no_path"] += 1

            if sr is not None:
                loc_served += float(sr.served_keys)
                for _rid, s, fn in (sr.serve_events or []):
                    loc_from_new += float(fn)
                    loc_events += 1
                    if fn < float(s) - 1.0e-9:
                        loc_partial += 1

        for k in tot:
            tot[k] += loc[k]
        from_new_sum += loc_from_new
        served_sum += loc_served
        n_events += loc_events
        n_partial += loc_partial
        per_seed.append({"seed": seed, "from_new": loc_from_new,
                         "served": loc_served, "events": loc_events,
                         "req_step": loc["req_step"], "served_req": loc["served"]})
        print("  seed %3d  服务事件 %5d  其中含存量 %5d ｜ from_new/served = %.6f"
              % (seed, loc_events, loc_partial,
                 loc_from_new / loc_served if loc_served else float("nan")))
        env.requests.serve = orig_serve

    print()
    print("=== 服务来源（%d 步 × %d 种子）===" % (a.steps, len(seeds)))
    print("  服务密钥总量        %14.1f" % served_sum)
    print("  其中来自本步新密钥  %14.1f" % from_new_sum)
    print("  ★ from_new / served = %.6f   （1.0 = 存量完全不参与）"
          % (from_new_sum / served_sum if served_sum else float("nan")))
    print("  服务事件数 %d ｜ 其中含存量成分的 %d 次（%.4f%%）"
          % (n_events, n_partial, 100.0 * n_partial / n_events if n_events else float("nan")))
    print()
    print("=== 请求·步 分类（每条在挂请求 × 每一步）===")
    rs = tot["req_step"]
    print("  总请求·步          %8d" % rs)
    for k, label in (("served", "本步被服务"),
                     ("path_active_full", "有全激活路径但没服务"),
                     ("path_positive_full", "有全存量路径但没服务"),
                     ("no_path", "两样都没有")):
        print("  %-18s %8d  (%.4f%%)" % (label, tot[k], 100.0 * tot[k] / rs if rs else float("nan")))
    print()
    print("=== 判据 ===")
    r = from_new_sum / served_sum if served_sum else float("nan")
    if r > 0.999:
        print("  ⟹ from_new/served = %.6f ≈ 1 ⟹ **存量确实完全不参与服务**" % r)
        print("     服务一条请求 = **同一步内把它的整条路径全激活**。")
        print("     这是**一步内的联合动作**，不是慢变量 ⟹ 该往相位/协调改，")
        print("     或让存量真的能用（现在的 TTL/容量其实允许，是路径选择不选它）")
    elif r < 0.9:
        print("  ⟹ from_new/served = %.6f < 0.9 ⟹ **旧记录在这套协议上不成立**，" % r)
        print("     存量在起作用 ⟹ 瓶颈回到「哪条跳没存量」，是**覆盖**问题")
    else:
        print("  ⟹ from_new/served = %.6f 居中" % r)
    if tot["path_active_full"] > tot["served"]:
        print("  ⟹ 「有全激活路径却没服务」(%d) 比「被服务」(%d) 还多 ⟹ "
              "路径激活了但服务阶段没吃上（deadline/路由选择），不在策略上"
              % (tot["path_active_full"], tot["served"]))

    Path(a.out).write_text(json.dumps(
        {"totals": tot, "from_new": from_new_sum, "served": served_sum,
         "ratio": r, "events": n_events, "events_with_history": n_partial,
         "per_seed": per_seed}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写 %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
