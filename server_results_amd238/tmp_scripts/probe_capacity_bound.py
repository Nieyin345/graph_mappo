"""探针 G：给"容量不够"判一个**可证的下界**（不看链路忙不忙，看物理速率）。

## 为什么需要这个口径

探针 E 把缺口拆成"不可达 5.3 点 + 可达没搬完 16.7 点"，探针 F 想用
"链路是否出现在 matching 里"再分 B/C —— **那个口径是坏的**：最忙的一批边全是
`GS__HAP`，占用率恒为 1.000（"被接上"不等于"被占满"），于是几乎所有请求的
hop_busy_share 都是 1.0，分不出容量和调度。

这里改用物理速率。机制（`qkd_rl/env/routing.py`）：请求要成功，路径上每一跳的
key 库存都要 ≥ 需求量，而库存来自生成，生成速率是物理给定且分时可用
（`rate_provider.get_rate / is_available`）。

于是给每条失败请求算一个**乐观上界**：

    cap_opt = max over 路径    min over 该路径的每一跳   Σ_{t ∈ 生命期} rate(hop, t)

即"这条路整段生命期的生成量全给它一个人，吞吐由最慢那跳决定"。它忽略与别的
请求竞争，所以是**上界**：

  * `cap_opt < 剩余量` → 连独占用都不够 → **谁也服务不了**，记入 B（容量下界）
  * `cap_opt ≥ 剩余量` → 物理上搬得动，没搬成就是调度/竞争 → 记入 C

关键是"max over 路径"那一层。**第一版漏了它**，只算静态最短路，结果量出
"所有失败请求都容量不足、cap_opt 中位 0.000" —— 因为第 0 天前 30 槽里
GS↔LEO 卫星链路大多不可用（实测 `E_Tokyo__Sat_LEO_1a` 只在 t=0..2 有速率，
t≥3 就是 0），而静态最短路恰好经过不可用的那颗卫星。换成全路径的
**最宽路径（max-min）** 之后，同一批请求就有一条"经由 HAP 走廊"的可行路。

用法（节点上）：
    bash .tmp/run_cap.sh
"""

from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from qkd_rl.baselines.path_greedy import PathScoreGreedy  # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe  # noqa: E402
from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.rl.algos.checkpoint import load_checkpoint  # noqa: E402
from qkd_rl.rl.algos.policy import MAPPOPolicy  # noqa: E402
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic  # noqa: E402


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


def build_rate_table(env, horizon: int) -> tuple[dict[str, np.ndarray], dict, int]:
    """每条边在 t=0..horizon 的生成速率（不可用时记 0），一次算好。

    不这么做的话，每条请求都要重扫 30 个槽 × 1978 条边，12 个种子下来是
    上亿次 provider 查询。
    """
    rp = env.rate_provider
    adj: dict[str, list[tuple[str, str]]] = {}
    table: dict[str, np.ndarray] = {}
    for edge in env.routing.edges:
        arr = np.zeros(horizon, dtype=np.float64)
        for t in range(horizon):
            if rp.is_available(edge.edge_id, t):
                r = float(rp.get_rate(edge.edge_id, t))
                if r > 0:
                    arr[t] = r
        table[edge.edge_id] = np.cumsum(arr)
        adj.setdefault(edge.src, []).append((edge.dst, edge.edge_id))
        adj.setdefault(edge.dst, []).append((edge.src, edge.edge_id))
    return table, adj, len(table)


def widest_path(adj: dict, src: str, dst: str, table: dict, a: int, d: int) -> float:
    """最大化 路径上最慢一跳 的 生命期生成总量（max-min / 最宽路径）。"""
    if src == dst:
        return float("inf")
    best: dict[str, float] = {src: float("inf")}
    pq: list[tuple[float, str]] = [(-float("inf"), src)]
    while pq:
        neg, u = heapq.heappop(pq)
        cur = -neg
        if u == dst:
            return cur
        if cur < best.get(u, -1.0):
            continue
        for v, e in adj.get(u, ()):
            arr = table.get(e)
            spec = 0.0 if arr is None else float(arr[min(d, len(arr)) - 1]
                                                 - (arr[a - 1] if a > 0 else 0.0))
            val = cur if spec > cur else spec
            if val > best.get(v, -1.0):
                best[v] = val
                heapq.heappush(pq, (-val, v))
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--policy", default="expert", choices=["expert", "rl"])
    ap.add_argument("--seeds", default="7-18")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--config", default=str(ROOT / "configs" / "var2_diag.yaml"))
    args = ap.parse_args()

    profile = _tp.load_validation_profile(Path(args.config))
    steps = args.steps or int(profile["episode_steps"]) or 240
    seeds = parse_seeds(args.seeds)
    config = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=steps,
        start_mode=profile["start_mode"])
    start_seed = int(profile.get("start_seed", 0))
    horizon = steps + 40                     # 覆盖 deadline（到达 + 30 槽）

    print(f"策略={args.policy}  种子={seeds}  步数={steps}  "
          f"窗={profile['window_start_day']}-{profile['window_end_day']}  "
          f"模式={profile['start_mode']}")

    tot_arrived = tot_served = 0.0
    rows: list[dict] = []
    for seed in seeds:
        env = build_env_from_config(config)
        if args.policy == "expert":
            policy = PathScoreGreedy(
                weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True,
                principles=False, router=ServeProbe(env))
            is_expert = True
        else:
            model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
            data = load_checkpoint(args.checkpoint, device="cpu")
            model.load_state_dict(data.model_state)
            model.eval()
            policy = MAPPOPolicy(model, "cpu")
            is_expert = False

        obs = env.reset(seed=seed, start_seed=start_seed + seed)
        table, adj, n_edges = build_rate_table(env, horizon)
        queue = env.requests
        orig_expire = queue.expire

        def expire_patched(t: int, _env=env, _rows=rows, _table=table, _adj=adj):
            out = orig_expire(t)
            for req in out:
                rem = max(0.0, float(req.amount) - float(req.served_amount))
                path = _env.routing.shortest_path(req.src_gs, req.dst_gs)
                a, d = int(req.arrival_t), int(req.deadline_t)
                row = {
                    "seed": seed, "src": req.src_gs, "dst": req.dst_gs,
                    "amount": float(req.amount), "remaining": rem,
                    "served": float(req.served_amount),
                    "arrival_t": a, "deadline_t": d,
                    "static_hops": -1 if path is None else len(path),
                    "cap_opt": 0.0, "ratio": 0.0,
                    "hops_used": 0,
                }
                if path is not None and rem > 0:
                    cap = widest_path(_adj, req.src_gs, req.dst_gs, _table, a,
                                      min(d, horizon))
                    row["cap_opt"] = cap
                    row["ratio"] = cap / max(1e-9, rem)
                _rows.append(row)
            return out

        queue.expire = expire_patched
        n = 0
        done = False
        while n < steps and not done:
            if is_expert:
                actions, scores = policy.act(obs)
            else:
                with torch.no_grad():
                    step = policy.act(obs)
                actions, scores = step.actions, step.action_scores
            obs, _r, term, trunc, _info = env.step(actions, action_scores=scores)
            n += 1
            done = term or trunc
        queue.expire = orig_expire

        s = env.metrics.episode_summary()
        tot_arrived += s["arrived_keys"]
        tot_served += s["served_keys"]
        print(f"  seed={seed} 成功率={s['success_rate']:.4f}  "
              f"过期请求={len(rows)}（累计）  边数={n_edges}")

    print(f"\n量加权成功率={tot_served / max(1e-9, tot_arrived):.4f}"
          f"  到达={tot_arrived:,.0f}  服务={tot_served:,.0f}")

    unreach = [r for r in rows if r["static_hops"] < 0]
    reach = [r for r in rows if r["static_hops"] >= 0 and r["remaining"] > 0]
    b_fail = [r for r in reach if r["ratio"] < 1.0]
    c_fail = [r for r in reach if r["ratio"] >= 1.0]

    def pct(x: float) -> str:
        return f"{x / max(1e-9, tot_arrived):.2%}"

    print("\n过期请求分类（占**到达量**的比例）：")
    for name, g in (("A 不可达（拓扑）", unreach),
                    ("B 容量不足（可证下界）", b_fail),
                    ("C 物理上搬得动却没搬", c_fail)):
        print(f"  {name:<22} n={len(g):>4}  "
              f"剩余量={sum(r['remaining'] for r in g):,.0f}  "
              f"{pct(sum(r['remaining'] for r in g))}")
    print(f"  过期总量                n={len(rows):>4}  "
          f"剩余量={sum(r['remaining'] for r in rows):,.0f}  "
          f"{pct(sum(r['remaining'] for r in rows))}")

    if c_fail:
        ratios = sorted(r["ratio"] for r in c_fail)
        print(f"\nC 档 cap_opt/剩余 分位：p10={ratios[int(0.10 * (len(ratios) - 1))]:.2f}  "
              f"中位={statistics.median(ratios):.2f}  "
              f"p90={ratios[int(0.90 * (len(ratios) - 1))]:.2f}")
    if b_fail:
        ratios = sorted(r["ratio"] for r in b_fail)
        print(f"B 档 cap_opt/剩余 中位={statistics.median(ratios):.3f}（<1 即独占用也不够）")

    print("\n按静态最短路跳数（可达且过期）：")
    for h in sorted({r["static_hops"] for r in reach}):
        g = [r for r in reach if r["static_hops"] == h]
        gb = [r for r in g if r["ratio"] < 1.0]
        print(f"  hops={h}: n={len(g):>4}  容量不足 {len(gb):>4} ({len(gb) / max(1, len(g)):.0%})  "
              f"容量不足剩余量={sum(r['remaining'] for r in gb):,.0f}")

    out = Path(f"outputs/eval/capacity_bound_{args.policy}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"policy": args.policy, "seeds": seeds, "steps": steps,
               "arrived": tot_arrived, "served": tot_served, "rows": rows},
              out.open("w", encoding="utf-8"), indent=2)
    print(f"\n已保存 {out}")


if __name__ == "__main__":
    main()