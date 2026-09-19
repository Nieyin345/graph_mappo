"""探针 F：过期请求的路径，在它活着的时候到底被谁占着。

背景：探针 E 把 22 个点的缺口拆开了 —— 5.3 点是拓扑不可达（A，谁都没办法），
剩下 ~16.7 点是"路径存在却没搬完"。但那 16.7 点还分不出两件事：

  B. **容量真的不够**：路径一直被别人占着，物理上搬不完这么多量；
  C. **调度没做**：路径大半时间是空的，却没把请求接上去。

修法完全不同 —— B 要改场景/目标，C 才说明 RL 还有空间。这个探针给一个粗分界。

做法（环境一行不改）：
  * 每一步记下 ``env.last_matched_arcs``（这一步真正被接上的定向弧），
    折算成无向 edge_id 的集合 —— "这一步这条链路忙不忙"。
  * 每条请求过期时（monkey-patch ``RequestQueue.expire`` 拿到 req 和 t），
    取它存活期 ``[arrival_t, deadline_t)`` 内每一步的忙闲状态。
  * 对这条请求的**静态最短路**（``routing.shortest_path``，A 那类返回 None），算：

      hop_busy_share   存活期里"至少有一跳在忙"的步数占比（链路视角的占用）
      all_idle_share   存活期里"所有跳都空着"的步数占比（真正能搬却没搬的窗口）
      completion       已服务量 / 需求量

  * 全时段还有一条系统级读数：**每条边的占用率 = 忙步数 / 总步数**，
    它的均值就是"整个网络到底有多满"。这条最直接：如果网络整体只用到两成，
    那 16.7 点的缺口就不可能是容量问题。

用法（节点上）：
    bash .tmp/run_occ.sh          # 专家 + BC 各跑一遍，见该脚本
"""

from __future__ import annotations

import argparse
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

import torch  # noqa: E402

from qkd_rl.baselines.path_greedy import PathScoreGreedy  # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe  # noqa: E402
from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.env.request import RequestQueue  # noqa: E402
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


def arc_to_edge_index(routing) -> dict[tuple[str, str], str]:
    """(src,dst) 与 (dst,src) 都映射到同一个 edge_id —— 路由把边当无向用。"""
    idx: dict[tuple[str, str], str] = {}
    for e in routing.edges:
        idx[(e.src, e.dst)] = e.edge_id
        idx[(e.dst, e.src)] = e.edge_id
    return idx


def run_episode(env, policy, seed: int, steps: int, start_seed: int, is_expert: bool,
                busy: list[set[str]], expired: list[dict]) -> dict:
    obs = env.reset(seed=seed, start_seed=start_seed + seed)
    busy.clear()
    expired.clear()

    queue = env.requests
    orig_expire = queue.expire
    arc2edge = arc_to_edge_index(env.routing)

    def expire_patched(t: int):
        rows = orig_expire(t)
        for req in rows:
            expired.append({
                "req": req.request_id,
                "src": req.src_gs,
                "dst": req.dst_gs,
                "arrival_t": int(req.arrival_t),
                "deadline_t": int(req.deadline_t),
                "amount": float(req.amount),
                "served": float(req.served_amount),
                "remaining": max(0.0, float(req.amount) - float(req.served_amount)),
            })
        return rows

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
        busy.append({arc2edge[a] for a in env.last_matched_arcs if a in arc2edge})
        n += 1
        done = term or trunc

    queue.expire = orig_expire
    return env.metrics.episode_summary()


def classify(expired: list[dict], busy: list[set[str]], routing) -> dict:
    per_req = []
    for r in expired:
        path = routing.shortest_path(r["src"], r["dst"])
        if not path:
            continue                      # 不可达，探针 E 已经处理过这一档
        a, d = r["arrival_t"], min(r["deadline_t"], len(busy))
        a = max(0, a)
        if d <= a:
            continue
        steps_life = d - a
        any_busy = sum(1 for t in range(a, d) if any(e in busy[t] for e in path))
        all_idle = sum(1 for t in range(a, d) if not any(e in busy[t] for e in path))
        per_req.append({
            **r,
            "hops": len(path),
            "completion": r["served"] / max(1e-9, r["amount"]),
            "hop_busy_share": any_busy / steps_life,
            "all_idle_share": all_idle / steps_life,
        })
    return {"rows": per_req, "n_steps": len(busy)}


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

    print(f"策略={args.policy}  种子={seeds}  步数={steps}  "
          f"窗={profile['window_start_day']}-{profile['window_end_day']}  "
          f"模式={profile['start_mode']}")

    all_rows: list[dict] = []
    edge_busy_total: dict[str, int] = {}
    edge_slots_total = 0
    n_edges = 0
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

        busy: list[set[str]] = []
        expired: list[dict] = []
        run_episode(env, policy, seed, steps, start_seed, is_expert, busy, expired)
        n_edges = len(env.routing.edges)
        edge_slots_total += len(busy) * n_edges
        for s in busy:
            for e in s:
                edge_busy_total[e] = edge_busy_total.get(e, 0) + 1
        all_rows.extend(classify(expired, busy, env.routing)["rows"])

    # ---- 系统级：整网占用率 ----
    util = sum(edge_busy_total.values()) / max(1, edge_slots_total)
    used = {e: c for e, c in edge_busy_total.items()}
    steps_total = edge_slots_total / max(1, n_edges)
    util_used = sum(used.values()) / max(1e-9, len(used) * steps_total)
    per_edge = sorted(used.values(), reverse=True)
    q = lambda f: (per_edge[int(f * (len(per_edge) - 1))] / steps_total) if per_edge else 0.0
    print(f"\n整网链路占用率（全部 {n_edges} 条边）          = {util:.4f}")
    print(f"只用过传输的 {len(used)} 条边的占用率（公平分母） = {util_used:.4f}")
    print(f"  单边占用率（在用过传输的边里）：p50={q(0.5):.4f}  p90={q(0.9):.4f}  "
          f"max={q(1.0):.4f}")
    print(f"  是死的（全程一次没传过）的边：{n_edges - len(used)}/{n_edges}"
          f" = {(n_edges - len(used)) / max(1, n_edges):.1%}")
    print(f"  种子数={len(seeds)}  每种子步数={steps_total:.0f}")
    if per_edge:
        top = sorted(used.items(), key=lambda kv: -kv[1])[:5]
        print("  最忙的 5 条边：" + "  ".join(
            f"{e}={c / steps_total:.2f}" for e, c in top))

    # ---- 请求级：可达但过期的那些 ----
    rows = [r for r in all_rows]
    low = [r for r in rows if r["completion"] < 0.5]
    hi = [r for r in rows if r["completion"] >= 0.5]
    print(f"\n可达且过期的请求数={len(rows)}（完成度<0.5 的 {len(low)} 条，"
          f"占过期量 {sum(r['remaining'] for r in low) / max(1e-9, sum(r['remaining'] for r in rows)):.1%}）")

    def stat(group: list[dict], name: str) -> None:
        if not group:
            print(f"  {name}: 无")
            return
        print(f"  {name}: n={len(group):>4}  "
              f"hop_busy_share 中位 {statistics.median(r['hop_busy_share'] for r in group):.3f}  "
              f"all_idle_share 中位 {statistics.median(r['all_idle_share'] for r in group):.3f}  "
              f"完成度中位 {statistics.median(r['completion'] for r in group):.3f}")

    print("\n按完成度分组（hops 全部可达）：")
    stat(rows, "全部过期")
    stat(low, "完成度<0.5")
    stat(hi, "完成度>=0.5")

    print("\n按跳数（完成度<0.5 组）：")
    for h in sorted({r["hops"] for r in low}):
        g = [r for r in low if r["hops"] == h]
        stat(g, f"  hops={h}")

    out = Path(f"outputs/eval/path_occupancy_{args.policy}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"policy": args.policy, "seeds": seeds, "steps": steps,
               "link_utilization": util, "link_utilization_used": util_used,
               "n_edges": n_edges, "edges_used": len(used),
               "edge_busy_total": used, "steps_total": steps_total,
               "rows": all_rows},
              out.open("w", encoding="utf-8"), indent=2)
    print(f"\n已保存 {out}")


if __name__ == "__main__":
    main()