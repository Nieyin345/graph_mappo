"""先知执行（2026-10-03）：离线最优的「可达下界」，与流体 LP 上界一起把离线最优夹在区间里。

做法：求流体 LP（含切换开销，--switch-decay cfg）的激活解 a[e,τ]；回合中每一步按 a 值
（+ 上一槽已激活的微小保持奖励，+ 速率破平）贪心选出每节点度 ≤2 的子图，定向成
「每点 ≤1 发 ≤1 收」后作为 mutual_choice 动作交给 env 真实执行（切换开销、EDF 服务、
单路径逐跳瓶颈、QKP 容量全部按 env 实际规则）。得到的成功率是「已知全部未来请求时
确实能做到」的值 ⟹ 离线最优 ∈ [oracle_exec, LP 上界]。只读诊断，不改 env。
用法：python scripts/milp/oracle_exec.py --run <run> --seeds 1000-1029 [--overlay x.yaml] --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qkd_rl.env.action_space import NodeActionSpace  # noqa: E402
from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.evaluation import test_protocol as tp  # noqa: E402
from scripts.milp.fluid_upper_bound import (  # noqa: E402
    parse_seeds, rollout_collect, solve_bound, task_config,
)


def orient(chosen: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """度 ≤2 的无向子图（路径/环）定向：每点至多一条出弧、一条入弧。"""
    adj: dict[str, list[str]] = defaultdict(list)
    for u, v in chosen:
        adj[u].append(v)
        adj[v].append(u)
    seen: set[frozenset] = set()
    arcs: list[tuple[str, str]] = []

    def walk(cur: str) -> None:
        while True:
            nxt = next((w for w in adj[cur] if frozenset((cur, w)) not in seen), None)
            if nxt is None:
                return
            seen.add(frozenset((cur, nxt)))
            arcs.append((cur, nxt))
            cur = nxt

    for n in list(adj):
        if len(adj[n]) == 1:
            walk(n)
    for n in list(adj):
        walk(n)
    return arcs


def oracle_actions(obs, plan: dict[int, dict[str, float]], t0: int, ends: dict[str, tuple[str, str]],
                   keep_bonus: float, fill: bool = False):
    """fill=True（opt-in）：LP 选中的边定下后，剩余端口再按（上一槽已激活优先, 速率）贪心补满。
    多激活的链路只会多产密钥、不占 LP 边的端口 ⟹ 不损害 LP 方案；也让示范不再「大量空闲」
    （BC 学生没有先知信息，学到大量 IDLE 会几乎不激活任何链路，实测 SR≈.01）。"""
    tau = int(obs.state.t) - t0
    cand = {n: set(obs.action_candidates[n]) for n in obs.node_ids}
    last = set(obs.state.last_activated_edges)
    windows = obs.state.edge_windows
    scored = []
    for eid, a in plan.get(tau, {}).items():
        u, v = ends[eid]
        if v not in cand.get(u, ()) or u not in cand.get(v, ()):
            continue
        try:
            rate = float(windows[eid].rates[0])
        except Exception:
            rate = 0.0
        scored.append((a + (keep_bonus if eid in last else 0.0), rate, eid, u, v))
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    if fill:
        in_plan = {x[2] for x in scored}
        extra = []
        for eid, (u, v) in ends.items():
            if eid in in_plan or v not in cand.get(u, ()) or u not in cand.get(v, ()):
                continue
            try:
                rate = float(windows[eid].rates[0])
            except Exception:
                rate = 0.0
            if rate <= 0:
                continue
            extra.append((-1.0 + (keep_bonus if eid in last else 0.0), rate, eid, u, v))
        extra.sort(key=lambda x: (-x[0], -x[1], x[2]))
        scored += extra
    deg: dict[str, int] = defaultdict(int)
    chosen = []
    for _s, _r, _e, u, v in scored:
        if deg[u] < 2 and deg[v] < 2:
            chosen.append((u, v))
            deg[u] += 1
            deg[v] += 1
    tx, rx = {}, {}
    for u, v in orient(chosen):
        tx[u] = v
        rx[v] = u
    idle = NodeActionSpace.IDLE
    return {n: (tx.get(n, idle), rx.get(n, idle)) for n in obs.node_ids}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", default="1000")
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--bucket", type=int, default=1)
    ap.add_argument("--keep-bonus", type=float, default=0.05)
    ap.add_argument("--no-expert", action="store_true")
    ap.add_argument("--fill", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(f"{out} 已存在（outputs 只增不改）")
    config, vcfg = task_config(a.run, a.overlay)
    fp = tp.task_fingerprint(config)
    T = int(config["env"]["episode_steps"])
    scfg = config.get("env", {}).get("switch_cost", {})
    sd = float(scfg.get("rate_decay_factor", 0.5)) if scfg.get("enabled", True) else 1.0
    base = int(vcfg.get("start_seed", 0) or 0)
    rows = []
    for seed in parse_seeds(a.seeds):
        env = build_env_from_config(config)
        obs = env.reset(seed=seed, start_seed=base + seed)
        arrivals, expert_sr, t0 = rollout_collect(env, obs, not a.no_expert)
        r = solve_bound(env, t0, T, arrivals, a.bucket, switch_decay=sd, return_activation=True)
        plan: dict[int, dict[str, float]] = defaultdict(dict)
        for eid, tau, val in r.pop("activation", []):
            plan[tau][eid] = val
        env2 = build_env_from_config(config)
        obs = env2.reset(seed=seed, start_seed=base + seed)
        assert int(obs.state.t) == t0, "episode start mismatch"
        ends = {e.edge_id: (e.src, e.dst) for e in env2.routing.edges}
        done = False
        while not done:
            acts = oracle_actions(obs, plan, t0, ends, a.keep_bonus, a.fill)
            obs, _r, term, trunc, _i = env2.step(acts, None)
            done = term or trunc
        orc = env2.metrics.episode_summary().get("success_rate")
        row = {"seed": seed, "t0": t0, "ub_success": r["ub_success"], "oracle_exec": orc,
               "expert_success": expert_sr, "lp_status": r["lp_status"], "switch_decay": sd,
               "n_requests": r["n_requests"], "arrived": r["arrived"]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"task_fingerprint": fp, "run": a.run, "overlays": a.overlay,
                               "bucket": a.bucket, "keep_bonus": a.keep_bonus, "fill": a.fill, "rows": rows},
                              indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
