"""流体松弛 LP 上界（2026-09-27，替代无效的 MILP 上界）。

★ 为什么有这个文件：scripts/milp/compute_milp_upper_bound.py 只用 t0 时刻
obs.physical_edge_ids 快照（194/1948 条边）建图，且未开 relax_multi_path ——
实测只覆盖 2.24% 的到达需求、并把无路请求移出分母，报出的 0.95/1.0 无效。
本文件用全部注册边 + 全部到达需求做分母，每一步都是松弛，结果是严格上界。
测试：tests/test_fluid_upper_bound.py（手算小算例 + 变异检查）。

对一个回合（同 RL 验证任务：--run 的 experiment_config + overlays，同种子），
求「任何策略 + 任何路由」能达到的成功率（served/arrived，按密钥量）的上界。

松弛（每一条都只放宽可行域 ⟹ LP 最优值 ≥ 真实最优 ⟹ 严格上界）：
  * 激活 a[e,t] ∈ [0,1] 连续；端口：每节点每时隙 Σ_{e∋v} a ≤ 2（一发一收匹配的松弛，
    整数时恰好等价：度≤2 的子图总能定向成每点 ≤1 出 ≤1 入）；
  * 生成 = a·rate·avail·slot_seconds，忽略切换半速（env 首槽 ×0.5）；
  * 忽略 QKP 库存上限（8e6–2e7 ≫ 单请求 ≤5e5）与 TTL（配置为 1e6 步）；
  * 任意多路径拆分、任意路由（env 用 BFS 最短路，不比这好）；
  * 截止期：请求 r 消耗的每条边的密钥必须在其最后可服务时隙之前生成 —— 按 B 槽分桶，
    截止时隙**向上**取整到桶末（再放宽）。库存递推 D[e,k] = D[e,k-1] + 桶内生成 − 桶内到期请求的流量 ≥ 0。
  * 初始库存 0（env: qkp.initial_level = 0）；回合结束仍未服务的请求算未服务（同 env 口径）。

分母 = 回合内全部到达量（含无路可走的请求）。同时跑专家做健全性检查：专家逐种子必须 ≤ 上界。

用法（服务器）：
  python scripts/milp/fluid_upper_bound.py --run outputs/experiments/v8/<run> --seeds 100-102 [--overlay x.yaml] \
      [--bucket 10] [--no-expert] --out outputs/eval/fluid_ub_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qkd_rl.core.config import deep_merge, load_config  # noqa: E402
from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.evaluation import test_protocol as tp  # noqa: E402

SCALE = 1e5  # 密钥量缩放，改善 LP 数值


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def task_config(run: str, overlays: list[str]) -> tuple[dict, dict]:
    cfg = load_config([Path(run) / "experiment_config.yaml"])
    for name in overlays:
        p = Path(name)
        cfg = deep_merge(cfg, load_config([p if p.exists() else ROOT / "configs" / name]))
    vcfg = cfg.get("validation", {}) or {}
    return tp.run_validation_env_config(cfg, vcfg), vcfg


def rollout_collect(env, obs, use_expert: bool) -> tuple[list, float | None, int]:
    """跑满一个回合，记录真实到达请求；use_expert 时同时得到专家成功率。"""
    arrivals: list = []
    gen = env.request_generator
    orig = gen.generate

    def rec(t, _orig=orig):
        out = _orig(t)
        arrivals.extend(out)
        return out

    gen.generate = rec
    t0 = int(obs.state.t)
    expert = None
    if use_expert:
        from qkd_rl.baselines.path_greedy import PathScoreGreedy
        from qkd_rl.baselines.serve_probe import ServeProbe
        expert = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True,
                                 principles=False, router=ServeProbe(env))
    done = False
    while not done:
        if expert is not None:
            actions, scores = expert.act(obs)
        else:
            actions, scores = {}, None
        obs, _r, term, trunc, _info = env.step(actions, scores)
        done = term or trunc
    gen.generate = orig
    sr = env.metrics.episode_summary().get("success_rate") if use_expert else None
    return arrivals, sr, t0


def solve_bound(env, obs0_t: int, T: int, arrivals: list, bucket: int, threads: int = 1,
                integer_activation: bool = False, time_limit: float = 3600.0,
                mip_rel_gap: float = 1e-4) -> dict:
    """integer_activation=True（改进项 5，2026-09-28）：激活 a[e,t] ∈ {0,1}。
    度 ≤2 的整数子图（路径/环）总能定向成每点 ≤1 发 ≤1 收 ⟹ 端口约束变为精确，
    其余松弛不变 ⟹ 仍是严格上界且 ≤ LP 上界。未在 time_limit 内证最优时取
    MILP 对偶界（HiGHS mip_dual_bound，仍是上界）。默认 False = 原 LP，逐位不变。"""
    t0 = obs0_t
    # ---- 边：env 注册的全部边（= QKP 池 = 路由图），不是 t0 快照
    edges = list(env.routing.edges)
    eids = [e.edge_id for e in edges]
    provider = env.rate_provider
    idx = provider.edge_link_ids(eids)
    slot = float(env.scenario.slot_seconds)
    G = np.zeros((T, len(eids)))
    for tau in range(T):
        blocks = provider.get_window_blocks(t0 + tau)
        rate = np.asarray(blocks[0][0][idx], dtype=float)
        av = np.asarray(blocks[1][0][idx], dtype=float) >= 0.5
        G[tau] = np.where(av, rate, 0.0) * slot / SCALE
    keep = G.max(axis=0) > 0  # 窗口内从不可生成的边不可能有密钥 → 删掉不影响最优值
    E = [i for i in range(len(eids)) if keep[i]]
    ecol = {i: j for j, i in enumerate(E)}
    src = [edges[i].src for i in E]
    dst = [edges[i].dst for i in E]
    nodes = sorted(set(src) | set(dst))
    nid = {v: k for k, v in enumerate(nodes)}
    nE, nV = len(E), len(nodes)
    K = (T + bucket - 1) // bucket

    reqs = []
    total = 0.0
    for r in arrivals:
        total += float(r.amount)
        last = min(int(r.deadline_t), t0 + T - 1) - t0
        if last < 0:
            continue
        if r.src_gs in nid and r.dst_gs in nid:
            reqs.append((nid[r.src_gs], nid[r.dst_gs], float(r.amount) / SCALE, last // bucket))
    R = len(reqs)

    # ---- 变量布局
    at = [(j, tau) for j, i in enumerate(E) for tau in range(T) if G[tau, i] > 0]
    nA = len(at)
    off_x, off_a = 0, R
    off_f = off_a + nA
    nF = R * nE * 2
    off_d = off_f + nF
    nD = nE * K
    nvar = off_d + nD

    def fcol(r, j, d):
        return off_f + (r * nE + j) * 2 + d

    rows_eq, cols_eq, vals_eq = [], [], []
    row = 0
    # 流守恒：每请求每节点  out − in − [v=s]x + [v=t]x = 0
    for r, (s, t, _amt, _k) in enumerate(reqs):
        base = row
        for j in range(nE):
            u, v = nid[src[j]], nid[dst[j]]
            # d=0: u→v, d=1: v→u
            rows_eq += [base + u, base + v, base + v, base + u]
            cols_eq += [fcol(r, j, 0), fcol(r, j, 0), fcol(r, j, 1), fcol(r, j, 1)]
            vals_eq += [1.0, -1.0, 1.0, -1.0]
        rows_eq += [base + s, base + t]
        cols_eq += [off_x + r, off_x + r]
        vals_eq += [-1.0, 1.0]
        row += nV
    # 库存递推：D[e,k] − D[e,k−1] − Σ_{τ∈桶k} g a + Σ_{r:k_r=k} (f0+f1) = 0
    stock_row0 = row
    for j in range(nE):
        for k in range(K):
            rr = stock_row0 + j * K + k
            rows_eq.append(rr); cols_eq.append(off_d + j * K + k); vals_eq.append(1.0)
            if k > 0:
                rows_eq.append(rr); cols_eq.append(off_d + j * K + k - 1); vals_eq.append(-1.0)
    for c, (j, tau) in enumerate(at):
        rows_eq.append(stock_row0 + j * K + tau // bucket)
        cols_eq.append(off_a + c)
        vals_eq.append(-G[tau, E[j]])
    for r, (_s, _t, _amt, k) in enumerate(reqs):
        for j in range(nE):
            rr = stock_row0 + j * K + k
            rows_eq += [rr, rr]
            cols_eq += [fcol(r, j, 0), fcol(r, j, 1)]
            vals_eq += [1.0, 1.0]
    row = stock_row0 + nE * K
    A_eq = sp.csr_matrix((vals_eq, (rows_eq, cols_eq)), shape=(row, nvar))
    b_eq = np.zeros(row)

    # 端口：每节点每时隙 Σ a ≤ 2
    rows_ub, cols_ub = [], []
    for c, (j, tau) in enumerate(at):
        for v in (nid[src[j]], nid[dst[j]]):
            rows_ub.append(v * T + tau); cols_ub.append(off_a + c)
    A_ub = sp.csr_matrix((np.ones(len(rows_ub)), (rows_ub, cols_ub)), shape=(nV * T, nvar))
    b_ub = np.full(nV * T, 2.0)

    lb = np.zeros(nvar)
    ub = np.full(nvar, np.inf)
    ub[off_x:off_x + R] = [q[2] for q in reqs]
    ub[off_a:off_a + nA] = 1.0
    c = np.zeros(nvar)
    c[off_x:off_x + R] = -1.0

    t_s = time.perf_counter()
    mip_info: dict = {}
    if not integer_activation:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=np.column_stack([lb, ub]), method="highs",
                      options={"presolve": True, "time_limit": 3600.0})
        solve_s = time.perf_counter() - t_s
        served = -res.fun * SCALE if res.status == 0 else float("nan")
        x = res.x[off_x:off_x + R] if res.status == 0 else None
        status, message = int(res.status), str(res.message)
    else:
        from scipy.optimize import Bounds, LinearConstraint, milp

        integrality = np.zeros(nvar)
        integrality[off_a:off_a + nA] = 1
        res = milp(c, integrality=integrality, bounds=Bounds(lb, ub),
                   constraints=[LinearConstraint(A_ub, -np.inf, b_ub),
                                LinearConstraint(A_eq, b_eq, b_eq)],
                   options={"presolve": True, "time_limit": float(time_limit),
                            "mip_rel_gap": float(mip_rel_gap), "disp": False})
        solve_s = time.perf_counter() - t_s
        dual = getattr(res, "mip_dual_bound", None)
        # 最小化 −Σx：对偶界是 min 的下界 ⟹ −dual 是服务量的上界。status 0 = 证最优。
        if res.status == 0 and res.fun is not None:
            bound = -float(res.fun) if dual is None or not np.isfinite(dual) else -float(dual)
        elif dual is not None and np.isfinite(dual):
            bound = -float(dual)
        else:
            bound = float("nan")
        served = bound * SCALE
        x = res.x[off_x:off_x + R] if res.x is not None else None
        # lp_status 在整数模式下表示"上界有效"（0）；原始求解状态放 mip_status。
        status = 0 if np.isfinite(served) else int(res.status) or 9
        message = str(res.message)
        mip_info = {
            "mip_status": int(res.status),
            "mip_gap": float(getattr(res, "mip_gap", float("nan")) or 0.0),
            "mip_node_count": int(getattr(res, "mip_node_count", 0) or 0),
            "incumbent_served": (-float(res.fun) * SCALE) if res.fun is not None else None,
            "n_binary": nA, "time_limit": float(time_limit),
        }
    routable = sum(q[2] for q in reqs) * SCALE
    return {
        "t0": t0, "arrived": total, "n_requests": len(arrivals), "n_modeled": R,
        "routable_amount": routable, "ub_served": served,
        "ub_success": served / total if total else 0.0,
        "lp_status": status, "lp_message": message[:200],
        "solve_s": round(solve_s, 1), "n_edges": nE, "n_nodes": nV, "n_vars": nvar,
        "n_rows": int(A_eq.shape[0] + A_ub.shape[0]), "bucket": bucket,
        "req_saturated": int(sum(1 for q, xv in zip(reqs, x) if xv >= q[2] - 1e-6)) if x is not None else None,
        "integer": bool(integer_activation), **mip_info,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", default="100")
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--bucket", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-expert", action="store_true")
    ap.add_argument("--integer", action="store_true",
                    help="整数激活（精确端口约束）；未证最优时报 MILP 对偶界")
    ap.add_argument("--time-limit", type=float, default=3600.0, help="--integer 的 HiGHS 时间限制（秒）")
    ap.add_argument("--mip-rel-gap", type=float, default=1e-4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(f"{out} 已存在（outputs 只增不改）")
    config, vcfg = task_config(a.run, a.overlay)
    fp = tp.task_fingerprint(config)
    T = int(config["env"]["episode_steps"])
    base = int(vcfg.get("start_seed", 0) or 0)
    print(f"task fingerprint {fp}; T={T}; bucket={a.bucket}", flush=True)
    rows = []
    for seed in parse_seeds(a.seeds):
        env = build_env_from_config(config)
        obs = env.reset(seed=seed, start_seed=base + seed)
        arrivals, expert_sr, t0 = rollout_collect(env, obs, not a.no_expert)
        r = solve_bound(env, t0, T, arrivals, a.bucket, a.threads,
                        integer_activation=a.integer, time_limit=a.time_limit,
                        mip_rel_gap=a.mip_rel_gap)
        r.update(seed=seed, expert_success=expert_sr)
        rows.append(r)
        print(json.dumps(r), flush=True)
        if expert_sr is not None and expert_sr > r["ub_success"] + 1e-6:
            print(f"!!! seed {seed}: 专家 {expert_sr:.4f} > 上界 {r['ub_success']:.4f} —— 上界有错", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"task_fingerprint": fp, "run": a.run, "overlays": a.overlay,
                               "bucket": a.bucket, "integer": a.integer,
                               "time_limit": a.time_limit, "rows": rows}, indent=2), encoding="utf-8")
    ok = [r for r in rows if r["lp_status"] == 0]
    if ok:
        print(f"mean UB = {np.mean([r['ub_success'] for r in ok]):.4f} over {len(ok)} seeds; "
              f"routable share = {np.mean([r['routable_amount']/r['arrived'] for r in ok]):.4f}")


if __name__ == "__main__":
    main()
