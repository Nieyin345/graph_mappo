"""探针 J：那 8.3% 的"结构上不可能"，有多少是 **deadline 太短**造成的？

## 为什么这个能离线算

探针 H 判出 B_avail = 8.3–8.8% 的到达量落在"源-目的在生命期内从未同时连通"。
这一档**和策略完全无关**：它只取决于 (src, dst, 到达时刻, deadline) 四件事，
而这四件事里前三件由请求流决定、第四件是场景常数（`deadline_steps`）。

所以不需要跑任何策略 —— 只要**记录请求流**（谁、从哪到哪、什么时候到、多少量），
再拿探针 H 那份"逐槽连通分量"表，就能直接算出：**把 deadline 从 30 槽拉到
60 / 120 / 240 槽，`avail_slots == 0` 的量会掉到多少**。

这就把"要不要换题"从一个争论变成一个数字。

## 口径

对每条到达的请求：
    d' = 到达 + D                （D 是候选 deadline 长度）
    avail_slots(D) = #{t ∈ [到达, min(d', horizon)) : src 与 dst 同分量}
统计"需求加权"的 `avail_slots(D) == 0` 占比 —— 这正是探针 H 里 B_avail 那条线。

注意 D 越大，能被计数的槽越多（因为窗口更长），所以这个对比天然偏向长 D；
但副产物 `平均可用槽数` 会一起报出来，避免只看单边。

用法（节点上）：
    cd /opt/qkd/graph_mappo && OMP_NUM_THREADS=2 /opt/qkd/venv/bin/python -u \
        .tmp/probe_deadline_sens.py
"""

from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

import numpy as np  # noqa: E402

from qkd_rl.baselines.path_greedy import PathScoreGreedy  # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe  # noqa: E402
from qkd_rl.env.factory import build_env_from_config  # noqa: E402

DEADLINES = (30, 60, 120, 240)


def slots_components(env, horizon: int):
    edges = env.routing.edges
    node_ids = sorted({n for e in edges for n in (e.src, e.dst)})
    node_idx = {n: i for i, n in enumerate(node_ids)}
    comp = np.full((horizon, len(node_ids)), -1, dtype=np.int32)
    rp = env.rate_provider
    for t in range(horizon):
        adj: dict[str, list[str]] = defaultdict(list)
        for e in edges:
            if rp.is_available(e.edge_id, t):
                adj[e.src].append(e.dst)
                adj[e.dst].append(e.src)
        row = comp[t]
        cid = 0
        for nd in node_ids:
            i = node_idx[nd]
            if row[i] >= 0:
                continue
            row[i] = cid
            stack = [nd]
            while stack:
                u = stack.pop()
                for v in adj.get(u, ()):
                    j = node_idx[v]
                    if row[j] < 0:
                        row[j] = cid
                        stack.append(v)
            cid += 1
    return comp, node_idx


def main() -> None:
    profile = _tp.load_validation_profile(ROOT / "configs" / "var2_diag.yaml")
    config = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=240,
        start_mode=profile["start_mode"])
    steps = 240
    horizon = steps + 40
    seeds = list(range(7, 19))

    print(f"协议：窗={profile['window_start_day']}-{profile['window_end_day']}  "
          f"模式={profile['start_mode']}  步数={steps}  种子={seeds[0]}-{seeds[-1]}")
    print(f"候选 deadline（槽）：{DEADLINES}\n")

    stats = {D: {"amt": 0.0, "zero": 0.0, "slots": 0.0, "n": 0,
                "famt": 0.0, "fzero": 0.0, "fslots": 0.0, "fn": 0}
             for D in DEADLINES}
    tot_amt = 0.0
    lifetimes: dict[int, float] = {}
    # 端点不在拓扑里的请求（如 Stockholm 这类孤立 GS）：它们在探针 H 里落在
    # 「A 拓扑不可达」那档（hop_distance = 10**6），这里也必须单独摊开，
    # 否则会被悄悄算进 "全通槽为 0"，把 deadline 的效应放大。
    off_topology_amt = 0.0
    h_tot = {"a": 0.0, "b": 0.0, "c": 0.0, "exp": 0.0,
             "seed_amt": 0.0, "na": 0, "nb": 0, "nc": 0, "n": 0}

    for seed in seeds:
        env = build_env_from_config(config)
        policy = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True,
                                 principles=False, router=ServeProbe(env))
        comp, node_idx = slots_components(env, horizon)

        arrivals: list[tuple[str, str, int, float, int]] = []
        queue = env.requests
        orig_add = queue.add_arrivals

        def add_patched(reqs, _arr=arrivals):
            for r in reqs:
                # 生命周期从请求自身读，不去猜配置键路径。
                _arr.append((r.src_gs, r.dst_gs, int(r.arrival_t), float(r.amount),
                             int(r.deadline_t) - int(r.arrival_t)))
            return orig_add(reqs)

        queue.add_arrivals = add_patched

        conn_cache: dict[tuple[str, str], np.ndarray] = {}

        def connected(src: str, dst: str) -> np.ndarray:
            key = (src, dst)
            arr = conn_cache.get(key)
            if arr is None:
                arr = comp[:, node_idx[src]] == comp[:, node_idx[dst]]
                conn_cache[key] = arr
            return arr

        # ── 把探针 H 的口径**搬进同一个进程**，直接和上面的离线口径对撞 ──
        # H 只统计 `expire()` 真的吐出来的请求，分子是**剩余量**（amount - served），
        # 且用 hop_distance 判可达。这里跑的是同一个专家策略，所以 H 口径的读数
        # 应该精确复现 outputs/eval/joint_avail_expert.json 的 B_avail（8.83% / 159 条）。
        # 不复现就说明两个口径之间还有没找到的差异，那离线那张表就不能用来判读。
        orig_expire = queue.expire
        h_rows: list[dict] = []

        def expire_patched(t: int, _rows=h_rows, _env=env):
            out = orig_expire(t)
            for req in out:
                hops = int(_env.routing.hop_distance(req.src_gs, req.dst_gs))
                reachable = (hops < 10 ** 6
                             and req.src_gs in node_idx and req.dst_gs in node_idx)
                a, d = int(req.arrival_t), int(req.deadline_t)
                avail = int(connected(req.src_gs, req.dst_gs)[a:min(d, horizon)].sum()) \
                    if reachable else 0
                _rows.append({
                    "reachable": reachable,
                    "remaining": max(0.0, float(req.amount) - float(req.served_amount)),
                    "avail_slots": avail,
                })
            return out

        queue.expire = expire_patched
        obs = env.reset(seed=seed, start_seed=seed)
        n = 0
        done = False
        while n < steps and not done:
            actions, scores = policy.act(obs)
            obs, _r, term, trunc, _info = env.step(actions, scores)
            n += 1
            done = term or trunc
        queue.add_arrivals = orig_add
        queue.expire = orig_expire

        for src, dst, a, amount, lif in arrivals:
            tot_amt += amount
            lifetimes[lif] = lifetimes.get(lif, 0.0) + amount
            if src not in node_idx or dst not in node_idx:
                off_topology_amt += amount
                continue
            arr = connected(src, dst)
            for D in DEADLINES:
                d = min(a + D, horizon)
                if d <= a:
                    continue
                k = int(arr[a:d].sum())
                s = stats[D]
                s["amt"] += amount
                s["n"] += 1
                s["slots"] += k
                if k == 0:
                    s["zero"] += amount
                # 探针 H 的分母只有**真的到期过**的请求：在 240 步内 deadline
                # 没到就被 episode 截断的那一批（尾部 30 槽）不算。为了能和
                # H 的 B_avail 对上，这里单独再记一遍"窗口完整落在 episode 内"的量。
                if a + D <= steps:
                    s["famt"] += amount
                    s["fn"] += 1
                    s["fslots"] += k
                    if k == 0:
                        s["fzero"] += amount

        # H 口径的读数：分母必须用**本种子**的到达量，不能用跨种子的累计值。
        seed_amt = sum(x[3] for x in arrivals)
        h_a = [r for r in h_rows if not r["reachable"]]
        h_b = [r for r in h_rows if r["reachable"] and r["avail_slots"] == 0]
        h_c = [r for r in h_rows if r["reachable"] and r["avail_slots"] > 0]
        h_amt = {"a": sum(r["remaining"] for r in h_a),
                 "b": sum(r["remaining"] for r in h_b),
                 "c": sum(r["remaining"] for r in h_c),
                 "exp": sum(r["remaining"] for r in h_rows)}
        for k in h_amt:
            h_tot[k] += h_amt[k]
        h_tot["na"] += len(h_a)
        h_tot["nb"] += len(h_b)
        h_tot["nc"] += len(h_c)
        h_tot["n"] += len(h_rows)
        h_tot["seed_amt"] += seed_amt
        print(f"  seed={seed} 到达请求={len(arrivals)}  [H 口径] 过期 {len(h_rows)} 条 / "
              f"剩余量 {h_amt['exp'] / max(1e-9, seed_amt):.2%}　"
              f"A {len(h_a)}/{h_amt['a'] / max(1e-9, seed_amt):.2%}"
              f"　B_avail {len(h_b)}/{h_amt['b'] / max(1e-9, seed_amt):.2%}"
              f"　C {len(h_c)}/{h_amt['c'] / max(1e-9, seed_amt):.2%}")

    print(f"\n到达请求总量（12 种子）= {tot_amt:,.0f}")
    print("请求生命周期（槽 -> 占到达量）：" + "  ".join(
        f"{k}={v / max(1e-9, tot_amt):.1%}" for k, v in sorted(lifetimes.items())))
    print(f"端点不在拓扑里的量（孤立节点，与 deadline 无关）="
          f"{off_topology_amt:,.0f}（{off_topology_amt / max(1e-9, tot_amt):.2%}，"
          f"对应探针 H 的 A 档，下面各行都已剔除）\n")

    print(f"{'deadline':>9}{'全样本全通槽为0':>18}{'占全体到达':>14}{'平均可用槽':>12}"
          f"　|{'':>2}{'完窗口全通槽为0':>18}{'占全体到达':>14}{'平均可用槽':>12}")
    for D in DEADLINES:
        s = stats[D]
        print(f"{D:>9}{s['zero'] / max(1e-9, s['amt']):>18.2%}"
              f"{s['zero'] / max(1e-9, tot_amt):>14.2%}"
              f"{s['slots'] / max(1, s['n']):>12.2f}"
              f"　|{'':>2}{s['fzero'] / max(1e-9, s['famt']):>18.2%}"
              f"{s['fzero'] / max(1e-9, tot_amt):>14.2%}"
              f"{s['fslots'] / max(1, s['fn']):>12.2f}")
    print("\n左半：所有到达请求（含 240 步尾部、deadline 没到就被 episode 截断的那批）。")
    print("右半：只有 D 窗口完整落在 episode 内的请求。\n")

    d = max(1e-9, h_tot["seed_amt"])
    print(f"【同进程对撞】这次 rollout 用的是**专家策略**，下面的数应该精确复现 "
          f"outputs/eval/joint_avail_expert.json：")
    print(f"  H 口径：过期 {h_tot['n']} 条 / 剩余量 {h_tot['exp'] / d:.2%}"
          f"　A {h_tot['na']} 条/{h_tot['a'] / d:.2%}"
          f"　B_avail {h_tot['nb']} 条/{h_tot['b'] / d:.2%}"
          f"　C {h_tot['nc']} 条/{h_tot['c'] / d:.2%}"
          f"　（分母=到达量 {h_tot['seed_amt']:,.0f}）")
    print(f"  探针 H 的专家读数：过期 344 条/19.22%　A 99 条/5.32%　"
          f"B_avail 159 条/8.83%　C 86 条/5.06%")
    print(f"  离线口径（左表 D=30 行）：全通槽为 0 = {stats[30]['zero'] / d:.2%}"
          f"　+ 被剔除的拓扑外 {off_topology_amt / d:.2%}"
          f"　= {stats[30]['zero'] / d + off_topology_amt / d:.2%}")
    print("  注意离线的分子是请求**原始量**，H 的分子是**剩余量**，两者差一个 served_frac。")
    print("\nDEADLINE_SENS_DONE")


if __name__ == "__main__":
    main()