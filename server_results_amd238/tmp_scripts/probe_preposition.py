"""探针 N：C 桶（曾全通却没搬完）的请求，在窗口到来那一刻，瓶颈边上有没有 key？

## 想验证什么

探针 M 把可争空间压到约 5 个点，而且全落在一件事上：**拥塞下的分配**；
探针 K 又发现 BC -> RL 的变化是「从完全没搬」变成「抓住窗口但搬不完」。
两句合起来指向一个机制问题：

    策略是不是**从来不在窗口到来之前，就把 key 预置到要用的边上**？

如果 C 桶请求在「源-目第一次物理全通」那一槽，服务路径的瓶颈库存 ≈ 0，
那就是**没预置**：窗口开了才开始攒，30 槽的寿命根本不够。
如果瓶颈库存本来就不小、却仍然搬不完，那是**速率硬约束**，策略再聪明也没用。

## 服务语义（决定了"瓶颈"怎么量）

`routing.partial_consume_for_request`：沿一条**每跳库存都为正**的路径，
每槽可搬 `min(该路径各跳库存, 剩余量)`，而且是从每一跳**各扣掉这么多**（逐跳中继）。

所以「这一槽这个请求最多能搬多少」 =
**正库存子图上 src->dst 路径各跳库存的最小值**（瓶颈）。注意它要求整条路径都为正，
只要有一跳是 0，这条路径当步就是废的 —— 这正是"预置"要解决的问题。

## 记什么

对每个到达请求（三个时点）：

    bn_arrival   到达当步末，正库存路径的瓶颈（None = 根本没有正库存路径）
    bn_first     源-目**第一次物理全通**那一槽的瓶颈
    bn_max / bn_sum
                 生命期内**所有全通槽**上瓶颈的最大值 / 之和
                 （bn_sum 是"如果这个请求独占它的窗口，最多能搬走多少"的上界）

另外记静态最短路（`routing.shortest_path`，拓扑最短路，不看库存）上的
`static_min` 与零库存跳占比，用来区分「完全没铺」和「铺了但差一跳」。

## 判读量

表 2 的 `bn_sum vs remaining` 是在**实际库存轨迹**上算的，所以它同时受"策略没预置"
和"窗口太短攒不出来"两种原因影响。为了把这两者分开，再加一个**与策略无关**的量：

    gen_raw = Σ_{窗口内物理相通的槽} (该槽可用边子图上 **max-min（最宽）路径** 的瓶颈速率)
              x scenario.slot_seconds

即"把这段窗口里路径每一跳**全部保持激活**、且不与任何别的请求竞争"，最多能攒多少。
它是纯物理上界，只取决于 rate_provider 与激活能力（`env._generate_keys` 只在被激活的
边上生成，所以"预置"本身是策略动作；这里假定策略愿意把每一跳都一直开着）。

两个容易写错的点（第一版都踩了）：
  * 必须乘 `slot_seconds`：`rate_provider.get_rate` 是 key/秒；
  * 必须取 max-min 路径，不能取最少跳路径 —— 后者不是上界。

于是有 2x2 交叉表（表 6）：

    库存够 + 物理够  -> 本来能成，纯策略失误（很少）
    库存不够 + 物理够 -> **有货可攒却没攒/没抢到** <- 唯一策略能挣的一格
    物理不够          -> 天花板钉死，改策略没用

用法（节点上，由 `.tmp/run_preposition.sh` 驱动）：
    cd /opt/qkd/graph_mappo && OMP_NUM_THREADS=2 /opt/qkd/venv/bin/python -u \
        .tmp/probe_preposition.py --policy expert
    ... --policy rl --checkpoint outputs/xxx.pt --deterministic
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import heapq
from collections import defaultdict, deque
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


def slots_components(env, horizon: int):
    """每个槽一份"物理可用边子图"的连通分量编号（与探针 H 同一份实现）。"""
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--policy", default="expert", choices=["expert", "rl"])
    ap.add_argument("--seeds", default="7-18")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--tag", default="expert")
    ap.add_argument("--config", default=str(ROOT / "configs" / "var2_diag.yaml"))
    ap.add_argument("--deterministic", action="store_true",
                    help="RL 策略用确定性动作（与训练侧 evaluate_validation 同口径）")
    args = ap.parse_args()

    profile = _tp.load_validation_profile(Path(args.config))
    steps = args.steps or int(profile["episode_steps"]) or 240
    seeds = parse_seeds(args.seeds)
    config = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=steps,
        start_mode=profile["start_mode"])
    start_seed = int(profile.get("start_seed", 0))
    horizon = steps + 40

    act_mode = ("确定性" if (args.policy == "expert" or args.deterministic) else "采样")
    print(f"策略={args.policy}  动作={act_mode}  种子={seeds}  步数={steps}  "
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

        comp, node_idx = slots_components(env, horizon)
        qkp = env.qkp
        routing = env.routing
        adj = routing.adj

        # 请求的度量记录，按 request_id 挂着（部分服务会用 dataclasses.replace
        # 生成新对象，但 request_id 不变，所以 id 是稳的）。
        meas: dict[str, dict] = {}
        # (src, dst, t) -> 该槽可用子图上最小跳速瓶颈，跨请求复用。
        gen_cache: dict[tuple[str, str, int], float] = {}

        def bottleneck(src: str, dst: str):
            """正库存子图上 src->dst 的瓶颈库存；None = 不存在全通的正库存路径。

            与 `routing._find_positive_path` 同语义（BFS，只看 positive 的边），
            但不依赖 `prepare_serve` 的缓存（我们是在 serve 之后自己调的），
            所以直接用 `routing.adj` + 活的 `qkp.positive` 过滤。
            """
            if src == dst:
                return 0.0, []
            positive = qkp.positive
            parent: dict[str, tuple[str | None, str | None]] = {src: (None, None)}
            dq = deque([src])
            while dq:
                u = dq.popleft()
                if u == dst:
                    path: list[str] = []
                    cur = dst
                    while parent[cur][0] is not None:
                        path.append(parent[cur][1])
                        cur = parent[cur][0]
                    path.reverse()
                    return min(qkp.get_level(e) for e in path), path
                for nb, eid in adj.get(u, ()):
                    if nb in parent or eid not in positive:
                        continue
                    parent[nb] = (u, eid)
                    dq.append(nb)
            return None, []

        slot_seconds = float(env.scenario.slot_seconds)

        def widest_rate(src: str, dst: str, t: int) -> float:
            """t 槽**可用边子图**里，所有 src->dst 路径中「各跳最小速率」的最大值。

            两条要点：
              * 不能用 `routing.shortest_path`（**静态拓扑**最短路）—— 这套 LEO 可见性
                是"打洞"的，静态路径几乎从不整条在同一槽里可用，拿它算会恒等于 0；
              * 也不能只取**最少跳**的那条路径 —— 那样得到的不是上界。真正的物理上界
                是"挑一条瓶颈最宽的路径"，所以这里跑的是 max-min（最宽路径），
                而不是 BFS 最短路。
            """
            if src == dst:
                return 0.0
            rp = env.rate_provider
            best: dict[str, float] = {src: float("inf")}
            heap: list[tuple[float, str]] = [(-float("inf"), src)]
            while heap:
                nrate, u = heapq.heappop(heap)
                rate = -nrate
                if rate < best.get(u, -1.0):
                    continue
                if u == dst:
                    return rate
                for nb, eid in adj.get(u, ()):
                    if not rp.is_available(eid, t):
                        continue
                    cand = min(rate, float(rp.get_rate(eid, t)))
                    if cand > best.get(nb, -1.0):
                        best[nb] = cand
                        heapq.heappush(heap, (-cand, nb))
            return 0.0

        def gen_ub_slot(src: str, dst: str, t: int) -> float:
            """该槽这条路径上最多能生成多少 key（与库存、与策略都无关的物理上界）。

            单位是 **key/槽**，所以必须乘 `scenario.slot_seconds`：`rate_provider.get_rate`
            给的是 key/秒，而 `env._generate_keys` 里正是 `rate * slot_seconds`
            （`receding_horizon_milp.py` 的上界项也是同一个口径）。漏乘这个 60
            会把物理上界压小两个数量级。

            另外把切换惩罚（`switch_cost.rate_decay_factor`）算成 1.0 —— 这里是上界，
            持续保持激活就没有惩罚（与 MILP 上界同处理）。按 (src, dst, t) 缓存。
            """
            key = (src, dst, t)
            v = gen_cache.get(key)
            if v is None:
                v = widest_rate(src, dst, t) * slot_seconds
                gen_cache[key] = v
            return v

        def static_profile(src: str, dst: str):
            """静态拓扑最短路上的（最小库存, 零库存跳占比, 跳数）。"""
            path = routing.shortest_path(src, dst)
            if not path:
                return None, None, 0
            levels = [qkp.get_level(e) for e in path]
            zeros = sum(1 for lv in levels if lv <= 1.0e-9)
            return min(levels), zeros / len(path), len(path)

        def new_rec(req, t: int) -> dict:
            hops = int(routing.hop_distance(req.src_gs, req.dst_gs))
            reachable = hops < 10 ** 6
            rec = {
                "seed": seed, "src": req.src_gs, "dst": req.dst_gs,
                "amount": float(req.amount), "arrival": int(req.arrival_t),
                "deadline": int(req.deadline_t), "hops": hops if reachable else -1,
                "reachable": reachable,
                "bn_arrival": None, "st_arrival": None, "sz_arrival": None,
                "first_conn": None, "bn_first": None,
                "st_first": None, "sz_first": None, "hl_first": 0,
                "bn_max": 0.0, "bn_sum": 0.0, "conn_slots": 0,
                # 窗口内"生成能力上界"：Σ_{窗口内的槽} max-min 路径瓶颈速率 x slot_seconds。
                # 这是"把这段窗口里路径每一跳全部保持激活、且不与任何别的请求竞争"
                # 时的可攒量上界 —— 与策略无关，只看物理生成率（见 gen_ub_slot）。
                #   gen_raw：不按剩余量截断（纯粹是物理量，绝不受策略轨迹影响）
                #   gen_ub：按当时的剩余量截断（受已服务量影响，只作参考）
                "gen_raw": 0.0, "gen_ub": 0.0,
                "gen_first": None, "gen_slots": 0,
            }
            if reachable:
                bn, _ = bottleneck(req.src_gs, req.dst_gs)
                st, sz, _ = static_profile(req.src_gs, req.dst_gs)
                rec["bn_arrival"] = None if bn is None else float(bn)
                rec["st_arrival"] = None if st is None else float(st)
                rec["sz_arrival"] = None if sz is None else float(sz)
            return rec

        def birth(req, t: int) -> dict:
            """请求的度量记录：到达当步末的库存画像（request_id 稳定）。"""
            rec = new_rec(req, t)
            meas[req.request_id] = rec
            return rec

        def touch(req, t: int) -> None:
            """在槽 t 末更新一个还活着的请求：窗口内瓶颈的逐步统计。"""
            if not rec_ok(req):
                return
            arr = connected(req.src_gs, req.dst_gs)
            if t >= horizon or not bool(arr[t]):
                return
            rec = meas[req.request_id]
            bn, _ = bottleneck(req.src_gs, req.dst_gs)
            bn = 0.0 if bn is None else float(bn)
            rec["conn_slots"] += 1
            rec["bn_sum"] += min(bn, max(0.0, float(req.amount) - float(req.served_amount)))
            if bn > rec["bn_max"]:
                rec["bn_max"] = bn
            # 窗口内生成能力上界（只看物理速率，与库存/策略无关）。
            # 注意：这里只在"源-目物理全通"的槽上累加（函数开头已经过滤），
            # 所以 gen_raw 是"这段窗口里能攒多少"的上界，不是"全程能攒多少"。
            g = gen_ub_slot(req.src_gs, req.dst_gs, t)
            if g > 0.0:
                rec["gen_slots"] += 1
                rec["gen_raw"] += g
                rec["gen_ub"] += min(
                    g, max(0.0, float(req.amount) - float(req.served_amount)))
                if rec["gen_first"] is None:
                    rec["gen_first"] = g
            if rec["first_conn"] is None:
                st, sz, hl = static_profile(req.src_gs, req.dst_gs)
                rec["first_conn"] = t - int(req.arrival_t)
                rec["bn_first"] = bn
                rec["st_first"] = None if st is None else float(st)
                rec["sz_first"] = None if sz is None else float(sz)
                rec["hl_first"] = hl

        conn_cache: dict[tuple[str, str], np.ndarray] = {}

        def connected(src: str, dst: str) -> np.ndarray:
            key = (src, dst)
            arr = conn_cache.get(key)
            if arr is None:
                arr = comp[:, node_idx[src]] == comp[:, node_idx[dst]]
                conn_cache[key] = arr
            return arr

        def rec_ok(req) -> bool:
            """端点都在拓扑里、且还没完成 —— 否则记了也没意义。"""
            if req.src_gs not in node_idx or req.dst_gs not in node_idx:
                return False
            return float(req.amount) - float(req.served_amount) > 1.0e-9

        queue = env.requests
        orig_serve = queue.serve
        orig_expire = queue.expire

        def serve_patched(_qkp, _routing, t: int, _rows=rows):
            res = orig_serve(_qkp, _routing, t)
            # 完成的请求不会再出现在 expire 里，必须在这里收口。
            for req in res.served_requests:
                rec = meas.pop(req.request_id, None)
                if rec is None:
                    # 到达当步就被服务完的请求：expire 钩子还没见过它，
                    # 这里补一条（三个时点都是空的，只贡献"成功"这一组）。
                    if req.src_gs not in node_idx or req.dst_gs not in node_idx:
                        continue
                    rec = new_rec(req, t)
                rec["outcome"] = "served"
                rec["remaining"] = 0.0
                rec["served_frac"] = 1.0
                _rows.append(rec)
            return res

        def expire_patched(t: int, _rows=rows, _env=env):
            out = orig_expire(t)
            # 先补测量：`queue.pending` 此刻已经被 serve 与 expire 清过，
            # 剩下的就是"严格还活着"的请求（含本步刚到的）。
            for req in queue.pending:
                if not rec_ok(req):
                    continue
                if req.request_id not in meas:
                    birth(req, t)
                touch(req, t)
            # 再分类到期请求（与探针 H 同口径，方便交叉核对）。
            for req in out:
                rec = meas.pop(req.request_id, None)
                if rec is None:
                    rec = new_rec(req, 0)
                hops = int(_env.routing.hop_distance(req.src_gs, req.dst_gs))
                reachable = (hops < 10 ** 6
                             and req.src_gs in node_idx and req.dst_gs in node_idx)
                a, d = int(req.arrival_t), int(req.deadline_t)
                avail = int(connected(req.src_gs, req.dst_gs)[a:min(d, horizon)].sum()) \
                    if reachable else 0
                rec["reachable"] = reachable
                rec["avail_slots"] = avail
                if not reachable:
                    rec["outcome"] = "A"
                elif avail == 0:
                    rec["outcome"] = "B_avail"
                else:
                    rec["outcome"] = "C"
                rec["remaining"] = max(0.0, float(req.amount) - float(req.served_amount))
                rec["served_frac"] = float(req.served_amount) / max(1e-9, float(req.amount))
                _rows.append(rec)
            return out

        queue.serve = serve_patched
        queue.expire = expire_patched
        obs = env.reset(seed=seed, start_seed=start_seed + seed)
        n = 0
        done = False
        while n < steps and not done:
            if is_expert:
                actions, scores = policy.act(obs)
            else:
                with torch.no_grad():
                    step = policy.act(obs, deterministic=args.deterministic)
                actions, scores = step.actions, step.action_scores
            obs, _r, term, trunc, _info = env.step(actions, action_scores=scores)
            n += 1
            done = term or trunc
        queue.serve = orig_serve
        queue.expire = orig_expire

        s = env.metrics.episode_summary()
        tot_arrived += s["arrived_keys"]
        tot_served += s["served_keys"]
        print(f"  seed={seed} 成功率={s['success_rate']:.4f}  记录请求={len(rows)}（累计）")

    # ────────────────────────── 汇总 ──────────────────────────
    def g(name: str) -> list[dict]:
        return [r for r in rows if r.get("outcome") == name]

    def amt(rs: list[dict]) -> float:
        return sum(r["amount"] for r in rs)

    def mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    def zero_frac(rs: list[dict], key: str) -> float:
        """该字段为 None（无正库存路径）或 0 的量加权占比。"""
        tot = 0.0
        z = 0.0
        for r in rs:
            v = r.get(key)
            if v is None:
                z += r["amount"]
            elif v <= 1.0e-9:
                z += r["amount"]
            tot += r["amount"]
        return z / max(1e-9, tot)

    print(f"\n量加权成功率={tot_served / max(1e-9, tot_arrived):.4f}"
          f"  到达={tot_arrived:,.0f}  服务={tot_served:,.0f}")
    print(f"记录的请求总数={len(rows)}")

    print("\n【表 1】各组在关键时点的瓶颈库存（量加权；bn=0 含「根本没有正库存路径」）")
    print(f"{'组':<10}{'请求数':>8}{'占到达量':>10}"
          f"{'bn@到达=0':>11}{'bn@到达均值':>13}"
          f"{'首连等待':>10}{'bn@首连=0':>11}{'bn@首连均值':>13}"
          f"{'静态零跳占比':>14}")
    for name in ("served", "C", "B_avail", "A"):
        rs = g(name)
        if not rs:
            continue
        waits = [r["first_conn"] for r in rs if r["first_conn"] is not None]
        print(f"{name:<10}{len(rs):>8}{amt(rs) / max(1e-9, tot_arrived):>10.2%}"
              f"{zero_frac(rs, 'bn_arrival'):>11.2%}"
              f"{mean([r['bn_arrival'] for r in rs if r['bn_arrival'] is not None]):>13,.0f}"
              f"{mean(waits):>10.1f}"
              f"{zero_frac(rs, 'bn_first'):>11.2%}"
              f"{mean([r['bn_first'] for r in rs if r['bn_first'] is not None]):>13,.0f}"
              f"{mean([r['sz_first'] for r in rs if r['sz_first'] is not None]):>14.2%}")

    c = g("C")
    print("\n【表 2】C 桶：窗口内机会（bn_sum）够不够搬完剩余量？")
    print("  bn_sum = 生命期内每个全通槽的瓶颈之和（= 该请求独占窗口时的搬运上界）")
    print(f"  {'bn_sum 是否 >= 剩余量':<24}{'请求数':>8}{'占到达量':>10}"
          f"{'剩余量占比':>12}{'平均完成度':>12}{'平均 bn_first':>14}")
    for label, sel in (("机会够 (bn_sum>=rem)", [r for r in c if r["bn_sum"] >= r["remaining"] - 1e-9]),
                       ("机会不够", [r for r in c if r["bn_sum"] < r["remaining"] - 1e-9])):
        if not sel:
            continue
        print(f"  {label:<24}{len(sel):>8}{amt(sel) / max(1e-9, tot_arrived):>10.2%}"
              f"{sum(r['remaining'] for r in sel) / max(1e-9, tot_arrived):>12.2%}"
              f"{mean([r['served_frac'] for r in sel]):>12.3f}"
              f"{mean([r['bn_first'] for r in sel if r['bn_first'] is not None]):>14,.0f}")
    hard = amt([r for r in c if r["bn_sum"] < r["remaining"] - 1e-9])
    print(f"  → 若最终数落「机会不够」这一侧偏大，C 桶是**速率硬约束**；"
          f"当前「机会不够」的到达量占全体 {hard / max(1e-9, tot_arrived):.2%}")

    print("\n【表 3】C 桶：bn_first / 请求到达量 分箱（窗口一开始手上有多少货）")
    print(f"  {'bn_first/amount':<18}{'请求数':>8}{'剩余量':>14}{'平均完成度':>12}"
          f"{'平均 bn_max':>13}{'平均 bn_sum':>13}")
    bins = (("=0", 0.0, 0.0), ("(0,10%]", 1e-9, 0.10), ("(10%,25%]", 0.10, 0.25),
            ("(25%,50%]", 0.25, 0.50), (">50%", 0.50, float("inf")))
    for label, lo, hi in bins:
        sel = []
        for r in c:
            if r.get("bn_first") is None:
                continue
            v = r["bn_first"] / max(1e-9, r["amount"])
            if label == "=0":
                ok = v <= 1e-9
            else:
                ok = (v > lo) and (v <= hi)
            if ok:
                sel.append(r)
        if not sel:
            continue
        print(f"  {label:<18}{len(sel):>8}{sum(r['remaining'] for r in sel):>14,.0f}"
              f"{mean([r['served_frac'] for r in sel]):>12.3f}"
              f"{mean([r['bn_max'] for r in sel]):>13,.0f}"
              f"{mean([r['bn_sum'] for r in sel]):>13,.0f}")
    print(f"  （bn_first 为 None 的 C 桶请求数={len([r for r in c if r.get('bn_first') is None])}，"
          f"这些在截止前从未全通、本不该进 C）")

    print("\n【表 5】窗口内「生成能力上界」gen_raw vs 请求到达量（纯物理上界，与策略/库存无关）")
    print("  gen_raw = Σ_{窗口内物理相通的槽} max-min路径瓶颈速率(e,t) x slot_seconds")
    print("  （= 该窗口里把路径每一跳**全部保持激活**、且不与任何别的请求竞争时的可攒量）")
    print("  注意这里比的是**到达量**而不是剩余量：物理上界不该随「已经搬了多少」变化。")
    print(f"  {'gen_raw 是否 >= 到达量':<24}{'请求数':>8}{'占到达量':>10}{'到达量占比':>12}"
          f"{'平均 gen_raw':>14}{'平均到达量':>13}{'平均窗口槽':>12}")
    for label, sel in (("物理够 (gen_raw>=amt)", [r for r in c if r["gen_raw"] >= r["amount"] - 1e-9]),
                       ("物理不够", [r for r in c if r["gen_raw"] < r["amount"] - 1e-9])):
        if not sel:
            continue
        print(f"  {label:<24}{len(sel):>8}{amt(sel) / max(1e-9, tot_arrived):>10.2%}"
              f"{sum(r['amount'] for r in sel) / max(1e-9, tot_arrived):>12.2%}"
              f"{mean([r['gen_raw'] for r in sel]):>14,.0f}"
              f"{mean([r['amount'] for r in sel]):>13,.0f}"
              f"{mean([r['gen_slots'] for r in sel]):>12.1f}")
    print(f"  （参考：按剩余量截断的旧口径 gen_ub 均值={mean([r['gen_ub'] for r in c]):,.0f}；"
          f"窗口内物理全通槽数均值={mean([r['gen_slots'] for r in c]):.1f}）")

    print("\n【表 6】交叉表：库存机会（bn_sum）x 生成上界（gen_raw）—— 谁才是真正的天花板？")
    print(f"  {'格':<30}{'请求数':>8}{'占到达量':>10}{'平均完成度':>12}")
    for label, pre in (
            ("库存够 + 物理够", lambda r: r["bn_sum"] >= r["remaining"] - 1e-9
             and r["gen_raw"] >= r["amount"] - 1e-9),
            ("库存够 + 物理不够", lambda r: r["bn_sum"] >= r["remaining"] - 1e-9
             and r["gen_raw"] < r["amount"] - 1e-9),
            ("库存不够 + 物理够", lambda r: r["bn_sum"] < r["remaining"] - 1e-9
             and r["gen_raw"] >= r["amount"] - 1e-9),
            ("库存不够 + 物理不够", lambda r: r["bn_sum"] < r["remaining"] - 1e-9
             and r["gen_raw"] < r["amount"] - 1e-9)):
        sel = [r for r in c if pre(r)]
        if not sel:
            continue
        print(f"  {label:<30}{len(sel):>8}{amt(sel) / max(1e-9, tot_arrived):>10.2%}"
              f"{mean([r['served_frac'] for r in sel]):>12.3f}")
    print("  → 「库存不够 + 物理够」才是策略能挣的那一格（有货可攒却没攒/没抢到）；")

    print("\n【表 4】首连等待（到达 -> 源-目第一次物理全通）分布")
    print(f"  {'等待槽数':<12}{'served':>10}{'C':>10}{'B_avail':>10}")
    for label, lo, hi in (("0", 0, 0), ("1-2", 1, 2), ("3-5", 3, 5),
                          ("6-10", 6, 10), ("11-20", 11, 20), (">20", 21, 10 ** 9),
                          ("从未全通", -1, -1)):
        cells = []
        for name in ("served", "C", "B_avail"):
            rs = g(name)
            if lo < 0:
                k = len([r for r in rs if r["first_conn"] is None])
            else:
                k = len([r for r in rs if r["first_conn"] is not None
                         and lo <= r["first_conn"] <= hi])
            cells.append(f"{k:>10}")
        print(f"  {label:<12}{''.join(cells)}")

    out = ROOT / "outputs" / "eval" / f"preposition_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "policy": args.policy, "deterministic": bool(args.deterministic),
        "checkpoint": args.checkpoint, "seeds": seeds, "steps": steps,
        "arrived": tot_arrived, "served": tot_served, "rows": rows,
    }, open(out, "w", encoding="utf-8"))
    print(f"\n已保存 {out}")
    print("PREPOSITION_DONE")


if __name__ == "__main__":
    main()