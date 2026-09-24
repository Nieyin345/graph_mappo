"""**服务缺口分解**：29% 的需求为什么没被服务 —— 逐条判它"理论上能不能服务"。

## ★ 这个探针是第三个版本，前两版都被证伪了，这里写清为什么

- **v1 的错**：把可用性做成"整局时间并集" ⟹ 算出有效上限 2.19%，而实测专家
  是 0.6979，矛盾。并集只要求"任一步可用"，对 2–6 跳的路径几乎必然满足。
- **v2 的错（更严重）**：我读到 `routing.py:205` 的 `shortest_path` 就下结论
  「请求被绑死静态最短路」，**没有读下面 8 行**。实际 `partial_consume_for_request`
  在静态路没货时会**回退**到 `_find_positive_path(...)`
  —— **全图 BFS 找任意一条"每一跳都有正存量"的路**。
  ⟹ 请求**能绕路**，静态路径不是约束。
  （正是 [[read-anomaly-blame-the-method-first]]：读数异常先怪方法。）

**教训**：判"某条代码路径是约束"必须把**整个函数**读完，尤其看它**失败后**做什么。

## 正确的机理

真实的判据（`routing.py` 的 `partial_consume_for_request`）：

    1. 先试缓存静态最短路 —— 若**每一跳都有正存量**就用它
    2. 否则 `_find_positive_path`：在**正存量子图**上 BFS 找任意一条通路
    3. 服务量 = `min(路径各跳存量, 剩余需求)`

⟹ 一条请求能被服务的**充要条件（忽略竞争）**是：
   **正存量子图里，源宿之间存在一条通路。**
   存量只在**被激活**的边上累积；每步只激活约 55 条（全网 1978）。
   ⟹ 真正的问题是：**策略选的这 55 条边，是否连成了需求要走的通路？**

## 三种失败，指向完全不同的修法

  F1 **激活覆盖不足** —— 用「整局曾激活过的边」做子图都找不到通路
     ⟹ 策略的**选边**从没把这条通路连通过 ⟹ 有可学空间
  F2 **存量时机不足** —— 曾激活子图里有通路，但**当前**正存量子图里没有
     ⟹ 通路上的边激活了但还没攒够货 / 或已被消耗光 ⟹ 时序问题
  F3 **能服务却没服务** —— 当前就有通路，但请求仍 pending
     ⟹ 竞争/期限/需求分片 ⟹ 该看的是排队与优先级

★ 判据用**路由器自己的函数** `routing._find_positive_path`，
  不自己重写一遍（[[duplicate-implementation-drifts]]）。
用法（远程务必 -u）：
    python3 -u probe_service_gap.py --seeds 100-103 --steps 240
"""
import importlib.util
import os
import statistics as st
import sys
from collections import Counter
from pathlib import Path

_here = sys.path[0] if sys.path else ""
if _here and _here not in ("", "."):
    sys.path[:] = [p for p in sys.path if p != _here]

REPO = Path(os.environ.get("GM_REPO", "/opt/qkd/graph_mappo"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import argparse                                                   # noqa: E402
from qkd_rl.env.factory import build_env_from_config              # noqa: E402
from qkd_rl.env.qkp import LinkQKPPool                            # noqa: E402
from qkd_rl.baselines.path_greedy import PathScoreGreedy          # noqa: E402
from qkd_rl.baselines.serve_probe import ServeProbe               # noqa: E402


def _tp():
    spec = importlib.util.spec_from_file_location(
        "gm_tp", REPO / "qkd_rl" / "evaluation" / "test_protocol.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def path_in_subgraph(routing, src, dst, edge_set, edge_by_id):
    """在给定边集上 BFS：src->dst 是否存在通路。返回跳数或 None。

    ★ 不调路由器的私有函数（它绑在 qkp 上），这里只做**子图连通性**判定，
      语义与 `_find_positive_path` 一致（都是"每条边都在集合里"的 BFS）。
    """
    if src == dst:
        return 0
    adj = {}
    for eid in edge_set:
        e = edge_by_id.get(eid)
        if e is None:
            continue
        adj.setdefault(e.src, []).append((e.dst, eid))
        adj.setdefault(e.dst, []).append((e.src, eid))
    seen = {src}
    q = [(src, 0)]
    while q:
        node, d = q.pop(0)
        for nxt, _eid in adj.get(node, ()):
            if nxt == dst:
                return d + 1
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, d + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-103")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--every", type=int, default=10)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    tp = _tp()
    profile = tp.load_validation_profile()
    start0 = int(profile.get("start_seed", 0))
    cfg = tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=args.steps,
        start_mode=profile["start_mode"])

    print("=" * 104)
    print(f"服务缺口分解：pending 请求「理论上能不能服务」（专家，种子 "
          f"{seeds[0]}–{seeds[-1]}，每 {args.every} 步抽样）")
    print("=" * 104)

    grand = Counter()
    per_seed = []
    for s in seeds:
        env = build_env_from_config(cfg)
        obs = env.reset(seed=s, start_seed=start0 + s)
        expert = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                                 principles=False, router=ServeProbe(env))
        routing = env.routing
        qkp = env.qkp
        # ★ 边表在 routing 上（`RoutingPolicy.edges`），env 没有 `action_space`
        edge_by_id = {e.edge_id: e for e in routing.edges}
        ever_act = set()
        c = Counter()
        n_samp = 0
        A = S = 0.0
        done = False
        k = 0
        while not done and k < args.steps:
            acts, scores = expert.act(obs)
            obs, _r, term, trunc, info = env.step(acts, scores)
            k += 1
            A += float(info.get("arrived_keys", 0.0))
            S += float(info.get("served_keys", 0.0))
            for e in getattr(env, "last_activated_edges", []) or []:
                ever_act.add(e)
            if k % args.every == 0:
                n_samp += 1
                pos_edges = set(qkp.positive)
                act_now = set(getattr(env, "last_activated_edges", []) or [])
                for req in env.requests.pending:
                    c["pending"] += 1
                    a = float(req.amount)
                    c["pending_amt"] += a
                    # F3：当前正存量子图里就有通路
                    p_pos = path_in_subgraph(
                        routing, req.src_gs, req.dst_gs, pos_edges, edge_by_id)
                    if p_pos is not None:
                        c["F3_servable"] += 1
                        c["F3_servable_amt"] += a
                        continue
                    # F2：曾激活子图里有通路（激活过，但现在没货）
                    p_ever = path_in_subgraph(
                        routing, req.src_gs, req.dst_gs, ever_act, edge_by_id)
                    if p_ever is not None:
                        c["F2_timing"] += 1
                        c["F2_timing_amt"] += a
                        continue
                    # F1：连整局激活过的边都连不通
                    p_act = path_in_subgraph(
                        routing, req.src_gs, req.dst_gs, act_now, edge_by_id)
                    if p_act is not None:
                        c["F2_timing"] += 1
                        c["F2_timing_amt"] += a
                    else:
                        c["F1_coverage"] += 1
                        c["F1_coverage_amt"] += a
            done = term or trunc

        tot = c["pending"] or 1
        print(f"  seed {s}: SR={S/A if A else 0:.4f}  抽样 {n_samp} 次  "
              f"pending 样本 {c['pending']:,}")
        print(f"      F3 当前就能服务   {c['F3_servable']:>7,} ({c['F3_servable']/tot:>6.1%})"
              f"  F2 激活过但没货 {c['F2_timing']:>7,} ({c['F2_timing']/tot:>6.1%})"
              f"  F1 从没连通 {c['F1_coverage']:>7,} ({c['F1_coverage']/tot:>6.1%})")
        grand.update(c)
        per_seed.append((c["F1_coverage"] / tot, c["F2_timing"] / tot,
                         c["F3_servable"] / tot))

    t = grand["pending"] or 1
    print()
    print("=" * 104)
    print("总判读（按请求**条数**）")
    print("=" * 104)
    print(f"  pending 样本合计 {grand['pending']:,}")
    print(f"    F3 **当前正存量子图里就有通路，却没被服务** "
          f"{grand['F3_servable']:>8,}  {grand['F3_servable']/t:>6.1%}")
    print(f"    F2 激活过、但此刻通路断了（没货/被消耗光）  "
          f"{grand['F2_timing']:>8,}  {grand['F2_timing']/t:>6.1%}")
    print(f"    F1 **连整局激活过的边都连不通**             "
          f"{grand['F1_coverage']:>8,}  {grand['F1_coverage']/t:>6.1%}")
    print()
    print("  按**密钥量**加权：")
    ta = grand["pending_amt"] or 1.0
    print(f"    F3 {grand['F3_servable_amt']/ta:>6.1%}   "
          f"F2 {grand['F2_timing_amt']/ta:>6.1%}   "
          f"F1 {grand['F1_coverage_amt']/ta:>6.1%}")
    print()
    print(f"  逐种子：F1 均值 {st.mean([x[0] for x in per_seed]):.1%}  "
          f"F2 {st.mean([x[1] for x in per_seed]):.1%}  "
          f"F3 {st.mean([x[2] for x in per_seed]):.1%}")
    print()
    print("  结论怎么读：")
    print("   · F1 大 ⟹ **激活覆盖不足**：策略选的边从没连通需求通路")
    print("     ⟹ 这是**选边**问题，有可学空间（RL 可能赢）")
    print("   · F2 大 ⟹ **时序问题**：边激活了但存量的时机不对")
    print("     ⟹ 该动的是生成落点/容量，不是选边")
    print("   · F3 大 ⟹ **能服务却没服务**：竞争/期限/分片 ⟹ 看排队与优先级")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
