"""**结构性可服务上限**：只考虑"整局里可曾用过的边"，多少需求根本无路可走。

## 这一条回答那个决定性的分叉

上一支探针把 pending 请求分成三类（专家，种子 100–103，240 步，每 10 步抽样）：

    F1 连整局激活过的边都连不通     59.6%    ← 最大的一块
    F2 激活过、但此刻通路断了       27.9%
    F3 当前就能服务却没服务         12.5%

F1 有**两个互斥解释**，指向完全相反的下一步：

  (a) **策略没选到那些边** ⟹ 可学 ⟹ 该改模型
  (b) **那些边整局都不可用** ⟹ 谁都做不到 ⟹ 该改**环境/链路生成**

判据：把「整局里 `available0` 报过 True 的边」并起来当子图，
      再问 F1 那些 (源,宿) 对在这张子图里**有没有通路**。

    · 有通路 ⟹ (a) 策略漏选 ⟹ **模型有空间**
    · 无通路 ⟹ (b) 结构不可达 ⟹ **模型再强也服务不了**

★ 为什么用「可用边的并集」而不是「激活边的并集」：
  存量**只在可用边上累积**（`_generate_keys` 只对 activated 边生成，
  而 activated ⊆ available），所以"某条边能否**曾经**持有存量"
  = "它是否曾经可用"。这是 (b) 的正确判据。

★ 与已有的**拓扑天花板 91.97%** 的区别：
  拓扑天花板只排除 `shortest_path is None`（Stockholm 孤岛，路径**不存在**）；
  本探针还排除"路径存在、但路径上的边整局不可用"⟹ 只会更紧。

★ 这一条**不依赖任何策略**（可用性是 H5 + 时钟的函数），
  所以结论对改动后的模型同样成立。
用法（远程务必 -u）：
    python3 -u probe_structural_serviceable.py --seeds 100-114 --steps 240
"""
import importlib.util
import os
import statistics as st
import sys
from pathlib import Path

_here = sys.path[0] if sys.path else ""
if _here and _here not in ("", "."):
    sys.path[:] = [p for p in sys.path if p != _here]

REPO = Path(os.environ.get("GM_REPO", "/opt/qkd/graph_mappo"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import argparse                                                   # noqa: E402
from qkd_rl.env.factory import build_env_from_config              # noqa: E402
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


def connected_pairs(edges_sub, edge_by_id, pairs):
    """在给定边集上做多源 BFS，返回 pairs 里**连通**的那些。"""
    adj = {}
    for eid in edges_sub:
        e = edge_by_id.get(eid)
        if e is None:
            continue
        adj.setdefault(e.src, []).append(e.dst)
        adj.setdefault(e.dst, []).append(e.src)
    # 并查集
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, nbrs in adj.items():
        for v in nbrs:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
    ok = set()
    for a, b in pairs:
        if a in parent and b in parent and find(a) == find(b):
            ok.add((a, b))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-114")
    ap.add_argument("--steps", type=int, default=240)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    tp = _tp()
    profile = tp.load_validation_profile()
    start0 = int(profile.get("start_seed", 0))
    cfg = tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=args.steps,
        start_mode=profile["start_mode"])

    # 先取全网边表（任何一局都一样）
    e0 = build_env_from_config(cfg)
    edge_by_id = {e.edge_id: e for e in e0.routing.edges}
    all_edges = sorted(edge_by_id)

    print("=" * 108)
    print(f"结构性可服务上限（官方种子 {seeds[0]}–{seeds[-1]}，{args.steps} 步）")
    print(f"全网物理边 {len(all_edges):,}")
    print("=" * 108)
    print(f"{'seed':>6}{'到达量':>16}{'拓扑不可达':>15}{'可用图不可达':>16}"
          f"{'可用图可达':>15}{'结构上限':>11}{'专家SR':>9}{'正对照':>9}")

    T = {"tot": 0.0, "topo": 0.0, "unav": 0.0, "ok": 0.0}
    caps = []
    for s in seeds:
        # ---- ① 外生需求：全量 (src,dst,amount)，含无路者 ----
        e = build_env_from_config(cfg)
        e.reset(seed=s, start_seed=start0 + s)
        t0 = int(e.t)
        agg = {}                     # (src,dst) -> amount
        topo_dead = 0.0
        a_tot = 0.0
        for k in range(args.steps):
            for req in e.request_generator.generate(t0 + k):
                key = (req.src_gs, req.dst_gs)
                agg[key] = agg.get(key, 0.0) + float(req.amount)
                a_tot += float(req.amount)
                if e.routing.shortest_path(req.src_gs, req.dst_gs) is None:
                    topo_dead += float(req.amount)

        # ---- ② 可用边的整局并集（跑一局覆盖时间窗；可用性与策略无关）----
        env = build_env_from_config(cfg)
        obs = env.reset(seed=s, start_seed=start0 + s)
        expert = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                                 principles=False, router=ServeProbe(env))
        avail_union = set()
        n = 0
        done = False
        while not done and n < args.steps:
            acts, scores = expert.act(obs)
            obs, _r, term, trunc, info = env.step(acts, scores)
            n += 1
            av = env._build_state().edge_windows.available0(all_edges)
            for ed, a in zip(all_edges, av):
                if bool(a):
                    avail_union.add(ed)
            done = term or trunc

        # ---- ③ 在可用子图上判每对 ----
        pairs = list(agg)
        connected = connected_pairs(avail_union, edge_by_id, pairs)
        a_unav = a_ok = 0.0
        for p in pairs:
            if e.routing.shortest_path(p[0], p[1]) is None:
                continue                      # 已计入拓扑不可达
            if p in connected:
                a_ok += agg[p]
            else:
                a_unav += agg[p]

        # 正对照：可用子图上至少有一对连通
        ctrl = len(connected)
        cap = a_ok / a_tot if a_tot else 0.0
        caps.append(cap)
        T["tot"] += a_tot
        T["topo"] += topo_dead
        T["unav"] += a_unav
        T["ok"] += a_ok

        # ---- ④ 专家实测（同种子同起点，独立一局）----
        e2 = build_env_from_config(cfg)
        obs2 = e2.reset(seed=s, start_seed=start0 + s)
        ex = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                             principles=False, router=ServeProbe(e2))
        A = S = 0.0
        done = False
        while not done:
            acts, scores = ex.act(obs2)
            obs2, _r, term, trunc, info = e2.step(acts, scores)
            A += float(info.get("arrived_keys", 0.0))
            S += float(info.get("served_keys", 0.0))
            done = term or trunc
        print(f"{s:>6}{a_tot:>16,.0f}{topo_dead:>15,.0f}{a_unav:>16,.0f}"
              f"{a_ok:>15,.0f}{cap:>11.4%}{S/A if A else 0:>9.4f}{ctrl:>9}")

    tot = T["tot"]
    print()
    print("=" * 108)
    print("总判读")
    print("=" * 108)
    print(f"  到达总量                          {tot:>18,.0f}")
    print(f"    ① 拓扑不可达（路径不存在）      {T['topo']:>18,.0f}   {T['topo']/tot:>7.3%}")
    print(f"    ② **可用图不可达**（路径存在但整局无可用通路）")
    print(f"                                    {T['unav']:>18,.0f}   {T['unav']/tot:>7.3%}"
          f"   ← 结构性")
    print(f"    ③ 可用图可达                    {T['ok']:>18,.0f}   {T['ok']/tot:>7.3%}")
    print()
    print(f"  拓扑天花板   = {1 - T['topo']/tot:.4%}")
    print(f"  **结构上限** = {1 - (T['topo']+T['unav'])/tot:.4%}"
          f"   （比拓扑天花板紧 {T['unav']/tot:.4%}）")
    print(f"  实测专家 0.6979 ⟹ 达到结构上限的 "
          f"{0.6979/(T['ok']/tot) if T['ok'] else 0:.1%}")
    print()
    print(f"  逐种子结构上限：均值 {st.mean(caps):.4%}  SD {st.pstdev(caps):.4%}  "
          f"极差 {min(caps):.3%}..{max(caps):.3%}")
    print()
    print("  结论怎么读：")
    print("   · 若 ② 远大于 0 ⟹ 这部分需求**任何策略都服务不了**")
    print("     ⟹ F1 是 (b) 结构不可达 ⟹ 该改**环境/链路可用性**，不是模型")
    print("   · 若 ② ≈ 0 ⟹ F1 全是策略漏选 ⟹ **模型有空间**，去改选边/奖励")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
