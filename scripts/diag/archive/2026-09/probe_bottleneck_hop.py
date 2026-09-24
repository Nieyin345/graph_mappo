"""定位 29% 失败到底卡在哪一环 —— 逐请求追它的最短路瓶颈跳。

## 为什么要这一条

实测（官方种子 100–114，240 步，专家，`initial_level=0`）：

    到达 75,326 条/步      服务 53,665 条/步      SR 0.6979
    生成 11,510,259 条/步   ⟹ **生成量是需求的 153 倍**

而负载扫描（amount_mean 100k→200k）显示 SR 只从 0.6979 掉到 0.6353，
**没有断崖**。所以「供给总量不足」被两路证据排除。

`routing.py:237-238` 给出真正的约束：

    hop_levels = [qkp.get_level(edge_id) for edge_id in path]
    serve_now  = min(hop_levels + [remaining])

**每条请求绑死一条预定的最短路，服务量 = 该路径上最弱那一跳的存量。**
路径上任一跳没货，整条请求就废。所以真问题不是「生成够不够」，
而是「**货有没有落在需求要走的那条边上**」。

三个互斥假说：
  C1 **覆盖不足** —— 需求路径上的跳根本没被激活过（策略选边选漏了）
  C2 **存量不足** —— 跳被激活了，但存量还没攒够（速率/容量问题）
  C3 **端口预算不够** —— 想覆盖全部需求路径，需要超过端口上限的边数
                    ⟹ 这是**结构性天花板**，任何策略都做不到

判据：
  · C1 主导 ⟹ 去改策略的选边（有空间，RL 可能赢）
  · C3 成立 ⟹ **端口预算就是硬约束**，RL 赢不了是环境的性质不是训练的问题
    ⟹ 该动的是端口/激活上限，不是奖励或结构

用法（远程务必 -u）：
    python3 -u probe_bottleneck_hop.py --seeds 100-104 --steps 240
"""
import importlib.util
import os
import statistics as st
import sys
from collections import Counter
from pathlib import Path

# ★ 服务器 /tmp 有 547 个 .py 会遮蔽标准库 ⟹ 先摘掉 sys.path[0] 再 import
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


def run_one(cfg, seed, start_seed, steps, verbose=False):
    env = build_env_from_config(cfg)
    obs = env.reset(seed=seed, start_seed=start_seed)
    expert = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                             principles=False, router=ServeProbe(env))
    qkp = env.qkp
    routing = env.routing

    # 需要的跳：所有请求的最短路并集（去重后的边集合）
    need_hops = Counter()        # edge -> 有多少次作为某请求的路径跳
    served_hops = Counter()      # edge -> 服务成功时被走过的次数
    # 瓶颈分解：每次尝试服务一条 pending 请求时，最弱那一跳的存量
    bn_zero = 0                  # 瓶颈跳存量 == 0（完全没货）
    bn_part = 0                  # 瓶颈跳有货但不够
    bn_ok = 0                    # 够
    path_never_activated = Counter()   # edge -> 它作为瓶颈且从未被激活的次数
    activated_ever = set()
    cover_need = []              # 每抽样步：覆盖全部 pending 路径所需边数
    cover_have = []              # 每抽样步：实际激活边数
    A = S = 0.0
    n_steps = 0

    done = False
    while not done:
        acts, scores = expert.act(obs)
        # ★ 先记本步激活了哪些边（step 之后 last_activated_edges 会更新，
        #   但我们要的是「到本步为止曾经激活过」的并集）
        obs, _r, term, trunc, info = env.step(acts, scores)
        n_steps += 1
        for e in getattr(env, "last_activated_edges", []) or []:
            activated_ever.add(e)
        A += float(info.get("arrived_keys", 0.0))
        S += float(info.get("served_keys", 0.0))

        # 每 10 步抽一次 pending 快照做瓶颈分解（全做太慢）
        if n_steps % 10 == 0:
            step_need_edges = set()
            for req in env.requests.pending:
                path = routing.shortest_path(req.src_gs, req.dst_gs)
                if not path:
                    continue
                for e in path:
                    need_hops[e] += 1
                    step_need_edges.add(e)
                remaining = max(0.0, req.amount - req.served_amount)
                lvl = [qkp.get_level(e) for e in path]
                m = min(lvl + [remaining])
                if m <= 0.0:
                    bn_zero += 1
                    # 瓶颈跳（第一处为零的）是否从未被激活
                    for e, l in zip(path, lvl):
                        if l <= 0.0:
                            if e not in activated_ever:
                                path_never_activated[e] += 1
                            break
                elif m < remaining:
                    bn_part += 1
                else:
                    bn_ok += 1
            # ★ C3：此刻覆盖全部 pending 需求路径，**至少**要同时激活多少条边？
            #   （只算被覆盖，不管存量；实际需求还会更高）
            cover_need.append(len(step_need_edges))
            # 上一步实际激活了多少条
            cover_have.append(len(getattr(env, "last_activated_edges", []) or []))
        if term or trunc:
            break
    return {
        "sr": S / A if A else 0.0, "A": A, "S": S,
        "bn_zero": bn_zero, "bn_part": bn_part, "bn_ok": bn_ok,
        "need_hops": need_hops, "never_act": path_never_activated,
        "activated_ever": activated_ever,
        "cover_need": cover_need, "cover_have": cover_have,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-104")
    ap.add_argument("--steps", type=int, default=240)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    tp = _tp()
    profile = tp.load_validation_profile()
    start0 = int(profile.get("start_seed", 0))
    cfg = tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=args.steps,
        start_mode=profile["start_mode"])

    print("=" * 100)
    print(f"① 逐请求瓶颈跳分解（专家，种子 {seeds[0]}–{seeds[-1]}，{args.steps} 步）")
    print("=" * 100)

    agg_need = Counter()
    agg_never = Counter()
    agg_act = set()
    rows = []
    for s in seeds:
        r = run_one(cfg, s, start0 + s, args.steps)
        rows.append(r)
        agg_need.update(r["need_hops"])
        agg_never.update(r["never_act"])
        agg_act |= r["activated_ever"]

        tot = r["bn_zero"] + r["bn_part"] + r["bn_ok"]
        if tot:
            print(f"  seed {s}: SR={r['sr']:.4f}  "
                  f"瓶颈=0 的样本 {r['bn_zero']:>7,} ({r['bn_zero']/tot:>6.1%})  "
                  f"部分 {r['bn_part']:>7,} ({r['bn_part']/tot:>6.1%})  "
                  f"够 {r['bn_ok']:>7,} ({r['bn_ok']/tot:>6.1%})")

    z = sum(r["bn_zero"] for r in rows)
    p = sum(r["bn_part"] for r in rows)
    o = sum(r["bn_ok"] for r in rows)
    tot = z + p + o
    print()
    print(f"  合计样本 {tot:,}")
    print(f"    瓶颈跳**存量恒为 0**（路径上有一跳完全没货）: {z:>9,}  {z/tot:>6.1%}")
    print(f"    瓶颈跳**有货但不够**                         : {p:>9,}  {p/tot:>6.1%}")
    print(f"    瓶颈**不卡**（能全额服务）                   : {o:>9,}  {o/tot:>6.1%}")

    print()
    print("=" * 100)
    print("② 覆盖：需求路径上的跳，有多少被激活过")
    print("=" * 100)
    need_edges = set(agg_need)
    covered = need_edges & agg_act
    print(f"  需求路径用到的**不同边**         : {len(need_edges):,}")
    print(f"  其中被激活过（进过 activated）   : {len(covered):,}  "
          f"({len(covered)/max(1,len(need_edges)):.1%})")
    print(f"  **从未被激活过**的需求路径边     : {len(need_edges - agg_act):,}  "
          f"({len(need_edges - agg_act)/max(1,len(need_edges)):.1%})")
    try:
        env0 = build_env_from_config(cfg)
        all_edges = set(env0.qkp.capacities.keys()) if hasattr(env0.qkp, "capacities") else set()
        print(f"  全网物理边总数                   : {len(all_edges):,}")
    except Exception as exc:                                      # noqa: BLE001
        print(f"  （取全网边数失败：{exc}）")

    cn = [x for r in rows for x in r["cover_need"]]
    ch = [x for r in rows for x in r["cover_have"]]
    if cn:
        print()
        print("=" * 100)
        print("③ 端口预算够不够（C3）—— 覆盖全部 pending 需求路径 需要多少条边")
        print("=" * 100)
        print(f"  覆盖全部 pending 路径所需边数：均值 {st.mean(cn):.1f}  "
              f"中位 {st.median(cn):.0f}  最大 {max(cn)}")
        if ch:
            print(f"  实际激活边数                ：均值 {st.mean(ch):.1f}  "
                  f"中位 {st.median(ch):.0f}  最大 {max(ch)}")
            over = sum(1 for a, b in zip(cn, ch) if b < a)
            print(f"  实际激活 < 覆盖所需 的抽样步  : {over}/{len(cn)} ({over/len(cn):.1%})")
        try:
            env0 = build_env_from_config(cfg)
            deg = [d for _, d in env0.graph_builder.graph.degree()] \
                if hasattr(env0.graph_builder, "graph") else []
            if deg:
                print(f"  图节点平均度 {st.mean(deg):.1f}，最大度 {max(deg)}")
                print(f"  （每节点有向端口上限 Tx≤1 / Rx≤1 ⟹ 无向边上限约 度/2）")
        except Exception:                                          # noqa: BLE001
            pass

    print()
    print("  判读：")
    print("   · ①里「存量恒为 0」占大头 ⟹ **C1 覆盖不足**：需求走的边上没货")
    print("     ⟹ 该动的是策略选边（RL 有空间）")
    print("   · ②里「从未被激活过的需求路径边」占比高 ⟹ 同上，覆盖漏了")
    print("   · 若 ①的「存量恒为 0」低但 SR 仍低 ⟹ 卡在别处（C2 速率/容量）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
