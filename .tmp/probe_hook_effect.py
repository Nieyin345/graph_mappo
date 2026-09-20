"""**受控对打**：钩子的两种写法（替换 vs 包装）会不会改变"可用边并集"？

## 为什么要这一条

两个探针给出**互相矛盾**的「全剧并集上限」：

  probe_simultaneous_ceiling.py（钩子 = **替换**）  →  86.4484%
  probe_deadline_headroom.py （钩子 = **包装**）    →  73.4667%

而两者的 `avail_by_t` 采集与并集循环**逐字相同**。唯一差别就是 `add_arrivals`
的写法：

  替换：env.requests.add_arrivals = lambda reqs: arrivals.extend(reqs)
        ⟹ `pending` **永远为空** ⟹ 专家一条也服务不了（SR 恒 0）
  包装：先调原方法再记账 ⟹ `pending` 正常填充

可用性看起来是**外生**的（`available0` 读 H5 的 blocks），与策略无关 ⟹
两者**应当**给出同一个并集。要么其中一个实现有 bug，要么"外生"这个假设错了。

## 这一条做什么

**同一进程、同一种子、同一配置**，建两个 env：
  A：替换钩子（复刻旧探针）
  B：包装钩子（复刻新探针）
各自跑 240 步专家，比较：
  ① `avail_by_t` 的**逐刻集合是否逐位相同**
  ② 全剧并集
  ③ 覆盖率（被记录到的 t 的个数与范围）
  ④ 专家 SR（A 应 ≈ 0 —— 这是 A 的钩子坏掉的**指纹**）

## 判读

· 若 ① 相同而 ② 不同 ⟹ **不可能**（并集是 ① 的函数）⟹ 我的代码有别的差异，继续查
· 若 ① 不同 ⟹ **可用性不是外生的**（或 t 的推进不同）⟹ 这是重大发现，要定位
· 若 A 的 SR ≈ 0 而 B ≈ 0.6979 ⟹ 确认 A 的钩子确实坏（已知），且并集差异**不是**
  钩子造成的（因为并集只看 ①）
· ★ 若 ① 里有 t 覆盖范围不同 ⟹ 说明**回合终止时机**被改变了

用法：
    python3 -u probe_hook_effect.py --seed 100 --steps 240
"""
import importlib.util
import os
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


def run_one(cfg, seed, start0, steps, all_edges, mode):
    env = build_env_from_config(cfg)
    obs = env.reset(seed=seed, start_seed=start0 + seed)

    arrivals = []
    if mode == "replace":
        env.requests.add_arrivals = lambda reqs: arrivals.extend(reqs)
    else:
        orig = env.requests.add_arrivals

        def wrapped(reqs, _o=orig):
            arrivals.extend(reqs)
            return _o(reqs)

        env.requests.add_arrivals = wrapped

    avail_by_t = {}
    expert = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                             principles=False, router=ServeProbe(env))
    A = S = 0.0
    n = 0
    done = False
    term_hit = trunc_hit = False
    while not done and n < steps:
        t_now = int(env.t)
        av = env._build_state().edge_windows.available0(all_edges)
        avail_by_t[t_now] = frozenset(ed for ed, x in zip(all_edges, av) if bool(x))
        acts, scores = expert.act(obs)
        obs, _r, term, trunc, info = env.step(acts, scores)
        A += float(info.get("arrived_keys", 0.0))
        S += float(info.get("served_keys", 0.0))
        n += 1
        done = term or trunc
        term_hit, trunc_hit = bool(term), bool(trunc)
    return {
        "mode": mode, "avail": avail_by_t, "A": A, "S": S, "n": n,
        "sr": (S / A if A else 0.0), "pending_end": len(env.requests.pending),
        "term": term_hit, "trunc": trunc_hit,
        "tot": sum(float(r.amount) for r in arrivals),
        "narr": len(arrivals),
        "arrivals": arrivals,
        "routing": env.routing,
    }


def _unused_union_ratio(avail, edge_by_id, arrivals, env_routing):  # noqa: 保留备用
    ue = set()
    for es in avail.values():
        ue |= es
    # 复用同一套 UF
    p = {}

    def find(x):
        p.setdefault(x, x)
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    for eid in ue:
        e = edge_by_id.get(eid)
        if e is None:
            continue
        ra, rb = find(e.src), find(e.dst)
        if ra != rb:
            p[ra] = rb

    def same(a, b):
        return a in p and b in p and find(a) == find(b)

    tot = uni = 0.0
    for req in arrivals:
        tot += float(req.amount)
        if env_routing.shortest_path(req.src_gs, req.dst_gs) is None:
            continue
        if same(req.src_gs, req.dst_gs):
            uni += float(req.amount)
    return uni / tot if tot else 0.0, len(ue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--steps", type=int, default=240)
    args = ap.parse_args()

    tp = _tp()
    profile = tp.load_validation_profile()
    start0 = int(profile.get("start_seed", 0))
    cfg = tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=args.steps,
        start_mode=profile["start_mode"])
    e0 = build_env_from_config(cfg)
    edge_by_id = {e.edge_id: e for e in e0.routing.edges}
    all_edges = sorted(edge_by_id)
    routing = e0.routing

    print("=" * 96)
    print(f"受控对打：钩子写法（替换 vs 包装）是否改变可用边并集   种子 {args.seed}  {args.steps} 步")
    print("=" * 96)

    res = {}
    for mode in ("replace", "wrap"):
        res[mode] = run_one(cfg, args.seed, start0, args.steps, all_edges, mode)

    a, b = res["replace"], res["wrap"]
    print(f"{'':<10}{'步数':>6}{'arrived':>10}{'served':>10}{'SR':>10}"
          f"{'pending末':>10}{'term':>6}{'trunc':>7}")
    for m in ("replace", "wrap"):
        r = res[m]
        print(f"{m:<10}{r['n']:>6}{r['A']:>10,.0f}{r['S']:>10,.0f}{r['sr']:>10.4f}"
              f"{r['pending_end']:>10}{str(r['term']):>6}{str(r['trunc']):>7}")

    print()
    print("=" * 96)
    print("可用集比对（这一节才是判据）")
    print("=" * 96)
    ta, tb = set(a["avail"]), set(b["avail"])
    print(f"  A(替换) 记录到的 t：{len(a['avail'])} 个  {min(ta) if ta else '-'}..{max(ta) if ta else '-'}")
    print(f"  B(包装) 记录到的 t：{len(b['avail'])} 个  {min(tb) if tb else '-'}..{max(tb) if tb else '-'}")
    print(f"  t 集合逐位相同？ {'是' if ta == tb else '否  ⟵ 终止时机/时钟被改变了'}")
    diff_t = sorted(ta ^ tb)
    if diff_t:
        print(f"    只在一边出现的 t（前 20）：{diff_t[:20]}")

    common = sorted(ta & tb)
    n_diff = 0
    first_diff = None
    for t in common:
        if a["avail"][t] != b["avail"][t]:
            n_diff += 1
            if first_diff is None:
                only_a = a["avail"][t] - b["avail"][t]
                only_b = b["avail"][t] - a["avail"][t]
                first_diff = (t, len(only_a), len(only_b), sorted(only_a)[:3], sorted(only_b)[:3])
    print(f"  共有 {len(common)} 个 t，其中可用集**不同**的有 {n_diff} 个")
    if first_diff:
        t, na, nb, sa, sb = first_diff
        print(f"    首个分歧 t={t}：只在A {na} 条 {sa}；只在B {nb} 条 {sb}")

    print()
    print("=" * 96)
    print("并集占比比对（**与两个探针逐字相同的算法**，这一节直接定位矛盾）")
    print("=" * 96)
    ratios = {}
    for m in ("replace", "wrap"):
        r = res[m]
        ratios[m] = _unused_union_ratio(r["avail"], edge_by_id, r["arrivals"], r["routing"])
    for m in ("replace", "wrap"):
        ratio, nedges = ratios[m]
        r = res[m]
        print(f"  {m:<9} 到达总量 {r['tot']:>14,.0f}  请求数 {r['narr']:>5}  "
              f"并集占比 {ratio:>8.4%}  并集边数 {nedges:,}")
    ra, rb = ratios["replace"][0], ratios["wrap"][0]
    print(f"  两者相差 {ra - rb:+.4%}"
          f"   {'✓ 相同 ⟹ 并集算法没问题，矛盾在别处' if abs(ra - rb) < 1e-9 else '✗ 不同 ⟹ 就在这里'}")
    # 请求明细差异
    aa, ab = res["replace"]["arrivals"], res["wrap"]["arrivals"]
    print(f"  请求数 replace={len(aa)}  wrap={len(ab)}")
    if len(aa) == len(ab):
        nd = 0
        for x, y in zip(aa, ab):
            if (x.src_gs, x.dst_gs, float(x.amount), int(x.arrival_t),
                    int(x.deadline_t)) != (y.src_gs, y.dst_gs, float(y.amount),
                                           int(y.arrival_t), int(y.deadline_t)):
                nd += 1
        print(f"  逐条相同的？ {'是' if nd == 0 else f'否，有 {nd} 条不同 ⟵ 请求内容被策略影响了'}")
    print()

    print()
    print("=" * 96)
    print("判读")
    print("=" * 96)
    if a["sr"] > 0.01:
        print(f"  ⚠ 替换钩子下 SR={a['sr']:.4f} 不是 0 ⟹ 我对'替换会清空 pending'的理解有误")
    else:
        print(f"  ✓ 替换钩子下 SR={a['sr']:.4f} ≈ 0 ⟹ 确认 pending 被清空（该探针的 SR 无效）")
    if ua_edges == ub_edges and ta == tb:
        print("  ⟹ 并集**不受钩子影响** ⟹ 旧读数 86.45% 与新读数 73.47% 的差异**另有原因**")
        print("     （不是钩子 ⟹ 要在别处找：种子集合? all_edges? 步数? 版本?）")
    else:
        print("  ⟹ 钩子**确实**改变了可用边 ⟹ 可用性不是外生的，或时钟推进不同 ⟹ 重大线索")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
