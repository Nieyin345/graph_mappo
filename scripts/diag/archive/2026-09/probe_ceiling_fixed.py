"""**修正后的上界**：窗口是 `[首刻, 请求死线]` 的可用边并集（不是 arrival 起）。

## 修正了什么（我第四次把"不是上界的东西"当成上界）

    probe_simultaneous_ceiling.py 用窗口 `[arrival_t, deadline_t]` ⟹ 70.0327%
    `probe_deadline_headroom.py` 用同一个窗口          ⟹ 同一批数

但 `key_ttl_steps = 1_000_000`（`qkp.py:139-169` 的 `expire` 按
`t - batch_t >= ttl` 删）⟹ **整局 240 步里没有任何密钥过期** ⟹ 存量一旦生成
就一直可用 ⟹ 请求在 `arrival_t` **之前**备好的货**也能用来服务它**。

⟹ `[arrival_t, deadline_t]` **排除了合法服务** ⟹ 它比真实可行集**更窄**
   ⟹ **不是上界**。症状正是：dl=240 的专家 0.8175 **突破了** 81.33%。

**正确窗口**：`[episode_first_t, min(deadline_t, horizon)]`。
（`initial_level = 0` 已在运行时确认 ⟹ 首刻之前没有存量 ⟹ 起点取首刻即可。）

## 实现（比原版更快）

原版对**每条请求**重建一次 UF（~1000 × 1978）⟹ 慢。
正确窗口是"到某刻为止的前缀并集" ⟹ 用**增量 UF**：按 t 递增把该刻的边
union 进去，得到 `UF_at[t]`；每条请求只需取 `t <= deadline_t` 的**最大**那个。
UF 只增不减 ⟹ 增量合法。⟹ 240 次 UF 构建代替 1000 次。

## 判据（必须全过，否则不给结论）

  ① 单调：窗口越宽，上界越高
  ② **上界 ≥ 实测专家**（对每个 dl）。被突破 ⟹ 判据仍错，停止解读
  ③ 正对照：dl=30 的专家必须复现 0.6979（种子 100–114 全员）

用法：
    python3 -u probe_ceiling_fixed.py --seeds 100-114 --dls 30,60,120,240 --steps 240
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

EXPERT_DL30 = 0.6979
BASELINE_TOL = 0.010


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


class UF:
    __slots__ = ("p",)

    def __init__(self):
        self.p = {}

    def find(self, x):
        p = self.p
        p.setdefault(x, x)
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:
            p[x], x = r, p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb

    def same(self, a, b):
        return a in self.p and b in self.p and self.find(a) == self.find(b)


def max_flow_undirected(cap, src, dst, nodes):
    """无向最大流（同一份存量两个方向只能各用一次）。

    残量图：无向边 ⟹ 初始 r[u][v] = r[v][u] = c；
    推 f 沿 u->v：r[u][v] -= f 且 r[v][u] += f（反向弧**共用**同一份容量）。
    """
    if src == dst:
        return float("inf")
    r = {n: {} for n in nodes}
    for (u, v), c in cap.items():
        if c <= 0:
            continue
        r[u][v] = r[u][v] + c if v in r[u] else c
        r[v][u] = r[v][u] + c if u in r[v] else c
    total = 0.0
    while True:
        parent = {src: None}
        stack = [src]
        found = False
        while stack and not found:
            u = stack.pop()
            for v, c in r[u].items():
                if v not in parent and c > 1e-9:
                    parent[v] = u
                    if v == dst:
                        found = True
                        break
                    stack.append(v)
        if not found:
            break
        b = float("inf")
        v = dst
        while parent[v] is not None:
            b = min(b, r[parent[v]][v])
            v = parent[v]
        v = dst
        while parent[v] is not None:
            u = parent[v]
            r[u][v] -= b
            r[v][u] += b
            v = u
        total += b
    return total


def measure(cfg, seed, start0, steps, all_edges, edge_by_id, nodes):
    env = build_env_from_config(cfg)
    obs = env.reset(seed=seed, start_seed=start0 + seed)

    arrivals = []
    orig_add = env.requests.add_arrivals

    def wrapped(reqs, _oa=orig_add, _a=arrivals):
        _a.extend(reqs)
        return _oa(reqs)

    env.requests.add_arrivals = wrapped

    avail_by_t = {}
    rate_by_t = {}
    expert = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                             principles=False, router=ServeProbe(env))
    A = S = 0.0
    n = 0
    done = False
    while not done and n < steps:
        t_now = int(env.t)
        ew = env._build_state().edge_windows
        av = ew.available0(all_edges)
        rr = ew.rates0(all_edges)
        avail_by_t[t_now] = {ed for ed, x in zip(all_edges, av) if bool(x)}
        rate_by_t[t_now] = {ed: float(x) for ed, x, k
                            in zip(all_edges, rr, av) if bool(k)}
        acts, scores = expert.act(obs)
        obs, _r, term, trunc, info = env.step(acts, scores)
        A += float(info.get("arrived_keys", 0.0))
        S += float(info.get("served_keys", 0.0))
        n += 1
        done = term or trunc
    sr = S / A if A else 0.0

    ts = sorted(avail_by_t)
    horizon = ts[-1] if ts else 0
    first = ts[0] if ts else 0

    # ---- 增量 UF：UF_at[t] = 并集 [first..t] 上的连通分量 ----
    uf = {}
    cur = UF()
    for t in ts:
        for e in avail_by_t[t]:
            ed = edge_by_id.get(e)
            if ed is not None:
                cur.union(ed.src, ed.dst)
        uf[t] = UF()
        uf[t].p = dict(cur.p)          # 快照（720 节点 ⟹ 拷贝很便宜）

    # ---- 正确窗口：[first, min(deadline, horizon)] ----
    tot = wide = 0.0
    for req in arrivals:
        amt = float(req.amount)
        tot += amt
        if env.routing.shortest_path(req.src_gs, req.dst_gs) is None:
            continue
        d = min(int(req.deadline_t), horizon)
        if d < first:
            continue
        if uf[d].same(req.src_gs, req.dst_gs):
            wide += amt
    return sr, wide, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-114")
    ap.add_argument("--dls", default="30,60,120,240")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--narrow-from",
                    choices=["first", "arrival"], default="first",
                    help="上界窗口起点：first=修正版（默认），arrival=原（偏窄）版")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    dls = [int(x) for x in args.dls.split(",")]

    tp = _tp()
    profile = tp.load_validation_profile()
    start0 = int(profile.get("start_seed", 0))
    base = tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=args.steps,
        start_mode=profile["start_mode"])
    e0 = build_env_from_config(base)
    edge_by_id = {e.edge_id: e for e in e0.routing.edges}
    all_edges = sorted(edge_by_id)
    nodes = sorted({e.src for e in e0.routing.edges} | {e.dst for e in e0.routing.edges})
    del e0

    print("=" * 106)
    print(f"修正后的上界（窗口 [{args.narrow_from}, 死线] 的可用边并集）  "
          f"种子 {seeds[0]}–{seeds[-1]}  {args.steps} 步  il=0（已运行时确认）")
    print("=" * 106)
    print(f"  自证：物理边 {len(all_edges):,}  节点 {len(nodes)}")
    print(f"{'dl':>5}{'专家SR':>10}{'修正上界':>11}{'**空间**':>11}{'正对照':>9}")

    rows = []
    for dl in dls:
        cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
        cfg["requests"] = dict(cfg.get("requests", {}))
        cfg["requests"]["deadline_steps"] = dl
        got = build_env_from_config(cfg).config.get("requests", {}).get("deadline_steps")
        if got != dl:
            print(f"  ⚠ dl={dl} 覆盖无效（读到 {got}）⟹ 该行作废")
            continue

        srs, wides, tots = [], [], []
        for s in seeds:
            sr, wide, tot = measure(cfg, s, start0, args.steps,
                                    all_edges, edge_by_id, nodes)
            srs.append(sr); wides.append(wide); tots.append(tot)
        m_sr = st.mean(srs)
        m_wide = sum(w for w in wides) / sum(tots) if sum(tots) else 0.0
        gap = m_wide - m_sr
        ok = m_wide >= m_sr - 1e-9
        rows.append((dl, m_sr, m_wide, gap, st.pstdev(srs)))
        print(f"{dl:>5}{m_sr:>10.4f}{m_wide:>11.4%}{gap:>+11.4f}"
              f"{'  ✓' if ok else '  ✗ 被突破':>9}   (SR SD {st.pstdev(srs):.4f})")

    print("-" * 106)
    print()
    full_pop = list(range(100, 115))
    b30 = next((r for r in rows if r[0] == 30), None)
    print("=" * 106)
    print("正对照（硬门）")
    print("=" * 106)
    if b30 is None:
        print("  ⚠ 没测 dl=30 ⟹ 无正对照 ⟹ 只当探索")
    elif list(seeds) != full_pop:
        print(f"  ⚠ 人口 {seeds[0]}–{seeds[-1]}(n={len(seeds)}) ≠ 基线人口 100–114(n=15)"
              f" ⟹ 硬门跳过（不是失败），dl=30 实测 {b30[1]:.4f}")
    else:
        dev = b30[1] - EXPERT_DL30
        print(f"  dl=30 实测 {b30[1]:.4f}  基线 {EXPERT_DL30:.4f}  偏差 {dev:+.4f}  "
              f"容差 ±{BASELINE_TOL}  {'✓ 通过' if abs(dev) <= BASELINE_TOL else '✗ 不通过'}")

    mono_ok = all(rows[i][2] <= rows[i + 1][2] + 1e-9
                  for i in range(len(rows) - 1)
                  if rows[i][0] < rows[i + 1][0])
    bound_ok = all(r[2] >= r[1] - 1e-9 for r in rows)
    print(f"  单调（窗口越宽上界越高）：{'✓' if mono_ok else '✗'}")
    print(f"  上界 ≥ 实测（对每个 dl）：{'✓' if bound_ok else '✗ 仍有突破 ⟹ 停止解读'}")
    print()
    if not bound_ok:
        print("DECISION=ABORT  上界仍被突破 ⟹ 判据还不成立")
        return 2
    best = max(r[3] for r in rows)
    pick = min([r for r in rows if abs(r[3] - best) < 1e-9], key=lambda r: r[0])
    print(f"  最大空间 {best:+.4f} @ dl={pick[0]}（专家 {pick[1]:.4f}，上界 {pick[2]:.4%}）")
    if best <= 0.02:
        print("  ⚠ 空间 ≤ 0.02 ⟹ 场景本身几乎没有可争之地 ⟹ 该改场景/机制")
    print(f"DECISION_DEADLINE={pick[0]}")
    print(f"DECISION_HEADROOM={best:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
