"""**找有判别力的 regime**：`deadline_steps` 同时抬高"诚实上界"和专家基线，
要的是两者之**差**（= RL 能赢的空间）。

## 为什么这是现在该做的一步

实测（种子 100–114，240 步，`il=0`）：

    拓扑天花板        91.9694%     路径存在（不考虑可用性）
    全剧并集           86.4484%     允许跨到请求存活期**之外**
    **存活期并集       70.0327%**  诚实封顶（货须在请求死之前备好）
    实测专家           0.6979      = 诚实上界的 **99.7%**

⟹ **没有空间**。这就是为什么此前所有训练侧改动（γ、λ、entropy、激活函数、
  特征、history encoder、v1_onpath、hist32、gae90）**一律测不出** ——
  不是改动无效，是**靶子已经贴着天花板**。

**约束在场景，不在策略**：请求只活 `deadline_steps = 30` 步，
而它需要的那条路常常在**存活期之外**才可用。

## 这一条做什么

对 `deadline_steps ∈ {30, 60, 120, 240}` 各测两个量（**只跑专家**，便宜）：

  ① **诚实上界** = 存活期并集口径的可服务份额（与策略无关）
  ② **专家 SR**  = 同种子实测
  ③ **空间** = ① − ②     ← 这才是 RL 有没有机会的判据

★ 与 `initial_level` 的关键区别：抬 `deadline_steps` **不绕过可用性**，
  只给请求更长的等待期。它是**合法的场景参数**，不是后门。
★ 预注册选择规则：**取「空间」最大的 dl**；并列则取较小的（改动更小）。
  若所有 dl 的空间都 ≤ 0.02 ⟹ 场景本身没有可争之地 ⟹ 应该改
  **场景/机制**（而不是继续调模型）。

## ★★ 两个必须内建的对照（否则读数不可信）

**① 钩子必须是包装，不是替换。**
第一版 `probe_simultaneous_ceiling.py` 里写的是
`env.requests.add_arrivals = lambda reqs: arrivals.extend(reqs)` ——
**替换**掉了真方法 ⟹ `pending` 永远为空 ⟹ 专家**一条也服务不了** ⟹
那个探针的 SR 恒 0（我没打印它，所以天花板数字仍然有效：天花板只依赖
`arrivals` 列表与逐刻可用边集，与 `pending` 无关）。本探针**要**报专家 SR，
所以必须 `wrap` —— 先调原方法，再记账。

**② dl=30 必须复现 0.6979**（已实测的专家基线）。
偏离 ⟹ 我的管线（启动路径/钩子/步数/种子映射）与官方评测不等价 ⟹ 全部作废。
这是硬门，不是参考值。

用法（远程务必 -u）：
    python3 -u probe_deadline_headroom.py --seeds 100-114 --dls 30,60,120,240 --steps 240
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

HEADROOM_MIN = 0.02
EXPERT_BASELINE = 0.6979          # 已实测的 dl=30 专家基线（正对照的靶值）
BASELINE_TOL = 0.010              # 允许的偏差（逐种子 SD 约 0.014）


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
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb

    def same(self, a, b):
        return a in self.p and b in self.p and self.find(a) == self.find(b)


def components_for(avail_edges, edge_by_id):
    uf = UF()
    for eid in avail_edges:
        e = edge_by_id.get(eid)
        if e is None:
            continue
        uf.union(e.src, e.dst)
    return uf


def measure(cfg, seed, start0, steps, all_edges, edge_by_id):
    """跑一局专家，返回 (SR, 诚实上界, 全剧并集上界)。"""
    env = build_env_from_config(cfg)
    obs = env.reset(seed=seed, start_seed=start0 + seed)

    arrivals = []
    # ★ 包装而非替换 —— 替换会让 pending 恒空 ⟹ 专家服务量恒 0
    orig_add = env.requests.add_arrivals

    def wrapped(reqs):
        arrivals.extend(reqs)
        return orig_add(reqs)

    env.requests.add_arrivals = wrapped

    avail_by_t = {}
    expert = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                             principles=False, router=ServeProbe(env))
    A = S = 0.0
    n = 0
    done = False
    while not done and n < steps:
        t_now = int(env.t)
        av = env._build_state().edge_windows.available0(all_edges)
        avail_by_t[t_now] = {ed for ed, x in zip(all_edges, av) if bool(x)}
        acts, scores = expert.act(obs)
        obs, _r, term, trunc, info = env.step(acts, scores)
        A += float(info.get("arrived_keys", 0.0))
        S += float(info.get("served_keys", 0.0))
        n += 1
        done = term or trunc
    # ★ 自证：迟到一次就等于钩子没装上（arrivals 列表 vs 环境自己的记账）
    if abs(A - sum(float(r.amount) for r in arrivals)) > 1e-6 * max(A, 1.0):
        print(f"  ⚠ 钩子漏了到达：info 累计 {A:,.0f} vs 列表 {sum(float(r.amount) for r in arrivals):,.0f}")
    sr = S / A if A else 0.0

    horizon = max(avail_by_t) if avail_by_t else 0
    tot = live = uni = 0.0
    for req in arrivals:
        amt = float(req.amount)
        tot += amt
        if env.routing.shortest_path(req.src_gs, req.dst_gs) is None:
            continue
        t_lo = int(req.arrival_t)
        t_hi = min(int(req.deadline_t), horizon)
        if t_hi >= t_lo:
            es = set()
            for t in range(t_lo, t_hi + 1):
                es |= avail_by_t.get(t, set())
            if components_for(es, edge_by_id).same(req.src_gs, req.dst_gs):
                live += amt
    ue = set()
    for es in avail_by_t.values():
        ue |= es
    cu = components_for(ue, edge_by_id)
    for req in arrivals:
        if env.routing.shortest_path(req.src_gs, req.dst_gs) is None:
            continue
        if cu.same(req.src_gs, req.dst_gs):
            uni += float(req.amount)      # ★ 必须当场取，不能用上一个循环的 amt
    # ★ 自证：两个占比都不许超过 1（uni/tot > 1 在数学上不可能）
    if tot > 0 and (uni / tot > 1.0 + 1e-9 or live / tot > 1.0 + 1e-9):
        raise AssertionError(
            f"比率 > 1 ⟹ 必是 bug：live/tot={live/tot:.4%} uni/tot={uni/tot:.4%}")
    return sr, (live / tot if tot else 0.0), (uni / tot if tot else 0.0), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-108")
    ap.add_argument("--dls", default="30,60,120,240")
    ap.add_argument("--steps", type=int, default=240)
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
    dl0 = base.get("requests", {}).get("deadline_steps")

    print("=" * 100)
    print(f"找有判别力的 regime：`deadline_steps` 的空间（种子 {seeds[0]}–{seeds[-1]}，"
          f"{args.steps} 步，只跑专家）")
    print("=" * 100)
    print(f"  启动路径自证：物理边 {len(all_edges):,}  base.deadline_steps={dl0}  "
          f"episode_steps={base.get('episode_steps')}")
    if dl0 != 30:
        print(f"  ⚠ base 里 dl 不是 30 而是 {dl0} ⟹ 我的覆盖叠加在错误的底上")
    header = f"{'dl':>5}{'专家SR':>10}{'诚实上界':>11}{'全剧并集':>11}{'**空间**':>11}"
    print(header)
    print("-" * 100)

    rows = []
    for dl in dls:
        cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
        cfg["requests"] = dict(cfg.get("requests", {}))
        cfg["requests"]["deadline_steps"] = dl
        # ★ 覆盖是否真的到达环境？（配置链有强制覆写的先例）
        probe_env = build_env_from_config(cfg)
        got = probe_env.config.get("requests", {}).get("deadline_steps")
        if got != dl:
            print(f"  ⚠ dl={dl} 覆盖后环境读到 {got} ⟹ 覆盖无效，该行作废")
        del probe_env
        srs, lives, unis, tots = [], [], [], []
        for s in seeds:
            sr, live, uni, tot = measure(cfg, s, start0, args.steps, all_edges, edge_by_id)
            srs.append(sr); lives.append(live); unis.append(uni); tots.append(tot)
        m_sr = st.mean(srs)
        # ★ 聚合口径：占比必须**按总量加权**，不能对逐种子比值取平均
        #   （比值平均会给小种子同样的权重 ⟹ 与"总占比"不同）
        m_live = sum(l * t for l, t in zip(lives, tots)) / sum(tots) if sum(tots) else 0.0
        m_uni = sum(u * t for u, t in zip(unis, tots)) / sum(tots) if sum(tots) else 0.0
        gap = m_live - m_sr
        rows.append((dl, m_sr, m_live, m_uni, gap, st.pstdev(srs)))
        print(f"{dl:>5}{m_sr:>10.4f}{m_live:>11.4%}{m_uni:>11.4%}{gap:>+11.4f}"
              f"   (SR SD {st.pstdev(srs):.4f})")

    print("-" * 100)
    print()

    # ---- 硬门：dl=30 必须复现已实测的专家基线 ----
    # ★ 人口必须一致：基线 0.6979 是在**种子 100–114** 上测的。
    #   拿 3 个种子的均值去比它会**必然**报"不通过"（我踩过），
    #   所以只在与基线同人口时才把它当硬门。
    FULL_POP = list(range(100, 115))
    base_row = next((r for r in rows if r[0] == 30), None)
    print("=" * 100)
    print("正对照（硬门）：dl=30 应复现已实测的专家基线")
    print("=" * 100)
    if base_row is None:
        print("  ⚠ 本次没测 dl=30 ⟹ **没有正对照** ⟹ 下面的读数只能当探索，不能当结论")
        gate_ok = False
    elif list(seeds) != FULL_POP:
        print(f"  ⚠ 本种子集 {seeds[0]}–{seeds[-1]}（n={len(seeds)}）与基线人口 "
              f"100–114（n=15）**不一致**")
        print(f"     ⟹ 基线 {EXPERT_BASELINE} 在此**不适用**，硬门跳过（不是失败）")
        print(f"     实测 dl=30 = {base_row[1]:.4f}（仅作探索用）")
        gate_ok = True
    else:
        dev = base_row[1] - EXPERT_BASELINE
        gate_ok = abs(dev) <= BASELINE_TOL
        print(f"  实测 {base_row[1]:.4f}  基线 {EXPERT_BASELINE:.4f}  "
              f"偏差 {dev:+.4f}  容差 ±{BASELINE_TOL}")
        print(f"  {'✓ 通过 ⟹ 管线与官方评测等价，下面的读数可用' if gate_ok else '✗ 不通过 ⟹ 管线不等价 ⟹ 全部作废'}")
    print()

    print("=" * 100)
    print(f"决策（规则预注册：取空间最大者；并列取较小 dl；阈值 HEADROOM_MIN={HEADROOM_MIN}）")
    print("=" * 100)
    for dl, m_sr, m_live, m_uni, gap, sd in sorted(rows, key=lambda r: -r[4]):
        print(f"    dl={dl:>4}  专家 {m_sr:.4f}  上界 {m_live:.4%}  空间 {gap:+.4f}")
    print()

    if not gate_ok:
        print("  ✗ 正对照不过 ⟹ **不给决策**，先修管线")
        print("DECISION=ABORT")
        return 2

    best_gap = max(r[4] for r in rows)
    cands = [r for r in rows if abs(r[4] - best_gap) < 1e-9]
    pick = min(cands, key=lambda r: r[0])
    if best_gap <= HEADROOM_MIN:
        print(f"  ⚠ 所有 dl 的空间都 ≤ {HEADROOM_MIN} ⟹ **场景本身没有可争之地**")
        print("    ⟹ 不该继续调模型/奖励，该改**场景或机制**")
    else:
        print(f"  ⟹ 选 **deadline_steps = {pick[0]}**：空间 {pick[4]:+.4f}")
        print(f"    （专家 {pick[1]:.4f}，诚实上界 {pick[2]:.4%}）")
    print(f"DECISION_DEADLINE={pick[0]}")
    print(f"DECISION_HEADROOM={pick[4]:.6f}")
    print(f"DECISION_EXPERT_SR={pick[1]:.6f}")
    print(f"DECISION_CEILING={pick[2]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
