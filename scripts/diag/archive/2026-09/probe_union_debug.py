"""**定位**：dlhead 的并集占比为何在种子 102 上超过 100%（`uni > tot` 不可能）。

## 已知

`tot` 与 `uni` 遍历的是**同一个** `arrivals` 列表 ⟹ `uni <= tot` 是恒等式。
出现 124.5%（3 种子均值，反推种子 102 ≈ 200%）⟹ 只可能是
**`arrivals` 在两个循环之间增长了**，或 `tot` 统计的人口 ≠ `uni` 统计的人口。

而 `hook_effect` 已证：同一种子下两个探针的并集占比**逐位相同**（83.9340%），
且可用边集、请求内容都逐条相同。

## 这一条做什么

在 dlhead 的算法里**逐种子打印输入**：
  · `len(arrivals)` 与 `tot`，在 `live` 循环**之前**与 `uni` 循环**之后**各打一次
  · `uni` 与 `tot` 的原始值（不是比值）
  · 硬断言 `uni <= tot`，违反则把两个时刻的 arrivals 快照长度一起打出
  · 额外：把 `arrivals` 里**重复的请求对象**（同一 id 出现多次）数出来

★ 若发现 arrivals 在两个循环之间变长 ⟹ 有东西在算并集时**又加了请求**
  （最可疑：`shortest_path` 或 `components_for` 的副作用）⟹ 这是真 bug

用法：
    python3 -u probe_union_debug.py --seeds 100-102 --dl 30 --steps 240
"""
import importlib.util
import os
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-102")
    ap.add_argument("--dl", type=int, default=30)
    ap.add_argument("--steps", type=int, default=240)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    tp = _tp()
    profile = tp.load_validation_profile()
    start0 = int(profile.get("start_seed", 0))
    cfg = tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=args.steps,
        start_mode=profile["start_mode"])
    cfg["requests"] = dict(cfg.get("requests", {}))
    cfg["requests"]["deadline_steps"] = args.dl
    e0 = build_env_from_config(cfg)
    edge_by_id = {e.edge_id: e for e in e0.routing.edges}
    all_edges = sorted(edge_by_id)
    del e0

    print("=" * 104)
    print(f"并集占比定位（dl={args.dl}，种子 {seeds}，{args.steps} 步）")
    print("=" * 104)
    print(f"{'seed':>6}{'N(循环前)':>11}{'tot':>16}{'N(uni前)':>11}{'uni':>16}"
          f"{'uni/tot':>10}{'重复请求':>10}")

    ratios = []
    for s in seeds:
        env = build_env_from_config(cfg)
        obs = env.reset(seed=s, start_seed=start0 + s)

        arrivals = []
        orig_add = env.requests.add_arrivals

        def wrapped(reqs, _oa=orig_add, _a=arrivals):
            _a.extend(reqs)
            return _oa(reqs)

        env.requests.add_arrivals = wrapped

        avail_by_t = {}
        expert = PathScoreGreedy(weights=(1, 10, 1, 0.5, 0.2), phased=True,
                                 principles=False, router=ServeProbe(env))
        A = S = 0.0
        n = 0
        done = False
        while not done and n < args.steps:
            t_now = int(env.t)
            ew = env._build_state().edge_windows
            av = ew.available0(all_edges)
            avail_by_t[t_now] = {ed for ed, x in zip(all_edges, av) if bool(x)}
            acts, scores = expert.act(obs)
            obs, _r, term, trunc, info = env.step(acts, scores)
            A += float(info.get("arrived_keys", 0.0))
            S += float(info.get("served_keys", 0.0))
            n += 1
            done = term or trunc
        sr = S / A if A else 0.0

        horizon = max(avail_by_t) if avail_by_t else 0
        n_before = len(arrivals)
        tot = live = 0.0
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

        # ★★ 关键观察点：uni 循环之前，arrivals 多长了？
        n_at_uni = len(arrivals)
        ue = set()
        for es in avail_by_t.values():
            ue |= es
        cu = components_for(ue, edge_by_id)
        uni = 0.0
        for req in arrivals:
            if env.routing.shortest_path(req.src_gs, req.dst_gs) is None:
                continue
            if cu.same(req.src_gs, req.dst_gs):
                uni += float(req.amount)
        n_after = len(arrivals)

        dup = sum(c - 1 for c in Counter(id(r) for r in arrivals).values() if c > 1)
        ratio = uni / tot if tot else 0.0
        ratios.append(ratio)
        flag = ""
        if n_at_uni != n_before or n_after != n_at_uni:
            flag = "  ⟵ ★ arrivals 在算占比时长长!"
        if ratio > 1.0 + 1e-9:
            flag += "  ⟵ ★★ 比值 > 1，不可能"
        print(f"{s:>6}{n_before:>11}{tot:>16,.0f}{n_at_uni:>11}{uni:>16,.0f}"
              f"{ratio:>10.4%}{dup:>10}{flag}")

    print("-" * 104)
    import statistics as st
    print(f"  逐种子并集占比 {(', '.join(f'{r:.4%}' for r in ratios))}")
    print(f"  均值 {st.mean(ratios):.4%}   （dlhead3 报的是 124.5455%）")
    print()
    print("  判读：")
    print("   · 若某行的「循环前」与「uni前」N 不同 ⟹ arrivals 在算占比期间增长")
    print("   · 若「重复请求」> 0 ⟹ 有请求被 add_arrivals 加了多次")
    print("   · 若 uni > tot ⟹ 上面两条之一必然成立（数学上不可能有第三种）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
