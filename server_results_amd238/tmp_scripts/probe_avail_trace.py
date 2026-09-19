"""探针 I：可用性的**时间结构** —— 8.3% 的"生命期内从未全通"是物理，还是数据截断？

探针 H 判出：诊断场景里 **8.3–8.8% 的到达量**落在"源-目的在该请求 30 槽生命期内
从未同时连通"这一档（平均完成度只有 0.016–0.033）。这是整个缺口里最大的一块，
所以必须先确认它是**真物理**（LEO 过境窗口本来就错开）还是**数据/口径问题**
（比如卫星链路只在 t=0..2 有速率，之后整段是 0）。

判据：
  * 若各链路类型的可用边数在整个 240 槽里都周期性起落 -> 真物理，B_avail 是场景设定；
  * 若卫星相关类型在开头几槽之后**恒为 0** -> 数据截断，B_avail 是人为的，
    该改的是数据覆盖或 `min_link_rate` / `out_of_range_policy`，不是策略。

同时给出"每个槽有多少比例的 GS 对是连通的"——这就是 B_avail 那条上界的分布形状。

用法（节点上）：
    cd /opt/qkd/graph_mappo && OMP_NUM_THREADS=2 /opt/qkd/venv/bin/python -u \
        .tmp/probe_avail_trace.py
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

from qkd_rl.env.factory import build_env_from_config  # noqa: E402


def main() -> None:
    profile = _tp.load_validation_profile(ROOT / "configs" / "var2_diag.yaml")
    config = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=240,
        start_mode=profile["start_mode"])
    env = build_env_from_config(config)
    env.reset(seed=7, start_seed=7)

    rp = env.rate_provider
    edges = env.routing.edges
    horizon = 240

    by_type: dict[str, list] = defaultdict(list)
    for e in edges:
        by_type[e.link_type.value if hasattr(e.link_type, "value") else str(e.link_type)].append(e)

    print(f"数据集时间长度 _T = {rp._T}   slot_seconds = {rp.slot_seconds}   "
          f"min_link_rate = {rp.min_link_rate}")
    print(f"可用性口径 availability_source = {rp.availability_source}  "
          f"越界策略 out_of_range_policy = {rp.out_of_range_policy}")
    print(f"边数 = {len(edges)}   链路类型 {len(by_type)} 种\n")

    # 每条边在 0..horizon-1 的可用性
    avail: dict[str, list[bool]] = {}
    for e in edges:
        avail[e.edge_id] = [bool(rp.is_available(e.edge_id, t)) for t in range(horizon)]

    print(f"{'链路类型':<22}{'边数':>6}{'可用槽比例':>12}{'首可用':>8}{'末可用':>8}{'最长连续':>10}")
    for lt, es in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        flat = [avail[e.edge_id] for e in es]
        frac = float(np.mean(np.array(flat, dtype=float)))
        firsts = [a.index(True) if any(a) else -1 for a in flat]
        lasts = [(horizon - 1 - a[::-1].index(True)) if any(a) else -1 for a in flat]
        best = 0
        for a in flat:
            run = cur = 0
            for v in a:
                cur = cur + 1 if v else 0
                best = max(best, cur)
        print(f"{lt:<22}{len(es):>6}{frac:>12.3f}"
              f"{(min(firsts) if min(firsts) >= 0 else -1):>8}"
              f"{(max(lasts)):>8}{best:>10}")

    # 每槽的可用边数（按类型），看是不是"开头之外全 0"
    print("\n每槽可用边数（每 20 槽采样）：")
    print(f"{'t':>5}" + "".join(f"{lt[:12]:>14}" for lt in
                                [kv[0] for kv in sorted(by_type.items(), key=lambda kv: -len(kv[1]))[:5]]))
    order = [kv[0] for kv in sorted(by_type.items(), key=lambda kv: -len(kv[1]))[:5]]
    for t in range(0, horizon, 20):
        row = f"{t:>5}"
        for lt in order:
            row += f"{sum(1 for e in by_type[lt] if avail[e.edge_id][t]):>14}"
        print(row)

    # 每槽有多少比例的 GS 对连通（B_avail 那条上界的形状）
    node_ids = sorted({n for e in edges for n in (e.src, e.dst)})
    idx = {n: i for i, n in enumerate(node_ids)}
    # GS 节点从 node_registry.csv 的 type 列取，别靠名字猜（节点名是城市名，
    # 不是 GS_ 前缀；只有 edge_id 才带 GS_ / HAP_ / Sat_ 前缀）。
    import csv
    reg = Path("dataset/global/node_registry.csv")
    gs = []
    with reg.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            name = row["name"].strip()
            if "GS" in row["type"].strip().upper() and name in idx:
                gs.append(name)
    gs = sorted(gs)
    pairs = [(a, b) for i, a in enumerate(gs) for b in gs[i + 1:]]
    print(f"\nGS 数 = {len(gs)}   GS 对 = {len(pairs)}")

    fracs: list[float] = []
    for t in range(horizon):
        adj: dict[str, list[str]] = defaultdict(list)
        for e in edges:
            if avail[e.edge_id][t]:
                adj[e.src].append(e.dst)
                adj[e.dst].append(e.src)
        comp = np.full(len(node_ids), -1, dtype=np.int32)
        cid = 0
        for nd in node_ids:
            i = idx[nd]
            if comp[i] >= 0:
                continue
            comp[i] = cid
            stack = [nd]
            while stack:
                u = stack.pop()
                for v in adj.get(u, ()):
                    j = idx[v]
                    if comp[j] < 0:
                        comp[j] = cid
                        stack.append(v)
            cid += 1
        ok = sum(1 for a, b in pairs if comp[idx[a]] == comp[idx[b]])
        fracs.append(ok / max(1, len(pairs)))

    arr = np.array(fracs)
    print(f"连通 GS 对比例：min={arr.min():.3f}  中位={np.median(arr):.3f}  "
          f"均值={arr.mean():.3f}  max={arr.max():.3f}")
    print("每 20 槽采样：" + " ".join(f"{arr[t]:.2f}" for t in range(0, horizon, 20)))
    print(f"恒为 0 的槽数 = {int((arr == 0).sum())} / {horizon}")
    print("AVAIL_TRACE_DONE")


if __name__ == "__main__":
    main()