"""核实"不可达"是真的拓扑断开，还是路由图构造的假象。

`.tmp/probe_failure_decomp.py` 报出约 25% 的过期需求落在 `hops == 10**6`
（DISCONNECTED 哨兵）上。如果这是真的，那这部分需求**从出生就注定失败**，
任何策略都救不回来 —— 是成功率天花板的一个硬缺口，必须先钉死。

可疑之处：`RoutingPolicy` 是用 `scenario.edges` 全量建的（qkd_rl/env/factory.py:68），
不含任何可用性过滤。所以 `_hop` 表里的连通性 = **静态拓扑连通性**，理论上不该有
断开的 GS 对（90 个节点的 FSO 网络）。所以要么

  a) 拓扑上确实有断开的 GS 对（那就有硬天花板），或
  b) 我的探针读 `hop_distance` 的方式有问题（比如节点 id 对不上）。

这个脚本直接、独立地数一遍：枚举全部 GS 对，用 `_hop` 表和一次独立 BFS
分别判断连通性，两者必须一致；再列出断开的具体是哪几对。

用法（节点上）：
    cd /opt/qkd/graph_mappo && /opt/qkd/venv/bin/python -u .tmp/verify_reachability.py
"""

from __future__ import annotations

import importlib.util
import sys
from collections import deque
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

from qkd_rl.env.factory import build_env_from_config  # noqa: E402


def main() -> None:
    profile = _tp.load_validation_profile(ROOT / "configs" / "global.yaml")
    config = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=240,
        start_mode=profile["start_mode"])
    env = build_env_from_config(config)
    routing = env.routing

    node_ids = list(env.node_ids)
    gs_ids = [n for n in node_ids if n.upper().startswith("GS") or "GS" in n[:6]]
    print(f"节点总数={len(node_ids)}  识别出的 GS={len(gs_ids)}")
    print(f"样例节点 id：{node_ids[:4]}")
    print(f"样例 GS id：{gs_ids[:4]}")
    print()

    # 全图连通分量（无向，静态拓扑）
    adj = {n: [] for n in node_ids}
    for e in routing.edges:
        adj.setdefault(e.src, []).append(e.dst)
        adj.setdefault(e.dst, []).append(e.src)
    comp = {}
    cid = 0
    for n in node_ids:
        if n in comp:
            continue
        q = deque([n])
        comp[n] = cid
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in comp:
                    comp[v] = cid
                    q.append(v)
        cid += 1
    print(f"静态拓扑连通分量数 = {cid}")
    from collections import Counter
    sizes = Counter(comp.values())
    print(f"各分量规模：{sorted(sizes.values(), reverse=True)[:8]}")
    print()

    # 独立 BFS 验证 GS 对连通性
    disconn = []
    for a, b in combinations(sorted(gs_ids), 2):
        hops = routing.hop_distance(a, b)
        bfs_ok = comp.get(a, -1) == comp.get(b, -2)
        sentinel_ok = hops >= routing.DISCONNECTED
        if bfs_ok == sentinel_ok:
            disconn.append((a, b, hops, bfs_ok))
    print(f"GS 对总数 = {len(list(combinations(sorted(gs_ids), 2)))}")
    print(f"BFS 与 _hop 哨兵**判定不一致**的对数 = {len(disconn)}"
          f"（0 才说明探针读法正确）")
    if disconn:
        for row in disconn[:10]:
            print("   不一致：", row)
    print()

    n_dis = sum(1 for a, b in combinations(sorted(gs_ids), 2)
                if routing.hop_distance(a, b) >= routing.DISCONNECTED)
    n_tot = len(list(combinations(sorted(gs_ids), 2)))
    print(f"真正断开的 GS 对 = {n_dis} / {n_tot}（{n_dis / max(1, n_tot):.1%}）")
    if n_dis:
        bad = [p for p in combinations(sorted(gs_ids), 2)
               if routing.hop_distance(*p) >= routing.DISCONNECTED][:20]
        print("  断开样例：", bad[:10])


if __name__ == "__main__":
    main()
