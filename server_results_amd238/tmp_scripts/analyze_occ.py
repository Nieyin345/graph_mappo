"""读 path_occupancy_*.json，看"路径一直忙"那一档到底是些什么请求。

触发原因：探针 F 的读数里，hops=2 的失败请求路径**全程空闲**（all_idle_share=1.0），
而 hops=3 的失败请求路径**全程有跳在忙**（hop_busy_share=1.0）—— 但整网占用率只有
3%。这两个数放在一起是矛盾的，先得看清 hops=3 那一档是不是集中在少数几条热点走廊上。

用法（节点上）：
    /opt/qkd/venv/bin/python .tmp/analyze_occ.py outputs/eval/path_occupancy_rl.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else
                "outputs/eval/path_occupancy_rl.json")
    d = json.load(path.open(encoding="utf-8"))
    rows = d["rows"]
    print(f"文件={path}  策略={d['policy']}  种子={d['seeds']}")
    if "link_utilization_used" in d:
        print(f"整网占用率={d['link_utilization']:.4f}  "
              f"用过传输的边占用率={d['link_utilization_used']:.4f}  "
              f"边数={d['n_edges']}  用过={d['edges_used']}")
        eb = d.get("edge_busy_total", {})
        st = d.get("steps_total", 1) or 1
        vals = sorted((c / st for c in eb.values()), reverse=True)
        if vals:
            # 注意：vals 是**降序**的，所以 index 0 是最大值、最后一个是最小值。
            # 上一版按升序的直觉去贴 p50/p90 标签，读出来是反的。
            print(f"  单边占用率（降序）：max={vals[0]:.4f}  "
                  f"p90={vals[int(0.10*(len(vals)-1))]:.4f}  "
                  f"中位={vals[len(vals)//2]:.4f}  "
                  f"p10={vals[int(0.90*(len(vals)-1))]:.4f}")
            top = sorted(eb.items(), key=lambda kv: -kv[1])[:8]
            print("  最忙的 8 条边（占用率）：")
            for e, c in top:
                print(f"    {e:<44}{c / st:.4f}")

    low = [r for r in rows if r["completion"] < 0.5]
    print(f"\n可达且过期={len(rows)}  完成度<0.5 的={len(low)}")

    # hop_busy_share 的实际取值分布（中位数会掩盖两个峰）
    for h in sorted({r["hops"] for r in low}):
        g = [r for r in low if r["hops"] == h]
        zeros = sum(1 for r in g if r["hop_busy_share"] == 0.0)
        ones = sum(1 for r in g if r["hop_busy_share"] == 1.0)
        mid = len(g) - zeros - ones
        print(f"\nhops={h}: n={len(g)}  "
              f"hop_busy_share==0 的 {zeros} 条，==1 的 {ones} 条，中间的 {mid} 条")
        print(f"  完成度：0 的 {sum(1 for r in g if r['completion'] == 0.0)} 条")

    print("\n完成度<0.5 的请求里，出现最多的 GS 对（前 12）：")
    c = Counter((r["src"], r["dst"]) for r in low)
    for (src, dst), n in c.most_common(12):
        g = [r for r in low if r["src"] == src and r["dst"] == dst]
        print(f"  {src:>10} -> {dst:<10} n={n:>3}  hops={g[0]['hops']}  "
              f"busy中位={sorted(r['hop_busy_share'] for r in g)[len(g)//2]:.3f}  "
              f"需求量={sum(r['amount'] for r in g):,.0f}")
    print(f"\n不同 GS 对总数={len(c)}  其中 n=1 的对={sum(1 for v in c.values() if v == 1)}")


if __name__ == "__main__":
    main()