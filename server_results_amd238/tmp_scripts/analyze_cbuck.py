"""C 桶的集中度分析：那 5 个百分点的"调度损失"是**摊在所有人身上**还是**集中在少数几条线上**？

## 为什么问这个

探针 H 把过期损失拆成三档，其中只有 **C（生命期内曾全通、却没搬完）** 是策略能动的，
约 5.2–5.5% 的到达量。最好模型（r3_m512ent，确定性 0.8283）离天花板 0.859–0.864
大约 3 个点 —— 也就是说策略已经回收了 C 的一大半，剩下 2–3 个点。

这 2–3 个点是"整体均匀偏慢"（那就只能靠更长训练/更好表示慢慢磨），
还是"卡在少数几对 GS 上"（那就有针对性的机制可做）？这决定了下一步方向，
所以必须先把 C 桶按 (src, dst) 拆开看集中度。

## 判据

* HHI / 前 10 对占比 —— 高（比如前 10 对吃掉 >50%）说明是局部问题。
* 与 `hops`（静态最短路跳数）的关系 —— 若剩余量随跳数单调上升，说明是长路径的
  逐跳瓶颈累积（结构性），不是某些对特别难。
* 各档的**平均完成度** —— 只完成一点点（<0.1）意味着"完全没抓住窗口"，
  完成一半左右意味着"抓到了但容量不够"。

用法（节点上）：
    /opt/qkd/venv/bin/python -u .tmp/analyze_cbuck.py outputs/eval/joint_avail_rl_r3ent_det.json ...
不带参数则扫 outputs/eval 下所有 joint_avail_*.json。
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNREACHABLE_HOPS = 10 ** 6


def buckets(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"A": [], "B_avail": [], "C": []}
    for r in rows:
        if r["hops"] < 0 or r["hops"] >= UNREACHABLE_HOPS:
            out["A"].append(r)
        elif r["avail_slots"] == 0:
            out["B_avail"].append(r)
        else:
            out["C"].append(r)
    return out


def report_node(name: str, pairs: list) -> None:
    """按节点归并：同一个 GS 出现在多少条 C 桶路径上、吃掉多少量。

    这一栏是为了回答"虽然不集中在少数**对**上，是不是集中在少数**节点**上"。
    """
    agg: dict[str, dict] = defaultdict(lambda: {"rem": 0.0, "end": 0})
    for (src, dst), d in pairs:
        for nd in (src, dst):          # 两端各记一次，同一个 GS 同时当源和目的就记两次
            agg[nd]["rem"] += d["rem"]
            agg[nd]["end"] += 1
    ranked = sorted(agg.items(), key=lambda kv: kv[1]["rem"], reverse=True)
    # 分母是"端点出现次数 × 该路径剩余量"，恒等于 2 × C 桶总量。
    tot = sum(a["rem"] for _, a in ranked)
    print(f"  {name:<14}{'端点次数':>9}{'归并剩余量':>14}{'占 2×C桶':>10}"
          f"（均匀时每节点 ≈ {1 / max(1, len(ranked)):.1%}）")
    for nd, a in ranked[:8]:
        print(f"  {nd:<14}{a['end']:>9}{a['rem']:>14,.0f}{a['rem'] / max(1e-9, tot):>10.1%}")


def report(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    arrived = float(data["arrived"])
    b = buckets(rows)
    print(f"\n===== {path.name} =====")
    print(f"策略={data.get('policy')}  确定性={data.get('deterministic')}  "
          f"checkpoint={Path(data.get('checkpoint') or '-').name}  到达量={arrived:,.0f}")
    print(f"过期请求数：A={len(b['A'])}  B_avail={len(b['B_avail'])}  C={len(b['C'])}")
    for k in ("A", "B_avail", "C"):
        g = b[k]
        if not g:
            continue
        rem = sum(r["remaining"] for r in g)
        print(f"  {k:<8} 剩余量={rem:>13,.0f}  占到达={rem / arrived:>6.2%}  "
              f"平均完成度={sum(r['served_frac'] for r in g) / len(g):>5.3f}")

    c = b["C"]
    if not c:
        return
    tot_c = sum(r["remaining"] for r in c)

    # ① 集中度：按 (src, dst) 归并剩余量
    by_pair: dict[tuple[str, str], dict] = defaultdict(lambda: {"rem": 0.0, "n": 0})
    for r in c:
        d = by_pair[(r["src"], r["dst"])]
        d["rem"] += r["remaining"]
        d["n"] += 1
    pairs = sorted(by_pair.items(), key=lambda kv: kv[1]["rem"], reverse=True)
    n_pairs = len(pairs)
    top10 = sum(d["rem"] for _, d in pairs[:10]) / max(1e-9, tot_c)
    hhi = sum((d["rem"] / max(1e-9, tot_c)) ** 2 for _, d in pairs)
    print(f"\n① 集中度：C 桶共 {n_pairs} 对 GS；前 10 对占 {top10:.1%}  "
          f"HHI={hhi:.4f}（均匀分布时 ≈ {1 / max(1, n_pairs):.4f}）")
    print(f"  {'src -> dst':<34}{'请求数':>7}{'剩余量':>14}{'占C桶':>9}{'完成度':>8}")
    for (src, dst), d in pairs[:10]:
        sub = [r for r in c if r["src"] == src and r["dst"] == dst]
        frac = sum(r["served_frac"] for r in sub) / len(sub)
        print(f"  {src + ' -> ' + dst:<34}{d['n']:>7}{d['rem']:>14,.0f}"
              f"{d['rem'] / max(1e-9, tot_c):>9.1%}{frac:>8.3f}")

    # ①b 按节点归并
    print(f"\n①b 按**节点**归并（同一 GS 出现在多少条 C 桶路径上）：")
    report_node("节点", pairs)

    # ② 与静态跳数的关系
    print(f"\n② 按静态最短路跳数分层：")
    by_hop: dict[int, list[dict]] = defaultdict(list)
    for r in c:
        by_hop[min(r["hops"], 12)].append(r)
    for h in sorted(by_hop):
        g = by_hop[h]
        rem = sum(r["remaining"] for r in g)
        print(f"  跳数={'12+' if h == 12 else h:<10}{'':<2}请求数={len(g):>4}  "
              f"剩余量={rem:>12,.0f}  占C桶={rem / max(1e-9, tot_c):>6.1%}  "
              f"平均完成度={sum(r['served_frac'] for r in g) / len(g):>5.3f}")

    # ③ 完成度分布：是"完全没抓住窗口"还是"抓到了但搬不完"
    print(f"\n③ C 桶完成度分布（0 = 一点没搬）：")
    edges = [0.0, 0.01, 0.1, 0.25, 0.5, 0.9, 1.01]
    for i in range(len(edges) - 1):
        g = [r for r in c if edges[i] <= r["served_frac"] < edges[i + 1]]
        rem = sum(r["remaining"] for r in g)
        print(f"  [{edges[i]:.2f},{edges[i + 1]:.2f})  请求数={len(g):>4}  "
              f"剩余量={rem:>12,.0f}  占C桶={rem / max(1e-9, tot_c):>6.1%}")

    # ④ 可用槽数 vs 完成度：窗口越长是不是就搬得越多（若否则说明是容量不是窗口）
    print(f"\n④ C 桶按可用槽数分层：")
    for lo, hi in ((1, 2), (3, 5), (6, 10), (11, 20), (21, 10 ** 9)):
        g = [r for r in c if lo <= r["avail_slots"] <= hi]
        if not g:
            continue
        rem = sum(r["remaining"] for r in g)
        lab = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        print(f"  {lab:<8}请求数={len(g):>4}  剩余量={rem:>12,.0f}  "
              f"占C桶={rem / max(1e-9, tot_c):>6.1%}  "
              f"平均完成度={sum(r['served_frac'] for r in g) / len(g):>5.3f}")


def main() -> None:
    args = sys.argv[1:]
    if args:
        paths = [Path(a) for a in args]
    else:
        paths = sorted((ROOT / "outputs" / "eval").glob("joint_avail_*.json"))
    if not paths:
        print("没有找到 joint_avail_*.json")
        return
    for p in paths:
        report(p)
    print("\nANALYZE_CBUCK_DONE")


if __name__ == "__main__":
    main()