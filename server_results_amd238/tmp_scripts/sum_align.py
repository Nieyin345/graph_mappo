"""汇总探针 P/O 的 JSON 产物（本地），避免逐字段手算。

用法：python .tmp/sum_align.py <demand_alignment.json|activation_budget.json>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def mean(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else float("nan")


def sum_p(path: Path) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    slots, reqs = d["slots"], d["reqs"]
    print(f"policy={d.get('policy')} ckpt={d.get('checkpoint')}")
    print(f"量加权成功率={d['served'] / max(1e-9, d['arrived']):.4f}"
          f"  槽数={len(slots)}  在途请求样本={len(reqs)}")
    print("\n【表 1】每槽均值")
    print(f"  act={mean(s['act'] for s in slots):.1f}"
          f"  needs={mean(s['needs'] for s in slots):.1f}"
          f"  重合={mean(s['inter'] for s in slots):.1f}"
          f"  覆盖率={mean(s['inter'] / max(1, s['needs']) for s in slots):.2%}"
          f"  多余={mean(s['extra'] for s in slots):.1f}"
          f"  live={mean(s['live'] for s in slots):.1f}")
    print("\n【表 2】槽级覆盖率分布（needs>0 才计入分母）")
    tot = sum(1 for s in slots if s["needs"] > 0)
    print(f"  needs>0 槽数={tot}  占全部槽 {tot / len(slots):.2%}")
    for lo, hi in ((0.0, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
        sel = [s for s in slots
               if s["needs"] > 0
               and (s["inter"] == 0 if lo == 0.0 else s["inter"] / s["needs"] > lo)
               and (s["inter"] / s["needs"] <= hi if hi < 1.0 else True)
               and (s["inter"] / s["needs"] <= hi or hi == 1.0)]
        sel = [s for s in sel if s["inter"] / s["needs"] <= hi + 1e-12]
        if not sel:
            continue
        label = "覆盖率=0" if lo == hi == 0.0 else f"({lo:.0%},{hi:.0%}]"
        print(f"  {label:<14}{len(sel):>8}{len(sel) / max(1, tot):>10.2%}"
              f"  平均needs={mean(s['needs'] for s in sel):.1f}"
              f"  平均重合={mean(s['inter'] for s in sel):.1f}")
    print("\n【表 3】逐请求（注意 JSON 无 H=0 记录，H=0 的请求没写进来）")
    for label, pre in (("全部跳激活", lambda r: r["cov"] == r["H"] and r["H"] > 0),
                       ("部分跳激活", lambda r: 0 < r["cov"] < r["H"]),
                       ("一跳没激活", lambda r: r["cov"] == 0 and r["H"] > 0)):
        sel = [r for r in reqs if pre(r)]
        if not sel:
            continue
        print(f"  {label:<10}{len(sel):>8}{len(sel) / max(1, len(reqs)):>9.2%}"
              f"  平均H={mean(r['H'] for r in sel):.1f}"
              f"  平均cov={mean(r['cov'] for r in sel):.1f}"
              f"  平均pos={mean(r['pos'] for r in sel):.1f}")


def sum_o(path: Path) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    slots = d["slots"]
    run_hist: dict[int, int] = defaultdict(int)
    for r, c in d.get("run_hist", {}).items():
        run_hist[int(r)] = c
    print(f"policy={d.get('policy')} ckpt={d.get('checkpoint')}")
    print(f"量加权成功率={d['served'] / max(1e-9, d['arrived']):.4f}")
    print("\n【表 1】每槽均值")
    print(f"  act={mean(s['act'] for s in slots):.1f}"
          f"  keep={mean(s['keep'] for s in slots):.1f}"
          f"  sw={mean(s['sw'] for s in slots):.1f}"
          f"  live={mean(s['live'] for s in slots):.1f}"
          f"  dem={mean(s['dem'] for s in slots):.1f}"
          f"  unr={mean(s['unr'] for s in slots):.1f}"
          f"  avail={mean(s['avail'] for s in slots):.1f}")
    over = [s for s in slots if s["dem"] > s["n_nodes"]]
    print(f"  dem>预算 的槽：{len(over)}/{len(slots)}={len(over) / len(slots):.2%}")
    tot_runs = sum(run_hist.values())
    avg_run = sum(r * c for r, c in run_hist.items()) / max(1, tot_runs)
    print(f"  平均游程={avg_run:.2f} 槽  总游程数={tot_runs}")
    for lo, hi in ((1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 10 ** 9)):
        k = sum(c for r, c in run_hist.items() if lo <= r <= hi)
        if k:
            label = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 9 else f">={lo}")
            print(f"  游程{label:<6}{k:>10}{k / max(1, tot_runs):>10.2%}")


if __name__ == "__main__":
    p = Path(sys.argv[1])
    if "activation" in p.name:
        sum_o(p)
    else:
        sum_p(p)
