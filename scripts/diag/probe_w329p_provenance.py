"""还原 doc 里 w329p 那组数字的来源：`per_seed_success` 在每一轮的均值。

动机（2026-09-19）：`paired_verdict.py` 取的是**每个 run 自己最后可用的那个
update** 的 `per_seed_success`。w329p 的 s43/s44 因为被提前判"达标"
（等待器数行数、把 eval 行也数进去了）而在 **u25** 就停止判读，
而对照臂 ent01_t8 三臂都到 **u30**。

若真是这样，doc 里那组 Δ 就是 **u30 的对照 vs u25 的实验**——
而验证曲线在 u25→u30 还在涨（本项目已实测 +1 点量级），
**这个落差与 Δ=−0.0062 同量级，足以单独解释掉整个结论。**

本脚本把 `mean_success_rate` 与 `mean(per_seed_success)` 两个口径
在每一轮都打出来，用于确认 doc 用的是哪个、以及它来自哪一轮。
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
SEEDS = [42, 43, 44]
RUNS = [("对照", "ent01_t8_s{}"), ("实验", "w329p_s{}")]

# doc 里 w329p 一节逐行抄下的数字，用来反查来源
DOC_CTRL = {42: 0.7156, 43: 0.6988, 44: 0.7121}
DOC_EXP = {42: 0.6991, 43: 0.7071, 44: 0.7018}


def rows(path: Path):
    n = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = r.get("eval_validation")
        if isinstance(ev, dict) and n:
            yield n, ev
        elif isinstance(r.get("update"), int):
            n = max(n, int(r["update"]))


def main() -> int:
    tab: dict[str, dict[int, dict[int, tuple[float, float]]]] = {}
    for tag, fmt in RUNS:
        tab[tag] = {}
        for s in SEEDS:
            d: dict[int, tuple[float, float]] = {}
            for u, ev in rows(OUT / fmt.format(s) / "metrics.jsonl"):
                msr = ev.get("mean_success_rate")
                pss = ev.get("per_seed_success")
                msr = float(msr) if isinstance(msr, (int, float)) else float("nan")
                pss_m = (statistics.mean(float(x) for x in pss)
                         if isinstance(pss, list) and pss else float("nan"))
                d[u] = (msr, pss_m)
            tab[tag][s] = d

    for tag in tab:
        print("=" * 76)
        print(f"【{tag}】msr = mean_success_rate ；pss = mean(per_seed_success)")
        print("=" * 76)
        print(f"  {'update':>6s} " + "".join(f"{('s%d msr / pss' % s):>22s}" for s in SEEDS))
        ups = sorted({u for s in SEEDS for u in tab[tag][s]})
        for u in ups:
            row = f"  {u:>6d} "
            for s in SEEDS:
                if u in tab[tag][s]:
                    a, b = tab[tag][s][u]
                    row += f"{('%.4f / %.4f' % (a, b)):>22s}"
                else:
                    row += f"{'—':>22s}"
            print(row)
        print()

    # ---- 反查 doc 数字来自哪一轮、哪个口径 ----
    print("=" * 76)
    print("反查：doc 里抄下的数字 ← 哪个口径 / 哪一轮")
    print("=" * 76)
    for tag, doc in (("对照", DOC_CTRL), ("实验", DOC_EXP)):
        print(f"  ── {tag} ──")
        for s in SEEDS:
            hits = []
            for u, (msr, pss) in sorted(tab[tag][s].items()):
                for name, v in (("msr", msr), ("pss", pss)):
                    if abs(v - doc[s]) < 0.00005:
                        hits.append(f"u{u}/{name}")
            print(f"    s{s} doc={doc[s]:.4f}  ⟹ " +
                  (", ".join(hits) if hits else "**没找到完全匹配**"))

    # ---- 同轮配对：u25 与 u30 ----
    print()
    print("=" * 76)
    print("同轮配对（口径 = mean(per_seed_success)，与 paired_verdict 一致）")
    print("=" * 76)
    for u in (25, 30):
        ds, lines = [], []
        for s in SEEDS:
            c = tab["对照"][s].get(u)
            e = tab["实验"][s].get(u)
            if c and e:
                ds.append(e[1] - c[1])
                lines.append(f"      s{s}: 对照 {c[1]:.4f}  实验 {e[1]:.4f}  Δ={e[1] - c[1]:+.4f}")
        if not ds:
            print(f"  ── u{u}：没有可配对的种子 ──\n")
            continue
        print(f"  ── u{u}（n={len(ds)}）──")
        for L in lines:
            print(L)
        if len(ds) >= 2:
            m = statistics.mean(ds)
            sd = statistics.stdev(ds)
            se = sd / len(ds) ** 0.5
            print(f"    Δ={m:+.4f} SD={sd:.4f} SE={se:.4f} t={m / se:+.3f}")
            if len(ds) == 3:
                print(f"    df=2 临界 4.303 ⟹ "
                      f"{'可测' if abs(m / se) > 4.303 else '测不出'}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
