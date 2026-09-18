#!/usr/bin/env python
"""汇总 outputs/eval/*.json 的基线评估结果（均值/标准误/种子/步数）。

为什么要它：`docs/训练诊断记录.md` 里到处引用"专家 0.698"作为参照，但那个数字
的**种子集、步数、标准误**从来没人核对过。本脚本把每个基线文件的口径摊开。

用法：python -m scripts.diag.summarize_eval --dir outputs/eval
      或 python scripts/diag/summarize_eval.py --dir <目录>
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/eval")
    ap.add_argument("--pair", nargs=2, default=None,
                    help="额外做两两配对差（两个文件名）")
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("*.json"))
    if not files:
        print(f"{args.dir} 下没有 json")
        return

    print(f"{'文件':<44}{'n':>4}{'步数':>7}{'种子起点':>10}"
          f"{'均值':>9}{'SE':>9}{'min':>8}{'max':>8}")
    print("-" * 100)
    store = {}
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"{f.name:<44} 读取失败 {e}")
            continue
        s = d.get("success") or []
        if not s:
            print(f"{f.name:<44} 无 success 字段（keys={list(d)[:5]}）")
            continue
        store[f.name] = (d, s)
        m = sum(s) / len(s)
        se = statistics.stdev(s) / len(s) ** 0.5 if len(s) > 1 else 0.0
        sd = (d.get("seeds") or ["?"])[0]
        print(f"{f.name:<44}{len(s):>4}{str(d.get('steps','?')):>7}"
              f"{str(sd):>10}{m:>9.4f}{se:>9.4f}{min(s):>8.3f}{max(s):>8.3f}")

    if args.pair:
        a, b = args.pair
        if a in store and b in store:
            da, sa = store[a]
            db, sb = store[b]
            sa_seeds = da.get("seeds") or []
            sb_seeds = db.get("seeds") or []
            common = [x for x in sa_seeds if x in sb_seeds]
            print(f"\n=== 配对：{a} vs {b} ===")
            print(f"  {a}: n={len(sa)} seeds={sa_seeds[:5]}...")
            print(f"  {b}: n={len(sb)} seeds={sb_seeds[:5]}...")
            print(f"  共同种子 {len(common)} 个")
            if len(common) < 3:
                print("  共同种子太少，无法配对 —— 这两个文件的数字**不可直接比较**")
                return
            ia = {x: i for i, x in enumerate(sa_seeds)}
            ib = {x: i for i, x in enumerate(sb_seeds)}
            diffs = [sa[ia[x]] - sb[ib[x]] for x in common]
            md = sum(diffs) / len(diffs)
            sed = (statistics.stdev(diffs) / len(diffs) ** 0.5
                   if len(diffs) > 1 else 0.0)
            print(f"  配对差均值 = {md:+.4f}  SE = {sed:.4f}  "
                  f"t = {md/sed if sed else float('nan'):.2f}")
            print(f"  逐种子差: " + " ".join(f"{d:+.3f}" for d in diffs))


if __name__ == "__main__":
    main()
