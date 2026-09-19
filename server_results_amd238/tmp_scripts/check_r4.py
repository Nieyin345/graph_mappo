"""核对 r4 六臂的验证序列（本地，读 metrics.jsonl 的 eval_validation 行）。

为什么单独写：harvest 输出里 sum_group.py 的表格列宽会把相邻列粘在一起
（`0.000070.811`），直接当数据抄会抄错。这里从原始 metrics 读，顺带算
sum_group 没给的配对：ent_ep2 vs ent（epochs 2 的独立增量）。

用法：python .tmp/check_r4.py
"""

from __future__ import annotations

import json
from pathlib import Path

M = Path(__file__).resolve().parents[1] / ".tmp" / "m"
ARMS = ["base", "ent", "ent_ep2", "ent_c128", "ent_lr2e4", "c128"]


def load(name: str):
    evals: list[dict] = []
    for line in (M / f"r4_{name}.jsonl").open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "eval_validation" in d:
            evals.append(d["eval_validation"])
    return evals


def paired(a, b):
    n = min(len(a), len(b))
    diffs = [a[i] - b[i] for i in range(n)]
    md = sum(diffs) / n
    var = sum((d - md) ** 2 for d in diffs) / (n - 1)
    se = (var / n) ** 0.5
    return md, se, (md / se if se > 0 else float("nan")), n


ev = {n: load(n) for n in ARMS}

print("验证序列（每个点是 12 种子均值；eval_interval=5）")
for n in ARMS:
    seq = [e["mean_success_rate"] for e in ev[n]]
    print(f"  {n:<11} n={len(seq):>2}  " + " ".join(f"{x:.3f}" for x in seq))

print("\n尾部平台（最后 6 次验证的均值 ± 标准差）")
for n in ARMS:
    tail = [e["mean_success_rate"] for e in ev[n]][-6:]
    m = sum(tail) / len(tail)
    sd = (sum((x - m) ** 2 for x in tail) / (len(tail) - 1)) ** 0.5
    print(f"  {n:<11} {m:.4f} ± {sd:.4f}   （末次 {tail[-1]:.4f}）")

print("\n配对（正数 = 前者更好）")
bseeds = ev["base"][-1]["per_seed_success"]
for n in ARMS:
    if n == "base":
        continue
    md, se, t, k = paired(ev[n][-1]["per_seed_success"], bseeds)
    print(f"  {n:<11} vs base     {md:+.4f}±{se:.4f}  t={t:+.2f}  ({k} 种子)")

print()
for a, b in (("ent_ep2", "ent"), ("ent_ep2", "ent_c128"), ("ent_c128", "ent"),
             ("ent_lr2e4", "ent")):
    md, se, t, k = paired(ev[a][-1]["per_seed_success"], ev[b][-1]["per_seed_success"])
    print(f"  {a:<11} vs {b:<9} {md:+.4f}±{se:.4f}  t={t:+.2f}  ({k} 种子)")