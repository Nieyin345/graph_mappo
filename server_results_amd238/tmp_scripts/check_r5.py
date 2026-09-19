"""r5（窗口修复）vs r4（旧窗口）的逐种子配对。

两组都是诊断场景、90 轮、2 线程，同 update 编号对应同一批种子（7-18），
所以按位置配对是合法的。r5 若还没跑满，就取两组共同的最长前缀。

用法：python .tmp/check_r5.py
"""

from __future__ import annotations

import json
from pathlib import Path

M = Path(__file__).resolve().parents[1] / ".tmp" / "m"


def evals(name: str) -> list[dict]:
    p = M / f"{name}.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    return [d["eval_validation"] for d in rows if "eval_validation" in d]


def paired(a, b):
    n = min(len(a), len(b))
    diffs = [a[i] - b[i] for i in range(n)]
    md = sum(diffs) / n
    var = sum((x - md) ** 2 for x in diffs) / (n - 1)
    se = (var / n) ** 0.5
    return md, se, (md / se if se > 0 else float("nan")), n


PAIRS = [("r5_impfix", "r4_base"), ("r5_impfix_ent", "r4_ent"),
         ("r5_impfix_ent_ep2", "r4_ent_ep2")]

for new, old in PAIRS:
    en, eo = evals(new), evals(old)
    if not en:
        print(f"{new}: 还没有验证记录")
        continue
    k = min(len(en), len(eo))
    md, se, t, n = paired(en[k - 1]["per_seed_success"], eo[k - 1]["per_seed_success"])
    print(f"{new} vs {old}")
    print(f"  共同验证轮数={k}（对应 update={5 * k}）")
    print(f"  {new} 最新 {en[k - 1]['mean_success_rate']:.4f}"
          f"   {old} 同轮 {eo[k - 1]['mean_success_rate']:.4f}")
    print(f"  配对差 {md:+.4f}±{se:.4f}  t={t:+.2f}  → "
          f"{'显著' if abs(t) >= 2 else '不显著'}")
    tail_n = [e["mean_success_rate"] for e in en[-6:]]
    tail_o = [e["mean_success_rate"] for e in eo[-6:]]
    print(f"  尾部6次均值  {new} {sum(tail_n) / len(tail_n):.4f}"
          f"   {old} {sum(tail_o) / len(tail_o):.4f}")
    print()