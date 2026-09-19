"""real 三臂训练侧信号：定位 base 0.701->0.622 退化的签名。

用法：python .tmp/check_real_base.py
"""
from __future__ import annotations

import json
from pathlib import Path

M = Path(__file__).resolve().parents[1] / ".tmp" / "m"

KEYS = [
    "entropy", "kl", "mean_ratio", "actor_grad_norm", "critic_grad_norm",
    "value_return_corr", "mean_abs_advantage", "mean_success_rate",
]

for name in ("real_base", "real_vcoef1", "real_mini512"):
    p = M / f"{name}.jsonl"
    rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    tr = [r for r in rows if "update" in r]
    ev = [r["eval_validation"] for r in rows if "eval_validation" in r]
    print(f"==== {name}  updates={len(tr)}  evals={len(ev)}")
    print("eval 序列: " + " ".join(f"{e['mean_success_rate']:.4f}" for e in ev))
    if not tr:
        continue
    # 首尾对比 + 中段走势
    print("  upd" + "".join(f"{k[:9]:>10}" for k in KEYS))
    step = max(1, len(tr) // 8)
    for r in tr[::step] + [tr[-1]]:
        if r is tr[::step][-1] and tr[-1] in tr[::step]:
            pass
        cells = []
        for k in KEYS:
            v = r.get(k)
            cells.append(f"{v:>10.4f}" if isinstance(v, float) else f"{str(v):>10}")
        print(f"{r['update']:>4}" + "".join(cells))
    # 分三段均值
    seg = max(1, len(tr) // 3)
    for i, tag in enumerate(("前1/3", "中1/3", "后1/3")):
        chunk = tr[i * seg: (i + 1) * seg] if i < 2 else tr[2 * seg:]
        if not chunk:
            continue
        avg = {k: sum(r.get(k, 0.0) for r in chunk) / len(chunk) for k in KEYS}
        print(f"  {tag}: " + " ".join(f"{k[:6]}={avg[k]:.4f}" for k in KEYS))
    print()
