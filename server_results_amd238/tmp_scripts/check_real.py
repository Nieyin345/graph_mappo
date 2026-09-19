"""核对 real 三臂的验证序列与配对（本地，读原始 metrics.jsonl）。

为什么不用 harvest 的表：`wait_and_harvest.sh` 的完成判据（日志出现 `update=30`）
早于 metrics 最后一条 eval 的写入，mini512 因此少读了一条（0.6241 → 真实 0.6392）。

用法：python .tmp/check_real.py
"""

from __future__ import annotations

import json
from pathlib import Path

M = Path(__file__).resolve().parents[1] / ".tmp" / "m"


def load(name: str) -> list[dict]:
    rows = [json.loads(l) for l in (M / f"real_{name}.jsonl").open(encoding="utf-8") if l.strip()]
    return [d["eval_validation"] for d in rows if "eval_validation" in d]


def paired(a, b):
    n = min(len(a), len(b))
    diffs = [a[i] - b[i] for i in range(n)]
    md = sum(diffs) / n
    var = sum((x - md) ** 2 for x in diffs) / (n - 1)
    se = (var / n) ** 0.5
    return md, se, (md / se if se > 0 else float("nan")), n


ev = {n: load(n) for n in ("base", "mini512", "vcoef1")}

for n, e in ev.items():
    seq = [x["mean_success_rate"] for x in e]
    print(f"  {n:<9} n={len(seq)}  " + " ".join(f"{x:.4f}" for x in seq))

bs = ev["base"][-1]["per_seed_success"]
print(f"\nbase 末次 = {ev['base'][-1]['mean_success_rate']:.4f}  ({len(bs)} 种子)")
for n in ("mini512", "vcoef1"):
    e = ev[n][-1]
    md, se, t, k = paired(e["per_seed_success"], bs)
    print(f"  {n:<9} 末次={e['mean_success_rate']:.4f}"
          f"  vs base {md:+.4f}±{se:.4f}  t={t:+.2f}  ({k} 种子)")