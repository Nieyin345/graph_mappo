"""r4 六臂对 BC 起点 / 专家 的逐种子配对（本地算，补 sum_group 缺的两列）。

参照物由节点补跑：`outputs/eval/bc_diag_perseed.json`（0.7680）、`expert_diag.json`（0.7828）。
两者与 docs/训练诊断记录.md 里历史上记录的逐种子值**逐位一致**（重建成功）。
注意专家参照是**旧窗口**（3/0.25）口径 —— 与 r4 同口径，配对才成立。

用法：python .tmp/check_refs.py
"""

from __future__ import annotations

import json
from pathlib import Path

M = Path(__file__).resolve().parents[1] / ".tmp" / "m"
ARMS = ["base", "ent", "ent_ep2", "ent_c128", "ent_lr2e4", "c128"]


def paired(a, b):
    n = min(len(a), len(b))
    diffs = [a[i] - b[i] for i in range(n)]
    md = sum(diffs) / n
    var = sum((x - md) ** 2 for x in diffs) / (n - 1)
    se = (var / n) ** 0.5
    return md, se, (md / se if se > 0 else float("nan")), n


def r4_last(name: str) -> dict:
    rows = [json.loads(l) for l in (M / f"r4_{name}.jsonl").open(encoding="utf-8") if l.strip()]
    ev = [d["eval_validation"] for d in rows if "eval_validation" in d]
    return ev[-1]


bc = json.loads((M / "bc_diag_perseed.json").read_text(encoding="utf-8"))["per_seed_success"]
ex = json.loads((M / "expert_diag.json").read_text(encoding="utf-8"))["success"]
print(f"BC 起点均值 {sum(bc) / len(bc):.4f}    专家均值 {sum(ex) / len(ex):.4f}\n")

hdr = f"{'变体':<11}{'末次':>8}{'对BC':>18}{'t':>8}{'对专家':>18}{'t':>8}"
print(hdr)
print("-" * len(hdr))
for n in ARMS:
    e = r4_last(n)
    s = e["per_seed_success"]
    rows = f"{n:<11}{e['mean_success_rate']:>8.4f}"
    for ref in (bc, ex):
        md, se, t, k = paired(s, ref)
        rows += f"{f'{md:+.4f}±{se:.4f}':>18}{t:>+8.2f}"
    print(rows)