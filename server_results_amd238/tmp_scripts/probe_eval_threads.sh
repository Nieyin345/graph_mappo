#!/usr/bin/env bash
# 评测本身对线程数敏感吗？
#
# 刚查实：**训练**在固定线程数下逐位可复现，但改 OMP_NUM_THREADS 会确定性地
# 改变结果（2 线程 vs 4 线程，末次验证差 +0.0176，t=+7.68，且重复两次完全一致）。
#
# 这直接影响判据：BC 起点和专家这两个参照点是我用 **4 线程**测的，而 12 个变体
# 的 eval_validation 是 **2 线程**跑的。如果**评测**也有同样量级的线程敏感性，
# 那"对 BC 配对差"里就混进了一个系统性偏移，结论会被夸大。
#
# 做法：同一个 checkpoint、同一套验证协议，只改线程数：
#   * 2 线程跑两次 —— 查评测自身在固定线程下是否可复现
#   * 4 线程再跑一次 —— 和已有的 outputs/eval/bc_diag_perseed.json 对照
#
# 用法（在节点上）：
#     bash .tmp/probe_eval_threads.sh
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt

cd "$MAIN" || exit 1
rm -f /tmp/bc_eval_*.json

echo "=== 起跑前负载 ==="
uptime
echo

run_eval() {  # $1=线程数 $2=标签
    echo "--- OMP_NUM_THREADS=$1  ($2) ---"
    OMP_NUM_THREADS="$1" MKL_NUM_THREADS="$1" \
        "$PY" -u .tmp/probe_bc_baseline.py "$CKPT" "/tmp/bc_eval_$2.json" 2>&1 \
        | grep -E "逐种子|均值|已写"
    echo
}

run_eval 2 t2_first
run_eval 2 t2_second
run_eval 4 t4

echo "=== 汇总对比 ==="
"$PY" - <<'PY'
import json
from pathlib import Path

runs = {}
for tag in ("t2_first", "t2_second", "t4"):
    p = Path(f"/tmp/bc_eval_{tag}.json")
    if p.exists():
        runs[tag] = json.loads(p.read_text(encoding="utf-8"))["per_seed_success"]
existing = Path("/opt/qkd/graph_mappo/outputs/eval/bc_diag_perseed.json")
if existing.exists():
    runs["先前那次(4线程)"] = json.loads(
        existing.read_text(encoding="utf-8"))["per_seed_success"]

for name, per in runs.items():
    print(f"{name:<20} 均值 {sum(per) / len(per):.6f}")

print()
names = list(runs)
ref = runs[names[0]]
for name in names[1:]:
    other = runs[name]
    n = min(len(ref), len(other))
    diffs = [ref[i] - other[i] for i in range(n)]
    md = sum(diffs) / n
    var = sum((d - md) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    se = (var / n) ** 0.5 if n > 1 else float("nan")
    ident = "  <-- 逐位相同" if all(abs(x) < 1e-12 for x in diffs) else ""
    print(f"{names[0]} vs {name:<20} 差 {md:+.6f} ± {se:.6f}{ident}")

print()
print("判读：")
print("  * t2_first vs t2_second 逐位相同 => 评测本身是确定性的。")
print("  * 2 线程 vs 4 线程之差若在 1e-6 量级 => 评测对线程数不敏感，")
print("    之前的'对 BC 配对差'不受影响。")
print("  * 若这个差也到 1e-2 量级 => 参照点和变体不在同一条件下，必须统一重测。")
print("EVAL_THREAD_DONE")
PY
