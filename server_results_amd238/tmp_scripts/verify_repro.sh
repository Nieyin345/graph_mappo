#!/usr/bin/env bash
# 验证可复现性修复：同配置同种子连跑两次，除计时字段外应逐位相同。
#
# 修复前：worker 的 Gumbel 采样用全局 torch RNG（由 OS 熵播种），
# 两次运行从第 1 轮 rollout 起就分歧。
# 修复后：应完全相同。
set -u
cd /opt/qkd/graph_mappo
source /opt/qkd/venv/bin/activate 2>/dev/null || true

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
UPDATES=${UPDATES:-4}
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
COMMON="--configs rl_algorithm.yaml train_mappo_smoke.yaml --checkpoint $CKPT --num-updates $UPDATES"

for i in 1 2; do
  rm -rf "outputs/verify_repro$i"
  echo "=== run $i ==="
  timeout 3600 python -u scripts/train/train_graph_mappo.py \
    $COMMON --run-name "verify_repro$i" >/dev/null 2>&1
  echo "  exit=$?"
done

echo
echo "=== 比较 ==="
python - <<'PY'
import json, pathlib
SKIP = {"rollout_s", "update_s", "elapsed_s", "wall_s", "timestamp", "run_name", "output_dir"}

def load(p):
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(j, dict):
            out.append({k: v for k, v in j.items() if k not in SKIP})
    return out

a = load(pathlib.Path("outputs/verify_repro1/metrics.jsonl"))
b = load(pathlib.Path("outputs/verify_repro2/metrics.jsonl"))
print(f"行数: run1={len(a)}  run2={len(b)}")
if len(a) != len(b):
    print("!! 行数不同，无法逐行比较")
    raise SystemExit(1)

diffs = 0
for i, (ra, rb) in enumerate(zip(a, b), 1):
    if ra != rb:
        diffs += 1
        if diffs <= 3:
            print(f"--- 第 {i} 行不同 ---")
            for k in sorted(set(ra) | set(rb)):
                va, vb = ra.get(k), rb.get(k)
                if va != vb:
                    print(f"    {k:<24} {va!r}  vs  {vb!r}")
print()
print(f"不同的行数: {diffs} / {len(a)}")
print("★ 可复现（逐位相同）" if diffs == 0 else "✗ 仍不可复现")
PY
