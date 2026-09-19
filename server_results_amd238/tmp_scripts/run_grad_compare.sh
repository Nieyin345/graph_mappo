#!/usr/bin/env bash
# 比新旧模型的**反向梯度**（前向已验过等价）。
set -u

MAIN=/opt/qkd/graph_mappo
OLD=/opt/qkd/eval_old_model
PY=/opt/qkd/venv/bin/python
SCRIPT="$MAIN/.tmp/grad_probe.py"
CKPT="$MAIN/outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"

if [ ! -d "$OLD" ]; then
    git -C "$MAIN" worktree add --detach "$OLD" d1930a2 >/dev/null || { echo "worktree add 失败"; exit 1; }
    ln -sfn "$MAIN/dataset" "$OLD/dataset"
    mkdir -p "$OLD/outputs"
    ln -sfn "$MAIN/outputs/supervised_pg_phased" "$OLD/outputs/supervised_pg_phased"
fi
mkdir -p "$OLD/.tmp"
cp "$SCRIPT" "$OLD/.tmp/grad_probe.py"

echo "=== 旧模型（d1930a2）反传 ==="
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout 900 \
    "$PY" -u "$OLD/.tmp/grad_probe.py" --repo "$OLD" --checkpoint "$CKPT" \
    --out /tmp/grad_old.pt 2>&1 | tail -n 8

echo
echo "=== 新模型（工作区）反传 ==="
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout 900 \
    "$PY" -u "$SCRIPT" --repo "$MAIN" --checkpoint "$CKPT" \
    --out /tmp/grad_new.pt 2>&1 | tail -n 8

echo
echo "=== 梯度对比 ==="
"$PY" -u "$SCRIPT" --compare /tmp/grad_old.pt /tmp/grad_new.pt 2>&1 | tail -n 25

echo
echo "GRAD_COMPARE_DONE"
