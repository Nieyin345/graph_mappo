#!/usr/bin/env bash
# 构造模型后，价值头的初始权重在新旧两版是否相同。
set -u
MAIN=/opt/qkd/graph_mappo
OLD=/opt/qkd/eval_old_model
PY=/opt/qkd/venv/bin/python
SCRIPT="$MAIN/.tmp/rng_probe.py"

mkdir -p "$OLD/.tmp"
cp "$SCRIPT" "$OLD/.tmp/rng_probe.py"

echo "=== 旧模型（d1930a2）==="
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout 600 \
    "$PY" -u "$OLD/.tmp/rng_probe.py" --repo "$OLD" --out /tmp/rng_old.pt 2>&1 | tail -n 8

echo
echo "=== 新模型（工作区）==="
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout 600 \
    "$PY" -u "$SCRIPT" --repo "$MAIN" --out /tmp/rng_new.pt 2>&1 | tail -n 8

echo
echo "=== 对比 ==="
"$PY" -u "$SCRIPT" --compare /tmp/rng_old.pt /tmp/rng_new.pt 2>&1 | tail -n 20

echo
echo "RNG_COMPARE_DONE"
