#!/usr/bin/env bash
# 冒烟：用修复后的代码跑 2 轮真实训练，确认整条路径不崩、GAE 不炸。
# 高线程只为快，结果不用来比较。
set -uo pipefail
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt

rm -rf outputs/smoke_tfix
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 timeout 1800 \
    $PY -u scripts/train/train_graph_mappo.py \
        --configs rl_algorithm.yaml train_full_rl.yaml \
        --checkpoint "$CKPT" \
        --num-updates 2 --run-name smoke_tfix \
    > /tmp/smoke_tfix.log 2>&1
rc=$?
echo "退出码=$rc"
echo "--- 关键指标 ---"
grep -E "^update=" /tmp/smoke_tfix.log | tail -3
echo "--- 是否有异常 ---"
grep -iE "error|traceback|nan|assert" /tmp/smoke_tfix.log | head -10 || echo "(无)"
echo "=== SMOKE_DONE rc=$rc ==="
