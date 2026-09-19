#!/usr/bin/env bash
# 并行跑一个种子（供 r9ext 并行化使用：3 个种子同时跑，而非串行）。
#
# 为什么不改线程数提速：线程数会确定性影响训练结果（实测差 0.018），
# r8 → r9ext 的对比必须同线程。所以走"多种子并行"这条路 ——
# 每个进程仍 4 线程，只是不再排队。
set -u
cd /opt/qkd/graph_mappo
SEED="$1"
UPDATES="${2:-30}"
LOG="/tmp/r9ext_s${SEED}.log"
echo "[$(date -Is)] r9ext_s${SEED} 启动（${UPDATES} 轮）" > "$LOG"
timeout 21600 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py \
    --configs rl_algorithm.yaml train_full_rl.yaml \
    --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
    --seed "${SEED}" --num-updates "${UPDATES}" --run-name "r9ext_s${SEED}" \
    >> "$LOG" 2>&1
echo "[$(date -Is)] r9ext_s${SEED} 退出码 $?" >> "$LOG"
