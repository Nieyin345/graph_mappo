#!/usr/bin/env bash
# 跑路径占用探针（专家 + BC），诊断场景协议（第 0 天、240 步、12 种子 7-18）。
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
CFG=configs/var2_diag.yaml
SEEDS=7-18
STEPS=240

cd "$MAIN" || exit 1

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -u .tmp/probe_path_occupancy.py \
    --policy expert --config "$CFG" --seeds "$SEEDS" --steps "$STEPS" \
    2>&1 | tee /tmp/occ_expert.log
echo
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -u .tmp/probe_path_occupancy.py \
    --policy rl --checkpoint "$CKPT" --config "$CFG" --seeds "$SEEDS" --steps "$STEPS" \
    2>&1 | tee /tmp/occ_rl.log
echo PATH_OCCUPANCY_DONE