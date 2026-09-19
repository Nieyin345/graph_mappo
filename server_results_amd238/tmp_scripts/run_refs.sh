#!/usr/bin/env bash
# 重建两个**判读参照物**。节点重装后 outputs/ 被清空，产物没了，而
#   * BC 起点（outputs/eval/bc_diag_perseed.json）
#   * 专家 PG-Phased（outputs/eval/expert_diag.json）
# 是"训练到底有没有把 BC 权重推上去 / 离启发式天花板还差多少"的唯一答案。
# 它们都跑诊断协议（第 0 天、fixed、240 步、12 种子 7-18），各约 1 分钟。
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python
export OMP_NUM_THREADS=2

echo "== BC 起点 =="
"$PY" -u .tmp/probe_bc_baseline.py \
    outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
    outputs/eval/bc_diag_perseed.json

echo
echo "== 专家 PG-Phased =="
"$PY" -u scripts/eval/eval_expert.py \
    --config configs/var2_diag.yaml --seeds 7-18 --steps 240 \
    --out outputs/eval/expert_diag.json

echo REFS_DONE