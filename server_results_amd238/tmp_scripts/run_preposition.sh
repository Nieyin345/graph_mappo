#!/usr/bin/env bash
# 探针 N：三策略（专家 / BC 起点 / r3_m512ent）在**确定性**口径下跑一遍
# "预置库存"度量。确定性是为了和探针 H 的 tbl、以及训练日志的验证数字同口径。
#
# 输出：outputs/eval/preposition_{expert,bc,r3ent}.json
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

BC=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
R3=outputs/r3_m512ent/checkpoint_final.pt

echo "== 专家 PG-Phased =="
timeout 2400 "$PY" -u .tmp/probe_preposition.py --policy expert --tag expert 2>&1 | tail -60

echo
echo "== BC 起点（确定性） =="
timeout 2400 "$PY" -u .tmp/probe_preposition.py \
    --policy rl --checkpoint "$BC" --deterministic --tag bc 2>&1 | tail -60

echo
echo "== r3_m512ent checkpoint_final（确定性） =="
timeout 2400 "$PY" -u .tmp/probe_preposition.py \
    --policy rl --checkpoint "$R3" --deterministic --tag r3ent 2>&1 | tail -60

echo "PREPOSITION_RUN_DONE"