#!/usr/bin/env bash
# 在节点上跑探针 P（激活边 vs 在途需求的错位度）：专家 + BC 两档。
#
# 为什么要这一跑：探针 O 发现专家的边被钉住得**很久**（游程 9.92 槽）、
# 预算**从没触顶**（dem/89 ≈ 0.08），可它在需求的路径上覆盖率只有 31.71%。
# 也就是说瓶颈是"钉住了不该钉的边"。BC 起点的成功率高 4.6 点低于专家，
# 那点差是否也落在同一个错位度上，就是这一跑要回答的。
#
# 线程数 2（钉死，见 docs/测试规范.md §6）。与 9 个训练进程并发，
# 本机无关（跑在 56 核节点上）。
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

BC=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt

echo "== 探针 P：专家 =="
timeout 2400 "$PY" -u .tmp/probe_demand_alignment.py \
    --policy expert --seeds 7-18 --steps 240 --tag expert 2>&1 | tail -30

echo
echo "== 探针 P：BC 起点（确定性） =="
timeout 2400 "$PY" -u .tmp/probe_demand_alignment.py \
    --policy rl --checkpoint "$BC" --deterministic \
    --seeds 7-18 --steps 240 --tag bc 2>&1 | tail -30

echo
echo "== 探针 O：BC 起点（确定性） =="
timeout 2400 "$PY" -u .tmp/probe_activation_budget.py \
    --policy rl --checkpoint "$BC" --deterministic \
    --seeds 7-18 --steps 240 --tag bc 2>&1 | tail -25

echo ALIGN_SERVER_DONE