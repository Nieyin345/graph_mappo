#!/usr/bin/env bash
# 把探针 H 拉到**训练侧同口径**（确定性动作）重跑两遍：
#
#   * BC 起点  —— 之前 joint_avail_rl.json 是采样跑的，和 bc_diag_perseed.json
#                 的 0.7680（确定性）不是一回事。
#   * r3_m512ent 的 checkpoint_final.pt —— 训练日志记 0.8283（确定性），
#                 但 joint_avail_rl_r3ent.json 只量出 0.7830，差 4.5 点。
#                 如果这一跑能回到 ~0.83，说明那 4.5 点是"采样 vs 确定性"，
#                 探针 H 原来的桶级数字（A/B_avail/C）只需在确定性口径下重读一次；
#                 如果回不去，说明两个口径之间还有别的东西，得先查清楚再下结论。
#
# 注意：B_avail / A 两档与策略无关，采样或确定性都不变（可作为这次的内部对照）。
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

BC=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
R3=outputs/r3_m512ent/checkpoint_final.pt

echo "== BC 起点（确定性） =="
timeout 1800 "$PY" -u .tmp/probe_joint_avail.py \
    --policy rl --checkpoint "$BC" --deterministic --tag _bc_det 2>&1 | tail -25

echo
echo "== r3_m512ent checkpoint_final（确定性） =="
timeout 1800 "$PY" -u .tmp/probe_joint_avail.py \
    --policy rl --checkpoint "$R3" --deterministic --tag _r3ent_det 2>&1 | tail -25

echo "JOINT_DET_DONE"