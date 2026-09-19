#!/usr/bin/env bash
# 探针 H 的驱动：专家 + RL 各跑一遍诊断协议，并汇总两边都跑完之后的对比。
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python
export OMP_NUM_THREADS=2
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt

"$PY" -u .tmp/probe_joint_avail.py --policy expert --seeds 7-18 --steps 240 \
    2>&1 | tee /tmp/ja_expert.log

echo
"$PY" -u .tmp/probe_joint_avail.py --policy rl --seeds 7-18 --steps 240 \
    --checkpoint "$CKPT" 2>&1 | tee /tmp/ja_rl.log

echo JOINT_DRIVER_DONE