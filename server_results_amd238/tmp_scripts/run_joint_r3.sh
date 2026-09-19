#!/usr/bin/env bash
# 用**训练出来的**策略重跑探针 H。
#
# 前面几趟探针里的 `--policy rl` 用的其实是 BC 权重（`supervised_pg_phased_latest.pt`），
# 标签有歧义。第三浪的最佳变体是 `m512ent`（末次验证 0.8283，对 BC 起点 +0.0603，
# t=+10.6），这里就用它的 final checkpoint，看 C 桶（"曾全通却没搬完"）缩了多少。
# 只有 C 桶能缩，才说明那 5.1–5.5 点确实在策略手里；如果 C 不缩，说明它是被别人
# 占着（竞争），而不是"没人搬"。
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python
export OMP_NUM_THREADS=2

CKPT=outputs/r3_m512ent/checkpoint_final.pt
ls -l "$CKPT" || { echo "缺 checkpoint：$CKPT"; exit 1; }

"$PY" -u .tmp/probe_joint_avail.py --policy rl --tag _r3ent \
    --checkpoint "$CKPT" --seeds 7-18 --steps 240

echo JOINT_R3ENT_DONE