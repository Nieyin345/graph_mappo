#!/usr/bin/env bash
# Where does the RL start, relative to the expert?
#
# The training runs begin from the BC checkpoint, and the docs measure the clone
# as losing ~5 points against its own expert (0.8104 -> 0.7581 on the standard
# protocol). If that gap is real on the protocol we actually evaluate on, then
# PPO is not trying to BEAT the expert -- it is trying to first climb back to it,
# and "improve the BC start" is worth more than any PPO knob.
#
# Measuring BC on the SAME 15 held-out seeds (7..21) that the expert and the
# trained checkpoint were measured on makes all three directly pairable.
#
# --policies is set to a name that does not exist, which disables every greedy
# baseline and leaves only the RL checkpoint -- the heuristics are already
# frozen in outputs/eval/frozen_baselines.json and re-running them is waste.
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python
export MPLBACKEND=Agg

OUT=outputs/eval/bc_heldout
rm -rf "$OUT"
"$PY" -u scripts/baselines/run_baselines.py \
    --config configs/global.yaml \
    --episodes 15 \
    --seeds 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21 \
    --out "$OUT" \
    --policies __none__ \
    --rl-checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
    --rl-name bc_warmstart \
    > .tmp/bc_heldout.log 2>&1
echo "exit=$?"
"$PY" .tmp/summarize_baselines.py "$OUT/summary.json"
echo "BC_EVAL_DONE"
