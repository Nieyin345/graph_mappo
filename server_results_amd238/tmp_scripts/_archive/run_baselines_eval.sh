#!/usr/bin/env bash
# Do the ~13% per-step failures move with the policy, or are they structural?
#
# Why this run: the reward A/B showed that raising the failure penalty 8x did
# not reduce failures (paired t = -2.59 the WRONG way). Two readings survive:
#   (a) the learning signal is too weak to act on any reward weight, or
#   (b) failures are not something the link-choice policy can move at all, so
#       the penalty is unactionable and no weight would help.
# Running several structurally different heuristics on the SAME protocol
# separates them: if random / greedy_* / the RL checkpoint all land on roughly
# the same failure rate, that rate is a property of the scenario (b), not of
# the policy.
#
# Protocol is the repo's canonical validation profile (configs/global.yaml:
# days 330-365, 240-step episodes) -- the same window the trainer's
# `eval_validation` uses, so these numbers sit directly beside the 0.64 that the
# two training arms reported there.
#
# 15 seeds instead of the profile's 5: a 5-seed mean has SEM ~0.03, far too
# coarse to see whether policies actually differ. The seeds stay the profile's
# own numbering (7..) so the run is still on-protocol, just deeper.
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python
export MPLBACKEND=Agg

OUT=outputs/eval/heur_vs_rl
rm -rf "$OUT"

"$PY" -u scripts/baselines/run_baselines.py \
    --config configs/global.yaml \
    --episodes 15 \
    --seeds 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21 \
    --out "$OUT" \
    --rl-checkpoint outputs/full_d8/checkpoint_final.pt \
    --rl-name rl_d8_final \
    > .tmp/baselines_eval.log 2>&1
echo "exit=$?"

echo "=== aggregate ==="
ls "$OUT"
cat "$OUT"/summary.json 2>/dev/null || cat "$OUT"/*.json 2>/dev/null | head -80
echo "BASELINES_DONE"
