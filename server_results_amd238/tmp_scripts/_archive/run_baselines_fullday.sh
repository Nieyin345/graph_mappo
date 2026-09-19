#!/usr/bin/env bash
# Heuristics + RL on the FULL-DAY, TRAINING-WINDOW scenario.
#
# The previous baselines run used configs/global.yaml's profile: days 330-365,
# 240-step episodes. That is a different task from training (one sixth of a day,
# held-out days), and it produced success rates far below the ~0.85 a full day
# gives -- so those numbers cannot be compared against the user's own heuristic
# figures. This run re-measures everyone on the training protocol instead:
# days 0-295, 1440-step episodes, seeds 7..14 (the same 8 days the fixed arms
# replay).
#
# Both runs are reported so the horizon effect is visible rather than asserted.
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python
export MPLBACKEND=Agg

# run_baselines.py reads its protocol from `global.validation` of the config
# passed to --config. configs/global.yaml's profile is the held-out protocol
# (days 330-365, 240-step episodes) -- a different task from training, which is
# why the heuristics look like 0.44 there while the same policies reach ~0.85 on
# a full day. These values mirror train_full_rl_fixdays.yaml instead:
#   * activation window days 0-295   -> the training window
#   * episode_steps 1440             -> one full day, matching env.episode_steps
#   * seeds 7..14, start_mode random_day -> the same 8 fixed days the training
#     arms replay (fixed_episode_seed pins base_seed to env_seed = 7).
#
# Written here rather than shipped as a .yaml because server_sync.sh only
# force-adds .tmp/*.py and .tmp/*.sh -- a .tmp/*.yaml silently does not travel.
PROFILE="$(pwd)/.tmp/eval_full_day.yaml"
cat > "$PROFILE" <<'YAML'
global:
  validation:
    window:
      start_day: 0
      end_day: 295
    request_seeds: [7, 8, 9, 10, 11, 12, 13, 14]
    episodes: 8
    episode_steps: 1440
    start_mode: random_day
YAML

RL=outputs/full_d8/checkpoint_final.pt

echo "############ FULL DAY (1440 steps), training window ############"
OUT=outputs/eval/heur_full_day
rm -rf "$OUT"
"$PY" -u scripts/baselines/run_baselines.py \
    --config "$PROFILE" \
    --out "$OUT" \
    --rl-checkpoint "$RL" --rl-name rl_d8_final \
    > .tmp/baselines_full_day.log 2>&1
echo "exit=$?"
"$PY" .tmp/summarize_baselines.py "$OUT/summary.json"

echo
echo "############ QUARTER DAY (240 steps), held-out window (previous run) ############"
"$PY" .tmp/summarize_baselines.py outputs/eval/heur_vs_rl/summary.json

echo "FULLDAY_DONE"
