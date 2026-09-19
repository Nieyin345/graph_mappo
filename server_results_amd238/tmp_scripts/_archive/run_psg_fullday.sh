#!/usr/bin/env bash
# The expert heuristic on the FULL-DAY training-window protocol.
#
# Completes the picture: .tmp/eval_psg.py measured the expert on the standard
# held-out protocol (0.7082, paired vs the RL checkpoint at 0.6857). The other
# half is the scenario the training arms actually run on -- one full 1440-step
# day, days 0-295, the same 8 seed-days (7..14) the fixed arms replay. Without
# this the "does RL beat the expert" question is only answered on one of the two
# protocols.
#
# The profile is written here rather than shipped as a .yaml: server_sync.sh
# only force-adds .tmp/*.py and .tmp/*.sh, so a .tmp/*.yaml silently does not
# travel (that mistake already cost one failed run).
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python

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

echo "############ expert on FULL DAY (1440 steps), training window ############"
"$PY" -u .tmp/eval_psg.py --config "$PROFILE" --seeds 7-14 --steps 1440
echo "PSG_FULLDAY_DONE"
