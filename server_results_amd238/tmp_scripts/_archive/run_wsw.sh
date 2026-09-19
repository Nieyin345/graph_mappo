#!/usr/bin/env bash
# Sweep n_rollout_workers on the canonical config. workers=4 is already known
# from the speed_canon run (rollout 227-231 s), so this only does the rest.
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python

for w in 8 16; do
    echo "########## workers=$w ##########"
    $PY -u .tmp/sweep_workers.py --workers "$w" --repeats 1 2>&1 | grep -E "rollout_steps=|rollout |RESULT|Error|Traceback"
done
echo "WSW_DONE"
