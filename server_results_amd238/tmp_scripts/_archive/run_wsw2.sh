#!/usr/bin/env bash
# Does the rollout scale with worker count now that the ship payload is fixed?
#
# Before the fix: 4 workers -> 227 s, 8 workers -> 238 s (no scaling at all),
# because the trainer's serial unpickle of ~1.3 GB was the critical path and
# more workers could not shorten it. After the fix a 4-worker rollout is 71 s.
# This re-runs the same comparison so the answer is measured, not inferred.
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python

for w in 4 8; do
    echo "########## workers=$w ##########"
    $PY -u .tmp/sweep_workers.py --workers "$w" --repeats 1 2>&1 \
        | grep -E "rollout_steps=|rollout |RESULT|Error|Traceback"
done
echo "WSW2_DONE"
