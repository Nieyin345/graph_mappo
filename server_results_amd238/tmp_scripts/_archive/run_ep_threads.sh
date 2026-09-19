#!/usr/bin/env bash
# Does a single rollout episode get faster with more torch threads?
#
# The rollout workers pin themselves to ONE thread each (rollout_workers.py:107),
# so a 4-worker rollout uses 4 of the node's 48 threads and the machine sits
# ~90% idle. Whether lifting that helps depends on what the episode is actually
# bound by: if it scales with threads it is torch/matmul bound and giving each
# worker more threads is the fix; if it is flat it is Python/numpy bound and
# more threads buy nothing -- only more workers (or less per-step Python) would.
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python

for t in 1 4 8 24; do
    echo "########## threads=$t ##########"
    $PY -u .tmp/one_episode.py --steps 1440 --repeats 1 --threads "$t" 2>&1 \
        | grep -E "steps=|torch threads|episode |RESULT|Error|Traceback"
done
echo "EPT_DONE"
