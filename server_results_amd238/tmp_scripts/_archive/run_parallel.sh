#!/usr/bin/env bash
# Is the OS-level parallelism there at all?
#
# Measured so far: one episode costs 28.9 s single-threaded and the 4- and
# 8-worker rollouts both take ~230 s -- i.e. exactly 8 x 28.9. And a single
# episode barely speeds up with 24 threads (24.6 s), so it is Python/numpy
# bound, not torch bound. That leaves one question: can this box run episodes
# in parallel at all, or does something serialise them?
#
# This runs 8 independent processes, each doing 2 episodes, and times the wall
# clock. Parallel  -> ~58 s (each process does its 2 sequentially).
# Serialised -> ~460 s (16 episodes end to end).
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python

rm -f .tmp/par_*.log
t0=$(date +%s)
for s in 0 1 2 3 4 5 6 7; do
    $PY -u .tmp/one_episode.py --steps 1440 --repeats 1 --threads 1 --seed "$s" \
        > ".tmp/par_$s.log" 2>&1 &
done
wait
t1=$(date +%s)
echo "WALL $((t1 - t0)) s for 8 processes x 2 episodes"
grep -h "RESULT" .tmp/par_*.log
echo "PAR_DONE"
