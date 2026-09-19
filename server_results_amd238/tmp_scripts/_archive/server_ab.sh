#!/usr/bin/env bash
# A/B the encoder+critic rewrite on the experiment node.
#
# The node's checkout is not a git repo (it arrived as a tarball), so the "before"
# file is shipped alongside as .tmp/graph_mappo_old.py and the two are swapped in
# place. A trap restores the new version, so an interrupt cannot leave the node
# running the old code.
#
# Run it detached and poll the log:
#   setsid nohup bash .tmp/server_ab.sh > .tmp/server_ab.log 2>&1 < /dev/null &
set -u

cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python
MODEL=qkd_rl/rl/models/graph_mappo.py
NEW=.tmp/graph_mappo_new.py
OLD=.tmp/graph_mappo_old.py

cp "$MODEL" "$NEW" || exit 1
restore() { cp "$NEW" "$MODEL"; echo "[restored new version]"; }
trap restore EXIT INT TERM

nproc
$PY -c "import torch; print('torch', torch.__version__, 'threads', torch.get_num_threads())"

run() {
    local tag="$1" threads="$2"
    echo "=== $tag (threads=$threads) ==="
    $PY -u .tmp/time_update.py --device cpu --steps 240 --episodes 8 --repeats 2 \
        --threads "$threads" 2>&1 | grep -E "rollout:|update |UPDATE_MEDIAN"
}

echo "########## BEFORE ##########"
cp "$OLD" "$MODEL"
grep -c "_segment_sum" "$MODEL" | sed 's/^/   segment_sum markers: /'
run before_t1 1
run before_t48 48

echo
echo "########## AFTER ##########"
cp "$NEW" "$MODEL"
grep -c "_segment_sum" "$MODEL" | sed 's/^/   segment_sum markers: /'
run after_t1 1
run after_t48 48

echo
echo "AB_DONE"
