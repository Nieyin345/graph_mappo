#!/usr/bin/env bash
# A/B the encoder+critic rewrite on the SAME machine, same seed, same buffer.
# Only graph_mappo.py differs between the two sides, and the rewrite is its sole
# modification in the working tree, so `git checkout` of that one path is exactly
# the "before" state. The new version is copied aside and restored by the trap,
# so an interrupt cannot leave the old code in place.
set -u

cd "$(dirname "$0")/.." || exit 1
MODEL="qkd_rl/rl/models/graph_mappo.py"
BAK=".tmp/graph_mappo_new.py"

cp "$MODEL" "$BAK" || exit 1
restore() { cp "$BAK" "$MODEL"; echo "[restored new version]"; }
trap restore EXIT INT TERM

run() {
    local tag="$1" threads="$2"
    echo "=== $tag (threads=$threads) ==="
    python .tmp/time_update.py --device cpu --steps 240 --episodes 8 --repeats 2 \
        --threads "$threads" 2>&1 | grep -E "rollout:|update |UPDATE_MEDIAN"
}

echo "########## BEFORE ##########"
git checkout -- "$MODEL"
grep -c "_segment_sum" "$MODEL" | sed 's/^/segment_sum markers: /'
run before 1
run before_default 0

echo
echo "########## AFTER ##########"
cp "$BAK" "$MODEL"
grep -c "_segment_sum" "$MODEL" | sed 's/^/segment_sum markers: /'
run after 1
run after_default 0
