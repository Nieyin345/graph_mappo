#!/usr/bin/env bash
# A/B: trainer thread count for the update phase (2 vs 16), rollout workers
# fixed at the new default 8. Interleaved 2 -> 16 -> 16 -> 2, drop each run's
# first update. Only update_s should move; rollout_s must stay ~47 s (the
# workers pin themselves to 1 torch thread, and the trainer is idle during
# rollout), which doubles as the sanity check that nothing else changed.
#
# Usage (on the server, after `bash deployment/sync.sh`):
#   nohup bash .tmp/ab_threads.sh > /tmp/ab_threads.log 2>&1 &
set -euo pipefail
cd /opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python

if [[ "${FORCE:-0}" != "1" ]] && pgrep -f train_graph_mappo.py | grep -v "^$$" >/dev/null; then
    echo "ERROR: another train_graph_mappo.py is running; timing would be contaminated." >&2
    echo "       Set FORCE=1 to override." >&2
    exit 1
fi

CHECKPOINT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt

run_arm() {
    local threads=$1; local name=$2
    echo "=== arm $name (threads=$threads) : $(date +%H:%M:%S) ==="
    OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads \
        $PY scripts/train/train_graph_mappo.py \
        --configs rl_algorithm.yaml train_full_rl.yaml \
        --checkpoint "$CHECKPOINT" \
        --num-updates 5 \
        --run-name "$name"
}

run_arm 2 thr2_a
run_arm 16 thr16_a
run_arm 16 thr16_b
run_arm 2 thr2_b

echo "=== done : $(date +%H:%M:%S) ==="
