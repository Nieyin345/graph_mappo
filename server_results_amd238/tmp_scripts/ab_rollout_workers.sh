#!/usr/bin/env bash
# A/B: n_rollout_workers 1 vs 8 on the real training config.
# Interleaved order (1 -> 8 -> 8 -> 1) cancels server-load drift; statistics
# drop each run's first update (warmup). Throughput-only conclusions: compare
# rollout_s / update_s / elapsed_s means; do NOT read learning quality from
# 5 updates.
#
# Usage (on the server, after `bash deployment/sync.sh`):
#   nohup bash .tmp/ab_rollout_workers.sh > /tmp/ab_rollout.log 2>&1 &
set -euo pipefail
cd /opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python

# THREADS=2 protocol for the trainer process; workers pin themselves to 1
# torch thread inside rollout_workers.py.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

# Refuse to run next to other training jobs: contention would poison the
# timing comparison. Override with FORCE=1 if you really mean it.
if [[ "${FORCE:-0}" != "1" ]] && pgrep -f train_graph_mappo.py | grep -v "^$$" >/dev/null; then
    echo "ERROR: another train_graph_mappo.py is running; timing would be contaminated." >&2
    echo "       Set FORCE=1 to override." >&2
    exit 1
fi

CHECKPOINT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt

run_arm() {
    local name=$1; shift
    echo "=== arm $name : $(date +%H:%M:%S) ==="
    $PY scripts/train/train_graph_mappo.py \
        --configs rl_algorithm.yaml train_full_rl.yaml "$@" \
        --checkpoint "$CHECKPOINT" \
        --num-updates 5 \
        --run-name "$name"
}

# Interleaved: 1w -> 8w -> 8w -> 1w
run_arm abw1_a
run_arm abw8_a rl_parallel.yaml
run_arm abw8_b rl_parallel.yaml
run_arm abw1_b

echo "=== done : $(date +%H:%M:%S) ==="
for r in abw1_a abw8_a abw8_b abw1_b; do
    echo "--- $r rollout_debug.jsonl exists: $([[ -f outputs/$r/rollout_debug.jsonl ]] && echo yes || echo NO)"
done
