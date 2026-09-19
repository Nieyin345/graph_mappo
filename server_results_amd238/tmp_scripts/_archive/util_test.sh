#!/usr/bin/env bash
# How many arms actually fit on this node at once?
#
# The update saturates around 12 threads (measured: 24/16/12 threads all land
# within 6% of each other), so a single arm leaves most of the box idle -- the
# update is memory-bandwidth-bound, not core-bound. That predicts several arms
# should run side by side at close to solo speed, which is exactly what "use
# the whole server" means for this workload.
#
# Each arm gets its own run dir, its own OMP thread count, and its own rollout
# workers, so nothing is shared except the machine. Wall time is measured per
# arm and the aggregate is compared against one arm alone.
#
#   THREADS=16 WORKERS=4 bash .tmp/util_test.sh 1 3
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python
THREADS="${THREADS:-16}"
WORKERS="${WORKERS:-4}"
UPDATES="${UPDATES:-1}"

override="$(pwd)/.tmp/arm_ovr.yaml"
cat > "$override" <<YAML
train:
  n_rollout_workers: $WORKERS
  rollout_worker_device: cpu
YAML
# build_config resolves --configs entries as configs/<name>, and pathlib lets an
# absolute path win the join, so the override has to be absolute to be found.

for k in "$@"; do
    echo "########## $k arm(s), $(($THREADS)) threads + $WORKERS workers each ##########"
    # "utilrun_" not "util_": the latter also matches this script's own name
    # (util_test.sh) and deleted it mid-run, which left the server's git work
    # tree dirty and silently blocked every subsequent sync.
    rm -rf .tmp/utilrun_* 2>/dev/null
    t0=$(date +%s)
    for i in $(seq 1 "$k"); do
        (
            OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
            "$PY" -u scripts/train/train_graph_mappo.py \
                --mode random_episode \
                --configs train_mappo.yaml "$override" \
                --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
                --run-name "utilrun_${k}_$i" --num-updates "$UPDATES" \
                > ".tmp/utilrun_${k}_$i.log" 2>&1
            echo "$(( $(date +%s) - t0 ))" > ".tmp/utilrun_${k}_$i.secs"
        ) &
    done
    wait
    t1=$(date +%s)
    wall=$((t1 - t0))
    echo "  wall for $k arm(s): ${wall} s"
    for i in $(seq 1 "$k"); do
        echo "    arm $i: $(cat ".tmp/utilrun_${k}_$i.secs" 2>/dev/null || echo '?') s"
    done
    awk -v k="$k" -v w="$wall" -v u="$UPDATES" \
        'BEGIN{printf "    aggregate throughput: %.4f updates/s\n", k*u/w}'
done
echo "UTIL_DONE"
