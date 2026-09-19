#!/usr/bin/env bash
# Wait until the named runs have logged N training updates, then print where
# they stand.
#   bash .tmp/wait_arms.sh <num_updates> <run1> [run2 ...]
#
# metrics.jsonl interleaves two record kinds -- training updates (`"update": N`)
# and held-out validation records (`"eval_validation": {...}`) -- so counting
# lines overstates the update count and makes the waiter fire early. Count the
# update records; that is the number the caller means by "N updates".
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python

WANT="${1:-2}"
shift || true
RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then
    RUNS=(full_rnd15 full_d8)
fi

count() {
    grep -c '"update":' "outputs/$1/metrics.jsonl" 2>/dev/null || echo 0
}

for _ in $(seq 1 900); do
    done_all=1
    for r in "${RUNS[@]}"; do
        [ "$(count "$r")" -ge "$WANT" ] || done_all=0
    done
    [ "$done_all" -eq 1 ] && break
    sleep 20
done

for r in "${RUNS[@]}"; do
    echo "=== $r ($(count "$r") training updates) ==="
    "$PY" scripts/show_metrics.py "outputs/$r/metrics.jsonl" 2>/dev/null | tail -3
done

PATHS=()
for r in "${RUNS[@]}"; do
    PATHS+=("outputs/$r/metrics.jsonl")
done

echo
"$PY" .tmp/show_evals.py "${PATHS[@]}" 2>/dev/null
echo
"$PY" .tmp/analyze_arms.py "${PATHS[@]}" 2>/dev/null
