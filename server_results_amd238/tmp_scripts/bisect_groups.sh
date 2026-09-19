#!/usr/bin/env bash
# Which session change killed the climb?
#
# Established so far:
#   pre-session code (d1930a2)  -> deterministic eval at update 5 = 0.8203
#   current working tree        -> 0.7627, and flat through 20 updates
#   docs section 5.2 (recorded) -> 0.7560 ... 0.8188 by update 15
# Same config, same BC checkpoint, same machine, same protocol. So the code is
# the variable, and one eval point is enough to tell the two states apart.
#
# Config is already exonerated: putting rollout_batch_envs back to HEAD's 4 left
# it flat (0.7753 at update 5, 0.7335 at update 10).
#
# The log-prob array path is very likely NOT the cause either -- a line-by-line
# trace of _matching_log_prob_entropy_arrays against the dict version found the
# candidate order, arc set, feasibility rows, STOP position and normalization
# all identical (tests/test_matching_log_prob_paths.py pins that). What changed
# there are float32-vs-float64 accumulation details, ~1e-7 on the ratio.
#
# So the remaining suspects are grouped by coupling, because the files cannot be
# reverted independently: policy.py's array path needs graph_mappo.py's
# `edge_arrays`, so those two move together. Each group gets its own git
# worktree at d1930a2 with ONLY that group's files overlaid from the working
# tree, so the main working tree is never touched.
#
#   group1  models/graph_mappo.py + algos/policy.py   forward + sampler refactor
#   group2  algos/mappo_trainer.py                     KL early-stop criterion
#   group3  env/env.py + algos/rollout_buffer.py + algos/rollout_workers.py
#
# Reading the result: whichever group, when overlaid ALONE onto d1930a2, drops
# update 5 from ~0.82 to ~0.76, contains the regression.
set -u

MAIN=/opt/qkd/graph_mappo
BASE=d1930a2
PY=/opt/qkd/venv/bin/python
UPDATES="${UPDATES:-5}"

GROUP1="qkd_rl/rl/models/graph_mappo.py qkd_rl/rl/algos/policy.py"
GROUP2="qkd_rl/rl/algos/mappo_trainer.py"
GROUP3="qkd_rl/env/env.py qkd_rl/rl/algos/rollout_buffer.py qkd_rl/rl/algos/rollout_workers.py"

run_group() {
    local name="$1"; shift
    local files="$1"
    local wt="/opt/qkd/bisect_${name}"

    git -C "$MAIN" worktree remove --force "$wt" 2>/dev/null || rm -rf "$wt"
    git -C "$MAIN" worktree add --detach "$wt" "$BASE" >/dev/null || return 1

    # Overlay only this group's files from the working tree.
    for f in $files; do
        cp "$MAIN/$f" "$wt/$f" || { echo "  overlay FAILED for $f"; return 1; }
    done

    ln -sfn "$MAIN/dataset" "$wt/dataset"
    mkdir -p "$wt/outputs"
    ln -sfn "$MAIN/outputs/supervised_pg_phased" "$wt/outputs/supervised_pg_phased"

    (
        cd "$wt" || exit 1
        rm -rf "outputs/diag_${name}"
        OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 \
            "$PY" -u scripts/train/train_graph_mappo.py \
                --configs rl_algorithm.yaml train_diag_fast.yaml \
                --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
                --num-updates "$UPDATES" --run-name "diag_${name}" \
                > "diag_${name}.log" 2>&1
    )
    echo "  ${name} run done"
}

echo "Overlaying one group at a time onto $BASE, $UPDATES updates each."
echo

for g in group1 group2 group3; do
    files=""
    case "$g" in
        group1) files="$GROUP1" ;;
        group2) files="$GROUP2" ;;
        group3) files="$GROUP3" ;;
    esac
    echo "=== $g: $files ==="
    run_group "$g" "$files"
done

echo
echo "=== results: deterministic eval on the fixed scenario (12 seeds) ==="
echo "  reference  pre-session (no overlay)     0.8203   [measured earlier]"
echo "  reference  current working tree         0.7627   [diag_repro]"
printf "  %-38s" "group1 forward+sampler refactor"
grep -h eval_validation /opt/qkd/bisect_group1/outputs/diag_group1/metrics.jsonl 2>/dev/null \
    | tail -1 | sed 's/.*"mean_success_rate": \([0-9.]*\).*/\1/' || echo "(no eval)"
printf "  %-38s" "group2 KL early-stop criterion"
grep -h eval_validation /opt/qkd/bisect_group2/outputs/diag_group2/metrics.jsonl 2>/dev/null \
    | tail -1 | sed 's/.*"mean_success_rate": \([0-9.]*\).*/\1/' || echo "(no eval)"
printf "  %-38s" "group3 env + buffer + workers"
grep -h eval_validation /opt/qkd/bisect_group3/outputs/diag_group3/metrics.jsonl 2>/dev/null \
    | tail -1 | sed 's/.*"mean_success_rate": \([0-9.]*\).*/\1/' || echo "(no eval)"

echo
echo "BISECT_GROUPS_DONE"
