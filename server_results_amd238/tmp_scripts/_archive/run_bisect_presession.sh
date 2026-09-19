#!/usr/bin/env bash
# Bisect step 2: run the diag config on the PRE-SESSION code.
#
# Step 1 ruled out config: putting rollout_batch_envs back to HEAD's value 4 did
# not restore the climb (eval at update 5 = 0.7753, update 10 = 0.7335, docs say
# 0.7560 -> 0.7858). So whatever removed the climb is in working-tree code.
#
# The server's deploy branch carries the project's full history, and the commit
# immediately before the first sync is d1930a2 -- which is also the tip of main,
# i.e. exactly the code as it stood before this session's changes. Checking that
# out gives a clean A/B on "session code vs pre-session code" with nothing else
# varying.
#
# A git worktree rather than a checkout: the main working tree stays untouched
# and running, and dataset/ + the BC checkpoint are symlinked in so the baseline
# sees identical data without a 374 MB copy.
set -u
BASE_COMMIT=d1930a2
MAIN=/opt/qkd/graph_mappo
WT=/opt/qkd/baseline

if [ -d "$WT" ]; then
    git -C "$MAIN" worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT"
fi
git -C "$MAIN" worktree add --detach "$WT" "$BASE_COMMIT" || exit 1

# Same data, same weights -- only the code differs.
ln -sfn "$MAIN/dataset" "$WT/dataset"
mkdir -p "$WT/outputs"
ln -sfn "$MAIN/outputs/supervised_pg_phased" "$WT/outputs/supervised_pg_phased"

echo "=== baseline worktree at $(git -C "$WT" rev-parse --short HEAD) ==="
ls -la "$WT/dataset" | head -2

cd "$WT" || exit 1
PY=/opt/qkd/venv/bin/python

rm -rf outputs/diag_base
OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 \
    "$PY" -u scripts/train/train_graph_mappo.py \
        --configs rl_algorithm.yaml train_diag_fast.yaml \
        --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
        --num-updates 20 --run-name diag_base \
        > "$WT/diag_base.log" 2>&1 &
PID=$!
echo "launched diag_base (pid $PID)"

sleep 40
"$PY" - <<'PYEOF' 2>/dev/null || echo "  (config not written yet -- check diag_base.log)"
import yaml
c = yaml.safe_load(open("outputs/diag_base/resolved_config.yaml"))
t = c["train"]
print(f"  rollout_batch_envs={t['rollout_batch_envs']} episodes={t['episodes_per_update']} "
      f"rollout_steps={t['rollout_steps']} chunk={t['ppo']['batch_chunk']} "
      f"failed_weight={c['reward']['failed_weight']}")
PYEOF

wait $PID
echo "DIAG_BASE_DONE"
