#!/usr/bin/env bash
# Bisect step 1: is the missing climb caused by rollout_batch_envs 4 -> 8?
#
# The docs' §5.2 curve for this exact config (12-seed deterministic eval on the
# fixed scenario) is
#     update  5      10     15     20
#             0.7560 0.7858 0.8188 0.8175
# Reproducing it today on the same machine gives
#             0.7627 0.7645 0.7488 0.7690
# -- same starting point, no climb. So something between the measurement and now
# removed it, and this is a regression, not a property of the scenario.
#
# `git diff HEAD -- configs/rl_algorithm.yaml` shows exactly ONE semantic change
# for this config: rollout_batch_envs 4 -> 8 (the working copy's edit of mine;
# HEAD still says 4). Everything else lives in working-tree code, not config.
#
# So: hold everything else fixed and put the cap back to HEAD's value. The change
# was documented as "run-to-run noise level (~0.003 success), not semantic" --
# but it alters which matchings get sampled, so the rollout data differs and a
# 20-update trajectory can diverge. Either way it is the cheapest thing to rule
# out, and if this run DOES climb, the regression is mine and the other session
# changes are exonerated.
#
# Why --num-updates 20: that is exactly how far the docs' table goes, so the two
# curves are comparable step for step.
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python

OVERRIDE="$(pwd)/.tmp/bisect_envs4_ovr.yaml"
cat > "$OVERRIDE" <<'YAML'
train:
  rollout_batch_envs: 4
YAML

rm -rf outputs/diag_envs4
OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 \
    "$PY" -u scripts/train/train_graph_mappo.py \
        --configs rl_algorithm.yaml train_diag_fast.yaml "$OVERRIDE" \
        --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
        --num-updates 20 --run-name diag_envs4 \
        > .tmp/arm_diag_envs4.log 2>&1 &
PID=$!
echo "launched diag_envs4 (pid $PID)"

sleep 40
"$PY" - <<'PYEOF' 2>/dev/null || echo "(config not written yet)"
import yaml
c = yaml.safe_load(open("outputs/diag_envs4/resolved_config.yaml"))
print(f"  rollout_batch_envs={c['train']['rollout_batch_envs']} "
      f"episodes={c['train']['episodes_per_update']} "
      f"rollout_steps={c['train']['rollout_steps']} "
      f"chunk={c['train']['ppo']['batch_chunk']}")
PYEOF

wait $PID
echo "BISECT_ENVS4_DONE"
