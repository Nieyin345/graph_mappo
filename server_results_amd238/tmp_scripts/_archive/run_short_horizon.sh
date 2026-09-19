#!/usr/bin/env bash
# Which axis blocks the climb: episode length, or scenario count?
#
# Two known data points disagree, and they differ on BOTH axes at once:
#
#   train_diag_fast.yaml   1 scenario (day 0),  240 steps  -> CLIMBS (+9.1 over
#                                                            20 updates, past
#                                                            the expert's 0.7765)
#   train_full_rl_fixdays  8 fixed days,       1440 steps -> FLAT
#
# Re-sampling days is already ruled out as the cause: the resampled-day arm and
# the fixed-8-day arm behaved identically (both flat), which also kills the
# "gradient signal-to-noise from re-sampling" explanation the docs leaned on.
#
# So this run fills the missing cell -- the SAME 8 fixed days as the flat arm,
# but at the diagnostic's 240-step horizon:
#
#   climbs here        -> the blocker is episode LENGTH (credit assignment over
#                         1440 steps), not scenario diversity
#   flat here too      -> the blocker is SCENARIO COUNT (1 vs 8 days is already
#                         enough to kill it), not horizon
#
# reward_scale is untouched. Nothing about the reward changes in this run; it is
# a pure scenario-shape experiment, which is the point -- every reward-side
# hypothesis has already been tested and rejected.
#
# eval_interval is 0: the held-out eval runs 240-step episodes on days 330-365,
# which is a different task from this one (docs/测试规范.md §4 rule 5) and costs
# ~44 s per call. The per-update training success rate IS the comparable readout
# here, because fixed_episode_seed replays the same 8 days every update.
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python

# Horizon means `train.rollout_steps`, not `env.episode_steps`: the diag config
# runs 720-step env episodes but collects only the first 240 steps per rollout
# (`_collect_rollout_serial` loops `while steps < self.rollout_steps`), so the
# credit-assignment span -- the axis in question -- is rollout_steps. This run
# mirrors the diag's shape (720 / 240) and changes only the scenario count.
OVERRIDE="$(pwd)/.tmp/short_horizon_ovr.yaml"
cat > "$OVERRIDE" <<'YAML'
env:
  episode_steps: 720
train:
  rollout_steps: 240
  n_rollout_workers: 8
  ppo:
    batch_chunk: 64
  logging:
    eval_interval: 0
YAML

CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt

rm -rf outputs/short8
OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 \
    "$PY" -u scripts/train/train_graph_mappo.py \
        --configs rl_algorithm.yaml train_full_rl_fixdays.yaml "$OVERRIDE" \
        --checkpoint "$CKPT" --num-updates 30 --run-name short8 \
        > .tmp/arm_short8.log 2>&1 &
PID=$!
echo "launched short8 (pid $PID)"

sleep 45
echo "=== resolved config (rollout_steps must be 240, episode_steps 720) ==="
"$PY" - <<'PYEOF' 2>/dev/null || echo "(config not written yet)"
import yaml
c = yaml.safe_load(open("outputs/short8/resolved_config.yaml"))
print(f"  env.episode_steps={c['env']['episode_steps']}  "
      f"train.rollout_steps={c['train']['rollout_steps']}  "
      f"episodes={c['train']['episodes_per_update']}  "
      f"workers={c['train']['n_rollout_workers']}  "
      f"fixed_episode_seed={c['train']['fixed_episode_seed']}")
PYEOF

wait $PID
echo "SHORT8_DONE"
