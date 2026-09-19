#!/usr/bin/env bash
# Reproduce the config the docs report as CLIMBING, on this machine, today.
#
# The whole 2x2 in .tmp/run_short_horizon.sh rests on one recorded fact:
# train_diag_fast.yaml (1 scenario, 240-step rollout) climbs +9.1 points over 20
# updates and passes the expert's 0.7765. That number predates the code changes
# made since (the slice-backward fix, the payload fix, the worker path), so
# before concluding "scenario count is the blocker" the baseline has to be shown
# to still climb here.
#
# If this is flat, the comparison is void and the real difference is something
# in the code history, not the scenario shape.
#
# Left exactly as the docs specify: n_rollout_workers 1 + rollout_batch (the only
# path that fills rollout_debug.jsonl), 4 episodes x 240 steps, fixed day 0,
# fixed request streams. eval_interval stays at the config's 5.
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python

rm -rf outputs/diag_repro
OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 \
    "$PY" -u scripts/train/train_graph_mappo.py \
        --configs rl_algorithm.yaml train_diag_fast.yaml \
        --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
        --num-updates 20 --run-name diag_repro \
        > .tmp/arm_diag_repro.log 2>&1 &
PID=$!
echo "launched diag_repro (pid $PID)"

sleep 40
"$PY" - <<'PYEOF' 2>/dev/null || echo "(config not written yet)"
import yaml
c = yaml.safe_load(open("outputs/diag_repro/resolved_config.yaml"))
print(f"  episode_steps={c['env']['episode_steps']} rollout_steps={c['train']['rollout_steps']} "
      f"episodes={c['train']['episodes_per_update']} start_mode={c['env']['episode_start_mode']} "
      f"start_day={c['env']['episode_start_day']} fixed_seed={c['train']['fixed_episode_seed']}")
PYEOF

wait $PID
echo "DIAG_REPRO_DONE"
