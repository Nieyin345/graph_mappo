#!/usr/bin/env bash
# Reward-component breakdown on the SAME 8 fixed days, at two policies.
#
# Why two runs: `rollout_debug.jsonl` is only filled by the single-process
# lockstep path (n_rollout_workers=1), and it is the only place the per-component
# reward split is recorded. Running it once at the BC checkpoint and once at the
# trained checkpoint answers a question the aggregate metrics cannot: did any
# reward term move without the success rate moving with it?
#
# Both runs are the fixed-8-day arm (train_full_rl_fixdays.yaml) so the two
# breakdowns describe identical scenarios and are directly comparable.
#
# --num-updates differs on purpose. load_checkpoint() restores update_count from
# the checkpoint (mappo_trainer.py:1167), so:
#   * the BC checkpoint carries update=0  -> 1 update runs 1 rollout
#   * checkpoint_final.pt carries update=30 -> 31 runs exactly 1 more
# In both cases the single rollout uses the checkpoint's own weights (the
# rollout happens before the update), which is what we want to measure.
#
# eval_interval: 0 -- validation is irrelevant here and costs ~44 s.
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python

OVERRIDE="$(pwd)/.tmp/reward_diag_ovr.yaml"
cat > "$OVERRIDE" <<'YAML'
train:
  n_rollout_workers: 1
  ppo:
    batch_chunk: 64
  logging:
    eval_interval: 0
YAML

run() {
    local name="$1" ckpt="$2" updates="$3"
    rm -rf "outputs/$name"
    OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 \
        "$PY" -u scripts/train/train_graph_mappo.py \
            --configs rl_algorithm.yaml train_full_rl_fixdays.yaml "$OVERRIDE" \
            --checkpoint "$ckpt" --num-updates "$updates" --run-name "$name" \
            > ".tmp/diag_${name}.log" 2>&1
    echo "  done $name (exit $?)"
}

echo "=== 1/2  BC checkpoint ==="
run diag_reward_bc outputs/supervised_pg_phased/supervised_pg_phased_latest.pt 1

echo "=== 2/2  trained checkpoint (full_d8 final) ==="
run diag_reward_trained outputs/full_d8/checkpoint_final.pt 31

echo
echo "=== breakdowns ==="
"$PY" .tmp/show_reward_breakdown.py \
    outputs/diag_reward_bc/rollout_debug.jsonl \
    outputs/diag_reward_trained/rollout_debug.jsonl
echo "REWARD_DIAG_DONE"
