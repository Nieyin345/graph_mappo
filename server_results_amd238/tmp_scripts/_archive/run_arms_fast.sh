#!/usr/bin/env bash
# Fast variants of the two discriminating arms -- same science as
# .tmp/run_arms.sh, but the rollout no longer runs on the slow path.
#
# What changes and why (all three are measured, see docs 5.2 / 5.3):
#
#   n_rollout_workers  1 -> 8    rollout 151 s -> 46 s.  The old value forced
#                                the single-process lockstep batch over 8 envs
#                                (.tmp/run_arms.sh header explains why it was
#                                kept); the pool path is 1.8x on the round and
#                                supports this config (not `continuous`,
#                                `episode_steps_fixed: true`).
#   eval_interval      5 -> 2    first held-out point at update 2 instead of 5.
#                                It matters here more than usual: arm A
#                                re-samples its days every update, so its
#                                *training* success rate is not comparable
#                                across updates at all -- only the held-out
#                                eval is.  Arm B's training number IS
#                                comparable (same 8 days replayed), which is
#                                the whole point of the pair.
#   batch_chunk      512 -> 64   as before: kernel merging, bit-identical
#                                results, removes the 30% variance.
#
# The one thing given up is rollout_debug.jsonl -- only the single-process path
# fills it (_update_rollout_debug).  That file is what reward tuning needs, not
# what the generalisation question needs; when it is wanted, take it from a
# short n_rollout_workers=1 diagnostic run rather than paying 1.8x on every
# training run.
#
# Load order note: this override file is passed LAST, and the resolved config
# is checked after launch -- a wrong key path would silently do nothing.
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python

OVERRIDE="$(pwd)/.tmp/arms_fast_ovr.yaml"
cat > "$OVERRIDE" <<'YAML'
train:
  n_rollout_workers: 8
  ppo:
    batch_chunk: 64
  logging:
    eval_interval: 2
YAML

CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
COMMON="--checkpoint $CKPT --num-updates 30"

launch() {
    local name="$1" cfg="$2"
    rm -rf "outputs/$name"
    OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 \
        "$PY" -u scripts/train/train_graph_mappo.py \
            --configs rl_algorithm.yaml "$cfg" "$OVERRIDE" \
            $COMMON --run-name "$name" \
            > ".tmp/arm_${name}.log" 2>&1 &
    echo "  launched $name (pid $!) -> .tmp/arm_${name}.log"
}

echo "=== launching fast arms ==="
launch full_rnd15 train_full_rl.yaml
launch full_d8    train_full_rl_fixdays.yaml

sleep 45
echo "=== resolved config check (must show 8 / 64 / 2) ==="
for name in full_rnd15 full_d8; do
    printf "%s: " "$name"
    "$PY" - "$name" <<'PY' 2>/dev/null || echo "(not written yet)"
import sys, yaml
c = yaml.safe_load(open(f"outputs/{sys.argv[1]}/resolved_config.yaml"))
t = c["train"]
print(
    f"workers={t['n_rollout_workers']} "
    f"chunk={t['ppo']['batch_chunk']} "
    f"eval_interval={t['logging']['eval_interval']} "
    f"episodes={t['episodes_per_update']} "
    f"minibatch={t['ppo']['minibatch_size']}"
)
PY
done

echo "=== waiting ==="
wait
echo "ARMS_DONE"
for name in full_rnd15 full_d8; do
    echo "--- $name ---"
    tail -2 ".tmp/arm_${name}.log"
done
