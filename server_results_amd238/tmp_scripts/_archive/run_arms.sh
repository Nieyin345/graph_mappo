#!/usr/bin/env bash
# Launch the two discriminating arms, side by side.
#
# They answer the question the user actually has: "the fixed scenario climbs
# but the broad window does not". The two config files differ in exactly one
# flag -- fixed_episode_seed -- which pins the per-update resampling:
#
#   A  train_full_rl.yaml          fixed_episode_seed: false  (a new 8-day
#                                  sample every update)         -> full_rnd15
#   B  train_full_rl_fixdays.yaml  fixed_episode_seed: true   (the same 8 days
#                                  and streams every update)    -> full_d8
#
#   A climbs, B does not -> the blocker is gradient signal-to-noise from
#                           re-sampling days each update (generalisation)
#   neither climbs       -> the blocker is optimisation on the 1440-step horizon
#
# Both keep n_rollout_workers: 1 from rl_algorithm.yaml on purpose: it is the
# only path that fills rollout_debug.jsonl with the per-component reward
# breakdown, which is what reward re-engineering needs. The worker path is
# ~3x faster (46 s vs 151 s) but drops that file -- see docs §5.2.
#
# Only one override: batch_chunk 512 -> 64. It is a pure kernel-merging change
# (measured identical success rates), and the cache argument is the same here:
# at minibatch 256 the effective chunk was min(512, 256) = 256, whose edge
# tensor is ~51 MB against a 128 MB L3.

set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python

OVERRIDE="$(pwd)/.tmp/arms_ovr.yaml"
cat > "$OVERRIDE" <<'YAML'
train:
  ppo:
    batch_chunk: 64
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

echo "=== launching arms ==="
# --configs entries resolve as configs/<name>, so pass bare filenames.
launch full_rnd15 train_full_rl.yaml
launch full_d8    train_full_rl_fixdays.yaml

echo "=== waiting ==="
wait
echo "ARMS_DONE"
for name in full_rnd15 full_d8; do
    echo "--- $name ---"
    tail -2 ".tmp/arm_${name}.log"
done
