#!/usr/bin/env bash
# Reward A/B: raise the failure penalty, change nothing else.
#
# Diagnosis this follows from (.tmp/show_reward_breakdown.py, 2026-09-15):
#   * 98.3% of the reward is `served`; `failed` is 1.5%; every other shaping term
#     is either explicitly disabled or contributes ~0.
#   * the physical counters reconcile exactly: ~75,700 requests arrive per step,
#     65,400 are served, 9,900 FAIL -- the 13% failure rate IS the headroom, and
#     it is worth only 1.5% of the reward.
#   * the trained policy made failures WORSE (9,932 -> 10,300) while generating
#     12.6% fewer keys, i.e. it drifted toward passivity.
#
# So the one knob: failed_weight 5.0 -> 40.0 (8x). That lifts `failed` from
# ~1.5% to ~12% of the reward, roughly matching the 13% failure rate it is
# supposed to price. A cheap, single-variable hypothesis.
#
# reward_scale is deliberately NOT touched, even though the config block warns
# to re-tune it whenever the reward block changes. Changing two knobs at once
# would confound "the penalty is heavier" with "the critic's target moved".
# The risk is pre-registered and observable instead: reward_scale 0.002 was
# chosen to hold return_std at O(1), so if the heavier penalty pushes the critic
# out of range, value_std / return_std / corr(V,R) in metrics.jsonl will show it
# and the arm is simply inconclusive rather than silently wrong.
#
# Both arms are the fixed-8-day config, so every update replays the same days
# and the same request streams and the per-update training success rate is
# directly comparable -- that is the readout, not the held-out eval (whose
# horizon is 240 steps against training's 1440, see validation.episode_steps).
#
# A FRESH control runs alongside rather than reusing full_d8, so both arms see
# the same machine conditions at the same time.
set -u
cd /opt/qkd/graph_mappo || exit 1
PY=/opt/qkd/venv/bin/python

BASE="$(pwd)/.tmp/rwd_ab_base.yaml"
cat > "$BASE" <<'YAML'
train:
  n_rollout_workers: 8
  ppo:
    batch_chunk: 64
  logging:
    eval_interval: 5
YAML

TREAT="$(pwd)/.tmp/rwd_ab_fail40.yaml"
cat > "$TREAT" <<'YAML'
reward:
  failed_weight: 40.0
YAML

CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
COMMON="--checkpoint $CKPT --num-updates 30"

launch() {
    local name="$1"; shift
    rm -rf "outputs/$name"
    OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 \
        "$PY" -u scripts/train/train_graph_mappo.py \
            --configs rl_algorithm.yaml train_full_rl_fixdays.yaml "$@" \
            $COMMON --run-name "$name" \
            > ".tmp/arm_${name}.log" 2>&1 &
    echo "  launched $name (pid $!)"
}

echo "=== launching ==="
launch rwd_ctrl   "$BASE"
launch rwd_fail40 "$BASE" "$TREAT"

sleep 45
echo "=== resolved reward block (fail40 must show 40.0, served 50.0, scale 0.002) ==="
for name in rwd_ctrl rwd_fail40; do
    printf "%-14s " "$name"
    "$PY" - "$name" <<'PY' 2>/dev/null || echo "(config not written yet)"
import sys, yaml
c = yaml.safe_load(open(f"outputs/{sys.argv[1]}/resolved_config.yaml"))
r, t = c["reward"], c["train"]
print(f"failed_weight={r['failed_weight']} served_weight={r['served_weight']} "
      f"scale={r['reward_scale']} mode={r['mode']} "
      f"| workers={t['n_rollout_workers']} chunk={t['ppo']['batch_chunk']}")
PY
done

echo "=== waiting ==="
wait
echo "REWARD_AB_DONE"
