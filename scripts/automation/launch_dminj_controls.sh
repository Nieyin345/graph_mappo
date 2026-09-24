#!/usr/bin/env bash
set -euo pipefail

root=/opt/qkd/ppo-stock-fix-20260922
cd "$root"
ulimit -n 65535

for seed in 42 43 44; do
  source_run="outputs/pairhist_v2_s${seed}_from_dminj30"
  if [[ ! -f "$source_run/checkpoint_final.pt" ]]; then
    echo "pair-history seed $seed has not finished; controls were not started" >&2
    exit 1
  fi
done

for seed in 42 43 44; do
  run="dminj_control_s${seed}_from_dminj30"
  dir="outputs/$run"
  if [[ -e "$dir/train.pid" || -e "$dir/checkpoint_final.pt" ]]; then
    echo "refusing to overwrite existing run: $dir" >&2
    exit 1
  fi
done

for seed in 42 43 44; do
  run="dminj_control_s${seed}_from_dminj30"
  dir="outputs/$run"
  mkdir -p "$dir"
  cp "outputs/pairhist_v2_s${seed}_from_dminj30/base_config.yaml" "$dir/base_config.yaml"
  case "$seed" in
    42) cpus=0-47 ;;
    43) cpus=48-95 ;;
    44) cpus=96-143 ;;
  esac
  nohup taskset -c "$cpus" env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    /opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py \
    --mode random_episode \
    --configs "../outputs/$run/base_config.yaml" train_dminj_control_equal_lr.yaml \
    --run-name "$run" --seed "$seed" --num-updates 30 \
    --checkpoint "/opt/qkd/graph_mappo/outputs/dminj_s${seed}/checkpoint_final.pt" \
    --device cpu > "$dir/train.log" 2>&1 < /dev/null &
  echo "$!" > "$dir/train.pid"
  echo "$run pid=$(cat "$dir/train.pid") cpus=$cpus"
done
