#!/usr/bin/env bash
set -euo pipefail

root=/opt/qkd/ppo-stock-fix-20260922
declare -a parents=()
declare -a children=()

# Validate every run before signalling any process. The trainer interprets
# --num-updates as ADDITIONAL updates after resume, so these runs would reach
# update 90 despite the intended update-60 comparison.
for seed in 42 43 44; do
  run="pairhist_v2_s${seed}_from_dminj30"
  dir="$root/outputs/$run"
  test -s "$dir/checkpoint_update_000060.pt"
  test ! -e "$dir/checkpoint_final.pt"
  /opt/qkd/venv/bin/python - "$dir/metrics.jsonl" <<'PY'
import json, sys
current = None
validated = set()
for line in open(sys.argv[1], encoding="utf-8"):
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "update" in row:
        current = int(row["update"])
    if "eval_validation" in row:
        validated.add(current)
assert 60 in validated, f"update 60 validation missing: {sys.argv[1]}"
PY
  pid="$(cat "$dir/train.pid")"
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$cmd" == *"$run"* ]] || { echo "PID does not match $run" >&2; exit 1; }
  parents+=("$pid")
  while read -r child; do
    [[ -z "$child" ]] || children+=("$child")
  done < <(pgrep -P "$pid" || true)
done

kill -TERM "${parents[@]}"
if ((${#children[@]})); then
  kill -TERM "${children[@]}" 2>/dev/null || true
fi
sleep 5
for pid in "${parents[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    state="$(ps -o stat= -p "$pid" | tr -d ' ')"
    [[ "$state" == Z* ]] || { echo "process $pid did not exit after SIGTERM" >&2; exit 1; }
  fi
done

for seed in 42 43 44; do
  dir="$root/outputs/pairhist_v2_s${seed}_from_dminj30"
  cp -p "$dir/checkpoint_update_000060.pt" "$dir/checkpoint_final.pt"
  cmp -s "$dir/checkpoint_update_000060.pt" "$dir/checkpoint_final.pt"
  printf '%s\n' 'Training was intentionally stopped at update 60 after validation.' \
    'checkpoint_final.pt is a byte-for-byte copy of checkpoint_update_000060.pt.' \
    'The trainer CLI --num-updates 60 would otherwise run 60 additional updates after resuming from update 30.' \
    > "$dir/finalization_note.txt"
  echo "finalized seed $seed at update 60"
done
