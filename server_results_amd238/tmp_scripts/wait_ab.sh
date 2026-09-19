#!/usr/bin/env bash
# Poll the A/B rollout-workers run on the node until it finishes, then dump
# the summary tail. 40 polls x 180 s ~= 2 h coverage.
set -u
for i in $(seq 1 40); do
    if ssh -o BatchMode=yes -o ConnectTimeout=15 qkd \
        "grep -q '=== done' /tmp/ab_rollout.log 2>/dev/null"; then
        echo "=== A/B finished (poll #$i) ==="
        ssh -o BatchMode=yes qkd "tail -30 /tmp/ab_rollout.log"
        exit 0
    fi
    sleep 180
done
echo "=== timeout: A/B not finished after 2 h ==="
ssh -o BatchMode=yes qkd "tail -10 /tmp/ab_rollout.log"
exit 1
