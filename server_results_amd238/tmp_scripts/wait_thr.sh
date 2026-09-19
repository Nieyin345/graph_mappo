#!/usr/bin/env bash
# Poll the threads A/B run until it finishes, then dump the summary tail.
# 30 polls x 180 s ~= 1.5 h coverage.
set -u
for i in $(seq 1 30); do
    if ssh -o BatchMode=yes -o ConnectTimeout=15 qkd \
        "grep -q '=== done' /tmp/ab_threads.log 2>/dev/null"; then
        echo "=== threads A/B finished (poll #$i) ==="
        ssh -o BatchMode=yes qkd "grep '=== arm' /tmp/ab_threads.log; tail -4 /tmp/ab_threads.log"
        exit 0
    fi
    sleep 180
done
echo "=== timeout: threads A/B not finished after 1.5 h ==="
ssh -o BatchMode=yes qkd "tail -6 /tmp/ab_threads.log"
exit 1
