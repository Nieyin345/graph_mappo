#!/usr/bin/env bash
# Verify the no_local_python hook blocks execution and nothing else.
# Feeds synthetic PreToolUse payloads to the hook and checks the exit code.
set -u

cd "$(dirname "$0")/.." || exit 1
HOOK=".claude/hooks/no_local_python.sh"
pass=0
fail=0

check() {
    local expect="$1" label="$2" cmd="$3"
    local payload
    payload="$(printf '%s' "$cmd" | sed 's/\\/\\\\/g; s/"/\\"/g')"
    printf '{"tool_name":"Bash","tool_input":{"command":"%s","description":"x"}}' "$payload" \
        | bash "$HOOK" >/dev/null 2>&1
    local got=$?
    local verdict
    if [ "$expect" = block ]; then
        [ "$got" = 2 ] && verdict=OK || verdict="**MISMATCH**"
    else
        [ "$got" = 0 ] && verdict=OK || verdict="**MISMATCH**"
    fi
    [ "$verdict" = OK ] && pass=$((pass + 1)) || fail=$((fail + 1))
    printf '%-6s exit=%d  %-9s %s\n' "$verdict" "$got" "$expect" "$label"
}

echo "--- must BLOCK (local execution) ---"
check block "benchmark on the laptop"        'python .tmp/profile_clean.py --device cpu --steps 240'
check block "pytest locally"                 'cd repo && python -m pytest tests/ -q'
check block "bare interpreter"               'py -3 scripts/show_metrics.py out.jsonl'
check block "windows-style path"             '/c/Python313/python .tmp/time_update.py'
check block "after a pipe"                   'cat data.json | python -c "import json"'
check block "pytest binary"                  'pytest tests/'

echo
echo "--- must ALLOW (remote, or not execution) ---"
check allow "ssh to the node"                "ssh qkd '/opt/qkd/venv/bin/python .tmp/time_update.py --steps 240'"
check allow "full-path ssh.exe"              "/c/Windows/System32/OpenSSH/ssh.exe qkd 'cd /opt/qkd/graph_mappo && /opt/qkd/venv/bin/python -c 1'"
check allow "scp upload"                     'scp .tmp/time_update.py qkd:/opt/qkd/graph_mappo/.tmp/'
check allow "ssh via \$SSH variable"         'SSH=/c/Windows/System32/OpenSSH/ssh.exe; "$SSH" -o BatchMode=yes qkd "cd /opt/qkd/graph_mappo && /opt/qkd/venv/bin/python scripts/show_metrics.py outputs/x/metrics.jsonl"'check allow "sync script"                    'bash scripts/server_sync.sh --with-tmp'
check allow "sync script"                    'bash scripts/server_sync.sh --with-tmp'
check allow "inspecting processes"           'ps -W | grep python'
check allow "mentioning a .py in git"        'git add qkd_rl/rl/models/graph_mappo.py'
check allow "editing a .py via shell"        'wc -l qkd_rl/rl/algos/mappo_trainer.py && grep -n kl repo.py'

echo
echo "passed $pass, failed $fail"
[ "$fail" = 0 ]
