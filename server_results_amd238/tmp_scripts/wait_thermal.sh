#!/usr/bin/env bash
# 等 thermal_test 跑完，打印冷/热对照结果 + 频率时间线。
set -uo pipefail
SSH="C:/Windows/System32/OpenSSH/ssh.exe"
SCP="C:/Windows/System32/OpenSSH/scp.exe"
OPTS="-o BatchMode=yes -o ConnectTimeout=15"

deadline=$(( $(date +%s) + 2400 ))
while true; do
    state=$($SSH $OPTS qkd 'if pgrep -f "thermal_tes[t].sh" >/dev/null 2>&1; then echo RUNNING; else echo GONE; fi' 2>/dev/null || true)
    [ "$state" = "GONE" ] && break
    if [ "$(date +%s)" -gt "$deadline" ]; then echo "!! 超时"; break; fi
    sleep 30
done

echo "=== thermal_test 结束 $(date +%H:%M:%S) ==="
echo
echo "=== 完整日志 ==="
$SSH $OPTS qkd 'cat /tmp/thermal_test.log'
echo
echo "=== 频率时间线（每 4 次取 1）==="
$SSH $OPTS qkd 'grep -v TEMP /tmp/thermal_freq.log | awk "NR%4==1"'
echo
echo "=== 温度（若有）==="
$SSH $OPTS qkd 'grep TEMP /tmp/thermal_freq.log | awk "NR%4==1" | head -20'
$SCP $OPTS qkd:/tmp/thermal_freq.log .tmp/thermal_freq.log 2>/dev/null
