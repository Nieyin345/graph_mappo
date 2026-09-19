#!/usr/bin/env bash
# 等 probe_restart 跑完，打印结果表 + 频率对照。
set -uo pipefail
SSH="C:/Windows/System32/OpenSSH/ssh.exe"
SCP="C:/Windows/System32/OpenSSH/scp.exe"
OPTS="-o BatchMode=yes -o ConnectTimeout=15"

deadline=$(( $(date +%s) + 2400 ))
while true; do
    state=$($SSH $OPTS qkd 'if pgrep -f "probe_restar[t].sh" >/dev/null 2>&1; then echo RUNNING; else echo GONE; fi' 2>/dev/null || true)
    [ "$state" = "GONE" ] && break
    if [ "$(date +%s)" -gt "$deadline" ]; then echo "!!! 超时"; break; fi
    sleep 30
done

echo "=== probe_restart 结果 $(date +%H:%M:%S) ==="
$SSH $OPTS qkd 'cat /tmp/probe_restart.log'
echo
echo "=== 全程频率（每 6 次采样取 1）==="
$SSH $OPTS qkd 'awk "NR==1 || NR%6==2" /tmp/freq_probe.log'
echo
$SCP $OPTS qkd:/tmp/freq_probe.log .tmp/freq_probe.log 2>/dev/null
echo "(频率原始日志已存 .tmp/freq_probe.log)"
