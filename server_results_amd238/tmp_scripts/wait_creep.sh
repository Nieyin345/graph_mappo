#!/usr/bin/env bash
# 等 diag_creep 跑完，把频率日志和 RSS 日志拉回本地做关联分析。
set -uo pipefail
SSH="C:/Windows/System32/OpenSSH/ssh.exe"
SCP="C:/Windows/System32/OpenSSH/scp.exe"
OPTS="-o BatchMode=yes -o ConnectTimeout=15"

deadline=$(( $(date +%s) + 2700 ))
while true; do
    # 中括号技巧：远端 bash -c 自己的命令行也含这串字符，裸写会匹配到自己。
    state=$($SSH $OPTS qkd 'if pgrep -f "diag_cre[e]p.sh" >/dev/null 2>&1; then echo RUNNING; else echo GONE; fi' 2>/dev/null || true)
    [ "$state" = "GONE" ] && break
    if [ "$(date +%s)" -gt "$deadline" ]; then echo "!!! 等待超时"; break; fi
    sleep 30
done
echo "=== diag_creep 已结束 $(date +%H:%M:%S) ==="
echo

echo "=== 逐轮耗时 ==="
$SSH $OPTS qkd '/opt/qkd/venv/bin/python /opt/qkd/graph_mappo/scripts/show_metrics.py /opt/qkd/graph_mappo/outputs/diag_creep/metrics.jsonl 2>/dev/null | tail -n +2 | grep -E "^ *[0-9]+ " | awk "{printf \"u%-3s rollout=%-6s update=%-7s\n\", \$1,\$2,\$3}"'
echo

echo "=== 拉回频率与 RSS 日志 ==="
$SCP $OPTS qkd:/tmp/freq_probe.log .tmp/freq_probe.log 2>&1
$SCP $OPTS qkd:/tmp/diag_creep.rss .tmp/diag_creep.rss 2>&1
wc -l .tmp/freq_probe.log .tmp/diag_creep.rss
