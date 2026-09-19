#!/usr/bin/env bash
# 等 corr_rss 跑完，把 RSS 与 metrics 拉回本地，跑 analyze_corr.py 出对齐表。
set -uo pipefail
SSH="C:/Windows/System32/OpenSSH/ssh.exe"
SCP="C:/Windows/System32/OpenSSH/scp.exe"
OPTS="-o BatchMode=yes -o ConnectTimeout=15"

deadline=$(( $(date +%s) + 3000 ))
while true; do
    state=$($SSH $OPTS qkd 'if pgrep -f "corr_rs[s].sh" >/dev/null 2>&1; then echo RUNNING; else echo GONE; fi' 2>/dev/null || true)
    [ "$state" = "GONE" ] && break
    if [ "$(date +%s)" -gt "$deadline" ]; then echo "!!! 超时"; break; fi
    sleep 30
done

echo "=== corr_rss 结束 $(date +%H:%M:%S) ==="
$SCP $OPTS qkd:/tmp/corr_rss.rss .tmp/corr_rss.rss 2>&1
$SCP $OPTS qkd:/opt/qkd/graph_mappo/outputs/corr_rss/metrics.jsonl .tmp/corr_rss_metrics.jsonl 2>&1
echo

# 用项目 venv 跑分析（本地 venv 在磁盘根，见 CLAUDE.md 的环境说明）
PY="D:/destop/work_space/learning_space/.venv/Scripts/python.exe"
[ -x "$PY" ] || PY=python
"$PY" .tmp/analyze_corr.py
