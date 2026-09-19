#!/usr/bin/env bash
# 回收遗留内存：孤儿 worker + 卡住的探针父进程。
# 自动保护**当前**在跑的 pss_trend / probe_loss_ab。
set -u
cd /opt/qkd/graph_mappo

PROT=""
for pat in pss_trend.py probe_loss_ab.py; do
    for p in $(pgrep -f "$pat" 2>/dev/null); do
        # 只保护存活 <20 分钟的（即当前这次），更老的按遗留处理
        et=$(ps -o etime= -p "$p" 2>/dev/null | tr -d ' ')
        mins=$(echo "$et" | awk -F'[-:]' '{
            if (NF==4) print $1*1440+$2*60+$3+$4/60;
            else if (NF==3) print $1*60+$2+$3/60;
            else print $1+$2/60 }')
        if [ -n "$mins" ] && [ "$(echo "$mins < 20" | bc 2>/dev/null || echo 1)" = "1" ]; then
            PROT="$PROT $p"
        fi
    done
done
echo "保护 PID:$PROT"

/opt/qkd/venv/bin/python .tmp/reap.py --apply --protect $PROT 2>&1 | tail -8
echo "=== 回收后 ==="
grep -E "^(MemAvailable|Committed_AS|SwapFree)" /proc/meminfo
