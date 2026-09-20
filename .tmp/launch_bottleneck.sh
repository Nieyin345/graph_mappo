#!/usr/bin/env bash
# 把逐请求瓶颈探针送上去后台跑。
set -u
HOST=qkd
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT=probe_regime_sweep.py
RDIR=/tmp/rlprobe
LOG=/tmp/regime.log

echo "=== 1. 建目录 + 传脚本（★ 节点会间歇性 reset ⟹ 必须带重试）==="
ok=0
for i in 1 2 3 4 5; do
  if timeout 25 ssh -o ConnectTimeout=10 "$HOST" "mkdir -p $RDIR" 2>/dev/null \
     && timeout 60 scp -o ConnectTimeout=10 "$LOCAL_DIR/$SCRIPT" "$HOST:$RDIR/$SCRIPT" 2>/dev/null; then
    ok=1; echo "  第 $i 次成功，已传 $SCRIPT"; break
  fi
  echo "  第 $i 次失败（节点 reset？），8 秒后重试"
  sleep 8
done
[ "$ok" -eq 1 ] || { echo "✗ 5 次都没传上去 ⟹ 节点可能真的掉了"; exit 1; }

echo "=== 2. 后台启动（-u 必须：重定向到文件时 Python 块缓冲，空日志看起来跟没跑一样）==="
timeout 25 ssh -o ConnectTimeout=10 "$HOST" \
  "cd /opt/qkd/graph_mappo && rm -f $LOG && setsid nohup /opt/qkd/venv/bin/python -u $RDIR/$SCRIPT \
     --seeds 100-114 --steps 240 > $LOG 2>&1 < /dev/null & echo started"
echo "  （ssh 被 timeout 掐掉是预期的）"

echo "=== 3. 启动后验证（确认真的在跑，不是静默空转）==="
sleep 25
timeout 25 ssh -o ConnectTimeout=10 "$HOST" \
  "echo '--- 进程 ---'; ps -eo pid,etime,args | grep '^ *[0-9]* .*rlprobe/$SCRIPT' | grep -v grep; \
   echo '--- 日志行数 ---'; wc -l $LOG; \
   echo '--- 日志 ---'; cat $LOG"
