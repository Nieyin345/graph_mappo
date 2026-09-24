#!/usr/bin/env bash
# 送 probe_deadline_headroom.py 上服务器后台跑。
# ★ 硬门在主脚本里：dl=30 必须复现 0.6979，否则它自己 return 2 不给决策。
set -u
HOST=qkd
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT=probe_deadline_headroom.py
RDIR=/tmp/rlprobe
LOG=/tmp/dlhead.log

echo "=== 1. 传脚本（带重试）==="
ok=0
for i in 1 2 3 4 5; do
  if timeout 25 ssh -o ConnectTimeout=10 "$HOST" "mkdir -p $RDIR" 2>/dev/null \
     && timeout 60 scp -o ConnectTimeout=10 "$LOCAL_DIR/$SCRIPT" "$HOST:$RDIR/$SCRIPT" 2>/dev/null; then
    ok=1; echo "  第 $i 次成功"; break
  fi
  echo "  第 $i 次失败，8 秒后重试"; sleep 8
done
[ "$ok" -eq 1 ] || { echo "✗ 5 次都没传上去 ⟹ 节点可能真的掉了"; exit 1; }

echo "=== 2. 语法预检（失败必须吵）==="
timeout 30 ssh -o ConnectTimeout=10 "$HOST" \
  "/opt/qkd/venv/bin/python -m py_compile $RDIR/$SCRIPT && echo '  语法 OK'" || { echo "✗ 语法不过"; exit 2; }

echo "=== 3. 后台启动（-u 必须）==="
timeout 25 ssh -o ConnectTimeout=10 "$HOST" \
  "cd /opt/qkd/graph_mappo && rm -f $LOG && setsid nohup /opt/qkd/venv/bin/python -u $RDIR/$SCRIPT \
     --seeds 100-114 --dls 30,60,120,240 --steps 240 > $LOG 2>&1 < /dev/null & echo started"
echo "  （ssh 被 timeout 掐掉是预期的）"

echo "=== 4. 启动后验证（失败必须吵）==="
sleep 40
timeout 25 ssh -o ConnectTimeout=10 "$HOST" \
  "ps -eo pid,etime,args | grep 'rlprobe/$SCRIPT' | grep -v grep || echo '  ⚠ 进程不在 ⟹ 启动失败'; \
   echo '--- 日志 ---'; cat $LOG"
