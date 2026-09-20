#!/usr/bin/env bash
# 停掉 `hist32_v3_s43`（已确认目录被两次启动写坏：update 号 1..30 后又 1..22，
# 且 checkpoint_update_000025.pt 的 mtime 早于 000005.pt）。
#
# ★ 为什么不 `kill <父pid>`：父进程被 SIGKILL 后，multiprocessing.spawn 的 8 个
#   rollout worker 会被 reparent 到 init **继续空转、永不退出**，每个约 1.1 GB
#   ⟹ 每杀一次多 9 GB 孤儿。（记忆 [[oom-orphan-workers]]）
# ★ 为什么不 `pkill -f train_graph`：在 `ssh host '...'` 里远程 bash 命令行自身
#   含该模式 ⟹ 自匹配，连后面的命令一起杀。
# ★ 正解：取**会话/进程组**，对整个组发 TERM，让子进程一起收到。
set -u
HOST=qkd
RUN=hist32_v3_s43

echo "=== 1. 定位进程（只匹配 python 解释器，不匹配 bash -c 包装器）==="
timeout 25 ssh -o ConnectTimeout=10 "$HOST" "
  ps -eo pid,ppid,sid,pgid,etime,args | awk '\$6 ~ /^python|^\\/opt/ && /train_graph_mappo/ && /$RUN/'
" || { echo "✗ ssh 失败"; exit 1; }

echo
echo "=== 2. 取 PID / PGID ==="
read -r PID PGID < <(timeout 25 ssh -o ConnectTimeout=10 "$HOST" "
  ps -eo pid,pgid,args | awk '\$3 ~ /^python|^\\/opt/ && /train_graph_mappo/ && /$RUN/ {print \$1, \$2; exit}'
")
if [ -z "${PID:-}" ]; then echo "  未找到在跑的 $RUN（可能已退出）"; exit 0; fi
echo "  PID=$PID  PGID=$PGID"
if [ "$PGID" = "1" ] || [ "$PGID" = "$PID" ] && [ "$(timeout 25 ssh -o ConnectTimeout=10 "$HOST" "ps -o pgid= -p $PID | tr -d ' '")" = "1" ]; then
  echo "  ⚠ PGID 为 1 ⟹ 不能按组杀，退回按 PID 杀会留孤儿，改用 pkill --parent 链式处理"
fi

echo
echo "=== 3. 对整个进程组发 TERM（负号 = 进程组）==="
timeout 25 ssh -o ConnectTimeout=10 "$HOST" "kill -TERM -$PGID 2>/dev/null; echo  TERM 已发往组 $PGID"

echo "=== 4. 等 15 秒后复查 ==="
sleep 15
LEFT=$(timeout 25 ssh -o ConnectTimeout=10 "$HOST" "ps -eo pid,args | awk '\$2 ~ /^python|^\\/opt/ && /train_graph_mappo/ && /$RUN/' | wc -l")
echo "  仍存活的 $RUN 相关进程: $LEFT"

if [ "$LEFT" -gt 0 ]; then
  echo "=== 5. 仍有残留，对组发 KILL ==="
  timeout 25 ssh -o ConnectTimeout=10 "$HOST" "kill -KILL -$PGID 2>/dev/null; echo  KILL 已发往组 $PGID"
  sleep 8
  LEFT=$(timeout 25 ssh -o ConnectTimeout=10 "$HOST" "ps -eo pid,args | awk '\$2 ~ /^python|^\\/opt/ && /train_graph_mappo/ && /$RUN/' | wc -l")
  echo "  KILL 后仍存活: $LEFT"
fi

echo
echo "=== 6. 内存复查（两把尺子都打）==="
timeout 25 ssh -o ConnectTimeout=10 "$HOST" "grep -E 'MemTotal|MemAvailable' /proc/meminfo; echo '--- 全部在跑的 run ---'; ps -eo pid,args | awk '\$2 ~ /^python|^\\/opt/ && /train_graph_mappo/' | wc -l"
