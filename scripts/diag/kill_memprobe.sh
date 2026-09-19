#!/usr/bin/env bash
# 撤掉 memprobe_t16：**杀整个进程组**，不是只杀父进程。
#
# 为什么按进程组杀：trainer 用 `setsid` 起，自己是组长；`multiprocessing.spawn`
# 的 8 个 worker 在父进程被 SIGKILL 后会 reparent 到 init 并**继续空转不退出**
# （记忆 oom-orphan-workers：每发生一次就新增约 9 GB 孤儿）。杀进程组一次带走全部。
#
# ⚠ 不动 outputs/memprobe_t16 目录 —— outputs 只增不删是红线。
#   它是 3 轮的点火探针，不是臂；抓结果时不镜像它即可。
set -u

PAT='memprobe_t1[6]'          # 字符类，避免 pgrep 匹配到自己这条命令行
PIDS=$(pgrep -f "$PAT" 2>/dev/null || true)
if [ -z "$PIDS" ]; then
  echo "没有 memprobe 进程（已退出）"
  exit 0
fi
echo "命中进程：$PIDS"

# 找组长（会话首进程）的 PGID：/proc/<pid>/stat 的第 5 个字段
PGIDS=""
for p in $PIDS; do
  if [ -r "/proc/$p/stat" ]; then
    g=$(awk '{print $5}' "/proc/$p/stat" 2>/dev/null)
    [ -n "$g" ] && PGIDS="$PGIDS $g"
  fi
done
PGIDS=$(echo "$PGIDS" | tr ' ' '\n' | sort -u | grep -v '^$' | tr '\n' ' ')
echo "进程组：$PGIDS"

for g in $PGIDS; do
  if [ "$g" -gt 1 ] 2>/dev/null; then
    kill -TERM "-$g" 2>/dev/null && echo "  TERM 进程组 $g"
  fi
done
sleep 5
for g in $PGIDS; do
  if [ "$g" -gt 1 ] 2>/dev/null; then
    kill -KILL "-$g" 2>/dev/null && echo "  KILL 进程组 $g（TERM 后仍在）"
  fi
done
sleep 2

LEFT=$(pgrep -cf "$PAT" 2>/dev/null || true)
echo "剩余 memprobe 进程数：${LEFT:-0}"
echo
echo "--- 现在在跑什么 ---"
/opt/qkd/venv/bin/python /tmp/mem_pss.py --to-update 30 2>&1 | sed -n '1,14p'
