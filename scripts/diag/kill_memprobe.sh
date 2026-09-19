#!/usr/bin/env bash
# 撤掉 memprobe_t16：**杀整个进程组**，不是只杀父进程。
#
# 为什么按进程组杀：trainer 用 `setsid` 起，自己是组长；`multiprocessing.spawn`
# 的 8 个 worker 在父进程被 SIGKILL 后会 reparent 到 init 并**继续空转不退出**
# （记忆 oom-orphan-workers：每发生一次就新增约 9 GB 孤儿）。杀进程组一次带走全部。
#
# ⚠ 不动 outputs/memprobe_t16 目录 —— outputs 只增不删是红线。
#   它是 3 轮的点火探针，不是臂；抓结果时不镜像它即可。
#
# ============================================================================
# ★★ B-12（2026-09-21 修）：旧版的**验证**与**杀**覆盖的是**不同人口**
#
#   旧版两处都用 `pgrep -f 'memprobe_t1[6]'`，而这个模式只匹配 cmdline 里
#   带 run-name 的进程 —— **spawn worker 的 cmdline 里没有 run-name**
#   （它长这样：`python -c "from multiprocessing.spawn import spawn_main; ..."`）。
#
#   ⟹ 两种失败：
#     1. **trainer 已被 OOM 杀掉、worker 还在空转**时，`pgrep` 返回空
#        ⟹ 脚本打印「没有 memprobe 进程（已退出）」并 **`exit 0`**，
#        **一个孤儿都不清** —— 而这恰是它被写出来要处理的那个场景。
#     2. 结尾 `LEFT=0` 报「已清理干净」，而 8 个 worker 可能还在。
#
#   ★ 这是 `cross-check-must-compare-same-population` 的又一例：
#     **验证的人口必须与被处置的人口相同**。杀的是进程组（含 worker），
#     验的却只有带 run-name 的那一个 ⟹ 验证在结构上看不见大多数人口。
#
#   修法：孤儿**单独检测、单独计数**，且在结尾把两个人口**都**印出来；
#   只有当**两者都为 0** 才说"已清理"。
#   ⚠ 归属：worker 自身 cmdline 里没有 run-name，**不试图从它自证归属**
#     （CLAUDE.md 明写）。所以孤儿走**显式开关** `--reap-orphans`，
#     默认只报告不杀 —— 宁可让操作者决定，也不要猜错杀别人的臂。
# ============================================================================
set -u

PAT='memprobe_t1[6]'          # 字符类，避免 pgrep 匹配到自己这条命令行
REAP_ORPHANS=0
[ "${1:-}" = "--reap-orphans" ] && REAP_ORPHANS=1

# PPid==1 且 cmdline 含 multiprocess 的 python 进程 = OOM 遗留的孤儿 worker。
# ★ 命令行里没有 run-name，所以**无法归属**；只报个数与总量。
orphan_pids() {
  local p ppid cl
  for p in /proc/[0-9]*; do
    p=${p#/proc/}
    [ -r "/proc/$p/cmdline" ] || continue
    cl=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    case "$cl" in *python*) ;; *) continue ;; esac
    case "$cl" in *multiprocess*) ;; *) continue ;; esac
    ppid=$(awk '/^PPid:/{print $2}' "/proc/$p/status" 2>/dev/null)
    [ "${ppid:-0}" = "1" ] && echo "$p"
  done
}

echo "=== 1) 找带 run-name 的 trainer ==="
PIDS=$(pgrep -f "$PAT" 2>/dev/null || true)
ORPH_BEFORE=$(orphan_pids | tr '\n' ' ')
echo "命中 trainer：${PIDS:-（无）}"
echo "启动前已存在的孤儿 worker：${ORPH_BEFORE:-（无）}"
echo

if [ -z "$PIDS" ]; then
  echo "⚠ **没有带 run-name 的 trainer 在跑。**"
  if [ -n "${ORPH_BEFORE// /}" ]; then
    # ★ 旧版在这里 `exit 0` 并说"已退出"—— 把 8 个空转的孤儿留在了原地。
    echo "   但**存在孤儿 worker**（$(echo $ORPH_BEFORE | wc -w) 个）——"
    echo "   这正是 OOM 遗留的那一类：父进程被杀，worker reparent 到 init 后继续空转。"
    echo "   ⚠ 它们 cmdline 里**没有 run-name**，**无法归属**某个 run。"
    if [ "$REAP_ORPHANS" = "1" ]; then
      echo "   --reap-orphans 已给 ⟹ 逐个杀（先 TERM 再 KILL）。"
      echo "$ORPH_BEFORE" | tr ' ' '\n' | grep -v '^$' | while read -r p; do
        kill -TERM "$p" 2>/dev/null && echo "     TERM $p"
      done
      sleep 5
      echo "$ORPH_BEFORE" | tr ' ' '\n' | grep -v '^$' | while read -r p; do
        kill -0 "$p" 2>/dev/null && { kill -KILL "$p" 2>/dev/null && echo "     KILL $p"; }
      done
      sleep 2
    else
      echo "   **默认不杀**（可能是别的臂的孤儿）。要清：\$0 --reap-orphans"
      echo "   或先看清单：.tmp/reap.py（默认只报告）。"
    fi
    echo
    echo "--- 现在在跑什么 ---"
    /opt/qkd/venv/bin/python /tmp/mem_pss.py --to-update 30 2>&1 | sed -n '1,14p'
    exit 3                      # ★ 第三态：既非"已退出"也非"已清理"
  fi
  echo "   也没有孤儿 worker。"
  exit 0
fi

# 找组长（会话首进程）的 PGID：/proc/<pid>/stat 的第 5 个字段
PGIDS=""
for p in $PIDS; do
  if [ -r "/proc/$p/stat" ]; then
    g=$(awk '{print $5}' "/proc/$p/stat" 2>/dev/null)
    [ -n "$g" ] && PGIDS="$PGIDS $g"
  fi
done
PGIDS=$(echo "$PGIDS" | tr ' ' '\n' | sort -u | grep -v '^$' | tr '\n' ' ')
echo "=== 2) 杀进程组（一次带走 worker）==="
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

echo
echo "=== 3) 验证（★ 两个人口都数）==="
LEFT=$(pgrep -cf "$PAT" 2>/dev/null || true)
LEFT=${LEFT:-0}
ORPH_AFTER=$(orphan_pids | tr '\n' ' ')
N_ORPH_AFTER=$(echo $ORPH_AFTER | wc -w)
echo "剩余带 run-name 的 trainer：$LEFT"
echo "剩余孤儿 worker：$N_ORPH_AFTER"
if [ "$LEFT" = "0" ] && [ "$N_ORPH_AFTER" = "0" ]; then
  echo "⟹ ✓ **两个人口都是 0**，确认清理干净。"
  RC=0
else
  echo "⟹ ✗ **还有残留**：trainer=$LEFT，孤儿=$N_ORPH_AFTER"
  [ "$N_ORPH_AFTER" != "0" ] && echo "   孤儿 PIDs：$ORPH_AFTER"
  echo "   （旧版只数 trainer，此时会报『已清理干净』）"
  RC=1
fi
echo
echo "--- 现在在跑什么 ---"
/opt/qkd/venv/bin/python /tmp/mem_pss.py --to-update 30 2>&1 | sed -n '1,14p'
exit "$RC"
