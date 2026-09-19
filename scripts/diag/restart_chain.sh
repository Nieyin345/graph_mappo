#!/usr/bin/env bash
# 2026-09-18 19:40 重启两条等待链。
#
# 为什么：两个**在跑的**进程用的是 BUGGY 文本。把文件改名成 .BUGGY.bak
# **不改变已经在内存里的东西** —— bash 是按需增量读取脚本文件的，进程启动时
# 已经把文本读进去了（证据：chain_prereg.log 里 44 条 `line 53` 报错）。
#
#   chain_prereg.sh    buggy → 只是白等到 19:57 才启动（多花 19 分钟）
#   chain_knobs_ent01.sh buggy 是**潜伏形态**：prereg 活着时 `[ "$p" -eq 0 ]`
#     先短路，`r="0\n0"` 从不被比较，所以现在日志里看不到任何报错；
#     一旦 prereg 退出就报 integer expression expected 且**永远为假**
#     → 白等满 1440×30s = **12 小时**（到 07:02，几乎吃掉整个剩余机时）。
#
# 重启为什么安全：两者都**还没启动任何 run**（无 marker、0 个 train 进程、123G 空闲），
# 所以不存在双跑 → 不存在 4 并发 → 不存在 OOM。
set -u

echo "=== 1. 重启安全性前提：确认没有在跑的 train ==="
n=$(pgrep -cf "train_graph_mappo.py" 2>/dev/null); n=${n:-0}
echo "  train 进程数: $n"
if [ "$n" -ne 0 ]; then
  echo "  !! 有 train 在跑 —— **不重启**（可能出现双跑/OOM）"; exit 1
fi
for m in /tmp/wave_prereg.go /tmp/wave_ep2e1.go /tmp/wave_mini512e1.go; do
  [ -f "$m" ] && { echo "  !! marker $m 已存在 —— **不重启**"; exit 1; }
done
for d in ent01_s45 ent01_s46; do
  [ -f "/opt/qkd/graph_mappo/outputs/$d/metrics.jsonl" ] && \
    { echo "  !! outputs/$d 已存在 —— **不重启**（防覆盖）"; exit 1; }
done
echo "  ok：无 train、无 marker、s45/s46 未产出 —— 可以安全重启"

echo "=== 2. 杀掉旧实例（**显式 PID**，不用 pkill -f —— 在 ssh 里会杀到自己）==="
for p in 119760 119762 124760 124762; do
  if kill -0 "$p" 2>/dev/null; then kill "$p" 2>/dev/null && echo "  SIGTERM $p"; else echo "  $p 已不在"; fi
done
sleep 4
for p in 119760 119762 124760 124762; do
  kill -0 "$p" 2>/dev/null && { kill -9 "$p" 2>/dev/null; echo "  SIGKILL $p"; }
done
sleep 1
echo "  剩余相关进程: $(pgrep -cf 'bash /tmp/chain_' 2>/dev/null || echo 0)"

echo "=== 3. 用**修好的**文本重启（wrapper 里先 rm marker，与初版一致）==="
cd /opt/qkd/graph_mappo || exit 1
rm -f /tmp/wave_prereg.go
setsid nohup bash /tmp/chain_prereg.sh > /tmp/chain_prereg.log 2>&1 < /dev/null &
sleep 3
setsid nohup bash /tmp/chain_knobs_ent01.sh > /tmp/chain_knobs_ent01.log 2>&1 < /dev/null &
sleep 4

echo "=== 4. 自检 ==="
pgrep -af "bash /tmp/chain_(prereg|knobs_ent01)\.sh" | cut -c1-80
echo "--- 确认跑的是修好的文本（pgrep_n 应存在）---"
grep -c "pgrep_n()" /tmp/chain_prereg.sh /tmp/chain_knobs_ent01.sh
echo "--- 确认没有残留的 buggy 写法 ---"
grep -n "|| echo 0" /tmp/chain_prereg.sh /tmp/chain_knobs_ent01.sh || echo "  无（正确）"
sleep 8
echo "=== 5. prereg 是否已启动 s45/s46 ==="
tail -6 /tmp/chain_prereg.log
pgrep -af "run-name ent01_s4[56]" | cut -c1-100 || echo "  （还没启动）"
