#!/usr/bin/env bash
# 状态速查：内存 / 在跑的 run / 已挂的 chain / 各 run 进度。
# 单独成文件是因为嵌套引号在 PowerShell→ssh→bash 三层转义下必崩。
cd /opt/qkd/graph_mappo || exit 1

echo "=== 内存 ==="
awk '/MemAvailable/{printf "MemAvailable  %.1f GB\n", $2/1048576}' /proc/meminfo

echo
echo "=== 在跑的 run（含轮次进度）==="
for pid in $(pgrep -f "train_graph_mappo.py"); do
  name=$(tr '\0' ' ' < "/proc/$pid/cmdline" | grep -o -- '--run-name [^ ]*' | awk '{print $2}')
  [ -z "$name" ] && continue
  n=$(wc -l < "outputs/$name/metrics.jsonl" 2>/dev/null || echo 0)
  rss=$(awk '/^VmRSS/{printf "%.1f", $2/1048576}' "/proc/$pid/status" 2>/dev/null)
  echo "  pid=$pid  $name  行数=$n  RSS=${rss}GB"
done

echo
echo "=== 已挂的 chain 脚本 ==="
pgrep -af "bash /tmp/chain_" | grep -v "pgrep" || echo "  （无）"

echo
echo "=== 最近 3 个 outputs 目录 ==="
ls -1dt outputs/*/ 2>/dev/null | head -3

echo
echo "=== 唤醒状态文件 ==="
cat /tmp/qkd_wake_state.json 2>/dev/null || echo "  （无）"
