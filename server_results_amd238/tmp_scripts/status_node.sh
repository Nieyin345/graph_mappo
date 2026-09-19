#!/usr/bin/env bash
# 节点侧状态一览（避免 PowerShell 内联引号被吞）。
set -u
cd /opt/qkd/graph_mappo || exit 1

echo "-- 训练进度（已跑 update 数 / 最好验证）--"
for f in /tmp/r4_*.log /tmp/real_*.log; do
    [ -e "$f" ] || continue
    n=$(grep -c 'update=' "$f" 2>/dev/null || echo 0)
    best=$(grep 'new best validation' "$f" 2>/dev/null | tail -1)
    printf '%-26s updates=%-4s %s\n' "$(basename "$f")" "$n" "$best"
done

echo "-- 跑着的训练进程数 --"
pgrep -c -f train_graph_mappo

echo "-- 探针进度 --"
for f in /tmp/ja_deadline3.log /tmp/load_sens.log /tmp/joint_det.log; do
    [ -e "$f" ] || continue
    printf '%-22s %s\n' "$(basename "$f")" "$(tail -1 "$f")"
done