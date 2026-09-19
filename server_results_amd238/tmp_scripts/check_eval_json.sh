#!/usr/bin/env bash
# 查看节点上 outputs/eval 下的产物头部（避免 PowerShell 内联引号被吞）。
set -u
cd /opt/qkd/graph_mappo || exit 1
ls -la outputs/eval/
for f in outputs/eval/joint_avail_*.json; do
    echo "== $f"
    head -c 400 "$f"
    echo
done