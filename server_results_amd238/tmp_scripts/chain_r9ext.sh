#!/usr/bin/env bash
# r8 全部结束后接力启动 r9ext（40 轮延长实验）。
# 用法：setsid nohup bash .tmp/chain_r9ext.sh > /tmp/chain_r9ext.out 2>&1 < /dev/null &
set -u
cd /opt/qkd/graph_mappo
echo "等待 r8 结束... $(date -Is)"
while pgrep -f "screen_r8_multiseed" >/dev/null 2>&1; do sleep 60; done
echo "r8 已结束，启动 r9ext $(date -Is)"
bash .tmp/screen_r9ext.sh
