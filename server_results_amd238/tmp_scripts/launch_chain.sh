#!/usr/bin/env bash
# 在服务器上启动 r8→r9ext 接力守护（detached）。
# 用法：ssh qkd 'bash /opt/qkd/graph_mappo/.tmp/launch_chain.sh'
set -u
cd /opt/qkd/graph_mappo
if pgrep -f "chain_r9ext.sh" >/dev/null 2>&1; then
  echo "接力守护已在运行"
  exit 0
fi
setsid nohup bash .tmp/chain_r9ext.sh > /tmp/chain_r9ext.out 2>&1 < /dev/null &
echo "接力已挂 pid=$!"
