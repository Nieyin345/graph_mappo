#!/usr/bin/env bash
# 等待一组训练臂跑完，然后自动汇总（sum_group.py），全程写日志。
#
# 为什么做成脚本：远程 ssh 命令行不支持 for 循环/内层引号（PowerShell 会剥），
# 写成文件传上去就随便用 bash 语法。跑在服务器上，本地只读结果文件。
#
# 用法（节点上）：
#   bash .tmp/wait_and_harvest.sh r4 90 /tmp/r4_ base,ent,ent_ep2,ent_c128,ent_lr2e4,c128
#   bash .tmp/wait_and_harvest.sh real 30 /tmp/real_ base,mini512,vcoef1
#
# 参数：组名  目标update  日志前缀  臂列表(逗号分隔)
# 输出：/tmp/harvest_<组名>.out，末尾打 HARVEST_<组名>_DONE

set -u

GROUP=$1
TARGET=$2
PREFIX=$3
ARMS=$(echo "$4" | tr ',' ' ')

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python

cd "$MAIN" || exit 1

echo "[$(date '+%F %T')] 等待 $GROUP：目标 update=$TARGET，臂：$ARMS"

while true; do
  n=0
  total=0
  missing=""
  for a in $ARMS; do
    total=$((total + 1))
    f="${PREFIX}${a}.log"
    if [ -f "$f" ] && grep -qE "^update= *${TARGET} " "$f"; then
      n=$((n + 1))
    else
      missing="$missing $a"
    fi
  done
  echo "[$(date '+%F %T')] $n/$total 已到 update=$TARGET；未完成：$missing"
  if [ "$n" -eq "$total" ]; then
    # 训练脚本是"先打印 update 行、后写该轮 eval"，所以日志出现 update=N 时
    # metrics.jsonl 的最后一条 eval 可能还没落盘 —— 直接汇总会少读一条
    # （实测 ent_ep2 少读 0.8336/真实 0.8359、mini512 少读 0.6241/真实 0.6392，
    #   都是该组最后收工的臂）。等一会再汇总。
    echo "[$(date '+%F %T')] 全部到齐，等 90s 让最后的 eval 写盘 ..."
    sleep 90
    break
  fi
  sleep 120
done

echo ""
echo "[$(date '+%F %T')] $GROUP 全部完成，开始汇总"
echo "===== sum_group.py $GROUP ====="
$PY -u .tmp/sum_group.py "$GROUP" 2>&1
echo ""
echo "[$(date '+%F %T')] 汇总结束"
echo "HARVEST_${GROUP}_DONE"