#!/usr/bin/env bash
# 远端状态查询（在服务器上跑）。把嵌套引号全关在这一个文件里，
# 本地轮询器只做 `ssh qkd 'bash .../pm_status.sh'`，不再套引号。
#
# 输出（逐行，供本地 grep 解析）：
#   ARM=<名字> N=<完成的 update 数> LOG=<日志字节数>
#   PROC=<真 python 训练进程数>   ← 用 comm 锚定，不用 pgrep -f
#   MEM=<MemAvailable GiB>
set -u
OUT=/opt/qkd/graph_mappo/outputs
for a in pm_decode_s42 pm_decode_s43 pm_decode_s44 pm_decode_s45 pm_decode_s46; do
    f="$OUT/$a/metrics.jsonl"
    n=0
    [ -f "$f" ] && n=$(grep -c '"update"' "$f")
    log="/tmp/pmlogs/$a.log"
    lb=0
    [ -f "$log" ] && lb=$(wc -c < "$log")
    echo "ARM=$a N=$n LOG=$lb"
done
echo "PROC=$(ps -eo comm,args | awk '$1 ~ /^python/ && /train_graph_mappo/' | wc -l)"
awk '/MemAvailable/ {printf "MEM=%.1f\n", $2/1048576}' /proc/meminfo
# 失败要吵：日志里出现 Traceback 就把臂名报出来
for a in pm_decode_s42 pm_decode_s43 pm_decode_s44 pm_decode_s45 pm_decode_s46; do
    log="/tmp/pmlogs/$a.log"
    if [ -f "$log" ] && grep -q "Traceback" "$log"; then
        echo "TRACE=$a"
    fi
done
