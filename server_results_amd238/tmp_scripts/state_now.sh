#!/usr/bin/env bash
# 现状快照（在服务器上跑）：确认三件事再启动新实验。
#   1. 快路径（查表替 np.flatnonzero）**真的在节点代码里**——本地提交了不等于服务器有了；
#   2. 有没有残留 run / 孤儿 worker 占内存；
#   3. 机器时间与余量。
#
# 为什么必须查第 1 条：sync.sh 是 git 工作区快照方式，若某次忘记同步，本地绿了
# 服务器还是旧的，而"新旧混跑"会让 A/B 变成伪 A/B。
cd /opt/qkd/graph_mappo || exit 1

echo "=== 1. 快路径是否在节点代码里 ==="
if grep -q "build_lookup\|_lookup" qkd_rl/rl/policy.py 2>/dev/null; then
  echo "  找到查表相关标识："
  grep -n "lookup" qkd_rl/rl/policy.py | head -8 | sed 's/^/    /'
else
  echo "  ✗ 没找到 lookup —— 快路径不在节点上"
fi
echo "  函数内是否残留 np.flatnonzero："
grep -n "flatnonzero" qkd_rl/rl/policy.py | sed 's/^/    /' || echo "    （无 flatnonzero）"
echo "  policy.py sha256: $(sha256sum qkd_rl/rl/policy.py | cut -c1-16)"

echo
echo "=== 2. 在跑的 run ==="
ps -eo pid,etime,args | grep train_graph_mappo.py | grep -v grep | sed 's/\(.\{140\}\).*/\1/' | sed 's/^/  /' || echo "  （无）"

echo
echo "=== 3. 孤儿 worker（PPid==1 且含 multiprocess）==="
c=0; kb=0
for pid in $(pgrep -f multiprocess 2>/dev/null); do
  ppid=$(awk '/^PPid:/{print $2}' /proc/$pid/status 2>/dev/null)
  [ "$ppid" = "1" ] || continue
  pss=$(awk '/^Pss:/{print $2}' /proc/$pid/smaps_rollup 2>/dev/null)
  [ -z "$pss" ] && continue
  c=$((c+1)); kb=$((kb+pss))
done
echo "  个数=$c  合计=$((kb/1048576)) GB"

echo
echo "=== 4. 内存与时间 ==="
awk '/MemAvailable/{printf "  MemAvailable %.1f GB\n", $2/1048576}
     /SwapFree/{printf "  SwapFree     %.1f GB\n", $2/1048576}' /proc/meminfo
echo "  现在: $(date -Is)"
echo "  开机: $(uptime -s)"
echo "  已运行: $(uptime -p)"

echo
echo "=== 5. outputs 里 ent01 家族的验证点（已有多少个种子）==="
for d in outputs/ent01_s4*; do
  [ -d "$d" ] || continue
  n=$(basename "$d")
  last=$(grep -o '"update": [0-9]*' "$d/metrics.jsonl" 2>/dev/null | tail -1)
  pts=$(grep -o 'u[0-9]*=[0-9.]*' "$d/metrics.jsonl" 2>/dev/null | tr '\n' ' ')
  printf "  %-28s %s | %s\n" "$n" "$last" "$pts"
done
