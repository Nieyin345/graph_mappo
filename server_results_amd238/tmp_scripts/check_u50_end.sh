#!/usr/bin/env bash
# s42/s43 是**跑完 u50 正常退出**还是**被杀**？两者在 ALERT 里长得一样。
cd /opt/qkd/graph_mappo || exit 1

for s in 42 43 44; do
  d=outputs/ent01_s${s}_u30to50
  echo "########## ent01_s${s}_u30to50"
  echo "  末轮: $(grep -o '"update": [0-9]*' $d/metrics.jsonl 2>/dev/null | tail -1)"
  echo "  验证点: $(grep -o 'u[0-9]*=[0-9.]*' $d/metrics.jsonl 2>/dev/null | tail -4 | tr '\n' ' ')"
  echo "  检查点:"
  ls -1 $d/checkpoint_*.pt 2>/dev/null | sed 's|.*/|    |' | tail -4
  echo "  最终检查点存在? $([ -f $d/checkpoint_final.pt ] && echo 是 || echo 否)"
  echo "  日志尾部:"
  tail -3 /tmp/ent01_s${s}_u30to50.log 2>/dev/null | sed 's/^/    /' | cut -c1-160
  echo
done

echo "=== 内核 OOM 记录 ==="
dmesg -T 2>/dev/null | grep -i "killed process\|out of memory" | tail -5 || echo "  （dmesg 无权限或空）"

echo
echo "=== 孤儿 worker ==="
c=0; gb=0
for pid in $(pgrep -f multiprocess); do
  ppid=$(awk '/^PPid:/{print $2}' /proc/$pid/status 2>/dev/null)
  [ "$ppid" = "1" ] || continue
  pss=$(awk '/^Pss:/{print $2}' /proc/$pid/smaps_rollup 2>/dev/null)
  [ -z "$pss" ] && continue
  c=$((c+1)); gb=$((gb+pss))
done
echo "  孤儿 worker 数=$c  合计=$((gb/1048576)) GB"

echo
echo "=== 内存 ==="
awk '/MemAvailable/{printf "  MemAvailable  %.1f GB\n", $2/1048576}' /proc/meminfo
date -Is
