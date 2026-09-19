#!/usr/bin/env bash
# g999_s42_r2 是正常跑完还是被杀？并查 OOM 痕迹与孤儿。
cd /opt/qkd/graph_mappo || exit 1

echo "=== g999_s42_r2 末尾几轮 ==="
tail -3 outputs/ent01_g999_s42_r2/metrics.jsonl 2>/dev/null | \
  /opt/qkd/venv/bin/python -c '
import sys, json
for l in sys.stdin:
    l=l.strip()
    if not l: continue
    try: r=json.loads(l)
    except Exception: continue
    if "update" in r:
        print(f"  update={r[\"update\"]} rollout_s={r.get(\"rollout_s\",0):.1f} update_s={r.get(\"update_s\",0):.1f}")
    elif "eval_validation" in r:
        print(f"  VALIDATION u=? mean={r[\"eval_validation\"].get(\"mean_success_rate\"):.4f}")
'
echo
echo "  该目录里的检查点（若最后一个是 u50 说明跑完）："
ls -1 outputs/ent01_g999_s42_r2/checkpoint_update_*.pt 2>/dev/null | tail -3
echo
echo "  日志尾部（正常结束会有收尾字样）："
tail -5 /tmp/ent01_g999_s42_r2.log 2>/dev/null

echo
echo "=== 内核是否发生过 OOM kill ==="
dmesg -T 2>/dev/null | grep -i "killed process\|out of memory" | tail -5 || echo "  （dmesg 无权限或无记录）"
journalctl -k --since "2 hours ago" 2>/dev/null | grep -i "out of memory\|killed process" | tail -5 || true
grep -i "oom" /var/log/syslog 2>/dev/null | tail -3 || echo "  （syslog 无记录/不可读）"

echo
echo "=== 孤儿 worker（PPid==1 且 cmdline 含 multiprocess）==="
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
echo "=== 三个延长臂的进度 ==="
for s in 42 43 44; do
  n=$(wc -l < "outputs/ent01_s${s}_u30to50/metrics.jsonl" 2>/dev/null || echo 0)
  echo "  ent01_s${s}_u30to50  行数=$n"
done
