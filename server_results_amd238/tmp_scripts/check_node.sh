#!/usr/bin/env bash
# 核对节点上的 policy.py 是否含已落地的快路径；并看 run 现状。
cd /opt/qkd/graph_mappo || exit 1

echo "=== 节点 policy.py 是否含快路径 ==="
if grep -q "arc_pos_lookup" qkd_rl/rl/algos/policy.py; then
  echo "  含 arc_pos_lookup（快路径已在节点上）"
  grep -n "arc_pos_lookup" qkd_rl/rl/algos/policy.py | head -3
else
  echo "  !! 不含 arc_pos_lookup —— 节点代码是旧的"
fi
echo "  本地/节点 policy.py sha256："
sha256sum qkd_rl/rl/algos/policy.py | awk '{print "    node " $1}'

echo
echo "=== 内存 ==="
awk '/MemAvailable/{printf "  MemAvailable  %.1f GB\n", $2/1048576}' /proc/meminfo

echo
echo "=== 在跑的 run ==="
for pid in $(pgrep -f "train_graph_mappo.py"); do
  name=$(tr '\0' ' ' < "/proc/$pid/cmdline" | grep -o -- '--run-name [^ ]*' | awk '{print $2}')
  [ -z "$name" ] && continue
  n=$(wc -l < "outputs/$name/metrics.jsonl" 2>/dev/null || echo 0)
  echo "  $name  行数=$n"
done

echo
echo "=== chain 状态 ==="
pgrep -af "chain_extend_u50" | grep -v pgrep || echo "  （chain 已退出）"
echo "--- chain 输出 ---"
cat /tmp/chain_extend_u50.out 2>/dev/null

echo
echo "=== u30 检查点是否都在 ==="
for s in 42 43 44; do
  f="outputs/ent01_s${s}/checkpoint_update_000030.pt"
  [ -f "$f" ] && echo "  有 $f" || echo "  !! 缺 $f"
done
