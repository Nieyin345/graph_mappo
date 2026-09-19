#!/usr/bin/env bash
# 布置唤醒链：上新的 wake_parse.py，设置 hold 原因，重启服务器端探针。
cd /opt/qkd/graph_mappo || exit 1

# hold 的原因要写**当前真正占用内存名额的那个实验**。
# 之前 hold 是给 ent01_s42_u25_base（起点配对对照），它已经跑完并出结论
# （0816f0e）。现在名额给了 ent01_s{42,43,44}_u30to50 这个延长臂。
cat > /tmp/qkd_g999_respawn.json <<'JSON'
{"hold": "内存名额让给 ent01_s4x_u30to50 延长臂（定位平台）；g999 系列已由 ent01_s42/43/44 覆盖"}
JSON

echo "=== hold 状态 ==="
cat /tmp/qkd_g999_respawn.json
echo
echo "=== 确认延长臂 chain 还挂着 ==="
pgrep -af "chain_extend_u50" | grep -v pgrep || echo "  !! chain 不在了"
echo
echo "=== chain 输出 ==="
cat /tmp/chain_extend_u50.out 2>/dev/null
