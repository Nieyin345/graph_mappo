#!/usr/bin/env bash
# 把服务器上的实验结果抓到本地（不要权重）。
#
# 为什么现在就要抓：CloudLab 实验期按节点启动时间计（本节点 2026-09-17 08:54），
# 到期自动清理，outputs/ 不可再生。metrics.jsonl 每个 run 只有几十 KB，
# checkpoint .pt 每个几十 MB —— 排除权重，只留结果与方法。
#
# 本机（Windows git bash）没有 rsync，用 tar over ssh。
#
# ⚠️ 本地路径按本机 bash 的挂载点写。若时报 "找不到路径"，先确认本机盘符
#    挂在哪（`ls /d` 或 `cygpath -u 'D:\'`），再改下面两行。
#
# 已在跑的 run 会被覆盖为最新版本；已完成的 run 内容不变，重复抓无副作用。
#
# 用法：bash .tmp/fetch_all.sh
set -u

HOST=qkd
LOCAL_ROOT="${LOCAL_ROOT:-/d/destop/work_space/learning_space/论文/QKD-SAGIN生产端调度/qkd_rl}"
DEST="$LOCAL_ROOT/server_results"

mkdir -p "$DEST"

echo "=== 1. outputs/（排除 .pt 权重与 figures 图片）==="
ssh "$HOST" 'cd /opt/qkd/graph_mappo && tar czf - --exclude="*.pt" --exclude="*/figures/*" outputs' \
    | tar xzf - -C "$DEST" && echo "  OK"

echo
echo "=== 2. .tmp 脚本与日志（复现方法本身也是结果）==="
ssh "$HOST" 'cd /opt/qkd/graph_mappo && tar czf - .tmp' \
    | tar xzf - -C "$DEST" 2>/dev/null && mv "$DEST/.tmp" "$DEST/tmp_scripts" 2>/dev/null; echo "  OK"

echo
echo "=== 3. 结果清单 ==="
n=$(find "$DEST" -name metrics.jsonl 2>/dev/null | wc -l)
echo "  run 数（有 metrics.jsonl）：$n"
du -sh "$DEST" 2>/dev/null | awk '{print "  体积: "$1}'
echo
echo "--- 最大的 10 个文件 ---"
find "$DEST" -type f -printf '%s %p\n' 2>/dev/null | sort -rn | head -10 \
    | awk '{printf "  %8.1f KB  %s\n", $1/1024, $2}'
echo
echo "若上面计数为 0，说明 LOCAL_ROOT 猜错了（见文件头注释）。"
