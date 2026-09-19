#!/usr/bin/env bash
# 抓取服务器结果到本地（只要结果与方法，不要权重）。
#
# 与原版 fetch_all.sh 的区别：**排除 /tmp/metrics_check**。
# 那是本会话建的隔离副本（705 MB 源码，非结果），而 /tmp 在根分区
# （63G 盘、只剩 40G），不排除的话 tar 要多读 705 MB 且本地多存一份派生物。
# 也没排除 outputs/ 里的 .pt —— 那一条原版已经做了。
#
# 用法：bash scripts/diag/fetch_results.sh
set -u
HOST=qkd
LOCAL="${LOCAL_ROOT:-/d/destop/work_space/learning_space/论文/QKD-SAGIN生产端调度/qkd_rl}"
DEST="$LOCAL/server_results"
mkdir -p "$DEST"

echo "=== 1. outputs/（排除 .pt 权重与 figures 图片）==="
ssh "$HOST" 'cd /opt/qkd/graph_mappo && tar czf - --exclude="*.pt" --exclude="*/figures/*" outputs' \
    | tar xzf - -C "$DEST" && echo "  OK"

echo
echo "=== 2. .tmp 与 /tmp 的探针脚本（复现方法本身也是结果）==="
echo "  （/tmp/metrics_check 是隔离副本，排除）"
# ⚠ tar 里存的是 `.tmp/...`，直接解到 $DEST 会嵌成 $DEST/.tmp —— 而
# tmp_scripts/ 已经存在时 `mv $DEST/.tmp` 会把它**塞进去**变成
# tmp_scripts/.tmp（第一次跑就踩了）。所以解到临时目录再拉平。
TMPD="$DEST/_unpack_tmp"
rm -rf "$TMPD"; mkdir -p "$TMPD"
ssh "$HOST" 'cd /opt/qkd/graph_mappo && tar czf - .tmp' | tar xzf - -C "$TMPD" 2>/dev/null
mkdir -p "$DEST/tmp_scripts"
cp -r "$TMPD"/.tmp/. "$DEST/tmp_scripts/" 2>/dev/null
rm -rf "$TMPD"
echo "  .tmp -> tmp_scripts/: $(ls "$DEST/tmp_scripts" 2>/dev/null | wc -l) 项"

mkdir -p "$DEST/tmp_scripts_server"
ssh "$HOST" 'cd /tmp && tar czf - --exclude="metrics_check" --exclude="*.pt" \
    $(ls /tmp/*.py /tmp/*.sh /tmp/*.log 2>/dev/null | head -200)' \
    | tar xzf - -C "$DEST/tmp_scripts_server" 2>/dev/null
echo "  /tmp 脚本: $(find "$DEST/tmp_scripts_server" -type f 2>/dev/null | wc -l) 个"

echo
echo "=== 3. 清单 ==="
n=$(find "$DEST" -name metrics.jsonl 2>/dev/null | wc -l)
echo "  run 数（有 metrics.jsonl）：$n"
du -sh "$DEST" 2>/dev/null | awk '{print "  体积: "$1}'
echo
echo "--- 最大的 8 个文件 ---"
find "$DEST" -type f -printf '%s %p\n' 2>/dev/null | sort -rn | head -8 \
    | awk '{printf "  %9.1f KB  %s\n", $1/1024, $2}'
echo
echo "指标（应无 .pt）："
find "$DEST" -name "*.pt" 2>/dev/null | wc -l | awk '{print "  .pt 文件数: "$1}'
