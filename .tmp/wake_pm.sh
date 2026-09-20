#!/usr/bin/env bash
# 本地唤醒器：等 5 条 pm_decode 臂跑到 u30。
#
# ★ 判据覆盖**所有终态**（`wait-handles-must-have-timeout` / `over-conservative-gate-is-silent-too`）：
#     0 = 全部跑满            → 叫醒，去判读
#     3 = 进程没了但没跑满     → ★ 失败，必须吵
#     5 = 日志里有 Traceback   → ★ 失败，必须吵
#     4 = 超时                → 也要报（"还在跑"与"坏了"必须可区分）
#   ★ 只等"成功"的轮询器在失败时会静默空转到超时，那正是上次
#     `wake_regime.out` 到 17:06:45 就断掉、臂从未起的原因之一。
#
# ★ 网络抖动不算终态：ssh 无输出 ⟹ 继续等（`node-loss-loses-unharvested-results`：
#   同一条 ssh 路径几分钟内会给出三种症状，单次探测不可信）。
set -u
SSH="timeout 40 ssh -o ConnectTimeout=15 -o BatchMode=yes qkd"
REMOTE='bash /opt/qkd/graph_mappo/.tmp/pm_status.sh'
ARMS="pm_decode_s42 pm_decode_s43 pm_decode_s44 pm_decode_s45 pm_decode_s46"
NEED=30
SLEEP=300            # 5 分钟一轮（一轮约 14 分钟，5 分钟足够细）
MAX=140              # 140 × 5min ≈ 11.7 小时上限

i=0
while [ "$i" -lt "$MAX" ]; do
    i=$((i + 1))
    OUT=$($SSH "$REMOTE" 2>/dev/null)

    if [ -z "$OUT" ]; then
        echo "[$i] ssh 无输出（网络抖动）—— 继续等，不判终态"
        sleep "$SLEEP"
        continue
    fi

    # 失败优先：有 Traceback 立刻报
    TR=$(echo "$OUT" | grep '^TRACE=' | cut -d= -f2 | tr '\n' ' ')
    if [ -n "${TR// /}" ]; then
        echo "=== ★ 日志里有 Traceback：$TR ==="
        echo "$OUT"
        exit 5
    fi

    DONE=1
    LINE=""
    for a in $ARMS; do
        n=$(echo "$OUT" | grep "^ARM=$a " | sed 's/.* N=\([0-9]*\).*/\1/')
        LINE="$LINE $a=${n:-?}"
        [ "${n:-0}" -ge "$NEED" ] || DONE=0
    done
    MEM=$(echo "$OUT" | grep '^MEM=' | cut -d= -f2)
    PROC=$(echo "$OUT" | grep '^PROC=' | cut -d= -f2)
    echo "[$i] u:$LINE | 进程=$PROC | 可用=${MEM}GiB"

    if [ "$DONE" = 1 ]; then
        echo "ALL_DONE  $LINE"
        exit 0
    fi

    if [ "${PROC:-1}" -eq 0 ]; then
        echo "=== ★ 没有训练进程在跑，但臂还没跑满 ⟹ 失败，不空等到超时 ==="
        echo "$OUT"
        exit 3
    fi

    sleep "$SLEEP"
done
echo "=== ★ 轮询 $MAX 次仍未跑满，放弃等待（超时也是终态，必须报）==="
exit 4
