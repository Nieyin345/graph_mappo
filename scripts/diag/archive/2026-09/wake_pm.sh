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

# ★★ 开轮之前先考判据自己。
#   `pm_status.sh` 的 classify 支持 LABEL_* 造反证模式；这里用**真臂的日志**
#   当输入考它。判据坏掉的表现是「静默」：恒真 ⟹ 无条件报完成；过保守 ⟹
#   永远不放行（`over-conservative-gate-is-silent-too`）。两种情况都不能靠
#   空转 11 小时才发现 —— 自检不过就直接退出，且用**不同的**退出码。
echo "--- 判据自检（用真臂日志考 classify）---"
# 取当前在跑的一臂当 GOING 样本；对照臂当 FIN 样本
SELF=$($SSH "LABEL_GOING='/tmp/pmlogs/pm_decode_s42.log|/opt/qkd/graph_mappo/outputs/pm_decode_s42' \
             LABEL_FIN='/nonexistent/x.log|/opt/qkd/graph_mappo/outputs/v2_bottleneck_s42' \
             bash /opt/qkd/graph_mappo/.tmp/pm_status.sh" 2>/dev/null)
echo "$SELF"
if ! echo "$SELF" | grep -q 'LABEL_OK'; then
    echo "=== ★ 判据自检没跑起来（ssh 失败或脚本坏了）⟹ 不拿坏判据空转 ==="
    exit 6
fi
if echo "$SELF" | grep -q 'LABEL_FAIL'; then
    echo "=== ★ 判据自检失败 ⟹ 立即退出，不要空等（坏判据的表现就是静默）==="
    exit 6
fi
echo "--- 自检通过 ---"

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
        end=$(echo "$OUT" | grep "^ARM=$a " | sed 's/.* END=\([A-Z]*\).*/\1/')
        age=$(echo "$OUT" | grep "^ARM=$a " | sed 's/.* AGE=\(-\{0,1\}[0-9]*\).*/\1/')
        LINE="$LINE $a=${n:-?}/${end:-?}/${age:-?}s"
        # ★ 终态判据用三态 END，不是「N 够不够」：N 在轮与轮之间不涨，
        #   只看 N 会把「正在跑第 k 轮」误判成死了（过保守），
        #   也会把「单条臂崩了但其它还在跑」漏掉（PROC>0）。
        [ "${n:-0}" -ge "$NEED" ] || [ "${end:-?}" = "FIN" ] || DONE=0
    done
    MEM=$(echo "$OUT" | grep '^MEM=' | cut -d= -f2)
    PROC=$(echo "$OUT" | grep '^PROC=' | cut -d= -f2)
    echo "[$i] u/END/AGE:$LINE | 进程=$PROC | 可用=${MEM}GiB"

    if [ "$DONE" = 1 ]; then
        echo "ALL_DONE  $LINE"
        exit 0
    fi

    # ★ END=DEAD ⟹ 崩了（Traceback 之外的死法也要抓：日志截断、进程被杀）
    DEAD=$(echo "$OUT" | grep '^ARM=' | grep 'END=DEAD' | sed 's/ARM=\([^ ]*\).*/\1/' | tr '\n' ' ')
    if [ -n "${DEAD// /}" ]; then
        echo "=== ★ 这些臂的日志处于死态（非在跑、非跑完）：$DEAD ==="
        echo "$OUT"
        exit 3
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
