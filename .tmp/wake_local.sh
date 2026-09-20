#!/usr/bin/env bash
# 本地唤醒器：轮询服务器上的完成标记，到点就退出（退出的通知即唤醒）。
#
# ★ 判据覆盖所有终态（`wait-handles-must-have-timeout` 一族）：不只等"成功"，
#   也等看门脚本自己死掉（那也是一种终态，必须报）。
# ★ 每次 ssh 都带 ConnectTimeout + 重试；单次探测不可信
#   （`node-loss-loses-unharvested-results`：同一条路径几分钟内会给出三种症状）。
set -u
SSH="timeout 40 ssh -o ConnectTimeout=15 -o BatchMode=yes qkd"
MAX=500          # 500 × 60s ≈ 8.3 小时上限
i=0
while [ "$i" -lt "$MAX" ]; do
    i=$((i + 1))

    # 1) 完成标记
    if $SSH 'test -f /tmp/watch_v2.done' 2>/dev/null; then
        echo "=== 看门脚本报完成（第 $i 次轮询）==="
        $SSH 'cat /tmp/watch_v2.done' 2>/dev/null
        echo
        echo "=== 看门日志尾部 ==="
        $SSH 'tail -20 /tmp/watch_v2.log' 2>/dev/null
        exit 0
    fi

    # 2) 看门脚本本身是否还活着（否则上面那条永远不会成立）
    if ! $SSH 'pgrep -u $(id -u) -f watch_v2.py >/dev/null' 2>/dev/null; then
        # 只有在网络确实通的时候才判"看门死了"，否则是网络问题
        if $SSH 'echo OK' 2>/dev/null | grep -q OK; then
            echo "=== ★ 看门脚本不在了，但完成标记没有 ==="
            $SSH 'tail -30 /tmp/watch_v2.log' 2>/dev/null
            exit 3
        fi
    fi

    # 3) 心跳（每 5 次）
    if [ $((i % 5)) -eq 0 ]; then
        echo "--- 心跳 $i（$((i)) 分钟）---"
        $SSH 'tail -3 /tmp/watch_v2.log' 2>/dev/null
    fi

    sleep 60
done
echo "=== ★ 轮询 $MAX 次未完成，放弃等待 ==="
$SSH 'tail -10 /tmp/watch_v2.log' 2>/dev/null
exit 4
