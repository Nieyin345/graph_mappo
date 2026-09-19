#!/usr/bin/env bash
# 唤醒器：等第七浪指定阶段结束，把结果拉回来。
#
#   bash .tmp/wake_r7.sh short    # 等 3 个 20 轮短臂结束（~2.5h）
#   bash .tmp/wake_r7.sh all      # 等全部（含 60 轮长臂，~7.5h）
#
# 判据用各臂 metrics.jsonl 的行数，不靠进程存活（长臂还活着时短臂已经写完）。
set -uo pipefail
SSH="C:/Windows/System32/OpenSSH/ssh.exe"
SCP="C:/Windows/System32/OpenSSH/scp.exe"
OPTS="-o BatchMode=yes -o ConnectTimeout=15"
R="cd /opt/qkd/graph_mappo"

MODE="${1:-short}"

count_updates() {   # 某臂已写出的 update 行数
    $SSH $OPTS qkd "$R && grep -c '\"update\"' outputs/r7_$1/metrics.jsonl 2>/dev/null || echo 0" 2>/dev/null | tr -d '\r' | tail -1
}

echo "=== 等第七浪（模式=$MODE）$(date +%H:%M:%S) ==="
deadline=$(( $(date +%s) + 40000 ))   # 上限 ~11h

while true; do
    b=$(count_updates base);        b=${b:-0}
    g=$(count_updates fix_g999);    g=${g:-0}
    e=$(count_updates fix_ent);     e=${e:-0}
    l=$(count_updates fix_g999_long); l=${l:-0}
    echo "  $(date +%H:%M:%S)  base=$b/20  g999=$g/20  ent=$e/20  long=$l/60"

    if [ "$MODE" = "short" ]; then
        [ "$b" -ge 20 ] && [ "$g" -ge 20 ] && [ "$e" -ge 20 ] && break
    else
        [ "$b" -ge 20 ] && [ "$g" -ge 20 ] && [ "$e" -ge 20 ] && [ "$l" -ge 60 ] && break
    fi

    # 训练进程全没了但轮数没到 -> 崩了，别死等
    alive=$($SSH $OPTS qkd 'pgrep -cf "train_graph_mappo[.]py" || echo 0' 2>/dev/null | tr -d '\r' | tail -1)
    alive=${alive:-0}
    if [ "$alive" -eq 0 ]; then
        echo "!! 训练进程已全部退出，但轮数未达标 —— 可能崩了"
        break
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then echo "!! 超时"; break; fi
    sleep 180
done

echo
echo "=== 拉取结果 $(date +%H:%M:%S) ==="
$SSH $OPTS qkd "$R && sed -n '/======== 结果 ========/,\$p' /tmp/screen_r7.out 2>/dev/null" || echo "(汇总还没写出)"
echo
echo "=== 各臂日志尾部（查错）==="
for n in base fix_g999 fix_ent fix_g999_long; do
    echo "--- r7_$n ---"
    $SSH $OPTS qkd "tail -3 /tmp/r7_$n.log 2>/dev/null | cut -c1-160" 2>/dev/null
done
echo "=== WAKE_DONE $(date +%H:%M:%S) ==="
