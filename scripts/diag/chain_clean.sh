#!/usr/bin/env bash
# 链式启动：等 cmax 波跑完 → 起 clean 波 → 起 resume 批（按内存预算）。
#
# ## 为什么是链式而不是定时
#
# cmax 跑完会释放 125 GiB，clean 必须在那之后才能起（否则撞内存）。
# 「等另一个启动器**进程退出**」比「等内存够」可靠 —— 后者是竞态
# （记忆 plan-is-a-claim-about-the-world-recheck-at-launch）。
#
# ## 判据锚在**地面真值**上，不锚在日志文本
#
#   完成 ⟺ `outputs/cmax_s42/checkpoint_final.pt` 存在
#   （trainer.train() 返回后才写一次；实测标定过）
#   死掉 ⟺ 进程没了但 final.pt 也没写
#
# ## 退出码（失败必须吵）
#
#   0 = clean 已起
#   2 = cmax 有臂失败（没起来 / 没写 final）
#   3 = clean 预检没过
#   4 = 超时（打印当时状态，便于判「慢」还是「冻结」）
#
# 用法（服务器上，脱离 ssh 通道）：
#     setsid nohup bash /opt/qkd/graph_mappo/.tmp/chain_clean.sh > /tmp/chain_clean.out 2>&1 &
set -uo pipefail

REPO=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
CMAX_ARMS="cmax_s42 cmax_s43 cmax_s44 cmax_s45 cmax_s46"
DEADLINE=$(( $(date +%s) + 7200 ))     # 2 小时上限（cmax 剩 ~8 分钟）

echo "=== chain_clean 启动 $(date -u +%H:%M:%S) UTC ==="

# ---------------------------------------------------------------- 1. 等 cmax
while :; do
    done_n=0
    alive=0
    for a in $CMAX_ARMS; do
        [ -f "$REPO/outputs/$a/checkpoint_final.pt" ] && done_n=$((done_n + 1))
    done
    alive=$(ps -eo comm,args | awk '$1 ~ /^python/ && /train_graph_mappo/' \
            | grep -c 'cmax_s' || true)

    echo "[$(date -u +%H:%M:%S)] cmax final.pt=${done_n}/5  进程在跑=${alive}"

    # ★ 终态一：全部写完 final.pt
    if [ "$done_n" -eq 5 ]; then
        echo "✓ cmax 五臂全部写 final.pt"
        break
    fi
    # ★ 终态二：进程全没了但 final 不齐 ⟹ 有臂挂了，必须吵
    if [ "$alive" -eq 0 ]; then
        echo "★ cmax 进程全没了，但只有 ${done_n}/5 写了 final.pt ⟹ 有臂失败"
        exit 2
    fi
    # ★ 终态三：超时
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        echo "★ 超时（2h）⟹ 当时的轮数："
        for a in $CMAX_ARMS; do
            n=$(grep -c '"update"' "$REPO/outputs/$a/metrics.jsonl" 2>/dev/null || echo 0)
            mt=$(date -r "$REPO/outputs/$a/metrics.jsonl" +%H:%M 2>/dev/null || echo '-')
            echo "    $a: ${n}/30  最后写入 $mt"
        done
        exit 4
    fi
    sleep 60
done

# ---------------------------------------------------------------- 2. 等内存落下去
echo "--- 等 60s 让内存释放 ---"
sleep 60
free -g | head -2

# ---------------------------------------------------------------- 3. 起 clean 波
cd "$REPO" || exit 3
echo "--- clean 波预检 + 启动 ---"
$PY -u .tmp/wave_clean.py
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "★ clean 波启动失败 rc=$rc"
    exit 3
fi
echo "✓ clean 波已起"

# ---------------------------------------------------------------- 4. resume 批（按内存预算）
sleep 30
avail=$(awk '/MemAvailable/{printf "%d", $2/1024/1024}' /proc/meminfo)
# 每条 25 GiB；留 30 GiB 余量
room=$(( avail - 30 ))
n=$(( room / 25 ))
echo "--- resume 批：MemAvailable=${avail}G ⟹ 最多 ${n} 条 ---"
if [ "$n" -ge 2 ]; then
    $PY -u .tmp/wave_resume.py || echo "★ resume 波 rc=$?（不致命，clean 已在跑）"
else
    echo "  内存不足以起 resume 批 ⟹ 跳过（clean 跑完再手动起）"
fi

echo "=== chain_clean 完成 $(date -u +%H:%M:%S) UTC ==="
exit 0
