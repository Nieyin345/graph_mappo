#!/usr/bin/env bash
# 判别实验：update_s 的爬升是【进程内累积】还是【CPU 升温/降频】？
#
# 之前的 probe_restart 有缺陷：每个进程只跑 1 个 update（~2 min），CPU 没热起来，
# 所以"新进程都 72s"既符合进程内累积，也符合"没热所以快"，分不清。
#
# 本实验用冷热对照：
#   A 冷机：直接跑 1 个 update，记 update_s
#   B 热机：先用满载负载烤 5 分钟，再跑 1 个 update，记 update_s
# 若 A ≈ B  -> 与温度无关，是进程内累积
# 若 B 明显 > A -> 升温/降频是真凶
#
# 全程每 5 秒采一次平均频率，事后按 update 边界对齐。
#
#   nohup bash .tmp/thermal_test.sh > /tmp/thermal_test.log 2>&1 &
set -uo pipefail
cd /opt/qkd/graph_mappo

PY=/opt/qkd/venv/bin/python
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
THREADS=16
FREQLOG=/tmp/thermal_freq.log

# 频率采样器（与训练并行，纯读 /proc/cpuinfo，几乎不占 CPU）
sample_freq() {
    while true; do
        # 必须用 -F: —— /proc/cpuinfo 的行是 "cpu MHz\t\t: 3286.575"，
        # 默认按空白分词时 $2 是字面量 "MHz"（转数字得 0），值在 $4。
        # 用 -F: 则 $1="cpu MHz\t\t"、$2=" 3286.575"。
        awk -F: -v ts="$(date +%H:%M:%S)" '
            /cpu MHz/ { v=$2+0; s+=v; n++; if(n==1||v<lo)lo=v; if(n==1||v>hi)hi=v }
            END { printf "%s %.0f %.0f %.0f\n", ts, s/n, lo, hi }
        ' /proc/cpuinfo
        sleep 5
    done
}

# 温度（有就采，没有就跳过）
sample_temp() {
    while true; do
        if [ -r /sys/class/thermal/thermal_zone0/temp ]; then
            echo "TEMP $(cat /sys/class/thermal/thermal_zone0/temp)"
        fi
        sleep 5
    done
}

run_one() {
    local name=$1
    rm -rf "outputs/$name"
    OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
        $PY scripts/train/train_graph_mappo.py \
            --configs rl_algorithm.yaml train_full_rl.yaml \
            --checkpoint "$CKPT" \
            --num-updates 1 --run-name "$name" > "/tmp/$name.log" 2>&1 < /dev/null
}

if pgrep -f "train_graph_mappo[.]py" >/dev/null 2>&1; then
    echo "ERROR: 已有训练在跑" >&2; exit 1
fi

: > "$FREQLOG"
sample_freq >> "$FREQLOG" &
FPID=$!
sample_temp >> "$FREQLOG" &
TPID=$!
trap 'kill $FPID $TPID 2>/dev/null' EXIT

echo "=== 阶段 A：冷机基线  $(date +%H:%M:%S) ==="
run_one th_cold
echo "    A 完成 $(date +%H:%M:%S)"

echo "=== 预热：满载烤 5 分钟  $(date +%H:%M:%S) ==="
# 用 nproc 个纯 shell 忙循环把核心烤热；不用 stress 是因为镜像里不一定有
for _ in $(seq 1 "$(nproc)"); do
    ( end=$((SECONDS+300)); while [ $SECONDS -lt $end ]; do :; done ) &
done
wait
echo "    预热结束 $(date +%H:%M:%S)  频率："
tail -1 "$FREQLOG"

echo "=== 阶段 B：热机重测  $(date +%H:%M:%S) ==="
run_one th_hot
echo "    B 完成 $(date +%H:%M:%S)"

echo
echo "=== 结果 ==="
for n in th_cold th_hot; do
    u=$(sed -n 's/.*update_s=\([0-9.]*\).*/\1/p' "/tmp/$n.log" | head -1)
    r=$(sed -n 's/.*rollout_s=\([0-9.]*\).*/\1/p' "/tmp/$n.log" | head -1)
    echo "  $n  rollout_s=${r:-?}  update_s=${u:-?}"
done
echo "=== done $(date +%H:%M:%S) ==="
