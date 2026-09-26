#!/usr/bin/env bash
# Step 1 of "PPO has no gain over BC": two cheap diagnostics that decide what to
# attack next.
#
#   B) Window-B check. All frozen arms were only evaluated on window A. If the
#      "PPO end == BC start" pattern also holds on window B, the BC ceiling is a
#      property of BC, not of one window. Cheap (6 x 35s).
#
#   A) Resume-limit curve. Is PPO under-trained (still rising at 20 updates) or
#      converged/misled (flat)? Train v3 seed 42 from the SAME shared BC for
#      1/5/10/40 updates and evaluate each on window A. The existing
#      scr240_v3_s42 gives the 20-update point for free.
#
# Concurrency: MAX_CONCURRENT x cpu_threads <= nproc (docs/实验产物规范.md §3.7);
# cpu_threads in config OVERRIDES OMP/MKL env vars, so cap by concurrency only.
set -u
ulimit -n 65535
cd /opt/qkd/graph_mappo

_nofile=$(ulimit -n)
[ "${_nofile}" -ge 65535 ] 2>/dev/null || { echo "ulimit -n=${_nofile} 过低" >&2; exit 1; }

CPU_THREADS=$(grep -E "^\s+cpu_threads:" configs/experiment_base.yaml | head -1 | awk '{print $2}')
CPU_THREADS="${CPU_THREADS:-16}"
MAXC="${MAXC:-4}"
[ $(( MAXC * CPU_THREADS )) -le "$(nproc)" ] || {
    echo "错误: ${MAXC}x${CPU_THREADS} > $(nproc) 核" >&2; exit 1; }
avail=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
echo "[预检] ulimit=$(ulimit -n) nproc=$(nproc) 并发=${MAXC} 线程=${CPU_THREADS} 内存=${avail}G"

BC="outputs/experiments/v3/screen_bc/bc/supervised_pg_phased.pt"
[ -f "$BC" ] || { echo "缺少 v3 共享 BC: $BC" >&2; exit 1; }

# ---------------- B) 窗 B 评测（现成 checkpoint） ----------------
echo
echo "=== B) 窗 B 评测（v3/v6 各 3 种子，已有 checkpoint）==="
for arm in v3/scr240_v3_s42 v3/scr240_v3_s43 v3/scr240_v3_s44 \
           v6/scr240_v6_s42 v6/scr240_v6_s43 v6/scr240_v6_s44; do
    tag=$(echo "$arm" | tr / _)
    if ls outputs/frozen_eval/ 2>/dev/null | grep -q "${tag}_B"; then
        echo "  跳过 ${arm}（窗B已有）"; continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$MAXC" ]; do wait -n; done
    env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        /opt/qkd/venv/bin/python -u scripts/eval/frozen_eval.py \
        "rl:outputs/experiments/${arm}" --window B \
        > "/tmp/evalB_${tag}.log" 2>&1 &
done
wait
echo "  窗 B 完成：$(ls outputs/frozen_eval/ | grep -c '_B.json') 个 JSON"

# ---------------- A) 续训时长曲线（v3 s42，同一 BC） ----------------
echo
echo "=== A) 续训曲线：v3 s42，从同一 BC 训练 1/5/10/40 轮 ==="
for u in 1 5 10 40; do
    name="gainlim_v3_u${u}_s42"
    if [ -f "outputs/experiments/v3/${name}/checkpoint_final.pt" ]; then
        echo "  跳过 ${name}（已完成）"; continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$MAXC" ]; do wait -n; done
    setsid nohup env OMP_NUM_THREADS=${CPU_THREADS} MKL_NUM_THREADS=${CPU_THREADS} \
        /opt/qkd/venv/bin/python -u scripts/experiment.py train \
        --model v3 --seed 42 --name "${name}" \
        --updates "${u}" --resume "${BC}" \
        > "/tmp/${name}.log" 2>&1 < /dev/null &
    echo "  启动 ${name}"
    sleep 5
done
# 训练是 setsid 后台，不受 jobs 管；轮询等它们出 checkpoint。
echo "  等待训练收尾..."
for u in 1 5 10 40; do
    name="gainlim_v3_u${u}_s42"
    for _ in $(seq 1 240); do
        [ -f "outputs/experiments/v3/${name}/checkpoint_final.pt" ] && break
        sleep 15
    done
done
echo "  训练段完成"

echo
echo "=== A) 各点窗 A 评测 ==="
for u in 1 5 10 40; do
    arm="v3/gainlim_v3_u${u}_s42"
    tag=$(echo "$arm" | tr / _)
    if ls outputs/frozen_eval/ 2>/dev/null | grep -q "${tag}_A"; then
        echo "  跳过 ${arm}"; continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$MAXC" ]; do wait -n; done
    env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        /opt/qkd/venv/bin/python -u scripts/eval/frozen_eval.py \
        "rl:outputs/experiments/${arm}" --window A \
        > "/tmp/evalA_${tag}.log" 2>&1 &
done
wait
echo "=== 全部完成 ==="
