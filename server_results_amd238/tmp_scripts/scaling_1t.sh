#!/usr/bin/env bash
# 补测吞吐曲线的端点：48 个进程，每个 1 线程（刚好 48 线程预算）。
#
# 已有：12×4 = 0.129 轮/s，24×2 = 0.171 轮/s —— 越薄越高，但边际递减。
# 这一组是趋势的终点，用来定"到底该开多少个"。
#
# 先量一个进程的常驻内存：48 个并发不能把 251 GB 吃爆。
#
# 用法（在节点上）：
#     bash /tmp/scaling_1t.sh
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
UPDATES="${UPDATES:-2}"
N="${N:-48}"
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
CFGS="rl_algorithm.yaml train_diag_fast.yaml"

cd "$MAIN" || exit 1

echo "=== 起来之前 ==="
free -g | awk '/Mem:/{printf "已用 %.1f GB，可用 %.1f GB\n", $3, $7}'

launch() {
    (
        cd "$MAIN" || exit 1
        rm -rf "outputs/thr_u$1"
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            "$PY" -u scripts/train/train_graph_mappo.py \
                --configs $CFGS --checkpoint "$CKPT" \
                --num-updates "$UPDATES" --run-name "thr_u$1" \
                > "/tmp/thr_u$1.log" 2>&1
    )
}

echo
echo "=== $N 个进程 × 1 线程 ==="
t0=$(date +%s)
for i in $(seq 1 "$N"); do launch "${N}_${i}" & done
# 等它们把环境建起来再采一次内存，这时才是稳态占用
sleep 60
free -g | awk '/Mem:/{printf "运行中：已用 %.1f GB，可用 %.1f GB\n", $3, $7}'
wait
wall=$(( $(date +%s) - t0 ))
echo "墙钟 ${wall}s"

echo
echo "======== 全部实测（含之前的组）========"
"$PY" /tmp/summarize_throughput.py
echo "THROUGHPUT_1T_DONE"
