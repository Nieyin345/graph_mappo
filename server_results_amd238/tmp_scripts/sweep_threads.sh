#!/usr/bin/env bash
# 扫描 update 阶段的线程数（当前配置 batch_chunk=64、minibatch_size=256、
# rollout worker 固定 8）。找这台 32 线程 EPYC 7302P 上的饱和点。
#
# 为什么要交错：update_s 在同一个进程内会爬升 15~20%（见 diag_creep），
# 按 8,8,16,16 顺序跑的话，后跑的臂处在更热的机器上，结论就废了。交错跑
# 让每个臂都是新进程（爬升重置），把顺序效应摊平。
#
#   nohup bash .tmp/sweep_threads.sh > /tmp/sweep_threads.log 2>&1 &
#
# 产出：outputs/sw_thr{8,16,24,32}_{r1,r2}/metrics.jsonl
set -uo pipefail
cd /opt/qkd/graph_mappo

PY=/opt/qkd/venv/bin/python
CHECKPOINT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
UPDATES=${UPDATES:-3}
REPS=${REPS:-2}
THREADS_LIST=${THREADS_LIST:-"8 16 24 32"}

if pgrep -f "train_graph_mappo[.]py" >/dev/null 2>&1; then
    echo "ERROR: 已有 train_graph_mappo.py 在跑，计时会被污染。" >&2
    exit 1
fi

run_arm() {
    local threads=$1 name=$2
    echo "=== arm $name (threads=$threads) : $(date +%H:%M:%S) ==="
    OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads \
        $PY scripts/train/train_graph_mappo.py \
        --configs rl_algorithm.yaml train_full_rl.yaml \
        --checkpoint "$CHECKPOINT" \
        --num-updates "$UPDATES" \
        --run-name "$name" 2>&1 | grep -E "^update=|^Final:" | sed "s/^/  [$name] /"
}

# 交错：r1 跑完所有线程值再跑 r2，而不是每个值连跑两遍
for r in $(seq 1 "$REPS"); do
    for t in $THREADS_LIST; do
        run_arm "$t" "sw_thr${t}_r${r}"
    done
done

echo "=== done : $(date +%H:%M:%S) ==="
