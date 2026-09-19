#!/usr/bin/env bash
# 决定性实验：update_s 的爬升是【进程内累积】还是【机器持续变慢】？
#
# diag_creep 用单进程 12 轮发现：u1=72.0 u2=78.9 u3=83.8（+16%）。
# 两种互斥解释：
#   A 进程内累积（replay/cache/内存增长）-> 每个新进程的 u1 都 ≈72 s
#   B 机器持续变慢（温度/降频）        -> 后面的进程 u1 越来越慢
#
# 做法：连跑 N 个独立进程，每个只跑 1 个 update。每个进程都只有"u1",
# 所以只要看这串 u1 是平的（A）还是上扬的（B）。
#
#   nohup bash .tmp/probe_restart.sh > /tmp/probe_restart.log 2>&1 &
set -uo pipefail
cd /opt/qkd/graph_mappo

PY=/opt/qkd/venv/bin/python
CHECKPOINT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
N=${N:-8}
THREADS=${THREADS:-16}

if pgrep -f "train_graph_mappo[.]py" >/dev/null 2>&1; then
    echo "ERROR: 已有 train_graph_mappo.py 在跑。" >&2
    exit 1
fi

echo "=== start $(date +%H:%M:%S)  threads=$THREADS  N=$N ==="
for i in $(seq 1 "$N"); do
    name="pr_$i"
    t0=$(date +%s)
    OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
        $PY scripts/train/train_graph_mappo.py \
        --configs rl_algorithm.yaml train_full_rl.yaml \
        --checkpoint "$CHECKPOINT" \
        --num-updates 1 \
        --run-name "$name" > "/tmp/$name.log" 2>&1
    t1=$(date +%s)
    # 从日志里抠三个数：该轮的 rollout_s / update_s / 进程峰值频率
    line=$(grep -m1 "^update=" "/tmp/$name.log" || echo "")
    roll=$(sed -n 's/.*rollout_s=\([0-9.]*\).*/\1/p' <<<"$line")
    upd=$(sed -n 's/.*update_s=\([0-9.]*\).*/\1/p' <<<"$line")
    echo "run $name  wall=$((t1-t0))s  rollout_s=${roll:-?}  update_s=${upd:-?}"
    rm -rf "outputs/$name"
done
echo "=== done : $(date +%H:%M:%S) ==="
