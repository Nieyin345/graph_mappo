#!/usr/bin/env bash
# 诊断：update_s 在同一个进程内爬升（thr16_a: 71->86 s, +20%；换臂即重置回 72）。
# 只从外部采 RSS，不改任何代码。12 个 update，约 26 分钟。
#
#   nohup bash .tmp/diag_creep.sh > /tmp/diag_creep.log 2>&1 &
#
# 产出：/tmp/diag_creep.rss  每 2 秒一行：elapsed update主进程RSS_MB worker总RSS_MB 进程数
#       /tmp/diag_creep.log  训练输出
#       outputs/diag_creep/metrics.jsonl  每轮的 rollout_s / update_s
set -uo pipefail
cd /opt/qkd/graph_mappo

PY=/opt/qkd/venv/bin/python
RUN=diag_creep
THREADS=${THREADS:-16}
UPDATES=${UPDATES:-12}
RSSLOG=/tmp/$RUN.rss
RUNLOG=/tmp/$RUN.log

if pgrep -f "train_graph_mappo[.]py" >/dev/null 2>&1; then
    echo "ERROR: 已有 train_graph_mappo.py 在跑，计时会被污染。" >&2
    exit 1
fi

rm -rf "outputs/$RUN"
: > "$RSSLOG"

echo "=== start $(date +%H:%M:%S)  threads=$THREADS updates=$UPDATES ==="

OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
  $PY scripts/train/train_graph_mappo.py \
    --configs rl_algorithm.yaml train_full_rl.yaml \
    --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
    --num-updates "$UPDATES" --run-name "$RUN" > "$RUNLOG" 2>&1 &
TPID=$!

START=$(date +%s)
while kill -0 "$TPID" 2>/dev/null; do
    # 主进程 args 含 train_graph_mappo.py；worker 是 multiprocessing.spawn。
    # 模式写成 train_graph_mappo[.]py，这样 awk 自己的命令行（含这串字面量）
    # 不会被自己的正则匹配到。
    read -r main wrk n <<<"$(ps -eo rss,args | awk '
        /train_graph_mappo[.]py/      { m += $1; nm++ }
        /multiprocessing[.]spawn/     { w += $1; nw++ }
        END { printf "%d %d %d", m/1024, w/1024, nm+nw }')"
    upd=$(wc -l < "outputs/$RUN/metrics.jsonl" 2>/dev/null || echo 0)
    echo "$(( $(date +%s) - START )) $upd ${main:-0} ${wrk:-0} ${n:-0}" >> "$RSSLOG"
    sleep 2
done
wait "$TPID"; rc=$?

echo "=== done $(date +%H:%M:%S)  exit=$rc ==="
