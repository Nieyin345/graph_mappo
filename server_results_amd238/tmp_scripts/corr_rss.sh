#!/usr/bin/env bash
# 关联实验：把「每轮的 RSS」和「每轮的 update_s」放在同一条时间轴上。
# 目的：RSS 是否会封顶？update_s 是否在同一轮封顶？
#   - 若两者同步封顶 -> 是某个缓存被填满，稳态耗时 = 封顶值
#   - 若 RSS 一直涨而 update_s 也一直涨 -> 无界累积，是真问题
#
#   nohup bash .tmp/corr_rss.sh > /tmp/corr_rss.log 2>&1 &
set -uo pipefail
cd /opt/qkd/graph_mappo

PY=/opt/qkd/venv/bin/python
RUN=corr_rss
THREADS=${THREADS:-16}
UPDATES=${UPDATES:-12}
RSSLOG=/tmp/$RUN.rss

if pgrep -f "train_graph_mappo[.]py" >/dev/null 2>&1; then
    echo "ERROR: 已有 train_graph_mappo.py 在跑。" >&2
    exit 1
fi

rm -rf "outputs/$RUN"; : > "$RSSLOG"
echo "=== start $(date +%H:%M:%S)  threads=$THREADS updates=$UPDATES ==="

OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
  $PY scripts/train/train_graph_mappo.py \
    --configs rl_algorithm.yaml train_full_rl.yaml \
    --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
    --num-updates "$UPDATES" --run-name "$RUN" > "/tmp/$RUN.log" 2>&1 &
TPID=$!

START=$(date +%s)
while kill -0 "$TPID" 2>/dev/null; do
    # 中括号技巧避免 awk 匹配到自己的命令行
    read -r main wrk <<<"$(ps -eo rss,args | awk '
        /train_graph_mappo[.]py/  { m += $1 }
        /multiprocessing[.]spawn/ { w += $1 }
        END { printf "%d %d", m/1024, w/1024 }')"
    # 注意：不能写 `grep -c ... || echo 0` —— 无匹配时 grep 既打印 "0" 又返回
    # 退出码 1，于是 || 再打一个 "0"，$upd 变成两行，把整行采样冲成两条。
    upd=$(grep -c "^update=" "/tmp/$RUN.log" 2>/dev/null); upd=${upd:-0}
    echo "$(( $(date +%s) - START )) ${upd:-0} ${main:-0} ${wrk:-0}" >> "$RSSLOG"
    sleep 2
done
wait "$TPID"; rc=$?
echo "=== done $(date +%H:%M:%S) exit=$rc ==="
