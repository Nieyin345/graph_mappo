#!/usr/bin/env bash
# 训练窗口扩展实验：0-295 → 0-329（吃下闲置的 34 天）
#
# 对照：r9ext（窗口 0-295，种子 52/53/54）正在跑，本实验用种子 62/63/64。
# 判读：同轮数下验证成功率是否高于 r9ext 的同期值。
#
# 入队方式：等 r9ext 的 3 个种子都结束再起（避免抢 CPU 把两边的
# 每轮耗时都拖长，更重要的是避免线程竞争改变数值）。
set -u
cd /opt/qkd/graph_mappo

UPDATES="${UPDATES:-30}"
SEEDS="${SEEDS:-62 63 64}"

echo "[$(date -Is)] 等 r9ext 全部结束..."
while pgrep -f "run-name r9ext_s" >/dev/null 2>&1; do sleep 60; done
echo "[$(date -Is)] r9ext 已结束，启动窗口扩展实验"

for seed in $SEEDS; do
  LOG="/tmp/w329_s${seed}.log"
  echo "[$(date -Is)] w329_s${seed} 启动" | tee -a "$LOG"
  timeout 21600 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    /opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py \
      --configs rl_algorithm.yaml train_full_rl.yaml train_window_329.yaml \
      --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
      --seed "${seed}" --num-updates "${UPDATES}" --run-name "w329_s${seed}" \
      >> "$LOG" 2>&1
  echo "[$(date -Is)] w329_s${seed} 退出码 $?" | tee -a "$LOG"
done

echo "======== 结果 ========"
/opt/qkd/venv/bin/python /tmp/server_status.py 2>&1 | tail -12
