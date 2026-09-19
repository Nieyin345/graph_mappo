#!/usr/bin/env bash
# 把「RL 比专家省 22% 密钥」从 1 个训练种子扩到 3 个（§4⑦ 的规矩）。
# 每个 eval 约 3 分钟、1~2 GB，相对两个 23 GB 的 run 可忽略；顺序跑不叠加。
set -u
cd /opt/qkd/graph_mappo || exit 1
for s in 42 43 44; do
  ck="outputs/ent01_s${s}/checkpoint_update_000030.pt"
  [ -f "$ck" ] || { echo "!! 缺 $ck"; continue; }
  out="outputs/eval/rl_ent01_s${s}_u30"
  [ -f "$out/summary.json" ] && { echo "已有 $out，跳过"; continue; }
  echo "[$(date -Is)] 评估 $ck"
  env OMP_NUM_THREADS=2 /opt/qkd/venv/bin/python -u scripts/baselines/run_baselines.py \
    --config configs/global.yaml --episodes 15 \
    --seeds 100,101,102,103,104,105,106,107,108,109,110,111,112,113,114 \
    --out "$out" --policies __none__ \
    --rl-checkpoint "$ck" --rl-name "rl_ent01_s${s}_u30" \
    > "/tmp/eval_eff_s${s}.log" 2>&1
  echo "[$(date -Is)] 完成 $s (exit $?)"
done
echo "[$(date -Is)] 三个种子的效率评估结束"
