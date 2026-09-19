#!/usr/bin/env bash
# 并行铺开所有待跑实验。服务器 32 核，每个 run 固定 OMP_NUM_THREADS=4，
# 6 个 run = 24 线程 < 32 核 → 不超订，数值不受影响（线程数才是决定因素，
# 不是同时跑几个进程）。
#
# 为什么不再让 w329 排队等 r9ext：
#   screen_w329.sh 的等待本意是"避免抢 CPU 把两边每轮耗时都拖长"，
#   但实测每个 run 只用 3 核（load 10/32），排队纯属浪费墙钟。
#
# 本脚本幂等：已在跑的 run 跳过，不会重复启动。
set -u
cd /opt/qkd/graph_mappo

CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
THREADS=4

# 先杀掉排队中的 screen_w329（它还在 sleep 等 r9ext）
pkill -f "screen_w329.sh" 2>/dev/null && echo "已取消排队中的 screen_w329"

launch() {  # $1=run-name  $2=seed  $3=configs
  local name="$1" seed="$2" cfgs="$3"
  if pgrep -f "run-name ${name}\$" >/dev/null 2>&1; then
    echo "跳过 ${name}（已在跑）"; return
  fi
  if [ -f "outputs/${name}/checkpoint_final.pt" ]; then
    echo "跳过 ${name}（已完成）"; return
  fi
  setsid nohup env OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
    /opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py \
      --configs $cfgs \
      --checkpoint "$CKPT" \
      --seed "$seed" --num-updates 30 --run-name "$name" \
      > "/tmp/${name}.log" 2>&1 < /dev/null &
  echo "启动 ${name} (seed=${seed})"
}

# r9ext：补齐缺失的种子 54（52/53 已在跑）
launch r9ext_s54 54 "rl_algorithm.yaml train_full_rl.yaml"

# 窗口扩展实验：0-329，三个种子
for s in 62 63 64; do
  launch "w329_s${s}" "$s" "rl_algorithm.yaml train_full_rl.yaml train_window_329.yaml"
done

sleep 8
echo "================"
pgrep -af "train_graph_mappo.py" | grep -o "run-name [a-z0-9_]*" | sort
echo "================ 负载 ================"
uptime
