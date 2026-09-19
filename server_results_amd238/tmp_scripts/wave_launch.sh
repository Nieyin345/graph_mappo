#!/usr/bin/env bash
# 按内存预算分波启动 run。
#
# 为什么不能一次全铺：每个 run 真实占用 ~13 GB（PSS），不是 1-2 GB。
# 125 GB 机器上 6 个就撞墙（已实测被 OOM 杀掉 4 个）。本脚本维持
# MAX_CONCURRENT 个并发，每 60s 检查一次空位，自动补下一个。
#
# 为什么用新 run-name：outputs/ 只增不删，且重跑同名会往同一个
# metrics.jsonl 追加，把两次运行的曲线混在一个文件里 → 无法解析。
#
# 用法：
#   bash .tmp/wave_launch.sh                 # 用内置任务列表
#   MAX_CONCURRENT=4 bash .tmp/wave_launch.sh
set -u
cd /opt/qkd/graph_mappo

MAX_CONCURRENT="${MAX_CONCURRENT:-5}"
UPDATES="${UPDATES:-30}"
THREADS="${THREADS:-4}"
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt

# 任务列表：run-name|seed|configs
# r9ext 臂：窗口 0-295（对照）；w329 臂：窗口 0-329（实验）
TASKS=(
  "r9ext_s52r|52|rl_algorithm.yaml train_full_rl.yaml"
  "r9ext_s54r|54|rl_algorithm.yaml train_full_rl.yaml"
  "w329_s62r|62|rl_algorithm.yaml train_full_rl.yaml train_window_329.yaml"
  "w329_s63r|63|rl_algorithm.yaml train_full_rl.yaml train_window_329.yaml"
)

running_now() {
  pgrep -f "train_graph_mappo.py" 2>/dev/null | wc -l
}

# 已在跑的（含本轮之前启动的 s53/s64）一起去重
already() {
  local name="$1"
  pgrep -f "run-name ${name}" >/dev/null 2>&1
}

echo "[$(date -Is)] 波浪启动器：最多 $MAX_CONCURRENT 并发，${#TASKS[@]} 个任务"
echo "当前已在跑 $(running_now) 个"
echo

for task in "${TASKS[@]}"; do
  IFS='|' read -r name seed cfgs <<< "$task"

  if [ -f "outputs/${name}/checkpoint_final.pt" ]; then
    echo "[$(date -Is)] 跳过 ${name}（已完成）"; continue
  fi
  if already "$name"; then
    echo "[$(date -Is)] 跳过 ${name}（已在跑）"; continue
  fi

  # 等空位
  while [ "$(running_now)" -ge "$MAX_CONCURRENT" ]; do
    sleep 60
  done

  # 空位等到了，但内存还得确认——进程数够了不代表内存够
  avail=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
  while [ "$avail" -lt 16 ]; do
    echo "[$(date -Is)] 内存仅 ${avail}G，等回收..."; sleep 60
    avail=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
  done

  setsid nohup env OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
    /opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py \
      --configs $cfgs \
      --checkpoint "$CKPT" \
      --seed "$seed" --num-updates "$UPDATES" --run-name "$name" \
      > "/tmp/${name}.log" 2>&1 < /dev/null &

  echo "[$(date -Is)] 启动 ${name} (seed=${seed}) — 并发 $(( $(running_now) ))/${MAX_CONCURRENT}，内存 ${avail}G"
  sleep 10
done

echo
echo "[$(date -Is)] 全部任务已投递。当前在跑："
pgrep -af "train_graph_mappo.py" | grep -o "run-name [a-z0-9_]*" | sort -u
