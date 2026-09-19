#!/usr/bin/env bash
# 启动链：4 个臂 × 3 种子，按内存分波排队。
#
# 内存判据（不是 CPU）：每 run 23.3 GB PSS 且每轮约 +0.21 GB，30 轮后约 30 GB/run。
#   125 GB 机器 → **并发 3 稳、4 勉强、5 必 OOM**（见 CLAUDE.md）。
#   所以 4 个臂必须分波，每波 3 个种子。每波 ~2.0h。
#
# 为什么不做"杀最晚启动的"看门狗：本项目记录里那条阈值设错、精确杀掉唯一想保的
# 对照臂的教训。这里只做**先算内存再排队**，不做任何自动杀。
#
# 可续跑：每波开始前写 /tmp/wave_<name>.go 标记；有标记就跳过。
#   之所以要它：本脚本是 `nohup bash 链`，**自身被 OOM/断连杀掉时子进程还活着**，
#   重跑脚本会因 `pgrep` 命中在跑的 run 而立刻"跳过"，把那波**当成已完成**。
#   上一版就是这样把第一波之后的臂全部跳过的（干跑自检发现的）。
set -u
cd /opt/qkd/graph_mappo || exit 1

CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
THREADS=4
UPDATES=30
BASE="rl_algorithm.yaml train_full_rl.yaml"

wait_mem() {   # $1 = 需要多少 GB
  for _ in $(seq 1 240); do
    a=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
    [ "$a" -ge "$1" ] && { echo "  [$(date -Is)] 内存够: ${a}G >= ${1}G"; return 0; }
    sleep 30
  done
  echo "  [$(date -Is)] ⚠ 等内存超时（需 ${1}G）"; return 1
}

launch() {     # $1=run-name  $2=seed  $3...=额外 config
  local name="$1"; local seed="$2"; shift 2
  setsid nohup env OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
    /opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py \
      --configs $BASE "$@" \
      --checkpoint "$CKPT" \
      --seed "$seed" --num-updates "$UPDATES" --run-name "$name" \
      > "/tmp/${name}.log" 2>&1 < /dev/null &
  echo "  [$(date -Is)] 启动 ${name} (seed $seed)"
  sleep 10
}

# 一波 = 一个臂的三个种子。$1=臂标签 $2=config 文件（可为空）
wave() {
  local label="$1"; local cfg="${2:-}"
  local marker="/tmp/wave_${label}.go"
  if [ -f "$marker" ]; then
    echo "=== 波 ${label}: 已有标记，跳过 ==="
    return 0
  fi
  if pgrep -f "run-name ${label}_s" > /dev/null; then
    echo "=== 波 ${label}: 检测到已有 run 在跑，等它结束（不重复启动）==="
    while pgrep -f "run-name ${label}_s" > /dev/null; do sleep 120; done
    touch "$marker"
    return 0
  fi
  wait_mem 100 || return 1
  echo "=== 波 ${label}: 启动 ==="
  for s in 42 43 44; do
    if [ -n "$cfg" ]; then launch "${label}_s${s}" "$s" "$cfg"
    else launch "${label}_s${s}" "$s"; fi
  done
  sleep 45
  echo "  --- 启动后自检 ---"
  local bad=0
  for s in 42 43 44; do
    f="/tmp/${label}_s${s}.log"
    if grep -qiE "error|traceback|parsererror" "$f" 2>/dev/null; then
      echo "  !! ${label}_s${s} 报错:"; grep -iE "error|traceback" "$f" | head -3 | sed 's/^/      /'
      bad=1
    else
      echo "  ok ${label}_s${s}: $(tail -1 "$f" 2>/dev/null | cut -c1-80)"
    fi
  done
  if [ "$bad" -ne 0 ]; then
    echo "  ⚠ 有臂启动失败 —— 不写标记，修好后可重跑本波"
    return 1
  fi
  echo "  [$(date -Is)] 等 ${label} 三个跑完（每 run ~2.0h）"
  while pgrep -f "run-name ${label}_s" > /dev/null; do sleep 120; done
  touch "$marker"
  echo "  [$(date -Is)] 波 ${label} 结束"
}

# 参数：只启动第一波（用于配合本地唤醒链，避免长链被断连打断）
#   bash chain_safe_gain.sh --wave1-only
ONLY1=0
[ "${1:-}" = "--wave1-only" ] && ONLY1=1

wave vcoef1 configs/train_safe_vcoef1.yaml
if [ "$ONLY1" -eq 1 ]; then
  echo "[$(date -Is)] --wave1-only：链条交回本地唤醒链接续"
  exit 0
fi

wave ep2    configs/train_safe_ep2.yaml
wave mini512 configs/train_safe_mini512.yaml
wave runt   ""                       # 时间锚点：与 ent01 完全同配置

echo
echo "[$(date -Is)] === 链条全部结束 ==="
pgrep -af train_graph_mappo.py | sed 's/^/  残留: /' || echo "  无残留 run"
awk '/MemAvailable/{printf "  MemAvailable %.1f GB\n", $2/1048576}' /proc/meminfo
