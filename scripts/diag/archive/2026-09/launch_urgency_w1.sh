#!/usr/bin/env bash
# urgency 剂量实验启动器 —— 严格按 docs/实验日志.md 第 4.5 节日程执行。
#
# 日程：5 组 × 5 种子 = 25 条
#   u0ctl (enabled:false) / u1half 0.17 / u2bal 0.35 / u3two 0.70 / u4quad 1.40
#   种子 42,43,44,45,46
#
# 分波（按内存预算，不一次全铺）：
#   每条稳态 25 GiB ⟹ 上限 (250-17)/25 ≈ 9 条
#   第 1 波 9 条（u0ctl 全 5 + u1half 全 4）—— 对照必须先起
#   第 2 波 9 条（u1half s46 + u2bal 全 5 + u3two 全 3）
#   第 3 波 7 条（u3two s45,s46 + u4quad 全 5）
#
# ★ 预检 + 启动后验证（本项目规矩：启动失败必须吵，不能被当成"还在跑"）
set -u
cd /opt/qkd/graph_mappo

CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
THREADS=8
UPDATES="${UPDATES:-30}"
SEEDS="42 43 44 45 46"

# ── 预检 ──────────────────────────────────────────────────────────────
echo "[$(date -Is)] 预检：配置文件"
missing=0
for arm in u0ctl u1half u2bal u3two u4quad; do
  f="configs/train_urgency_${arm}.yaml"
  if [ ! -f "$f" ]; then echo "  !! 缺 $f"; missing=1; fi
done
if [ ! -f "$CKPT" ]; then echo "  !! 缺 checkpoint $CKPT"; missing=1; fi
if [ "$missing" = "1" ]; then echo "!! 预检失败，中止"; exit 1; fi
echo "  ✓ 5 个臂配置 + checkpoint 都在"

# ── 起一批 ────────────────────────────────────────────────────────────
# 参数：<波号> <空格分隔的 arm:seed 列表>
launch_wave() {
  local wave="$1"; shift
  echo
  echo "[$(date -Is)] 第 ${wave} 波：$# 条"
  local launched=0 failed=0
  for spec in "$@"; do
    local arm="${spec%%:*}"
    local seed="${spec##*:}"
    local name="${arm}_s${seed}"
    # 已在跑就不重起（去重问世界，不问账本）
    if pgrep -f "run-name ${name}\b" >/dev/null 2>&1; then
      echo "  跳过 ${name}（已在跑）"; continue
    fi
    ulimit -n 65536 2>/dev/null || true
    setsid nohup env OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
      /opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py \
        --configs rl_algorithm.yaml train_full_rl.yaml train_ent01.yaml \
                   "train_urgency_${arm}.yaml" \
        --checkpoint "$CKPT" \
        --seed "$seed" --num-updates "$UPDATES" --run-name "$name" \
        > "/tmp/${name}.log" 2>&1 < /dev/null &
    echo "  启动 ${name} (pid $!)"
    launched=$((launched+1))
    sleep 12
  done

  # ── 启动后验证：等 90s 再确认真的在跑且没立刻报错 ──
  echo "[$(date -Is)] 等 90s 后验证..."
  sleep 90
  local alive=0 dead=0
  for spec in "$@"; do
    local name="${spec%%:*}_s${spec##*:}"
    if pgrep -f "run-name ${name}\b" >/dev/null 2>&1; then
      alive=$((alive+1))
    else
      dead=$((dead+1))
      echo "  ★ ${name} 没起来！日志尾部："
      tail -5 "/tmp/${name}.log" 2>/dev/null | sed 's/^/      /'
    fi
  done
  echo "[$(date -Is)] 第 ${wave} 波：启动 ${launched}，存活 ${alive}，失败 ${dead}"
  echo "  内存：$(free -g | awk '/Mem:/{print $7}') GiB 可用"
  if [ "$dead" -gt 0 ]; then
    echo "  ★★ 第 ${wave} 波有失败的臂 —— 停下来报告，不自动往下推"
    return 1
  fi
  return 0
}

# ── 第 1 波：对照全 5 + u1half 4 条 ──
if ! launch_wave 1 \
  u0ctl:42 u0ctl:43 u0ctl:44 u0ctl:45 u0ctl:46 \
  u1half:42 u1half:43 u1half:44 u1half:45; then exit 1; fi

echo
echo "第 1 波已铺开。等它们到 u30（约 3-4 小时），用 .tmp/check_urgency.py 查进度。"
echo "第 2/3 波由 wave2.sh / wave3.sh 在内存允许时启动。"
