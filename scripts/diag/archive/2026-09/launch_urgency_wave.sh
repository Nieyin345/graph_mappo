#!/usr/bin/env bash
# urgency 剂量实验 —— 第 2/3 波启动器（按日程，不自己加臂）。
#
# 用法：bash /tmp/launch_urgency_wave.sh 2    # 或 3
#
# 日程（docs/实验日志.md 4.5 节，25 条 = 5 组 × 5 种子）：
#   第 1 波 9 条  u0ctl 全 5 + u1half s42-45      （launch_urgency_w1.sh 已起）
#   第 2 波 9 条  u1half s46 + u2bal 全 5 + u3two s42-44
#   第 3 波 7 条  u3two s45,s46 + u4quad 全 5
#
# ★ 内存判据：MemAvailable/30（每 run 稳态 25 GiB，不是 23），地板留 17 GiB
set -u
cd /opt/qkd/graph_mappo

WAVE="${1:?用法: launch_urgency_wave.sh <2|3>}"
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
THREADS=8
UPDATES="${UPDATES:-30}"

case "$WAVE" in
  2) SPECS="u1half:46 u2bal:42 u2bal:43 u2bal:44 u2bal:45 u2bal:46 u3two:42 u3two:43 u3two:44" ;;
  3) SPECS="u3two:45 u3two:46 u4quad:42 u4quad:43 u4quad:44 u4quad:45 u4quad:46" ;;
  *) echo "!! 未知波号 $WAVE"; exit 1 ;;
esac

# ── 预检：配置存在 ──
for spec in $SPECS; do
  arm="${spec%%:*}"
  f="configs/train_urgency_${arm}.yaml"
  [ -f "$f" ] || { echo "!! 缺 $f，中止"; exit 1; }
done
[ -f "$CKPT" ] || { echo "!! 缺 checkpoint，中止"; exit 1; }
echo "[$(date -Is)] 预检通过（$(echo $SPECS | wc -w) 条）"

# ── 内存门：必须打印自己的输入（恒真的门不报错，直到 OOM 才现形）──
avail=$(free -g | awk '/Mem:/{print $7}')
running=$(ps -eo args | grep -c "[t]rain_graph_mappo")
# 在跑的 run 还没到稳态：按 25 GiB 全额预留（保守）
reserved=$(( running * 25 ))
usable=$(( avail - 17 ))
echo "[$(date -Is)] 内存门：可用 ${avail} GiB，在跑 ${running} 条（预留 ${reserved} GiB），"
echo "                     地板 17 GiB ⟹ 可再放 $(( usable / 25 )) 条（usable=${usable} GiB）"
if [ "$usable" -lt 25 ]; then
  echo "!! 内存不足（余量 ${usable} GiB < 25），本波不启动"; exit 2
fi

# ── 启动 ──
launched=0
for spec in $SPECS; do
  arm="${spec%%:*}"; seed="${spec##*:}"
  name="${arm}_s${seed}"
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

# ── 启动后验证：失败必须吵 ──
echo "[$(date -Is)] 等 90s 后验证..."
sleep 90
alive=0; dead=0
for spec in $SPECS; do
  name="${spec%%:*}_s${spec##*:}"
  if pgrep -f "run-name ${name}\b" >/dev/null 2>&1; then
    alive=$((alive+1))
  else
    dead=$((dead+1))
    echo "  ★ ${name} 没起来！日志尾部："
    tail -6 "/tmp/${name}.log" 2>/dev/null | sed 's/^/      /'
  fi
done
echo "[$(date -Is)] 第 ${WAVE} 波：启动 ${launched}，存活 ${alive}，失败 ${dead}"
echo "  内存：$(free -g | awk '/Mem:/{print $7}') GiB 可用"
[ "$dead" -gt 0 ] && { echo "★★ 有臂失败 —— 停下来报告，不自动往下推"; exit 1; }
exit 0
