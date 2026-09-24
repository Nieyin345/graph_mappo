#!/usr/bin/env bash
# 补种子：u4quad（4× urgency 剂量）+ 它自己的对照 u0ctl，各补 9 个种子。
#
# 为什么两个都要补：配对需要**同种子**。现有 s42..46 有配对，
# 补的 s47..55 若只补实验组，对照组没有对应臂 ⟹ 不能配对。
#
# 目标：n=5 → 14（s42..s55）
# 算过：n=14 时 SE = 0.0135/√14 = 0.0036，t = 0.0101/0.0036 = 2.80 > 2.160（df=13）
#       ⟹ 能判定。
#
# 分 2 波（18 条 × 25 GiB = 450 > 250）：
#   波 A：9 条（u0ctl s47..55）
#   波 B：9 条（u4quad s47..55）
#
# 用法：bash /tmp/launch_urgency_ext.sh A|B
set -u
cd /opt/qkd/graph_mappo

WAVE="${1:?用法: launch_urgency_ext.sh <A|B>}"
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
THREADS=8
UPDATES="${UPDATES:-30}"

case "$WAVE" in
  A) ARM="u0ctl";  SEEDS="47 48 49 50 51 52 53 54 55" ;;
  B) ARM="u4quad"; SEEDS="47 48 49 50 51 52 53 54 55" ;;
  *) echo "!! 未知波 $WAVE"; exit 1 ;;
esac

f="configs/train_urgency_${ARM}.yaml"
[ -f "$f" ] || { echo "!! 缺 $f"; exit 1; }
[ -f "$CKPT" ] || { echo "!! 缺 checkpoint"; exit 1; }
echo "[$(date -Is)] 预检通过：$ARM，9 个种子"

# 内存门（必须打印输入）
avail=$(free -g | awk '/Mem:/{print $7}')
running=$(ps -eo args | grep -c "[t]rain_graph_mappo")
usable=$(( avail - running * 25 - 17 ))
echo "[$(date -Is)] 内存门：可用 ${avail} GiB，在跑 ${running} 条 ⟹ 余量 ${usable} GiB"
if [ "$usable" -lt 200 ]; then
  echo "!! 余量不足 200 GiB（本波需 9×25=225），不启动"; exit 2
fi

for seed in $SEEDS; do
  name="${ARM}_s${seed}"
  if pgrep -f "run-name ${name}\b" >/dev/null 2>&1; then
    echo "  跳过 ${name}（已在跑）"; continue
  fi
  ulimit -n 65536 2>/dev/null || true
  setsid nohup env OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
    /opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py \
      --configs rl_algorithm.yaml train_full_rl.yaml train_ent01.yaml \
                 "train_urgency_${ARM}.yaml" \
      --checkpoint "$CKPT" \
      --seed "$seed" --num-updates "$UPDATES" --run-name "$name" \
      > "/tmp/${name}.log" 2>&1 < /dev/null &
  echo "  启动 ${name} (pid $!)"
  sleep 12
done

echo "[$(date -Is)] 等 90s 后验证..."
sleep 90
alive=0; dead=0
for seed in $SEEDS; do
  name="${ARM}_s${seed}"
  if pgrep -f "run-name ${name}\b" >/dev/null 2>&1; then
    alive=$((alive+1))
  else
    dead=$((dead+1))
    echo "  ★ ${name} 没起来："; tail -5 "/tmp/${name}.log" 2>/dev/null | sed 's/^/      /'
  fi
done
echo "[$(date -Is)] 波 ${WAVE}（${ARM}）：启动 9，存活 ${alive}，失败 ${dead}"
echo "  内存：$(free -g | awk '/Mem:/{print $7}') GiB 可用"
[ "$dead" -gt 0 ] && { echo "★★ 有失败，停下报告"; exit 1; }
exit 0
