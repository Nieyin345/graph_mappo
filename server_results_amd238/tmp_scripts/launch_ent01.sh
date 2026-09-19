#!/usr/bin/env bash
# ent01 配对实验（修正版）：同种子对照 r8_base（ent=0.001）。
#
# 设计：种子 42/43/44 —— 与 r8_base_s42/43/44 **完全相同**，
# 只改 entropy_coef=0.01，轮数 30。
#   - 与 r8_base 在 u5/u10/u15 三点配对（r8_base 只跑到 15）
#   - 额外跑到 u30，看曲线是否像 r7_fix_ent 那样在 u20 后继续爬
#
# 为什么 30 不是 20：r7_fix_ent 的曲线 u15 0.7063 → u20 0.7113 **仍在上升**，
# 切在 20 会看不到平台，也就不知道这个改动能走多远。
#
# 先停掉 w329_s62r/s63r —— 它们用的是已被证明更差的 ent=0.001，
# 结果会被混淆，不如把内存让给本实验。
set -u
cd /opt/qkd/graph_mappo

CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
THREADS=4
UPDATES="${UPDATES:-30}"

# 前置检查：配置文件必须在
if [ ! -f configs/train_ent01.yaml ]; then
  echo "!! 缺少 configs/train_ent01.yaml，中止"; exit 1
fi

echo "[$(date -Is)] 停掉 ent=0.001 的残留 run"
for p in $(pgrep -f "train_graph_mappo.py"); do
  nm=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -o 'run-name [a-z0-9_]*' | cut -d' ' -f2)
  echo "  停 pid=$p (${nm:-?})"
  kill -TERM "$p" 2>/dev/null
done
sleep 15
for p in $(pgrep -f "train_graph_mappo.py"); do kill -KILL "$p" 2>/dev/null; done
sleep 5
/opt/qkd/venv/bin/python /tmp/kill_orphans.py --apply 2>&1 | tail -2

echo
echo "[$(date -Is)] 启动 ent01_s42/43/44（30 轮，entropy_coef=0.01）"
for seed in 42 43 44; do
  name="ent01_s${seed}"
  setsid nohup env OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
    /opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py \
      --configs rl_algorithm.yaml train_full_rl.yaml train_ent01.yaml \
      --checkpoint "$CKPT" \
      --seed "$seed" --num-updates "$UPDATES" --run-name "$name" \
      > "/tmp/${name}.log" 2>&1 < /dev/null &
  echo "[$(date -Is)] 启动 ${name}"
  sleep 10
done

sleep 40
echo
echo "=== 启动后 40s 检查（应能看见 update=1 之前无报错）==="
for seed in 42 43 44; do
  echo "--- ent01_s${seed} ---"
  tail -3 "/tmp/ent01_s${seed}.log" 2>/dev/null || echo "(无日志)"
done
echo
echo "=== 在跑 ==="
pgrep -af "train_graph_mappo.py" | grep -o "run-name [a-z0-9_]*" | sort -u
echo "=== 内存 ==="
free -g | head -2
