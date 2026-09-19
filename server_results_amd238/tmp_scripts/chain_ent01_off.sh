#!/usr/bin/env bash
# 等 ent01 三条臂退出（释放内存）后，启动"同起点配对对照"：
# 从 ent01_s42 的 u25 检查点恢复，只把 entropy_coef 改回 0.001，同种子 42。
#
# 为什么不立刻起：本机 125 GB，每 run 实测 23.3 GB 且随轮数增长。
# 现在 4 个 run 占着 103 GB、MemAvailable 25.7 GB，第 5 个 run（需 ~24 GB）
# 会把它自己或某个 ent01 臂 OOM 掉 —— 那不是省时间，是毁掉对照。
# ent01 三条臂约 18 分钟后跑到 u30 并退出，届时释放约 72 GB。
#
# 用法：setsid nohup bash .tmp/chain_ent01_off.sh > /tmp/chain_ent01_off.out 2>&1 < /dev/null &
set -u
cd /opt/qkd/graph_mappo

echo "等待 ent01_s42/s43/s44 退出... $(date -Is)"
while pgrep -f "run-name ent01_s42\b" >/dev/null 2>&1 \
   || pgrep -f "run-name ent01_s43\b" >/dev/null 2>&1 \
   || pgrep -f "run-name ent01_s44\b" >/dev/null 2>&1; do
  sleep 60
done
echo "ent01 三条臂已退出 $(date -Is)"
free -g | head -2

# 对照臂期间不要有第二个 run 抢内存：等 MemAvailable 足够再说
while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 40 ]; do
  echo "  MemAvailable 不足 40 GB，继续等... $(date -Is)"
  sleep 60
done

echo "启动 ent01_s42_u25_base $(date -Is)"
timeout 10800 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py \
    --configs rl_algorithm.yaml train_full_rl.yaml \
               train_ent01.yaml train_ent01_off.yaml \
    --checkpoint outputs/ent01_s42/checkpoint_update_000025.pt \
    --seed 42 --num-updates 5 --run-name ent01_s42_u25_base
echo "--- 退出码 $? $(date -Is) ---"

echo
echo "======== 结果 ========"
/opt/qkd/venv/bin/python scripts/diag/val_align.py ent01_s42 ent01_s42_u25_base || true
echo
echo "参照: 专家 0.6979（种子 100-114 配对）；ent01_s42 的 u25 = 0.7077"
