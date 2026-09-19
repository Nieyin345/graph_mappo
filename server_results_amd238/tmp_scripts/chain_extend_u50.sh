#!/usr/bin/env bash
# 延长训练到 u50：定位平台在哪。
#
# 依据（docs/训练诊断记录.md，2026-09-19）：
#   ent01 三种子在 u25 全部超专家，u30 继续上行（s42 0.7049，对照臂 0.7176）。
#   同起点配对对照已经把 entropy_coef 这条线索否掉（净效应 +0.0128 < 分辨率
#   0.035，且符号在同起点两臂间翻转）。于是现在**唯一还站得住的机制**是
#   "训练得更久"，而它的**天花板**完全未知——历史 run 最多只跑到 u30。
#
#   所以下一步不是再调超参，而是**看曲线走到哪**。从 u30 检查点恢复，
#   配置一律不动（保持 ent01 的 ent=0.01，使 u5..u50 是一条连续曲线），
#   跑到 u50 → 新增 u35/u40/u45/u50 四个验证点。
#
# 三个种子**并发**（内存实测 98 GB 可用，3 个 run × ~24 GB = 72 GB，
# 符合"3 稳"）。先等 g999_s42_r2 退出，避免第 4 个 run 把谁 OOM 掉。
#
# 用法：setsid nohup bash /tmp/chain_extend_u50.sh > /tmp/chain_extend_u50.out 2>&1 < /dev/null &
set -u
cd /opt/qkd/graph_mappo

echo "等待 ent01_g999_s42_r2 退出... $(date -Is)"
while pgrep -f "run-name ent01_g999_s42_r2" >/dev/null 2>&1; do sleep 60; done
echo "g999_s42_r2 已退出 $(date -Is)"

# 三个 run 并发需要 ~72 GB；留足余量再起
while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 80 ]; do
  echo "  MemAvailable 不足 80 GB，继续等... $(date -Is)"
  sleep 60
done

for seed in 42 43 44; do
  ckpt="outputs/ent01_s${seed}/checkpoint_update_000030.pt"
  if [ ! -f "$ckpt" ]; then
    echo "!! 缺 $ckpt，跳过 seed $seed"
    continue
  fi
  name="ent01_s${seed}_u30to50"
  echo "--- 启动 $name $(date -Is) ---"
  setsid nohup env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    /opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py \
      --configs rl_algorithm.yaml train_full_rl.yaml train_ent01.yaml \
      --checkpoint "$ckpt" \
      --seed "${seed}" --num-updates 20 --run-name "$name" \
      > "/tmp/${name}.log" 2>&1 < /dev/null &
  sleep 20   # 错开启动，避免同时抢内存峰值
done

echo
echo "三个延长臂已全部启动 $(date -Is)"
echo "→ u35/u40/u45/u50 会各出一个验证点；用 scripts/diag/val_align.py 看曲线"
echo "→ 手动收结果："
echo "   /opt/qkd/venv/bin/python scripts/diag/val_align.py ent01_s42 ent01_s43 ent01_s44 ent01_s42_u30to50 ent01_s43_u30to50 ent01_s44_u30to50"
