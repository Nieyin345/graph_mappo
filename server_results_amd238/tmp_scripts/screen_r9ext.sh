#!/usr/bin/env bash
# 决策实验：r8 的配置（RNG 修复后）延长训练，验证曲线是否继续上行？
#
# 依据（2026-09-19 晨 r8 多种子）：
#   u5 → u10 → u15   0.648/0.652 → 0.673/0.680 → 0.678/0.700/0.690
#   三种子一致上行，s43 在 u15 已超专家（0.7001 vs 0.6980）。
#   历史 run 只跑 15–30 轮，从没看过曲线的后半段。
#
# 本实验：3 个新种子 × 30 轮，每 5 轮验证一次，看平台在哪。
# 线程固定 4（线程数会确定性影响训练结果，A/B 必须同线程）。
set -u
cd /opt/qkd/graph_mappo

UPDATES="${UPDATES:-30}"
SEEDS="${SEEDS:-52 53 54}"

echo "延长训练：种子 [$SEEDS] × ${UPDATES} 轮   线程=4   $(date -Is)"
echo "代码版本：$(git log --oneline -1)"
echo

for seed in $SEEDS; do
  echo "--- r9ext_s${seed} 开始 $(date -Is) ---"
  timeout 21600 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    /opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py \
      --configs rl_algorithm.yaml train_full_rl.yaml \
      --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
      --seed "${seed}" --num-updates "${UPDATES}" --run-name "r9ext_s${seed}"
  echo "--- r9ext_s${seed} 退出码 $? $(date -Is) ---"
  echo
done

echo "======== 结果 ========"
for seed in $SEEDS; do
  d="outputs/r9ext_s${seed}"
  echo "== r9ext_s${seed} =="
  if [ -f "$d/metrics.jsonl" ]; then
    /opt/qkd/venv/bin/python /tmp/summarize_r8.py "r9ext_s${seed}" || true
  else
    echo "  (无 metrics.jsonl)"
  fi
done
echo
echo "参照: BC 起点 0.6483 / 专家 0.6980（验证 regime，15 种子）"
