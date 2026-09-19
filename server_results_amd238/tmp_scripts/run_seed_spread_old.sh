#!/usr/bin/env bash
# 旧代码（d1930a2）在不同训练种子下的评估。
#
# 为什么必须补这一刀：新代码四个种子是 0.7414 / 0.7501 / 0.7535 / 0.7627
# （均值 0.752，标准差 0.009，很紧），旧代码只在 seed 42 上测过，是 0.8174。
# 但「旧代码稳在 0.82 附近」这个前提本身还没验过 —— 如果旧代码换个种子也在
# 0.74-0.82 之间散开，那 0.06 就落在自然波动里，前面所有归因都不成立。
#
#   - 旧代码 4 个种子都聚在 ~0.82 -> 两版代码的分布清晰分离，前向数值差确实
#     系统性地把训练带到了更差的解，得让前向保持逐位一致。
#   - 旧代码也散开 ~0.06       -> 「回归」是这个指标的正常波动，先修测量。
set -u

OLD=/opt/qkd/eval_old_model
PY=/opt/qkd/venv/bin/python
UPDATES="${UPDATES:-5}"
SEEDS="${SEEDS:-42 43 44 45}"

for s in $SEEDS; do
    name="old_seed${s}"
    echo "=== $name (--seed $s) ==="
    rm -rf "$OLD/outputs/diag_${name}"
    (
        cd "$OLD" || exit 1
        timeout 1500 env OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 \
            "$PY" -u scripts/train/train_graph_mappo.py \
                --configs rl_algorithm.yaml train_diag_fast.yaml \
                --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
                --num-updates "$UPDATES" --run-name "diag_${name}" --seed "$s" \
                > "/tmp/diag_${name}.log" 2>&1
    )
    grep -h 'best validation' "/tmp/diag_${name}.log" 2>/dev/null | sed 's/^/  /' || echo "  (无评估)"
done

echo
echo "=== 汇总：update 5 后的确定性评估 ==="
echo "  当前工作区（新模型）：seed42 0.7627 / seed43 0.7414 / seed44 0.7535 / seed45 0.7501"
for s in $SEEDS; do
    printf '  d1930a2（旧模型）seed%-4s' "$s"
    L="/tmp/diag_old_seed${s}.log"
    if grep -q 'best validation' "$L" 2>/dev/null; then
        grep -h 'best validation' "$L" | sed 's/.*success=//; s/ ->.*//'
    else
        echo "(无)"
    fi
done
echo
echo "OLD_SEED_SPREAD_DONE"
