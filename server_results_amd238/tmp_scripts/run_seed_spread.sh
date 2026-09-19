#!/usr/bin/env bash
# 这个指标本身的分辨率是多少？
#
# 已经确立的事实：
#   1. 训练+评估逐位可复现（ref1 与 ref2 完全相同，包括每轮指标）。
#   2. 新模型的前向在**相同权重**下不改变任何决策 —— 三份 checkpoint（BC、
#      旧模型训出的、新模型训出的）在新旧两版模型下给出六位小数相同的评估。
#      也就是说重构在动作选择层面完全等价，split 出来的 0.06 不是前向 bug。
#   3. 于是 0.06 只能来自训练：前向里 ~1e-6 的浮点重结合（critic 池化换成了
#      分段求和）改变了梯度，5 轮之后权重分叉。
#
# 剩下唯一要回答的问题：0.06 算不算「有意义的差距」？
#
# 测法：同一份代码，只换训练随机种子，跑几个。评估用的 seeds 来自验证配置、
# 不随训练种子变，所以这是同一把尺子量几个不同的策略。
#   - 各种子散开 ~0.06 或更多 -> 0.06 落在自然波动里，「回归」不成立，
#     该先修测量方法，而不是继续找代码 bug。
#   - 各种子都聚在 0.81 附近 -> 0.06 是真的，那个 1e-6 的前向差异确实把训练
#     带到了更差的解，那就让前向保持逐位一致。
#
# 日志写到 /tmp，不落在仓库里：仓库根目录多出来的未跟踪文件会让下一次
# server_sync.sh 的推送被拒（updateInstead 要求工作区干净）。
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
UPDATES="${UPDATES:-5}"
SEEDS="${SEEDS:-43 44 45}"

for s in $SEEDS; do
    name="seed${s}"
    echo "=== $name (--seed $s) ==="
    rm -rf "$MAIN/outputs/diag_${name}"
    (
        cd "$MAIN" || exit 1
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
echo "  seed 42  当前工作区（对照组）  0.7627"
echo "  seed 42  d1930a2 裸跑          0.8174"
for s in $SEEDS; do
    printf '  seed %-4s' "$s"
    if grep -q 'best validation' "/tmp/diag_seed${s}.log" 2>/dev/null; then
        grep -h 'best validation' "/tmp/diag_seed${s}.log" | sed 's/.*success=//; s/ ->.*//'
    else
        echo "(无)"
    fi
done
echo
echo "SEED_SPREAD_DONE"
