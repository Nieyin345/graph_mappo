#!/usr/bin/env bash
# 把「前向差异」和「训练放大」分开。
#
# 做法：拿同一份权重，分别用旧模型和新模型跑同一条确定性评估。
#   - 若两边数字相同 -> 前向等价，之前 update 5 的 0.06 差距只能来自训练中的数值放大。
#   - 若两边差 ~0.06 -> 新模型的前向本身就改变了决策，与训练无关，是真 bug。
#
# 喂三份权重，因为只看 BC 权重不够：
#   BC          —— 起点，update 0。前向若在这里就分叉，说明与训练无关。
#   g2_final    —— **旧模型**训练 5 轮后的权重（评估 0.8174）。
#                  用它喂新模型，是最直接的隔离：权重一模一样，只换前向。
#   g1a_final   —— 新模型训练 5 轮后的权重（评估 0.7572），反向对照。
set -u

MAIN=/opt/qkd/graph_mappo
OLD=/opt/qkd/eval_old_model
PY=/opt/qkd/venv/bin/python
BASE=d1930a2
SCRIPT="$MAIN/.tmp/eval_same_weights.py"

CKPTS=(
    "BC|$MAIN/outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
    "g2_final|$MAIN/../bisect_group2/outputs/diag_group2/checkpoint_final.pt"
    "g1a_final|$MAIN/../bisect_g1a/outputs/diag_g1a/checkpoint_final.pt"
)

# 旧模型的干净 worktree。独立目录，不碰正在跑的 bisect worktree。
git -C "$MAIN" worktree remove --force "$OLD" 2>/dev/null || rm -rf "$OLD"
git -C "$MAIN" worktree add --detach "$OLD" "$BASE" >/dev/null || { echo "worktree add 失败"; exit 1; }
ln -sfn "$MAIN/dataset" "$OLD/dataset"
mkdir -p "$OLD/outputs"
ln -sfn "$MAIN/outputs/supervised_pg_phased" "$OLD/outputs/supervised_pg_phased"
mkdir -p "$OLD/.tmp"
cp "$SCRIPT" "$OLD/.tmp/eval_same_weights.py"

run_one() {
    local label="$1" repo="$2" ckpt="$3" script="$4"
    echo "--- $label ---"
    if [ ! -s "$ckpt" ]; then echo "  (缺 checkpoint: $ckpt)"; return; fi
    OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 timeout 1800 \
        "$PY" -u "$script" --repo "$repo" --checkpoint "$ckpt" 2>&1 \
        | grep -E 'success_rate|mean_reward|served_keys|episodes|Error|error|Traceback' \
        | sed 's/^/  /'
}

for entry in "${CKPTS[@]}"; do
    name="${entry%%|*}"
    ckpt="${entry#*|}"
    echo "================ $name ================"
    run_one "旧模型 (d1930a2)  / $name" "$OLD" "$ckpt" "$OLD/.tmp/eval_same_weights.py"
    run_one "新模型 (工作区)    / $name" "$MAIN" "$ckpt" "$SCRIPT"
    echo
done

echo "SAME_WEIGHTS_DONE"
