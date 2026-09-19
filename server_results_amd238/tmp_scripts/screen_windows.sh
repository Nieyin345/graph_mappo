#!/usr/bin/env bash
# 留出窗口(330-365)是真的更难，还是模型过拟合？
#
# 背景：专家在留出窗口上 0.7082，RL 训练 30 轮后 0.6857 —— 差 2.2 个点。
# 但同时 RL 在**训练窗口**上的 rollout 成功率是 0.86，和专家的 fullday 0.8673 相当。
# 于是有两种读法：
#   (a) 留出期（第 330-365 天，接近年末）本身就比训练期（0-295 天）难 ——
#       卫星几何、天气、需求分布都不同。那 2.2 个点不是策略的错。
#   (b) 留出期难度相当，是策略泛化不出去。
#
# 判法：让**专家**在几个不同窗口上跑同一个协议（12 种子、240 步）。
# 专家没有训练过程，它在各窗口的差异只反映窗口本身的难度。
#   - 专家在 330-365 上也明显更低  -> 支持 (a)，先别怪策略。
#   - 专家在各窗口差不多           -> 支持 (b)，问题在泛化。
#
# eval_expert.py 从配置的 global.validation 读窗口，所以给每个窗口造一个小配置。
# 配置必须落在 configs/ 下（build_config 拼的是 ROOT/configs/<name>），并加进
# 节点的 .git/info/exclude，否则会把工作区弄脏、挡掉下一次推送。
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
SEEDS="${SEEDS:-7-18}"
STEPS="${STEPS:-240}"

WINDOWS=(
    "heldout:330:365"
    "late_train:200:235"
    "mid_train:100:135"
    "early_train:0:35"
)

cd "$MAIN" || exit 1
if ! grep -qxF 'configs/var_*.yaml' .git/info/exclude 2>/dev/null; then
    printf 'configs/var_*.yaml\n' >> .git/info/exclude
fi

for spec in "${WINDOWS[@]}"; do
    name="${spec%%:*}"
    rest="${spec#*:}"
    start="${rest%%:*}"
    end="${rest##*:}"
    cat > "configs/var_win_${name}.yaml" <<YAML
global:
  validation:
    window:
      start_day: ${start}
      end_day: ${end}
YAML
done

for spec in "${WINDOWS[@]}"; do
    name="${spec%%:*}"
    rest="${spec#*:}"
    start="${rest%%:*}"
    end="${rest##*:}"
    echo "=== $name  第 $start-$end 天 ==="
    # --out 必须按窗口分开！eval_expert.py 默认写 expert_<steps>.json，
    # 四个窗口轮着跑会把前三个覆盖掉，只剩最后一个 —— 那就没法按种子配对了。
    OMP_NUM_THREADS=4 "$PY" -u scripts/eval/eval_expert.py \
        --config "configs/var_win_${name}.yaml" \
        --seeds "$SEEDS" --steps "$STEPS" \
        --out "outputs/eval/expert_win_${name}.json" 2>&1 \
        | grep -E "profile:|mean success|failure rate|min |wrote"
    echo
done

echo "WINDOW_SCREEN_DONE"
