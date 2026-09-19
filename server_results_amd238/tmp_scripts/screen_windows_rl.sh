#!/usr/bin/env bash
# 让**策略**（BC 权重或某个训练后的 checkpoint）在各窗口上跑，和专家做差中之差。
#
# 已知专家在 12 种子 / 240 步下：留出(330-365) 0.7359、训练中段(100-135) 0.7763、
# 训练后段(200-235) 0.7756、训练前段(0-35) 0.7242。留出窗口确实略难（-0.040）。
#
# 现在要回答的是：策略相对**专家**的差距，在训练窗口和留出窗口上是不是一样大？
#   * 训练窗口追平专家、只在留出窗口掉 -> 泛化问题（训练分布覆盖不到留出期）。
#   * 各窗口都低差不多的量         -> 纯粹的能力问题。
#
# 用法（在节点上）：
#     bash .tmp/screen_windows_rl.sh                                   # BC 权重
#     bash .tmp/screen_windows_rl.sh outputs/screen_noadvnorm/checkpoint_final.pt  noadvnorm
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
CKPT="${1:-outputs/supervised_pg_phased/supervised_pg_phased_latest.pt}"
TAG="${2:-bc}"
# run_baselines.py 的 --seeds 是逗号列表，不认 "7-18" 这种区间写法
# （eval_expert.py 认，两者不一样，别混）。专家那个窗口对照用的就是 7-18，
# 所以这里必须展开成同一批种子，配对才有意义。
SEEDS="${SEEDS:-7,8,9,10,11,12,13,14,15,16,17,18}"
EPISODES="${EPISODES:-12}"
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
    # 必须显式写步数！run_baselines.py 是从这里取 episode_steps 的，
    # 而窗口配置里不写就回落到默认的 1440，和专家那组（eval_expert.py
    # 传 --steps 240）就不是同一个协议了 —— 第一版就是这么跑错的，
    # 结果里 meta.episode_steps 显示 1440 才发现。
    episode_steps: ${STEPS}
    episodes: ${EPISODES}
YAML
done

echo "策略 = $CKPT  标识 = $TAG  种子 = $SEEDS  episodes = $EPISODES"
echo

for spec in "${WINDOWS[@]}"; do
    name="${spec%%:*}"
    rest="${spec#*:}"
    start="${rest%%:*}"
    end="${rest##*:}"
    echo "=== $name  第 $start-$end 天 ==="
    OMP_NUM_THREADS=8 "$PY" -u scripts/baselines/run_baselines.py \
        --config "configs/var_win_${name}.yaml" \
        --episodes "$EPISODES" --seeds "$SEEDS" \
        --out "outputs/eval/win_${TAG}_${name}" \
        --policies __none__ \
        --rl-checkpoint "$CKPT" --rl-name "$TAG" 2>&1 \
        | grep -Ei "success|mean|policy|episode" | tail -n 6
    echo
done

echo "=== 汇总 ==="
"$PY" - "$TAG" "${WINDOWS[@]}" <<'PY'
import json, sys
from pathlib import Path

tag = sys.argv[1]
out = {}
for spec in sys.argv[2:]:
    name = spec.split(":")[0]
    p = Path(f"outputs/eval/win_{tag}_{name}/summary.json")
    if not p.exists():
        print(f"  {name:<14}(缺 summary.json)")
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    pol = (d.get("policies") or {}).get(tag)
    if pol is None:
        print(f"  {name:<14}(summary 里没有策略 {tag})")
        continue
    log = (pol.get("runs") or [{}])[-1].get("episode_log", [])
    seeds = [e.get("seed") for e in log]
    vals = [e.get("success_rate") for e in log]
    print(f"  {name:<14}{pol.get('success_rate_mean', float('nan')):>10.4f}"
          f"   （{len(vals)} 个种子）")
    out[name] = {"seeds": seeds, "success": vals,
                 "mean": pol.get("success_rate_mean")}

Path(f"outputs/eval/win_{tag}_perseed.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print(f"  已写 outputs/eval/win_{tag}_perseed.json（供按种子配对）")
PY
echo
echo "参照（专家，同 12 种子 240 步）：heldout 0.7359 / late_train 0.7756 / mid_train 0.7763 / early_train 0.7242"
echo "WINDOW_RL_DONE"
