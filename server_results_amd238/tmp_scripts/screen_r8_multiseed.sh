#!/usr/bin/env bash
# 多种子复制：按项目自己的规程（docs/测试规范.md §4 ⑦）重测声称的效应。
#
# 规程原文："比较两个训练配置时，训练种子本身是比评估种子更大的方差来源。
# 单种子下的差距不算数 —— 至少 3 个训练种子，比分布。"
# 实测 5 条数据流（--seed 7/42/43/44/45）update 5 的值：0.8174 / 0.7965 /
# 0.7730 / 0.7134 / 0.7591 —— 跨度 0.104。配对后标准误约 0.017，
# **低于约 0.035 的差异测不出来**。
#
# 第六/七浪违反了这条：每臂只跑 1 次训练，报的 ±0.010 只是**评估侧**噪声
# （15 个固定验证 episode 上的配对），真实分辨率是 0.035。更糟的是当时
# rollout 采样 RNG 没播种（提交 d4bd79c 才修），所以每臂实际是一次
# **不受控的随机抽样** —— 这解释了 r7_fix_g999 与 r7_fix_g999_long 在
# 同配置下的 0.037 差异。
#
# 按 0.035 重新判读历史结果：
#   entropy_coef 0.01   +0.0954  → 2.7× 阈值，**仍可信**
#   终止语义修复        -0.0399  → 略高于阈值，方向大概真实（已完成回滚）
#   gamma 0.999         +0.0279  → 低于阈值，**从来不是真效应**
#
# 设计：
#   臂 = base（rl_algorithm.yaml 的 entropy_coef 0.001，即第六/七浪实际跑的）
#      / ent01（entropy_coef 0.01，即调过的 train_mappo.yaml 的值）
#   种子 = 42 43 44（三条已知互异的数据流，见上）
#   → 2 臂 × 3 种子 = 6 次运行，每次 15 轮
#
# 为什么 15 轮：critic 冷启动要 8 轮（前 5 轮优势是噪声），eval_interval=5，
# 所以 15 轮给出 update 5/10/15 三个点，端点在冷启动之后还有 7 轮余量。
#
# 判读：同臂跨种子的离散度 σ_run 才是误差尺度。
#   若 |Δ| < 2√2·σ_run，则该效应在单次 run 下不可分辨 —— 与 §4⑦ 的
#   0.035 阈值互相印证。
#
# 线程数固定 4。注意线程数确定性影响训练结果（2 vs 4 线程差 +0.0176，
# docs/测试规范.md），所以本实验的绝对值不可与第六/七浪（THREADS=2）
# 直接比，只能本实验内部比较 —— 这正是重点。
#
# 用法（在节点上）：nohup bash .tmp/screen_r8_multiseed.sh > /tmp/r8.out 2>&1 &
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
THREADS=4
UPDATES=${UPDATES:-15}
SEEDS=(42 43 44)
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
BASE_CFGS="rl_algorithm.yaml train_full_rl.yaml"

write_yaml() {
    if ! grep -qxF 'configs/var8_*.yaml' .git/info/exclude 2>/dev/null; then
        printf 'configs/var8_*.yaml\n' >> .git/info/exclude
    fi
    case "$1" in
        base)  rm -f configs/var8_base.yaml ;;
        ent01) printf 'train:\n  ppo:\n    entropy_coef: 0.01\n' > configs/var8_ent01.yaml ;;
    esac
}

cd "$MAIN" || exit 1

if pgrep -f "train_graph_mappo[.]py" >/dev/null 2>&1; then
    echo "ERROR: 已有训练在跑，本轮会被污染。" >&2; exit 1
fi

for arm in base ent01; do write_yaml "$arm"; done

echo "多种子复制：2 臂 × ${#SEEDS[@]} 种子 × $UPDATES 轮   线程=$THREADS"
echo "代码版本：$(git log --oneline -1)"
echo

# 顺序跑（并发会互相争抢 CPU，而这正是要消除的噪声源之一）
for arm in base ent01; do
    for seed in "${SEEDS[@]}"; do
        name="${arm}_s${seed}"
        extra=""
        [ -f "configs/var8_${arm}.yaml" ] && extra="var8_${arm}.yaml"
        echo "--- $name ---"
        rm -rf "outputs/r8_${name}"
        timeout $(( UPDATES * 1200 + 3600 )) \
            env OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
            "$PY" -u scripts/train/train_graph_mappo.py \
                --configs $BASE_CFGS $extra \
                --checkpoint "$CKPT" \
                --seed "$seed" \
                --num-updates "$UPDATES" --run-name "r8_${name}" \
                > "/tmp/r8_${name}.log" 2>&1
        echo "    退出码 $?"
    done
done

echo
echo "======== 结果 ========"
"$PY" - <<'PY'
import json, statistics as st
from pathlib import Path
from itertools import combinations

MAIN = Path("/opt/qkd/graph_mappo")
SEEDS = [42, 43, 44]
ARMS = ["base", "ent01"]


def evals(run):
    p = MAIN / "outputs" / run / "metrics.jsonl"
    out = []
    if not p.exists():
        return out
    for line in p.open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and "eval_validation" in d:
            out.append(float(d["eval_validation"]["mean_success_rate"]))
    return out


data = {}
for arm in ARMS:
    for seed in SEEDS:
        data[(arm, seed)] = evals(f"r8_{arm}_s{seed}")

print("=== 每次运行的 eval 轨迹 ===")
for (arm, seed), seq in data.items():
    s = " ".join(f"{v:.4f}" for v in seq)
    print(f"  {arm:<7} s{seed}  n={len(seq):<2} {s}")

print()
print("=== 末次 eval（按臂分组）===")
groups = {}
for arm in ARMS:
    vals = [seq[-1] for seed in SEEDS if (seq := data[(arm, seed)])]
    groups[arm] = vals
    if not vals:
        print(f"  {arm}: 无数据")
        continue
    m = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    print(f"  {arm:<7} n={len(vals)}  mean={m:.4f}  sd={sd:.4f}  {[round(v,4) for v in vals]}")

print()
base_v, ent_v = groups.get("base", []), groups.get("ent01", [])
if base_v and ent_v:
    d = st.mean(ent_v) - st.mean(base_v)
    print(f"Δ(ent01 - base) = {d:+.4f}")
    print()
    # 跨种子离散度才是可信的误差尺度
    pooled = []
    for v in (base_v, ent_v):
        if len(v) > 1:
            pooled.append(st.stdev(v))
    if pooled:
        sd_run = st.mean(pooled)
        print(f"单次 run 的方差 σ_run ≈ {sd_run:.4f}（同臂跨种子）")
        print(f"  → 单次 run 的 A/B 差异噪声 ≈ {sd_run * 2**0.5:.4f} (√2·σ)")
        print(f"  → 要 2σ 分辨需要 |Δ| > {sd_run * 2**0.5 * 2:.4f}")
        print(f"  → 实测 |Δ| = {abs(d):.4f}  " +
              ("可分辨" if abs(d) > sd_run * 2**0.5 * 2 else "**不可分辨（落在噪声内）**"))
    print()
    # 同一 config 同一种子跨运行应当逐位相同 —— 这里没有重复跑，
    # 用不同种子之间的离散度作为 run 方差的下界估计。
    print("注：本实验每个 (臂,种子) 只跑一次；σ_run 由不同种子间的离散度估计，")
    print("    包含种子效应本身。严格分离需要同种子重复跑（见 verify_repro.sh）。")
PY
