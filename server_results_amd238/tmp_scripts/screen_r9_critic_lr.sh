#!/usr/bin/env bash
# 第九浪：针对 H1 根因——降低 critic 学习率。
#
# 依据（r8_base_s42 干净代码上的 H1 复现）：
#   u      |A|    corr   V_std  R_std
#   1   1.0817   0.215   0.027  0.395
#   5   0.5608   0.862   0.650  0.892
# |A| 单调跌到 0.52×，corr 爬到 0.862，V_std 追平 R_std。
# → critic 拟合回报越好，真实优势越接近 0，per-minibatch 归一化把噪声
#   放大回单位尺度。训练越久，喂给 actor 的越是纯噪声。
#
# 为什么动 critic_lr 而不是继续加 entropy_coef：
#   entropy_coef 是对抗**症状**（维持熵），不阻止优势塌陷 —— r7 的 ent 臂
#   优势照样塌。而 critic_lr 是**驱动项**：critic_lr 0.001 是 actor_lr
#   0.0003 的 3.3 倍，critic 学得快得多，正是"拟合太好"的直接原因。
#
# 臂：
#   base  —— 复用 r8_base_s42/43/44（同代码、同线程、同种子、15 轮）。
#            **不重跑**：它们就是本实验的对照组，省 3 次运行 ≈ 2.2 小时。
#   clr3  —— critic_lr 0.0003（与 actor_lr 齐平）。3 个种子。
#
# 判读：同 r8 的多种子协议 —— 同臂跨种子离散度 σ_run 才是误差尺度，
# |Δ| 要超过 2√2·σ_run 才算分得开（与 docs/测试规范.md §4⑦ 的 0.035 阈值互印证）。
#
# 用法（在节点上）：nohup bash .tmp/screen_r9_critic_lr.sh > /tmp/r9.out 2>&1 &
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
THREADS=4
UPDATES=${UPDATES:-15}
SEEDS=(42 43 44)
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
BASE_CFGS="rl_algorithm.yaml train_full_rl.yaml"
ARM=${ARM:-clr3}
CRITIC_LR=${CRITIC_LR:-0.0003}

cd "$MAIN" || exit 1

if pgrep -f "train_graph_mappo[.]py" >/dev/null 2>&1; then
    echo "ERROR: 已有训练在跑，本轮会被污染。" >&2; exit 1
fi

# 对照组必须存在，否则这次实验没有意义（不重跑 base 是省机时的前提）
for seed in "${SEEDS[@]}"; do
    if [ ! -f "outputs/r8_base_s${seed}/metrics.jsonl" ]; then
        echo "ERROR: 对照组 outputs/r8_base_s${seed} 不存在，先跑完 screen_r8_multiseed.sh。" >&2
        exit 1
    fi
done
echo "对照组 r8_base_s{${SEEDS[*]}} 已就位，复用（不重跑）"
echo

if ! grep -qxF 'configs/var9_*.yaml' .git/info/exclude 2>/dev/null; then
    printf 'configs/var9_*.yaml\n' >> .git/info/exclude
fi
printf 'train:\n  optimizer:\n    critic_lr: %s\n' "$CRITIC_LR" > "configs/var9_${ARM}.yaml"

echo "第九浪：critic_lr -> $CRITIC_LR （臂名 $ARM）"
echo "种子=${SEEDS[*]}  轮数=$UPDATES  线程=$THREADS"
echo "代码版本：$(git log --oneline -1)"
echo

for seed in "${SEEDS[@]}"; do
    name="${ARM}_s${seed}"
    echo "--- $name ---"
    rm -rf "outputs/r9_${name}"
    timeout $(( UPDATES * 1200 + 3600 )) \
        env OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
        "$PY" -u scripts/train/train_graph_mappo.py \
            --configs $BASE_CFGS "configs/var9_${ARM}.yaml" \
            --checkpoint "$CKPT" \
            --seed "$seed" \
            --num-updates "$UPDATES" --run-name "r9_${name}" \
            > "/tmp/r9_${name}.log" 2>&1
    echo "    退出码 $?"
done

echo
echo "======== 结果 ========"
"$PY" - <<'PY'
import json, statistics as st
from pathlib import Path

MAIN = Path("/opt/qkd/graph_mappo")
SEEDS = [42, 43, 44]
# (标签, base 前缀, 处理前缀)
ARMS = [("critic_lr", "r8_base", "r9_clr3")]


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


for label, base_pfx, treat_pfx in ARMS:
    print(f"=== {label} ===")
    base_v, treat_v = [], []
    for seed in SEEDS:
        b, t = evals(f"{base_pfx}_s{seed}"), evals(f"{treat_pfx}_s{seed}")
        bs = " ".join(f"{v:.4f}" for v in b)
        ts = " ".join(f"{v:.4f}" for v in t)
        print(f"  s{seed}  base   n={len(b):<2} {bs}")
        print(f"  s{seed}  treat  n={len(t):<2} {ts}")
        if b:
            base_v.append(b[-1])
        if t:
            treat_v.append(t[-1])

    print()
    if not base_v or not treat_v:
        print("  数据不全，无法判读")
        continue
    mb, mt = st.mean(base_v), st.mean(treat_v)
    print(f"  末次 eval  base  mean={mb:.4f}  {[round(v,4) for v in base_v]}")
    print(f"  末次 eval  treat mean={mt:.4f}  {[round(v,4) for v in treat_v]}")
    d = mt - mb
    print(f"  Δ = {d:+.4f}")
    sds = [st.stdev(v) for v in (base_v, treat_v) if len(v) > 1]
    if sds:
        sd_run = st.mean(sds)
        need = sd_run * 2 ** 0.5 * 2
        print(f"  单次 run 方差 σ_run ≈ {sd_run:.4f}（同臂跨种子）")
        print(f"  2σ 分辨需要 |Δ| > {need:.4f}")
        print(f"  实测 |Δ| = {abs(d):.4f}  " +
              ("**可分辨**" if abs(d) > need else "**不可分辨（落在噪声内）**"))
    print()
    # 逐轮配对（同一 seed 内 base vs treat），看趋势是否一致
    print("  逐轮配对差（treat - base，同 seed）：")
    for seed in SEEDS:
        b, t = evals(f"{base_pfx}_s{seed}"), evals(f"{treat_pfx}_s{seed}")
        n = min(len(b), len(t))
        if n:
            diffs = " ".join(f"{t[i]-b[i]:+.4f}" for i in range(n))
            print(f"    s{seed}: {diffs}")
    print()
PY
