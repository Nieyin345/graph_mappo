#!/usr/bin/env bash
# 第三浪：围绕前两名做组合与延伸。
#
# 前两浪读出来的规律是"降低 actor 更新方差有效"（mini512 +0.0546、vcoef1 +0.0542，
# 几乎并列；而放大步长的 epochs>1 和 lr0.001 都变差）。这一浪就是在验证这条规律，
# 并顺带回答两个具体问题：
#
#   * mini512 和 vcoef1 是**同一机制的不同实现**，还是两个独立收益？
#     -> m512v1 组合：若 ≈ 两者各自的水位，就是同一机制；若显著更高，是独立收益。
#   * "学习率太大"能不能靠"batch 更大"救回来？
#     -> m512lr1e3：若它接近 mini512，说明两者是同一个量的两面（步长/噪声比）。
#
# 另外三个方向性试探：batch 再大一档（mini1024）、critic 权重再翻倍（vcoef2）、
# 关掉优势归一化（m512noadv —— 第一浪里 noadvnorm 是在 bug 未修 + 4 线程下测的，
# 结论作废，值得在干净条件下重测）。
#
# 全部 2 线程（换线程数会确定性改变结果，见 docs/测试规范.md §6）。
#
# 用法（在节点上）：
#     bash .tmp/screen_r3.sh
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
UPDATES="${UPDATES:-30}"
JOBS="${JOBS:-9}"
THREADS=2
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
BASE_CFGS="rl_algorithm.yaml train_diag_fast.yaml"

# `base` 必须在浪里：本浪要按**逐种子**配对比，而配对的另一半必须在同一个节点、
# 同一份代码、同一线程数下跑出来。旧节点的 outputs/r2_base 已经随重装没了，
# 拿日志里的 0.8119 只能对均值，对不了种子。
NAMES=(base m512v1 mini1024 m512ent m512lr4 vcoef2 m512clip02 m512noadv m512lr1e3)

write_yaml() {
    if ! grep -qxF 'configs/var2_*.yaml' .git/info/exclude 2>/dev/null; then
        printf 'configs/var2_*.yaml\n' >> .git/info/exclude
    fi
    case "$1" in
        base)       printf 'train:\n  ppo:\n    epochs: 1\n' > configs/var2_r3_base.yaml ;;
        m512v1)     printf 'train:\n  ppo:\n    minibatch_size: 512\n    value_coef: 1.0\n' > configs/var2_r3_m512v1.yaml ;;
        mini1024)   printf 'train:\n  ppo:\n    minibatch_size: 1024\n' > configs/var2_r3_mini1024.yaml ;;
        m512ent)    printf 'train:\n  ppo:\n    minibatch_size: 512\n    entropy_coef: 0.01\n' > configs/var2_r3_m512ent.yaml ;;
        m512lr4)    printf 'train:\n  ppo:\n    minibatch_size: 512\n  optimizer:\n    actor_lr: 0.0001\n' > configs/var2_r3_m512lr4.yaml ;;
        vcoef2)     printf 'train:\n  ppo:\n    value_coef: 2.0\n' > configs/var2_r3_vcoef2.yaml ;;
        m512clip02) printf 'train:\n  ppo:\n    minibatch_size: 512\n    clip_eps: 0.2\n' > configs/var2_r3_m512clip02.yaml ;;
        m512noadv)  printf 'train:\n  ppo:\n    minibatch_size: 512\n    normalize_advantages: false\n' > configs/var2_r3_m512noadv.yaml ;;
        m512lr1e3)  printf 'train:\n  ppo:\n    minibatch_size: 512\n  optimizer:\n    actor_lr: 0.001\n' > configs/var2_r3_m512lr1e3.yaml ;;
    esac
}

run_one() {
    local name="$1"
    (
        cd "$MAIN" || exit 1
        rm -rf "outputs/r3_${name}"
        timeout 10800 env OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
            "$PY" -u scripts/train/train_graph_mappo.py \
                --configs $BASE_CFGS "var2_r3_${name}.yaml" \
                --checkpoint "$CKPT" \
                --num-updates "$UPDATES" --run-name "r3_${name}" \
                > "/tmp/r3_${name}.log" 2>&1
    )
    echo "  [$name] 结束（退出码 $?）"
}

cd "$MAIN" || exit 1
for n in "${NAMES[@]}"; do write_yaml "$n"; done
echo "第三浪：${NAMES[*]}"
echo "轮数=$UPDATES  并发=$JOBS  线程=$THREADS（钉死）"
echo

n=0
for name in "${NAMES[@]}"; do
    run_one "$name" &
    n=$((n + 1))
    if [ $((n % JOBS)) -eq 0 ]; then wait; fi
done
wait

echo
echo "======== 第三浪结果（对照：base 0.8119、mini512 0.8226、vcoef1 0.8221）========"
"$PY" - <<'PY'
import json
from pathlib import Path

MAIN = Path("/opt/qkd/graph_mappo")
refs = {"base": 0.8119, "mini512": 0.8226, "vcoef1": 0.8221}


def load(name):
    p = MAIN / "outputs" / name / "metrics.jsonl"
    if not p.exists():
        return None, None
    tr, ev = [], []
    for line in p.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "update" in d:
            tr.append(d)
        elif "eval_validation" in d:
            ev.append(d["eval_validation"])
    return tr, ev


rows = []
for p in sorted(MAIN.glob("outputs/r3_*")):
    tr, ev = load(p.name)
    if not ev:
        continue
    rows.append((p.name.replace("r3_", ""), ev[-1]["mean_success_rate"],
                 ev[-1].get("per_seed_success"), tr))
rows.sort(key=lambda r: r[1], reverse=True)

base = load("r3_base")
base_seeds = base[1][-1]["per_seed_success"] if base[1] else None

print(f"{'变体':<12}{'末次验证':>10}{'验证序列':>32}{'对base':>14}{'t':>7}")
for name, last, seeds, tr in rows:
    series = " ".join(f"{e['mean_success_rate']:.3f}" for e in load(f"r3_{name}")[1])
    if base_seeds and seeds:
        n = min(len(base_seeds), len(seeds))
        diffs = [seeds[i] - base_seeds[i] for i in range(n)]
        md = sum(diffs) / n
        var = sum((d - md) ** 2 for d in diffs) / (n - 1)
        se = (var / n) ** 0.5
        ds = f"{md:+.4f}±{se:.4f}"
        ts = f"{md / se if se else float('nan'):+.2f}"
    else:
        ds = ts = "—"
    print(f"{name:<12}{last:>10.4f}{series:>32}{ds:>14}{ts:>7}")
print()
print("参照水位：base " + " ".join(f"{k}={v}" for k, v in refs.items()))
print("SCREEN_R3_DONE")
PY
