#!/usr/bin/env bash
# 第六浪（amd238 重跑版）：clnode118 上的原始第六浪随公钥失效丢失，
# 只抢救回 eval@5/@10 中期快照。本脚本在新节点上一次跑全 A/B 两线。
#
# 预算约束（用户）：总时长 ≤5 小时 → bootstrap ~40min + 5 臂 × 20 轮 ≈ 3.5h。
# 每臂 20 轮（eval 每 5 轮一次，共 4 个读数），足够看退化/改善的方向与斜率，
# 但**结论判读按"末次共同 eval 配对 + |t|>=2"**，见 docs/测试规范.md。
#
# 五臂（THREADS=2 钉死）：
#   base        锚（新机器 BLAS 有 ~0.003 微差，配对必须同机同批重跑）
#   lowv025     value_coef 0.25     —— H1：弱 critic 让优势变大，退化应消失
#   base_lr1e4  actor_lr 1e-4       —— H1：小步长减慢随机游走
#   g999        gamma 0.999         —— 信用视野 100 步 -> 1000 步（养链路记账）
#   stor10      storage 激励 x10    —— 让"囤货"有当步回报，撤边不再免费
#
# 用法（在节点上）：nohup bash .tmp/screen_r6c.sh > /tmp/screen_r6c.out 2>&1 &
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
THREADS=2                     # 钉死，见 docs/测试规范.md §6
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
BASE_CFGS="rl_algorithm.yaml train_full_rl.yaml"
UPDATES="${UPDATES:-20}"

NAMES=(base lowv025 base_lr1e4 g999 stor10)

write_yaml() {
    if ! grep -qxF 'configs/var4_*.yaml' .git/info/exclude 2>/dev/null; then
        printf 'configs/var4_*.yaml\n' >> .git/info/exclude
    fi
    case "$1" in
        base)        printf 'train:\n  ppo:\n    epochs: 1\n' > configs/var4_base.yaml ;;
        lowv025)     printf 'train:\n  ppo:\n    value_coef: 0.25\n' > configs/var4_lowv025.yaml ;;
        base_lr1e4)  printf 'train:\n  optimizer:\n    actor_lr: 0.0001\n' > configs/var4_base_lr1e4.yaml ;;
        g999)        printf 'train:\n  gamma: 0.999\n' > configs/var4_g999.yaml ;;
        stor10)      printf 'reward:\n  storage_reward_weight: 5.0\n' > configs/var4_stor10.yaml ;;
    esac
}

run_one() {
    local name="$1"
    local timeout_s=$((UPDATES * 900 + 2400))
    (
        cd "$MAIN" || exit 1
        rm -rf "outputs/r6_${name}"
        timeout "$timeout_s" env OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
            "$PY" -u scripts/train/train_graph_mappo.py \
                --configs $BASE_CFGS "var4_${name}.yaml" \
                --checkpoint "$CKPT" \
                --num-updates "$UPDATES" --run-name "r6_${name}" \
                > "/tmp/r6_${name}.log" 2>&1
    )
    echo "  [$name] 结束（退出码 $?）"
}

cd "$MAIN" || exit 1
for n in "${NAMES[@]}"; do write_yaml "$n"; done

echo "第六浪（amd238）：${NAMES[*]}"
echo "轮数=$UPDATES  线程=$THREADS（钉死）"
echo "配置：$BASE_CFGS + 各自 var4 变体"
echo

for name in "${NAMES[@]}"; do run_one "$name" & done
wait

echo
echo "======== 结果 ========"
"$PY" - <<'PY'
import json
from pathlib import Path

MAIN = Path("/opt/qkd/graph_mappo")
ARMS = ("r6_base", "r6_lowv025", "r6_base_lr1e4", "r6_g999", "r6_stor10")


def load(run):
    p = MAIN / "outputs" / run / "metrics.jsonl"
    if not p.exists():
        return [], []
    train, evals = [], []
    for line in p.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "update" in d:
            train.append(d)
        elif "eval_validation" in d:
            evals.append(d["eval_validation"])
    return train, evals


def paired(a, b):
    n = min(len(a), len(b))
    if n < 2:
        return float("nan"), float("nan"), n
    diffs = [a[i] - b[i] for i in range(n)]
    md = sum(diffs) / n
    var = sum((d - md) ** 2 for d in diffs) / (n - 1)
    return md, (var / n) ** 0.5, n


res = {r: load(r) for r in ARMS}
print("== eval 序列（每 5 轮一个读数）==")
for r in ARMS:
    ev = res[r][1]
    if not ev:
        print(f"{r:<16} (没有验证记录)")
        continue
    seq = " ".join(f"{e['mean_success_rate']:.4f}" for e in ev)
    print(f"{r:<16} evals={len(ev)}  {seq}")

ctrl = res["r6_base"][1]
if ctrl:
    print()
    print("== 逐种子配对 vs r6_base（末次共同 eval；正数=更好；|t|>=2 才算分得开）==")
    for r in ARMS[1:]:
        ev = res[r][1]
        if not ev:
            continue
        k = min(len(ev), len(ctrl))
        md, se, n = paired(ev[k - 1]["per_seed_success"], ctrl[k - 1]["per_seed_success"])
        t = md / se if se else float("nan")
        print(f"  {r:<16} vs r6_base @update{5 * k:<3} {md:+.4f} ± {se:.4f}  t={t:+.2f}"
              f"  {'显著' if abs(t) >= 2 else '不显著'}")
print("SCREEN_R6C_DONE")
PY
