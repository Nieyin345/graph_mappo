#!/usr/bin/env bash
# 第六浪 B 线：真实配置 —— 信用视野与养链激励探针（与 A 线 screen_r6.sh 并发）。
#
# 依据（real_base 的 rollout_debug 分解，.tmp/check_reward_components.py）：
#   奖励几乎纯稀疏：served ~102%，failed 罚 1-3%，storage/keep_active ~0.1%
#   （比 served 小 700 倍），dense=0（dense_enabled: false）、waiting=0。
#   base 退化签名：激活边 59.3->54.0（撤边）、QKP 利用率 0.099->0.082、
#   积压 +17%、失败 +15%，served 持平 —— 策略在学"少激活"：
#   撤边当步零成本，而养链路的收益要等很久之后的送达，γ=0.99（视野~100 步）
#   记不到账，storage/keep_active 又小到可以忽略。
#
# 两臂（30 轮，与 real_base 逐种子配对；THREADS=2 钉死）：
#   g999    train.gamma 0.99 -> 0.999 —— 信用视野 ~100 步 -> ~1000 步；
#           若退化消失（末次 >= 0.70），支持"视野不够"假说。
#   stor10  reward.storage_reward_weight 0.5 -> 5.0 —— 10 倍养链激励；
#           若撤边停止（激活边不降）、失败不升，支持"撤边无成本"假说。
#
# 用法（在节点上）：nohup bash .tmp/screen_r6b.sh > /tmp/screen_r6b.out 2>&1 &
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
THREADS=2                     # 钉死
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
BASE_CFGS="rl_algorithm.yaml train_full_rl.yaml"

NAMES=(g999 stor10)
UPDATES_g999=30
UPDATES_stor10=30

updates_for() { eval "echo \${UPDATES_$1:-30}"; }

write_yaml() {
    if ! grep -qxF 'configs/var3_*.yaml' .git/info/exclude 2>/dev/null; then
        printf 'configs/var3_*.yaml\n' >> .git/info/exclude
    fi
    case "$1" in
        g999)   printf 'train:\n  gamma: 0.999\n' > configs/var3_g999.yaml ;;
        stor10) printf 'reward:\n  storage_reward_weight: 5.0\n' > configs/var3_stor10.yaml ;;
    esac
}

run_one() {
    local name="$1"
    local n=$(updates_for "$name")
    local timeout_s=$((n * 800 + 3600))
    (
        cd "$MAIN" || exit 1
        rm -rf "outputs/r6_${name}"
        timeout "$timeout_s" env OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
            "$PY" -u scripts/train/train_graph_mappo.py \
                --configs $BASE_CFGS "var3_${name}.yaml" \
                --checkpoint "$CKPT" \
                --num-updates "$n" --run-name "r6_${name}" \
                > "/tmp/r6_${name}.log" 2>&1
    )
    echo "  [$name] 结束（退出码 $?）"
}

cd "$MAIN" || exit 1
for n in "${NAMES[@]}"; do write_yaml "$n"; done

echo "第六浪 B 线：${NAMES[*]}"
echo "轮数=$(updates_for g999)/$(updates_for stor10)  线程=$THREADS（钉死）"
echo

for name in "${NAMES[@]}"; do run_one "$name" & done
wait

echo
echo "======== 结果 ========"
"$PY" - <<'PY'
import json
from pathlib import Path

MAIN = Path("/opt/qkd/graph_mappo")


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


runs = ("r6_g999", "r6_stor10", "real_base")
res = {r: load(r) for r in runs}
for r in runs:
    tr, ev = res[r]
    if not ev:
        print(f"{r:<12} (没有验证记录)")
        continue
    seq = " ".join(f"{e['mean_success_rate']:.4f}" for e in ev)
    print(f"{r:<12} evals={len(ev)}  {seq}")

ctrl = res["real_base"][1]
if ctrl:
    print()
    print("逐种子配对 vs real_base（末次共同 eval；|t|>=2 才算分得开）：")
    for r in ("r6_g999", "r6_stor10"):
        ev = res[r][1]
        if not ev:
            continue
        k = min(len(ev), len(ctrl))
        md, se, n = paired(ev[k - 1]["per_seed_success"], ctrl[k - 1]["per_seed_success"])
        t = md / se if se else float("nan")
        print(f"  {r:<10} @update{5 * k:<3} {md:+.4f} ± {se:.4f}  t={t:+.2f}")

# stor10 的物理签名：激活边是否还在撤
for r in ("r6_stor10", "r6_g999"):
    tr = res[r][0]
    if len(tr) >= 9:
        seg = len(tr) // 3
        a = sum(x["mean_activated_edges"] for x in tr[:seg]) / seg
        b = sum(x["mean_activated_edges"] for x in tr[-seg:]) / seg
        print(f"  {r:<10} 激活边 前1/3={a:.1f} 后1/3={b:.1f}（real_base 59.3->54.0）")
print("SCREEN_R6B_DONE")
PY
