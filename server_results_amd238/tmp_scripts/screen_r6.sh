#!/usr/bin/env bash
# 第六浪 A 线：真实配置 —— base 退化定位 + vcoef1 平台复验。
#
# 已知事实（real 三臂 30 轮，.tmp/m/real_*.jsonl）：
#   base    验证 0.701->0.622 一路退化（落地阻断项，比排名更优先）；
#   vcoef1  0.6839 唯一胜者（+6.14 点 t=2.49 刚过线）。
# 训练侧签名（.tmp/check_real_base.py）：base 的 KL 极小(0.001-0.004)、
#   熵不塌(3.8->4.0)、mean_abs_advantage 1.08->0.32 单调收缩、
#   value_return_corr ->0.9、train 成功率缓降(0.873->0.856)。
#   → 主假说 H1：critic 把 return 学得太好，真优势趋零，归一化把
#     噪声放大到单位尺度，策略在噪声方向小步随机游走，漂移出好区域。
#
# 三臂（THREADS=2 钉死，见 docs/测试规范.md §6）：
#   vcoef1_long  value_coef 1.0 拉长 90 轮 —— 唯一胜者是平台还是继续涨；
#                前 30 轮应逐位复现 real_vcoef1（同机同线程确定性）。
#   lowv025      value_coef 0.25 —— 弱 critic 让优势变大；若退化消失
#                （末次 >= 0.70），支持 H1。
#   base_lr1e4   actor_lr 1e-4 —— 小步长减慢随机游走；若退化变缓，
#                支持 H1。
#
# 对照组：outputs/real_base、outputs/real_vcoef1（服务器上已有）。
# 每臂 30 轮 ≈ 5.2h；vcoef1_long 90 轮 ≈ 15.6h。三臂并发 ~12GB，56 核吃得下。
#
# 用法（在节点上）：nohup bash .tmp/screen_r6.sh > /tmp/screen_r6.out 2>&1 &
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
THREADS=2                     # 钉死
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
BASE_CFGS="rl_algorithm.yaml train_full_rl.yaml"

NAMES=(vcoef1_long lowv025 base_lr1e4)
UPDATES_vcoef1_long=90
UPDATES_lowv025=30
UPDATES_base_lr1e4=30

updates_for() { eval "echo \${UPDATES_$1:-30}"; }

write_yaml() {
    if ! grep -qxF 'configs/var3_*.yaml' .git/info/exclude 2>/dev/null; then
        printf 'configs/var3_*.yaml\n' >> .git/info/exclude
    fi
    case "$1" in
        vcoef1_long) printf 'train:\n  ppo:\n    value_coef: 1.0\n' > configs/var3_vcoef1_long.yaml ;;
        lowv025)     printf 'train:\n  ppo:\n    value_coef: 0.25\n' > configs/var3_lowv025.yaml ;;
        base_lr1e4)  printf 'train:\n  optimizer:\n    actor_lr: 0.0001\n' > configs/var3_base_lr1e4.yaml ;;
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

echo "第六浪 A 线：${NAMES[*]}"
echo "轮数=$(updates_for vcoef1_long)/$(updates_for lowv025)/$(updates_for base_lr1e4)  线程=$THREADS（钉死）"
echo "配置：$BASE_CFGS + 各自 var3 变体；对照组 real_base / real_vcoef1"
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


runs = ("r6_vcoef1_long", "r6_lowv025", "r6_base_lr1e4", "real_base", "real_vcoef1")
res = {r: load(r) for r in runs}
for r in runs:
    tr, ev = res[r]
    if not ev:
        print(f"{r:<16} (没有验证记录)")
        continue
    seq = " ".join(f"{e['mean_success_rate']:.4f}" for e in ev)
    print(f"{r:<16} evals={len(ev)}  {seq}")

print()
print("逐种子配对（末次共同 eval；正数=新臂更好；|t|>=2 才算分得开）：")
ctrl = res["real_base"][1]
if ctrl:
    for r in ("r6_lowv025", "r6_base_lr1e4", "r6_vcoef1_long"):
        ev = res[r][1]
        if not ev:
            continue
        k = min(len(ev), len(ctrl))
        md, se, n = paired(ev[k - 1]["per_seed_success"], ctrl[k - 1]["per_seed_success"])
        t = md / se if se else float("nan")
        print(f"  {r:<16} vs real_base @update{5 * k:<3} {md:+.4f} ± {se:.4f}  t={t:+.2f}")

vc = res["real_vcoef1"][1]
lv = res["r6_vcoef1_long"][1]
if vc and lv:
    k = min(len(vc), len(lv))
    same = all(
        abs(a["mean_success_rate"] - b["mean_success_rate"]) < 1e-9
        for a, b in zip(lv[:k], vc[:k])
    )
    print(f"  vcoef1_long 前 {k} 次 eval 与 real_vcoef1 逐位复现: {same}")
    tail = [e["mean_success_rate"] for e in lv[-6:]]
    print(f"  vcoef1_long 尾部6次均值 {sum(tail) / len(tail):.4f}（vcoef1 30轮末次 {vc[-1]['mean_success_rate']:.4f}）")
print("SCREEN_R6_DONE")
PY
