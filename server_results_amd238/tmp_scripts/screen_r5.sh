#!/usr/bin/env bash
# 第五浪：验证 relay_importance 窗口放开（3→8 跳、衰减 0.25→0.8）对 RL 的作用。
#
# 依据（见 docs/训练诊断记录.md「探针 Q」「窗口放开验证」两节）：
#   * 探针 Q：旧窗口 3 跳让 72.47% 的（请求,槽）需求路径一跳都拿不到 importance 分
#     （诊断场景需求路径平均 6.2 跳），得分边每槽仅 1.6 条 → 特征列与 dense 奖励
#     "看不见"长需求路径。
#   * 本地专家档验证：放开后 imp 覆盖率 7.09%→50.78%，专家量加权成功率
#     0.7801→0.7886（12 种子 9-10 个为正）。
#
# 本浪只跑 RL —— 专家侧的窗口参数是 GreedyRelayDiffusionPolicyV3 的实例默认值
# （不走 config），所以这里用 **config 覆盖**（不动全局 features.yaml），
# 既能隔离实验，也不影响节点上正在跑的 r4/real 读到的配置。
#
#   impfix          窗口放开，其余与 r4 的 base 相同（epochs 1）
#   impfix_ent      窗口放开 + entropy_coef 0.01
#   impfix_ent_ep2  窗口放开 + entropy_coef 0.01 + epochs 2（= r4 胜者配置 + 窗口修复）
#
# 对照：r4 的 base / ent / ent_ep2（旧窗口跑的），逐种子配对。r4 结论：
#   ent_ep2 0.8336 (+0.0384, t=+7.24) > ent_c128 0.8297 > ent 0.8236 > base 0.7952
# 所以最关键的一行是 impfix_ent_ep2 vs ent_ep2 —— 窗口修复在最强配置上还有没有增量。
#
# 全部 2 线程（换线程数会确定性改变结果，见 docs/测试规范.md §6）。
#
# 用法（在节点上，等 r4 跑完后）：
#     bash .tmp/screen_r5.sh
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
UPDATES="${UPDATES:-90}"
JOBS="${JOBS:-3}"
THREADS=2
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
BASE_CFGS="rl_algorithm.yaml train_diag_fast.yaml"

NAMES=(impfix impfix_ent impfix_ent_ep2)

IMP='features:\n  edge:\n    relay_importance:\n      max_path_links: 8\n      hop_decay_factor: 0.8\n'

write_yaml() {
    if ! grep -qxF 'configs/var2_*.yaml' .git/info/exclude 2>/dev/null; then
        printf 'configs/var2_*.yaml\n' >> .git/info/exclude
    fi
    case "$1" in
        impfix)
            printf "$IMP" > configs/var2_r5_impfix.yaml ;;
        impfix_ent)
            printf "${IMP}train:\n  ppo:\n    entropy_coef: 0.01\n" \
                > configs/var2_r5_impfix_ent.yaml ;;
        impfix_ent_ep2)
            printf "${IMP}train:\n  ppo:\n    entropy_coef: 0.01\n    epochs: 2\n" \
                > configs/var2_r5_impfix_ent_ep2.yaml ;;
    esac
}

run_one() {
    local name="$1"
    (
        cd "$MAIN" || exit 1
        rm -rf "outputs/r5_${name}"
        timeout 21600 env OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
            "$PY" -u scripts/train/train_graph_mappo.py \
                --configs $BASE_CFGS "var2_r5_${name}.yaml" \
                --checkpoint "$CKPT" \
                --num-updates "$UPDATES" --run-name "r5_${name}" \
                > "/tmp/r5_${name}.log" 2>&1
    )
    echo "  [$name] 结束（退出码 $?）"
}

cd "$MAIN" || exit 1
for n in "${NAMES[@]}"; do write_yaml "$n"; done
echo "第五浪（窗口放开）：${NAMES[*]}"
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
echo "======== 第五浪结果（vs r4 的旧窗口臂逐种子配对）========"
"$PY" - <<'PY'
import json
from pathlib import Path

MAIN = Path("/opt/qkd/graph_mappo")


def last_eval(prefix, name):
    p = MAIN / "outputs" / f"{prefix}_{name}" / "metrics.jsonl"
    if not p.exists():
        return None
    last = None
    for line in p.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "eval_validation" in d:
            last = d["eval_validation"]
    return last


def paired(a, b):
    n = min(len(a), len(b))
    diffs = [a[i] - b[i] for i in range(n)]
    md = sum(diffs) / n
    var = sum((d - md) ** 2 for d in diffs) / (n - 1)
    se = (var / n) ** 0.5
    return md, se, (md / se if se > 0 else float("nan"))


refs = {n: last_eval("r4", n) for n in ("base", "ent", "ent_ep2")}
print("参照（r4，旧窗口 3/0.25）末次验证：")
for n, ev in refs.items():
    if ev:
        print(f"  r4_{n:<10} {ev['mean_success_rate']:.4f}"
              f"  ({len(ev['per_seed_success'])} 种子)")
print()
for name in ("impfix", "impfix_ent", "impfix_ent_ep2"):
    ev = last_eval("r5", name)
    if not ev:
        print(f"  r5_{name}: 无验证记录")
        continue
    line = f"  r5_{name:<14} {ev['mean_success_rate']:.4f}"
    for rn, rev in refs.items():
        if not rev:
            continue
        md, se, t = paired(ev["per_seed_success"], rev["per_seed_success"])
        line += f"   vs r4_{rn}: {md:+.4f}±{se:.4f} (t={t:+.2f})"
    print(line)
print()
print("判据 |t|>=2；最关键的是 impfix_ent_ep2 vs r4_ent_ep2（窗口修复的增量）。")
PY
echo SCREEN_R5_DONE