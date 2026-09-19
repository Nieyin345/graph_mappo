#!/usr/bin/env bash
# 把诊断场景的结论搬到**真实训练配置**上验证。
#
# 诊断场景（train_diag_fast：240 步、固定第 0 天）的价值只是"算法能不能学"，
# 它自己的注释就写着 Do NOT quote its absolute numbers as results。真在用的是
# train_full_rl：1440 步 rollout、随机日、激活窗 0–295 天；留出验证窗 330–365、
# 15 个种子 100–114、240 步、random_day。
#
# 要回答的问题：诊断场景里读出来的规律（小学习率、大 batch、大 value_coef 更好；
# epochs>1 有害）在真实配置上**排名是否保持**？
#
# 只挑三个代表性配置，不铺开 —— 真实配置每轮约 10 分钟，30 轮 + 验证约 6 小时：
#   base      基线（rl_algorithm 默认）
#   mini512   诊断场景第一名
#   vcoef1    诊断场景第二名（和 mini512 几乎并列）
#
# **必须 2 线程**：换线程数会确定性改变结果（见 docs/测试规范.md §6）。
# 这三个各占 ~4 GB，三个并发完全吃得下，没必要压到极限 —— 这里要的是可比的
# 结果，不是最大吞吐。
#
# 用法（在节点上）：
#     bash .tmp/screen_real.sh
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
UPDATES="${UPDATES:-30}"
THREADS=2                     # 钉死，见文件头
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
# 注意顺序：rl_algorithm 在前，train_full_rl 覆盖 rollout/窗口，变体最后。
BASE_CFGS="rl_algorithm.yaml train_full_rl.yaml"

NAMES=(base mini512 vcoef1)

write_yaml() {
    if ! grep -qxF 'configs/var2_*.yaml' .git/info/exclude 2>/dev/null; then
        printf 'configs/var2_*.yaml\n' >> .git/info/exclude
    fi
    case "$1" in
        base)    printf 'train:\n  ppo:\n    epochs: 1\n' > configs/var2_real_base.yaml ;;
        mini512) printf 'train:\n  ppo:\n    minibatch_size: 512\n' > configs/var2_real_mini512.yaml ;;
        vcoef1)  printf 'train:\n  ppo:\n    value_coef: 1.0\n' > configs/var2_real_vcoef1.yaml ;;
    esac
}

run_one() {
    local name="$1"
    (
        cd "$MAIN" || exit 1
        rm -rf "outputs/real_${name}"
        timeout 43200 env OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
            "$PY" -u scripts/train/train_graph_mappo.py \
                --configs $BASE_CFGS "var2_real_${name}.yaml" \
                --checkpoint "$CKPT" \
                --num-updates "$UPDATES" --run-name "real_${name}" \
                > "/tmp/real_${name}.log" 2>&1
    )
    echo "  [$name] 结束（退出码 $?）"
}

cd "$MAIN" || exit 1
for n in "${NAMES[@]}"; do write_yaml "$n"; done

echo "真实配置筛选：${NAMES[*]}"
echo "轮数=$UPDATES  线程=$THREADS（钉死）"
echo "配置：$BASE_CFGS + 各自变体"
echo

for name in "${NAMES[@]}"; do run_one "$name" & done
wait

echo
echo "======== 结果 ========"
"$PY" - <<'PY'
import json
from pathlib import Path

MAIN = Path("/opt/qkd/graph_mappo")


def load(name):
    p = MAIN / "outputs" / f"real_{name}" / "metrics.jsonl"
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


res = {n: load(n) for n in ("base", "mini512", "vcoef1")}
print(f"{'变体':<10}{'末次验证':>10}{'验证次数':>9}{'熵首->末':>16}{'平均kl':>10}")
for n, (tr, ev) in res.items():
    if not ev:
        print(f"{n:<10}  (没有验证记录)")
        continue
    last = ev[-1]["mean_success_rate"]
    ent = f"{tr[0]['entropy']:.3f}->{tr[-1]['entropy']:.3f}" if tr else "—"
    kl = sum(r.get("kl", 0.0) for r in tr) / len(tr) if tr else float("nan")
    print(f"{n:<10}{last:>10.4f}{len(ev):>9}{ent:>16}{kl:>10.5f}")

base_seeds = res["base"][1][-1]["per_seed_success"] if res["base"][1] else None
if base_seeds:
    print()
    print("相对 base 的逐种子配对差（正数=更好）：")
    for n in ("mini512", "vcoef1"):
        ev = res[n][1]
        if not ev:
            continue
        md, se, k = paired(ev[-1]["per_seed_success"], base_seeds)
        print(f"  {n:<10} {md:+.4f} ± {se:.4f}  t={md / se if se else float('nan'):+.2f}  ({k} 种子)")
print()
print("注意：真实配置的验证是 random_day + 留出窗 330-365，起点不同，")
print("所以这里的绝对值**不能**和诊断场景的 0.81 比；只有排名才可比。")
print("SCREEN_REAL_DONE")
PY
