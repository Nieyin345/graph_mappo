#!/usr/bin/env bash
# 第七浪：在**修复了终止语义**的代码上，重测第六浪的两个结论。
#
# 背景（第六浪 screen_r6c 的 t 检验，20 轮，THREADS=2）：
#   r6_g999   vs r6_base  +0.0279 ± 0.0100  t=+2.79  显著  <- 唯一胜者
#   r6_lowv025           -0.0010 ± 0.0096  t=-0.11  不显著
#   r6_base_lr1e4        -0.0041 ± 0.0093  t=-0.45  不显著
#   r6_stor10            +0.0065 ± 0.0095  t=+0.68  不显著
#   且 r6_base 在这台机器上验证分是平的（0.6494->0.6532），
#   并未复现 real_base 的 0.701->0.622 退化 —— 所以问题是"停滞"，不是"漂移"。
#
# 本轮改动（提交 1394b02）：env.step 把 episode_steps 到达报成 truncated
# 而非 terminated，GAE 据此恢复 bootstrap。此前每集末端不 bootstrap，
# 系统性低估晚段状态价值 —— 而中继链路的收益恰在几百步之后，
# gamma=0.999 争的就是这个视野。**所以这个修复可能直接吃掉 g999 的收益。**
#
# 四臂（THREADS=2 钉死，与第六浪同线程数保证可比）：
#   base        锚：修复后的代码 + 原配置。对比 r6_base = 修复本身有没有用。
#   fix_g999    修复 + gamma 0.999。g999 的收益在修复后还在吗？
#   fix_ent     修复 + entropy_coef 0.001->0.01。直接对抗优势塌陷（H1）。
#   fix_g999_long  修复 + gamma 0.999，**60 轮**。g999 在 20 轮末仍在爬
#               （0.6461 0.6626 0.6731 0.6811 单调），20 轮可能截断了曲线。
#
# 判读：末次共同 eval 的逐种子配对，|t|>=2 才算分得开（docs/测试规范.md）。
# 用法（在节点上）：nohup bash .tmp/screen_r7.sh > /tmp/screen_r7.out 2>&1 &
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
THREADS=2
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
BASE_CFGS="rl_algorithm.yaml train_full_rl.yaml"

NAMES=(base fix_g999 fix_ent fix_g999_long)
UPDATES_base=20
UPDATES_fix_g999=20
UPDATES_fix_ent=20
UPDATES_fix_g999_long=60

updates_for() { eval "echo \${UPDATES_$1:-20}"; }

write_yaml() {
    if ! grep -qxF 'configs/var7_*.yaml' .git/info/exclude 2>/dev/null; then
        printf 'configs/var7_*.yaml\n' >> .git/info/exclude
    fi
    case "$1" in
        base)          rm -f configs/var7_base.yaml ;;   # 原配置，无覆盖
        fix_g999)      printf 'train:\n  gamma: 0.999\n' > configs/var7_fix_g999.yaml ;;
        fix_ent)       printf 'train:\n  ppo:\n    entropy_coef: 0.01\n' > configs/var7_fix_ent.yaml ;;
        fix_g999_long) printf 'train:\n  gamma: 0.999\n' > configs/var7_fix_g999_long.yaml ;;
    esac
}

run_one() {
    local name="$1"
    local n=$(updates_for "$name")
    local timeout_s=$((n * 900 + 3600))
    local extra=""
    [ -f "configs/var7_${name}.yaml" ] && extra="var7_${name}.yaml"
    (
        cd "$MAIN" || exit 1
        rm -rf "outputs/r7_${name}"
        timeout "$timeout_s" env OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
            "$PY" -u scripts/train/train_graph_mappo.py \
                --configs $BASE_CFGS $extra \
                --checkpoint "$CKPT" \
                --num-updates "$n" --run-name "r7_${name}" \
                > "/tmp/r7_${name}.log" 2>&1
    )
    echo "  [$name] 结束（退出码 $?）"
}

cd "$MAIN" || exit 1

if pgrep -f "train_graph_mappo[.]py" >/dev/null 2>&1; then
    echo "ERROR: 已有训练在跑，本轮会被污染。" >&2; exit 1
fi

for n in "${NAMES[@]}"; do write_yaml "$n"; done

echo "第七浪（修复终止语义后）：${NAMES[*]}"
echo "轮数=$(updates_for base)/$(updates_for fix_g999)/$(updates_for fix_ent)/$(updates_for fix_g999_long)  线程=$THREADS"
echo "代码版本：$(git log --oneline -1)"
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
    tr, ev = [], []
    for line in p.open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict):
            continue
        if "update" in d:
            tr.append(d)
        elif "eval_validation" in d:
            ev.append(d["eval_validation"])
    return tr, ev


def paired(a, b):
    n = min(len(a), len(b))
    if n < 2:
        return float("nan"), float("nan"), 0
    diffs = [a[i] - b[i] for i in range(n)]
    md = sum(diffs) / n
    var = sum((d - md) ** 2 for d in diffs) / (n - 1)
    return md, (var / n) ** 0.5, n


def show(run):
    tr, ev = load(run)
    if not ev:
        print(f"{run:<22} (没有验证记录)")
        return tr, ev
    seq = " ".join(f"{e.get('mean_success_rate', float('nan')):.4f}" for e in ev)
    print(f"{run:<22} n={len(ev):<2} {seq}")
    return tr, ev


print("== eval 序列（每 5 轮一个读数）==")
R7 = ["r7_base", "r7_fix_g999", "r7_fix_ent", "r7_fix_g999_long"]
R6 = ["r6_base", "r6_g999", "r6_lowv025", "r6_stor10", "r6_base_lr1e4"]
data = {r: show(r) for r in R7 + R6}

print()
print("== 逐种子配对：第七浪内部 vs r7_base（末次共同 eval；|t|>=2 才算分得开）==")
ctrl = data["r7_base"][1]
if ctrl:
    for r in R7[1:]:
        ev = data[r][1]
        if not ev:
            continue
        k = min(len(ev), len(ctrl))
        md, se, n = paired(ev[k - 1]["per_seed_success"], ctrl[k - 1]["per_seed_success"])
        t = md / se if se else float("nan")
        print(f"  {r:<20} @update{5 * k:<3} {md:+.4f} ± {se:.4f}  t={t:+.2f}"
              f"  {'显著' if abs(t) >= 2 else '不显著'}")

print()
print("== 关键问题：修复本身有没有用？（r7_base vs r6_base，同配置不同代码）==")
a = data["r7_base"][1]
b = data["r6_base"][1]
if a and b:
    k = min(len(a), len(b))
    for i in range(k):
        pa = a[i]["per_seed_success"]
        pb = b[i]["per_seed_success"]
        md, se, n = paired(pa, pb)
        t = md / se if se else float("nan")
        print(f"  @update{5 * (i + 1):<3} {md:+.4f} ± {se:.4f}  t={t:+.2f}"
              f"  {'显著' if abs(t) >= 2 else '不显著'}")

print()
print("== g999 的收益在修复后还在吗？（r7_fix_g999 vs r6_g999）==")
a = data["r7_fix_g999"][1]
b = data["r6_g999"][1]
if a and b:
    k = min(len(a), len(b))
    md, se, n = paired(a[k - 1]["per_seed_success"], b[k - 1]["per_seed_success"])
    t = md / se if se else float("nan")
    print(f"  @update{5 * k:<3} {md:+.4f} ± {se:.4f}  t={t:+.2f}")

print()
print("== 优势塌陷签名（H1）：前 1/3 vs 后 1/3 ==")
print(f"  {'run':<22}{'|A|前':>9}{'|A|后':>9}{'收缩':>7}{'corr后':>8}{'kl后':>9}{'ent后':>8}")
for r in R7:
    tr = data[r][0]
    if len(tr) < 6:
        continue
    k = max(1, len(tr) // 3)

    def m(key, lo, hi):
        xs = [x.get(key) for x in tr[lo:hi]]
        xs = [x for x in xs if isinstance(x, (int, float))]
        return sum(xs) / len(xs) if xs else float("nan")

    a1, a2 = m("mean_abs_advantage", 0, k), m("mean_abs_advantage", len(tr) - k, len(tr))
    print(f"  {r:<22}{a1:>9.3f}{a2:>9.3f}{a2 / a1 if a1 else float('nan'):>7.2f}"
          f"{m('value_return_corr', len(tr) - k, len(tr)):>8.3f}"
          f"{m('kl', len(tr) - k, len(tr)):>9.4f}"
          f"{m('entropy', len(tr) - k, len(tr)):>8.3f}")
print("SCREEN_R7_DONE")
PY
