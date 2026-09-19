#!/usr/bin/env bash
# 线程数会不会改变训练结果？—— 这是所有 A/B 对比的前提。
#
# 起因：第一轮 base（跑在 0.001 上，因为学习率 bug 还没修）验证末次 0.8166、
# 熵 3.92->4.15；第二轮 lr1e3（**显式**设 0.001，且 bug 已修）末次 0.7576、
# 熵 3.79->3.63。配置上这两者应该完全等价（三组 lr 都是 0.001），却差 0.06。
# 唯一还能解释它的差异是**运行时环境**：第一轮是 6 进程 × 4 线程，第二轮
# 是 24 进程 × 2 线程。
#
# rl_algorithm.yaml 里自己写着 8 图块对角前向的 BLAS 分块和 4 图不同，
# "argmax 平局时采样可能不同……单次噪声 ~0.003"。0.003 不至于变成 0.06，
# 但如果线程数真的会改变结果，那：
#   * 跨轮次、跨并行度的比较全部不成立；
#   * 上面那句"噪声 ~0.003"的注释是错的，得改。
#
# 做法：同一份 base 配置、同一个 checkpoint，只改 OMP_NUM_THREADS，
# 每组跑两次。四种运行**同时**起（2×2线程 + 2×4线程），背景负载一致。
#
# 判据：同组两次之间的差 = 该线程数下的重复性；两组之间的差 = 线程数的影响。
# 两者都报，直接比大小。
#
# 用法（在节点上）：
#     bash .tmp/probe_thread_repro.sh
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
UPDATES="${UPDATES:-30}"
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
CFGS="rl_algorithm.yaml train_diag_fast.yaml var2_base.yaml"

cd "$MAIN" || exit 1
if ! grep -qxF 'configs/var2_*.yaml' .git/info/exclude 2>/dev/null; then
    printf 'configs/var2_*.yaml\n' >> .git/info/exclude
fi
cat > configs/var2_base.yaml <<'YAML'
# 对照：显式声明默认值，避免全注释 YAML 解析成 None。
train:
  ppo:
    epochs: 1
YAML

launch() {  # $1=线程数 $2=第几次
    (
        cd "$MAIN" || exit 1
        rm -rf "outputs/repro_t$1_$2"
        OMP_NUM_THREADS="$1" MKL_NUM_THREADS="$1" \
            "$PY" -u scripts/train/train_graph_mappo.py \
                --configs $CFGS --checkpoint "$CKPT" \
                --num-updates "$UPDATES" --run-name "repro_t$1_$2" \
                > "/tmp/repro_t$1_$2.log" 2>&1
    )
    echo "  完成 t$1_$2"
}

for th in 2 4; do
    for rep in 1 2; do
        launch "$th" "$rep" &
    done
done
wait

echo
"$PY" - <<'PY'
import json
from pathlib import Path

MAIN = Path("/opt/qkd/graph_mappo")


def series(name):
    p = MAIN / "outputs" / name / "metrics.jsonl"
    if not p.exists():
        return None, None
    cur = None
    per_seed = []
    for line in p.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "update" in d:
            cur = d
        elif "eval_validation" in d:
            per_seed.append(d["eval_validation"]["per_seed_success"])
    return cur, per_seed


def paired(a, b):
    n = min(len(a), len(b))
    if n < 2:
        return float("nan"), float("nan"), 0
    diffs = [a[i] - b[i] for i in range(n)]
    md = sum(diffs) / n
    var = sum((d - md) ** 2 for d in diffs) / (n - 1)
    se = (var / n) ** 0.5
    return md, se, n


runs = {}
for th in (2, 4):
    for rep in (1, 2):
        name = f"repro_t{th}_{rep}"
        last, ps = series(name)
        runs[(th, rep)] = (last, ps)
        if last is None:
            print(f"  (缺 {name})")
            continue
        final = ps[-1] if ps else []
        mean = sum(final) / len(final) if final else float("nan")
        print(f"{name:<14} 熵 {last['entropy']:.4f}  kl {last['kl']:.6f}  "
              f"末次验证 {mean:.4f}  ({len(ps)} 次验证)")

print()
print("=== 同线程数的两次重复（纯随机噪声有多小）===")
for th in (2, 4):
    a, b = runs.get((th, 1)), runs.get((th, 2))
    if not a or not b or not a[1] or not b[1]:
        continue
    md, se, n = paired(a[1][-1], b[1][-1])
    print(f"  线程 {th}: 末次验证差 {md:+.4f} ± {se:.4f}（{n} 种子配对）")

print()
print("=== 2 线程 vs 4 线程（这是关键：同一个配置）===")
a, b = runs.get((2, 1)), runs.get((4, 1))
if a and b and a[1] and b[1]:
    md, se, n = paired(a[1][-1], b[1][-1])
    print(f"  rep1: {md:+.4f} ± {se:.4f}  t = {md / se if se else float('nan'):+.2f}")
a, b = runs.get((2, 2)), runs.get((4, 2))
if a and b and a[1] and b[1]:
    md, se, n = paired(a[1][-1], b[1][-1])
    print(f"  rep2: {md:+.4f} ± {se:.4f}  t = {md / se if se else float('nan'):+.2f}")
print()
print("判读：同线程重复差 << 跨线程差 => 线程数真的改变结果，跨并行度不可比；")
print("      两者同量级           => 只是随机漂移，跨并行度可比。")
print("THREAD_REPRO_DONE")
PY
echo "THREAD_REPRO_DONE"
