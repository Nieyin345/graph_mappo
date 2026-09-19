#!/usr/bin/env bash
# 这台 56 核 / 251 GB 的机器该怎么吃满：单进程加线程 vs 多进程并发。
#
# 背景：一次训练只用掉 ~4.1 核（ps 里 411% CPU / 22 线程），机器 91% 空闲。
# 所以问题不是"能不能并行"，而是"哪一层能并行"。
#
# 两个假设，分开测：
#   H1 单进程加线程能加速 —— 张量只有 128 宽，BLAS 在这种小矩阵上收益递减，
#      外加 rollout 每步有几十次 Python/采样调用，OMP 线程再多也堵在 GIL 上。
#   H2 多进程并发能接近线性 —— 每个进程独立持有 env + GNN，进程之间一个
#      rollout 内不通信，理论上应该各自吃自己的核。
#
# 判据用 metrics.jsonl 里的 elapsed_s（不含启动和 H5 加载），不是墙钟：
# 墙钟被启动开销污染，而我们要比的是稳态成本。
#
# 用法（在节点上）：
#     bash /tmp/scaling_test.sh threads     # 单进程线程扩展
#     bash /tmp/scaling_test.sh procs       # 多进程并发
#
# 注意：跑的时候机器上还有别的作业（比如 BC 窗口对比）会是背景负载，
# 两个模式内部各自可比，跨模式比时要知道这一点。
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
UPDATES="${UPDATES:-2}"
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
CFGS="rl_algorithm.yaml train_diag_fast.yaml"

launch() {  # $1=后缀 $2=线程数
    (
        cd "$MAIN" || exit 1
        rm -rf "outputs/scale_$1"
        OMP_NUM_THREADS="$2" MKL_NUM_THREADS="$2" \
            "$PY" -u scripts/train/train_graph_mappo.py \
                --configs $CFGS --checkpoint "$CKPT" \
                --num-updates "$UPDATES" --run-name "scale_$1" \
                > "/tmp/scale_$1.log" 2>&1
        echo "  完成 $1（线程 $2）"
    )
}

summarize() {
    "$PY" - <<'PY'
import glob, json, os
rows = []
for f in sorted(glob.glob("/opt/qkd/graph_mappo/outputs/scale_*/metrics.jsonl")):
    name = os.path.basename(os.path.dirname(f))
    tot = ro = up = 0.0
    n = 0
    for line in open(f, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "update" not in d:
            continue
        tot += d.get("elapsed_s", 0.0)
        ro += d.get("rollout_s", 0.0)
        up += d.get("update_s", 0.0)
        n += 1
    if n:
        rows.append((name, n, tot / n, ro / n, up / n))
print(f"{'实验':<16}{'轮':>4}{'每轮秒':>10}{'rollout':>10}{'update':>10}")
for name, n, t, r, u in rows:
    print(f"{name:<16}{n:>4}{t:>10.1f}{r:>10.1f}{u:>10.1f}")
PY
}

case "${1:-threads}" in
threads)
    for th in 1 4 16; do
        echo "--- 单进程，OMP_NUM_THREADS=$th ---"
        t0=$(date +%s)
        launch "t${th}" "$th"
        echo "    墙钟 $(( $(date +%s) - t0 )) s"
    done
    ;;
procs)
    for n in 1 4 8 12; do
        echo "--- $n 个进程并发，每个 OMP_NUM_THREADS=4 ---"
        t0=$(date +%s)
        for i in $(seq 1 "$n"); do launch "p${n}_${i}" 4 & done
        wait
        echo "    墙钟 $(( $(date +%s) - t0 )) s"
    done
    ;;
*)
    echo "用法: $0 {threads|procs}" >&2
    exit 2
    ;;
esac

echo
echo "======== 汇总（每轮平均值）========"
summarize
echo "SCALING_DONE"
