#!/usr/bin/env bash
# 同样 48 线程的预算，怎么切吞吐最高？
#
# 已有数据（train_diag_fast 2 轮，节点上实测）：
#   单进程：1 线程 119.7 s/轮，4 线程 69.8，16 线程 55.9
#     -> 线程越多越快，但**每核效率**是反的：1 线程 0.0084 轮/(s·核)，
#        4 线程 0.0036，16 线程 0.0011（差 7.6 倍）
#   多进程（各 4 线程）：1 个 70.0 s/轮，4 个 70.9，8 个 79.3，12 个 93.3
#     -> 到 12 个进程（48 线程）为止，吞吐 9.2 倍，只有 33% 的退化
#
# 推论：如果每核效率随线程数下降这么快，那**同样 48 个线程，切成更多、更瘦的
# 进程应该吞吐更高**。这个脚本就是去验证它：固定总线程数 ~48，只改切的份数。
#
# 用法（在节点上）：
#     bash /tmp/scaling_throughput.sh
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
UPDATES="${UPDATES:-2}"
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
CFGS="rl_algorithm.yaml train_diag_fast.yaml"

launch() {  # $1=后缀 $2=线程数
    (
        cd "$MAIN" || exit 1
        rm -rf "outputs/thr_$1"
        OMP_NUM_THREADS="$2" MKL_NUM_THREADS="$2" \
            "$PY" -u scripts/train/train_graph_mappo.py \
                --configs $CFGS --checkpoint "$CKPT" \
                --num-updates "$UPDATES" --run-name "thr_$1" \
                > "/tmp/thr_$1.log" 2>&1
    )
}

# 一个组合：$1=标签 $2=进程数 $3=每进程线程数
combo() {
    local label="$1" n="$2" th="$3"
    echo "--- $label：$n 个进程 × $th 线程 = $((n * th)) 线程 ---"
    local t0
    t0=$(date +%s)
    for i in $(seq 1 "$n"); do launch "${label}_${i}" "$th" & done
    wait
    local wall=$(( $(date +%s) - t0 ))
    local peak
    peak=$(free -g | awk '/^Mem:/{print $3}')
    echo "    墙钟 ${wall}s   峰值已用内存 ${peak} GB"
    echo "$label $n $th $wall" >> /tmp/thr_results.txt
}

rm -f /tmp/thr_results.txt
combo t2 24 2
combo t6 8 6

echo
echo "======== 吞吐对比 ========"
"$PY" - <<'PY'
import glob, json, os
rows = []
for line in open("/tmp/thr_results.txt", encoding="utf-8"):
    label, n, th, wall = line.split()
    n, th, wall = int(n), int(th), int(wall)
    per = []
    for f in glob.glob(f"/opt/qkd/graph_mappo/outputs/thr_{label}_*/metrics.jsonl"):
        s, c = 0.0, 0
        for ln in open(f, encoding="utf-8"):            try:
                d = json.loads(ln)
            except Exception:
                continue
            if "update" in d:
                s += d["elapsed_s"]
                c += 1
        if c:
            per.append(s / c)
    if not per:
        continue
    mean = sum(per) / len(per)
    # 吞吐 = 单位时间内完成的"训练轮"数（全部进程合计）
    thru = n / mean
    rows.append((label, n, th, mean, wall, thru))
print(f"{'组合':<8}{'进程':>5}{'线程/个':>8}{'每轮秒':>10}{'墙钟':>8}{'吞吐(轮/s)':>12}{'相对 12x4':>11}")
base = None
for label, n, th, mean, wall, thru in rows:
    if label == "t2":
        base = thru
    print(f"{label:<8}{n:>5}{th:>8}{mean:>10.1f}{wall:>8}{thru:>12.3f}{'':>11}")
print()
print("参照：12 进程 × 4 线程 每轮 93.3 s -> 吞吐 0.129 轮/s")
PY
echo "THROUGHPUT_DONE"
