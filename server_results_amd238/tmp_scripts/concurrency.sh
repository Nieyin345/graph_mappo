#!/usr/bin/env bash
# 并发臂扫描：这台 32 核节点上，同时跑 N 个训练进程的聚合吞吐是多少？
#
# 背景：文档 §5.3 在 48 核的 amd276 上测过「2 臂几乎白送（每臂慢 5%，聚合 1.91×），
# 3 臂是上限」。当前节点是 32 核，核数少 1/3，甜点位置可能不同，值得重测。
# 而且并发臂是**唯一不依赖单臂提速**的吞吐杠杆 —— 对"跑一批筛选实验"最有用。
#
# 每个臂：rollout worker 固定 8，update 线程 8（并发时按核数分摊）
# 计时口径：所有臂都用「各自第 4~8 轮」的稳态值（前 3 轮预热，见 corr_rss 结论）
#
#   nohup bash .tmp/concurrency.sh > /tmp/concurrency.log 2>&1 &
set -uo pipefail
cd /opt/qkd/graph_mappo

PY=/opt/qkd/venv/bin/python
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
UPDATES=${UPDATES:-8}
LEVELS=${LEVELS:-"1 2 3"}

if pgrep -f "train_graph_mappo[.]py" >/dev/null 2>&1; then
    echo "ERROR: 已有训练在跑，并发测试会被污染。" >&2; exit 1
fi

echo "=== 并发臂扫描开始 $(date +%H:%M:%S) ==="

for n in $LEVELS; do
    # 每臂分到的 update 线程数：总核数 / 臂数，但不低于 4、不高于 16
    thr=$(( 32 / n ))
    [ "$thr" -gt 16 ] && thr=16
    [ "$thr" -lt 4 ] && thr=4

    echo "--- $n 臂 开始 $(date +%H:%M:%S)  每臂 update 线程=$thr  worker=8"
    t0=$(date +%s)

    pids=()
    for i in $(seq 1 "$n"); do
        name="cc${n}_a${i}"
        rm -rf "outputs/$name"
        OMP_NUM_THREADS=$thr MKL_NUM_THREADS=$thr \
            $PY scripts/train/train_graph_mappo.py \
                --configs rl_algorithm.yaml train_full_rl.yaml \
                --checkpoint "$CKPT" \
                --num-updates "$UPDATES" --run-name "$name" \
                > "/tmp/$name.log" 2>&1 < /dev/null &
        pids+=($!)
    done

    # 等这一组全部结束
    for p in "${pids[@]}"; do wait "$p" || true; done
    t1=$(date +%s)
    echo "--- $n 臂 结束 $(date +%H:%M:%S)  墙钟 $((t1-t0))s"

    # 汇总：每臂取稳态（第 4 轮起）均值
    $PY - "$n" <<'PYEOF'
import json, pathlib, statistics, sys
n = int(sys.argv[1])
per_arm = []
for i in range(1, n + 1):
    m = pathlib.Path(f"outputs/cc{n}_a{i}/metrics.jsonl")
    if not m.exists():
        print(f"    臂 {i}: 无 metrics"); continue
    rows = []
    for line in m.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except json.JSONDecodeError: continue
        if "update" in d and "update_s" in d: rows.append(d)
    steady = rows[3:] or rows
    if not steady: continue
    upd = statistics.mean(r["update_s"] for r in steady)
    roll = statistics.mean(r["rollout_s"] for r in steady)
    succ = statistics.mean(r["mean_success_rate"] for r in steady)
    per_arm.append((i, roll, upd, roll + upd, succ))
    print(f"    臂 {i}: rollout={roll:.1f} update={upd:.1f} 一轮={roll+upd:.1f} 成功率={succ:.4f}")
if per_arm:
    avg = statistics.mean(a[3] for a in per_arm)
    thr = 1.0 / avg
    print(f"  平均一轮 {avg:.1f}s -> 聚合 {thr:.5f} 更新/秒/进程，"
          f"总吞吐 {thr * len(per_arm):.5f} 更新/秒")
PYEOF
done

echo "=== 全部结束 $(date +%H:%M:%S) ==="
