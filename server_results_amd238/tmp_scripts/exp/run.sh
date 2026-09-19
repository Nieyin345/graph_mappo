#!/usr/bin/env bash
# 自动化实验 runner：按 plan 顺序跑实验，每个实验把指标追加到 results.csv。
# 整个 plan 跑完（或中途全部失败）脚本才退出 —— 本机侧的监视靠这个退出被唤醒。
#
#   nohup bash .tmp/exp/run.sh .tmp/exp/plan_speed.txt > /tmp/exp_speed.log 2>&1 &
#
# plan 格式（每行一个实验，# 开头是注释）：
#   <实验名>|<config1,config2>|<线程数>
#   chunk256|_exp_chunk256.yaml|16
# 第三段可省略，省略则用 $THREADS。
#
# 设计要点：
#   * 线程数固定（线程数会确定性影响训练结果，混用就没法比）
#   * 每个实验 8 轮，前 3 轮是预热（corr_rss 实测 u1=72.3 -> u6=83.9 封顶），
#     分析时丢弃，见 harvest.py
#   * 训练输出走块缓冲，日志不实时；指标一律读 metrics.jsonl（每轮 flush）
set -uo pipefail

ROOT=/opt/qkd/graph_mappo
PY=$ROOT/venv/bin/python
EXP_DIR=$ROOT/.tmp/exp
PLAN=${1:?用法: run.sh <plan 文件>}

THREADS=${THREADS:-16}          # 固定，别混
UPDATES=${UPDATES:-8}
CHECKPOINT=${CHECKPOINT:-outputs/supervised_pg_phased/supervised_pg_phased_latest.pt}

cd "$ROOT" || exit 1
mkdir -p "$EXP_DIR"

if pgrep -f "train_graph_mappo[.]py" >/dev/null 2>&1; then
    echo "ERROR: 已有 train_graph_mappo.py 在跑，会污染计时。先清空再跑。" >&2
    exit 2
fi

RESULT_CSV="$EXP_DIR/results.csv"
[ -f "$RESULT_CSV" ] || echo "tag,update,rollout_s,update_s,elapsed_s,success_rate,served_keys,configs" > "$RESULT_CSV"

echo "=== EXP START $(date +%H:%M:%S)  plan=$PLAN  threads=$THREADS  updates=$UPDATES ==="
n_ok=0; n_fail=0

while IFS='|' read -r tag cfgs thr || [ -n "$tag" ]; do
    tag=$(echo "${tag:-}" | xargs)
    [ -z "$tag" ] && continue
    case "$tag" in \#*) continue ;; esac
    thr=$(echo "${thr:-}" | xargs); thr=${thr:-$THREADS}

    cfg_args=""
    if [ -n "${cfgs:-}" ]; then
        IFS=',' read -ra arr <<< "$cfgs"
        for c in "${arr[@]}"; do cfg_args="$cfg_args $(echo "$c" | xargs)"; done
    fi

    echo "--- [$tag] 开始 $(date +%H:%M:%S)  threads=$thr  configs:${cfg_args:- (默认)}"
    rm -rf "outputs/$tag"

    OMP_NUM_THREADS=$thr MKL_NUM_THREADS=$thr \
        $PY scripts/train/train_graph_mappo.py \
            --configs rl_algorithm.yaml train_full_rl.yaml $cfg_args \
            --checkpoint "$CHECKPOINT" \
            --num-updates "$UPDATES" \
            --run-name "$tag" > "/tmp/exp_$tag.log" 2>&1 < /dev/null
    rc=$?

    if [ $rc -ne 0 ]; then
        echo "--- [$tag] 失败 rc=$rc，日志末尾："
        tail -5 "/tmp/exp_$tag.log" | sed 's/^/      /'
        echo "$tag,FAILED,,,,,,${cfg_args:-}" >> "$RESULT_CSV"
        n_fail=$((n_fail + 1))
        continue
    fi

    # 指标只认 metrics.jsonl（每轮 flush，可靠）；训练 stdout 是块缓冲的，不可靠
    $PY - "$tag" "${cfg_args:-} thr=$thr" "$RESULT_CSV" <<'PYEOF'
import json, pathlib, sys
tag, cfg, csv_path = sys.argv[1], sys.argv[2].strip(), sys.argv[3]
m = pathlib.Path(f"outputs/{tag}/metrics.jsonl")
if not m.exists():
    print(f"    [{tag}] 没有 metrics.jsonl"); sys.exit(0)
rows = []
for line in m.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line: continue
    try: d = json.loads(line)
    except json.JSONDecodeError: continue
    if "update" not in d or "update_s" not in d: continue
    rows.append(d)
with open(csv_path, "a", encoding="utf-8") as f:
    for d in rows:
        f.write(",".join(str(x) for x in [
            tag, d.get("update"), f"{d.get('rollout_s', 0):.1f}",
            f"{d.get('update_s', 0):.1f}", f"{d.get('elapsed_s', 0):.1f}",
            f"{d.get('mean_success_rate', 0):.4f}", f"{d.get('mean_served_keys', 0):.0f}",
            cfg,
        ]) + "\n")
print(f"    [{tag}] 记录 {len(rows)} 轮")
PYEOF
    echo "--- [$tag] 完成 $(date +%H:%M:%S)"
    n_ok=$((n_ok + 1))
done < "$PLAN"

echo "=== EXP DONE $(date +%H:%M:%S)  成功 $n_ok / 失败 $n_fail ==="
echo "=== 结果表：$RESULT_CSV ==="
