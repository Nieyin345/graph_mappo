#!/usr/bin/env bash
# 两个对照，用来判定「回归」到底存不存在。
#
# 起因：bisect 的参照值 0.8203 是在**旧节点**（amd276, EPYC 7402P）上测的，
# 而所有 bisect 臂都跑在**新节点**（Xeon E5-2683 v3）上。配置里自己写着
# block-diagonal 的批宽会改变 BLAS 分块，而罕见的 argmax 平局下采样的匹配
# 可能不同 —— 也就是说，同一份代码在两台机器上跑出的确定性评估未必相等。
#
# 而且子代理核对了作者自己留下的前向快照（.tmp/_archive/ref_forward/）：
# before.pt 与 after.pt 前 70100/71457 字节完全相同，剩下的差异只有
# ~1e-6 相对量级（float32 的 1-2 ulp，纯归约顺序噪声），且落在 critic 池化那
# 一块上 —— actor 的输出是逐位相同的。模型重写在数值上等于没改。
#
# 所以现在有两种读法，必须用实验分开：
#   A. 回归真实存在，group1（模型+policy）是原因
#   B. group1 的下降是评估噪声，真正的差异来自没测过的 group3（env 等）
#      或者根本不存在回归，0.8203 只是旧机器的数
#
# 本脚本跑三个：
#   ref1    —— d1930a2 裸跑，不覆盖任何文件。在**新节点上**的参照值。
#   ref2    —— 和 ref1 逐字节相同的一份代码，再跑一遍。
#              **这是整个方法论的前提**：bisect 拿「不同代码 → 不同评估」当证据，
#              而每个臂只跑一次。如果同一份代码跑两遍评估就能差 0.05，那么
#              group1 的 0.7572 和 group2 的 0.8174 之间的差距根本不能归因给代码，
#              前面所有分组结论都要作废。先把它测掉。
#   group3  —— 只覆盖 env.py + rollout_buffer.py + rollout_workers.py
#              （上一次跑到第 1 轮进程就被 ssh 会话带走了，从没测到）
#
# env.py 是这几个文件里唯一能改变观测/plan 的，也是唯一没被排除的。
set -u

MAIN=/opt/qkd/graph_mappo
BASE=d1930a2
PY=/opt/qkd/venv/bin/python
UPDATES="${UPDATES:-5}"

run_one() {
    local name="$1" files="$2"
    local wt="/opt/qkd/bisect_${name}"
    git -C "$MAIN" worktree remove --force "$wt" 2>/dev/null || rm -rf "$wt"
    git -C "$MAIN" worktree add --detach "$wt" "$BASE" >/dev/null || return 1
    local f
    for f in $files; do
        cp "$MAIN/$f" "$wt/$f" || { echo "  overlay FAILED for $f"; return 1; }
    done
    ln -sfn "$MAIN/dataset" "$wt/dataset"
    mkdir -p "$wt/outputs"
    ln -sfn "$MAIN/outputs/supervised_pg_phased" "$wt/outputs/supervised_pg_phased"
    (
        cd "$wt" || exit 1
        rm -rf "outputs/diag_${name}"
        # timeout 是必需的，不是保险：训练本身跑完后进程会卡在解释器退出
        # （多进程 worker 池没干净收尾），脚本就一直等下去。g1b 那次训练和
        # 图表都写完了还多挂了 20 分钟。25 分钟足够 5 轮 + 评估。
        timeout 1500 env OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 \
            "$PY" -u scripts/train/train_graph_mappo.py \
                --configs rl_algorithm.yaml train_diag_fast.yaml \
                --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
                --num-updates "$UPDATES" --run-name "diag_${name}" \
                > "diag_${name}.log" 2>&1
    )
    echo "  ${name} run 结束（退出码 $?）"
}

echo "对照实验，各 $UPDATES 轮。"
echo

echo "=== ref1: d1930a2 裸跑（不覆盖任何文件）—— 本节点参照值 ==="
run_one ref1 ""

echo "=== ref2: 同一份代码再跑一遍 —— 复现性前提 ==="
run_one ref2 ""

echo "=== group3: 只覆盖 env.py + rollout_buffer.py + rollout_workers.py ==="
run_one group3 "qkd_rl/env/env.py qkd_rl/rl/algos/rollout_buffer.py qkd_rl/rl/algos/rollout_workers.py"

echo
echo "=== 汇总（update 5 后的确定性评估，全部同机同配置）==="
echo "  旧节点 amd276 上的历史参照  d1930a2        0.8203"
echo "  本节点结果："
for g in group1 group2 ref1 ref2 group3 g1a g1b; do
    f="/opt/qkd/bisect_${g}/outputs/diag_${g}/metrics.jsonl"
    printf "  %-34s" "$g"
    if [ -f "$f" ]; then
        "$PY" - "$f" <<'PY'
import json, sys
vals = [d["eval_validation"]["mean_success_rate"]
        for d in (json.loads(l) for l in open(sys.argv[1]))
        if "eval_validation" in d]
print("%.4f" % vals[-1] if vals else "(无评估记录)")
PY
    else
        echo "(无 metrics)"
    fi
done
echo
echo "REF_G3_DONE"
