#!/usr/bin/env bash
# group1 内部再拆一刀：是模型前向改坏了，还是 policy 的 log-prob 改坏了？
#
# 已知（服务端实测，update 5 的确定性评估）：
#   d1930a2 全量（旧 trainer + 旧 policy）        0.8203
#   当前工作区（新 trainer + 新 policy）          0.7627
#   group1 = 新 graph_mappo + 新 policy，旧 trainer  0.7572   <- 坏
#   group2 = 新 trainer，旧 graph_mappo + 旧 policy  0.8174   <- 好
# 所以坏的东西跟着 policy.py/graph_mappo.py 走，跟 trainer 无关。
#
# 两组的 rollout 成功率逐轮几乎相同（~0.775-0.782），差别只在确定性评估。
# 而 policy.py 的改动全在「更新期」的 log-prob 计算里，采样和 deterministic
# 分支一行没动 —— 也就是说这是训练把策略带偏了，不是评估期选动作坏了。
#
# 拆法：
#   g1a = 只覆盖 graph_mappo.py（旧 policy + 新模型）
#   g1b = 只覆盖 policy.py      （新 policy + 旧模型）
#
# g1b 很可能直接崩：新 policy 的 evaluate_actions_batched 会调
# batched_forward(..., want_edge_maps=False) 并读 outputs.edge_arrays，旧模型
# 没有这两样。崩了也是有信息量的 —— 说明两者耦合，那么 g1a 能跑通且结果正常
# 就基本锁定在 policy.py。
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
        OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 \
            "$PY" -u scripts/train/train_graph_mappo.py \
                --configs rl_algorithm.yaml train_diag_fast.yaml \
                --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
                --num-updates "$UPDATES" --run-name "diag_${name}" \
                > "diag_${name}.log" 2>&1
    )
    echo "  ${name} run 结束（退出码 $?）"
}

echo "group1 拆分：模型前向 vs policy log-prob，各 $UPDATES 轮。"
echo

echo "=== g1a: 只换 graph_mappo.py（旧 policy + 新模型）==="
run_one g1a "qkd_rl/rl/models/graph_mappo.py"

echo "=== g1b: 只换 policy.py（新 policy + 旧模型）==="
run_one g1b "qkd_rl/rl/algos/policy.py"

echo
echo "=== 结果：update 5 后的确定性评估 ==="
echo "  参照  d1930a2 全量                0.8203"
echo "  参照  当前工作区                  0.7627"
echo "  参照  group1（两个文件都换）      0.7572"
echo "  参照  group2（两个文件都不换）    0.8174"
for g in g1a g1b; do
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
        echo "(无 metrics —— 大概率崩了，看 diag_${g}.log)"
    fi
done
echo
echo "G1SPLIT_DONE"
