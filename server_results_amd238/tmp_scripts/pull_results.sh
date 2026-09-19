#!/usr/bin/env bash
# 把服务器上的实验记录拉回本地 `server_results/`。
#
# 为什么要做：服务器是临时实验节点，随时可能到期（上次 amd276 就是在到期前几分钟
# 才抢回来）。而本地留下的往往只有**结论**，一旦要复核"这个数是哪来的"就没证据了。
#
# 只拉**证据级**的小文件：
#   - 每次运行的 metrics.jsonl（每轮一行的指标，含 eval_validation 的逐种子成绩）
#   - rollout_debug.jsonl（奖励分项拆解）
#   - eval/ 下的专家与 BC 参照（perseed_matrix.json、expert_diag.json 等）
# checkpoint（*.pt，每个 ~12 MB）**不拉** —— 体积大，且结论不依赖它。
#
# 用法：
#     bash .tmp/pull_results.sh              # 拉全部
#     bash .tmp/pull_results.sh r2_ repro_   # 只拉名字带这些前缀的
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

SSH="/c/Windows/System32/OpenSSH/ssh.exe"
[ -x "$SSH" ] || SSH="ssh"
HOST="$(tr -d '\r\n' < .tmp/qkd_host 2>/dev/null || echo qkd)"
REMOTE=/opt/qkd/graph_mappo
DEST=server_results/runs

FILTERS=("$@")
mkdir -p "$DEST"

want() {  # 没有过滤器就全要
    [ ${#FILTERS[@]} -eq 0 ] && return 0
    local name="$1"
    for f in "${FILTERS[@]}"; do
        case "$name" in *"$f"*) return 0 ;; esac
    done
    return 1
}

echo "== 列出节点上的运行 =="
mapfile -t RUNS < <("$SSH" -o BatchMode=yes "$HOST" \
    "cd $REMOTE && ls -d outputs/*/ 2>/dev/null | sed 's|outputs/||; s|/||'")
echo "  节点上共 ${#RUNS[@]} 个运行目录"

n=0
for run in "${RUNS[@]}"; do
    want "$run" || continue
    for f in metrics.jsonl rollout_debug.jsonl; do
        # 先问存不存在，避免每个缺失的文件都报一次错
        if "$SSH" -o BatchMode=yes "$HOST" "test -f $REMOTE/outputs/$run/$f"; then
            mkdir -p "$DEST/outputs/$run"
            "$SSH" -o BatchMode=yes "$HOST" "cat $REMOTE/outputs/$run/$f" \
                > "$DEST/outputs/$run/$f"
            n=$((n + 1))
        fi
    done
done

echo "== eval/ 下的参照 =="
mapfile -t EVALS < <("$SSH" -o BatchMode=yes "$HOST" \
    "cd $REMOTE && ls outputs/eval/*.json 2>/dev/null | sed 's|outputs/eval/||'")
mkdir -p "$DEST/outputs/eval"
for f in "${EVALS[@]}"; do
    case "$f" in perseed|expert|bc|frozen) ;; esac
    "$SSH" -o BatchMode=yes "$HOST" "cat $REMOTE/outputs/eval/$f" \
        > "$DEST/outputs/eval/$f"
    n=$((n + 1))
done

echo
echo "拉到 $n 个文件，共 $(du -sh "$DEST" | cut -f1)"
echo "PULL_DONE"
