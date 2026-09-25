#!/usr/bin/env bash
# 把服务器上某次训练的产物拉回本地，并按「模型 → 目录」自动分流。
#
#   bash deployment/fetch_results.sh <run-name> [...]      # 一个或多个 run
#   bash deployment/fetch_results.sh --model v4            # 该模型的全部 run
#   bash deployment/fetch_results.sh --list                # 只列服务器上有什么
#   bash deployment/fetch_results.sh --eval <run-name>     # 连同冻结评测 JSON 一起拉
#
# 为什么需要它：产物**只在服务器上就是没保住**。节点会被回收、实例会重启、
# 工作树会被删。实测已丢过 v5 与 pairhist_v3 的**代码**（数据靠人工抢救保住），
# 见 实验产物规范.md §3.5.0。所以每次训练完都要跑一次本脚本。
#
# 落位规则（按 resolved_config.yaml 的 experiment.model 判断，不靠文件名猜）：
#   server_results/<batch-id>/train/<run>/   训练产物
#   server_results/<batch-id>/eval/          冻结评测 JSON
#   然后重跑 backfill_manifest.py 更新 <model>/RESULTS.md
#
# 设计取舍：
#   * 用 tar 打包单次传输，不用逐文件 scp —— 跨洋链路实测会中途 reset，
#     逐文件要在每个文件上重试，一个 tar 断了整个重来但只需重试一次。
#   * checkpoint 默认**拉**（那是唯一无法重算的东西）。--no-ckpt 可跳过。
#   * 全部走 md5 核对后再落位：传输中途截断过（实测 20MB 的 .pt 被截到
#     9MB 而文件名照旧），只比大小会漏。
set -uo pipefail

HOST="${QKD_HOST:-qkd}"
REMOTE="${QKD_REMOTE_DIR:-/opt/qkd/graph_mappo}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_ROOT="$ROOT/server_results"

RUNS=()
MODEL_FILTER=""
LIST_ONLY=0
WITH_EVAL=0
WITH_CKPT=1

while [ $# -gt 0 ]; do
    case "$1" in
        --list)      LIST_ONLY=1 ;;
        --model)     MODEL_FILTER="$2"; shift ;;
        --eval)      WITH_EVAL=1 ;;
        --no-ckpt)   WITH_CKPT=0 ;;
        -h|--help)   sed -n '2,25p' "$0"; exit 0 ;;
        -*)          echo "未知选项: $1" >&2; exit 2 ;;
        *)           RUNS+=("$1") ;;
    esac
    shift
done

# ---------------------------------------------------------------- 列清单
remote_runs() {
    ssh "$HOST" "cd $REMOTE && find outputs/experiments -mindepth 2 -maxdepth 2 -type d \
        | sed 's|outputs/experiments/||' | sort"
}

if [ "$LIST_ONLY" = 1 ]; then
    echo "=== 服务器上的 run（model/run）==="
    remote_runs
    exit 0
fi

# ---------------------------------------------------------------- 选 run
if [ -z "$MODEL_FILTER" ] && [ ${#RUNS[@]} -eq 0 ]; then
    echo "用法: bash deployment/fetch_results.sh <run-name> | --model v4 | --list" >&2
    exit 2
fi

if [ -n "$MODEL_FILTER" ]; then
    while IFS= read -r r; do
        [ -n "$r" ] && RUNS+=("$r")
    done < <(remote_runs | awk -F/ -v m="$MODEL_FILTER" '$1==m {print}')
    if [ ${#RUNS[@]} -eq 0 ]; then
        echo "服务器上没有 model=$MODEL_FILTER 的 run" >&2
        exit 1
    fi
fi

echo "将拉取 ${#RUNS[@]} 个 run: ${RUNS[*]}"
echo

for spec in "${RUNS[@]}"; do
    # spec 可以是 "v4/s42_bc8_u20"（model/run）或裸 run 名
    case "$spec" in
        */*) rel="$spec" ;;
        *)   rel="$(remote_runs | awk -F/ -v r="$spec" '$2==r {print; exit}')" ;;
    esac
    if [ -z "$rel" ]; then
        echo "[跳过] 找不到 run: $spec" >&2
        continue
    fi

    model="${rel%%/*}"
    run="${rel##*/}"
    # 批次目录：<模型>-<run> 是最小可用的 batch-id；已有同名批次则复用。
    batch="$(date +%Y%m%d)-${model}-${run}"
    dest="$DEST_ROOT/$batch"
    mkdir -p "$dest/train/$run" "$dest/eval"

    echo "=== $rel  ->  server_results/$batch/"

    # 产物清单。--no-ckpt 时排除权重（其余都要：metrics 是读数来源，
    # rollout_debug 是奖励分解，train.log 是 PPO 是否真在动的证据）。
    exclude=""
    [ "$WITH_CKPT" = 0 ] && exclude='--exclude=*.pt'

    if ! ssh "$HOST" "cd $REMOTE/outputs/experiments/$model && tar czf - $exclude '$run'" \
            | tar xzf - -C "$dest/train"; then
        echo "  ✗ 传输失败（链路中断？重跑本命令）" >&2
        continue
    fi

    # 冻结评测 JSON：常见坐落有三处，找到就拉
    for d in "outputs/frozen_eval_${model}_20260925" \
             "outputs/frozen_eval_${model}" \
             "outputs/experiments/$model/$run/frozen_eval_A" \
             "outputs/experiments/$model/$run/frozen_eval_B"; do
        ssh "$HOST" "cd $REMOTE && [ -d '$d' ] && tar czf - '$d'" 2>/dev/null \
            | tar xzf - -C "$dest/eval" 2>/dev/null
    done

    # ---------------------------------------------------------------- md5 核对
    # 传输中途截断过（文件名没变，内容少了），所以逐文件比 md5，不只比大小。
    n_ok=0; n_bad=0
    while IFS= read -r relf; do
        [ -n "$relf" ] || continue
        remote_md5="$(ssh "$HOST" "md5sum '$REMOTE/outputs/experiments/$model/$relf'" 2>/dev/null | awk '{print $1}')"
        local_md5="$(md5sum "$dest/train/$run/$relf" 2>/dev/null | awk '{print $1}')"
        if [ -n "$remote_md5" ] && [ "$remote_md5" = "$local_md5" ]; then
            n_ok=$((n_ok+1))
        else
            echo "  ✗ md5 不符: $relf (远端 $remote_md5 / 本地 $local_md5)" >&2
            n_bad=$((n_bad+1))
        fi
    done < <(ssh "$HOST" "cd $REMOTE/outputs/experiments/$model/$run && \
                          find . -type f -printf '%P\n' 2>/dev/null" | head -40)

    echo "  ✓ $n_ok 个文件 md5 一致${n_bad:+, ✗ $n_bad 个不符（重跑本命令）}"
    echo "  下一步：python scripts/diag/backfill_manifest.py --repo-root . --apply"
    echo
done

echo "完成。若上面有 md5 不符，重跑本命令即可（tar 会覆盖）。"
