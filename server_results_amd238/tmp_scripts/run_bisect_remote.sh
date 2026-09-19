#!/usr/bin/env bash
# 在节点上启动一个 .tmp/ 下的长跑脚本（nohup，脱离 ssh 会话），然后立刻返回。
#
#   bash .tmp/run_bisect_remote.sh <user@host>                      # 默认跑 bisect_groups.sh
#   bash .tmp/run_bisect_remote.sh <user@host> --script .tmp/x.sh   # 指定脚本
#   bash .tmp/run_bisect_remote.sh <user@host> --status             # 只看进度
#
# 为什么 nohup：这类实验常在 10-25 分钟，而节点随时可能被回收。挂在前台
# 的话，ssh 一断脚本就跟着死；nohup 之后日志落在 /opt/qkd/bisect.log，
# 换会话重连也能接着看。
set -euo pipefail

HOST="${1:?usage: run_bisect_remote.sh <user@host> [--script PATH | --status]}"
shift || true

MODE=start
SCRIPT=".tmp/bisect_groups.sh"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --status) MODE=status ;;
        --script) SCRIPT="${2:?--script 需要一个路径}"; shift ;;
        *) echo "未知参数: $1" >&2; exit 2 ;;
    esac
    shift
done

# 跑在节点上的脚本必须已经在节点仓库里 —— server_sync.sh 只同步 .tmp/*.py 和
# .tmp/*.sh，所以新写的脚本要先同步一次，否则节点上根本没有这个文件。
case "$SCRIPT" in
    .tmp/*.sh|.tmp/*.py) ;;
    *) echo "只支持 .tmp/ 下的 .sh/.py（同步通道只带这些）" >&2; exit 2 ;;
esac

# 这台机器上必须用系统自带的 OpenSSH，conda 的 MSYS 版连不上 ssh-agent。
SSH_BIN=""
for cand in "C:/Windows/System32/OpenSSH/ssh.exe" \
            "/c/Windows/System32/OpenSSH/ssh.exe" \
            "$(command -v ssh 2>/dev/null || true)"; do
    [ -n "$cand" ] && [ -x "$cand" ] && { SSH_BIN="$cand"; break; }
done
[ -n "$SSH_BIN" ] || { echo "找不到可用的 ssh" >&2; exit 1; }

if [ "$MODE" = status ]; then
    "$SSH_BIN" -o BatchMode=yes "$HOST" "bash -s -- $SCRIPT" <<'REMOTE'
set -u
cd /opt/qkd/graph_mappo 2>/dev/null || exit 1
if ps -eo args | grep -q "[b]$(basename "$1")"; then
    echo "状态  运行中（$1）"
else
    echo "状态  已结束（或未启动）（$1）"
fi
echo "--- 日志尾部 ---"
tail -n 20 /opt/qkd/bisect.log 2>/dev/null || echo "(无日志)"
REMOTE
    exit 0
fi

"$SSH_BIN" -o BatchMode=yes "$HOST" "bash -s -- $SCRIPT" <<'REMOTE'
set -eu
cd /opt/qkd/graph_mappo

# worktree add 需要 BASE 是个真实的 commit 对象。同步推的是工作区快照，
# 它的父链一路回到本地 HEAD，所以 d1930a2 应该在 —— 但不该假定，先查。
if ! git cat-file -e 'd1930a2^{commit}' 2>/dev/null; then
    echo "BASE_COMMIT_MISSING: 节点仓库里没有 d1930a2，无法做 worktree 回退"
    exit 1
fi
echo "base commit d1930a2 存在"

# 这个节点上没有 pgrep/pkill（exit 127），只能用 ps + grep。
if ps -eo args | grep -q "[b]$(basename "$1")"; then
    echo "已经在跑了，不重复启动。看进度: --status"
    exit 0
fi

nohup bash "$1" > /opt/qkd/bisect.log 2>&1 &
echo "已启动 $1，pid $!，日志 /opt/qkd/bisect.log"
REMOTE
