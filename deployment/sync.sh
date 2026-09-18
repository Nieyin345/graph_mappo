#!/usr/bin/env bash
# 把本地工作区同步到实验节点 —— 用 git 语义，不是复制文件。
#
#   bash deployment/sync.sh --setup      # 一次性：在节点上建受管仓库（幂等）
#   bash deployment/sync.sh              # 同步（含未提交改动 + .tmp/ 下的探针脚本）
#   bash deployment/sync.sh --dry-run    # 只报将要改哪些文件
#
# `.tmp/` 下只有 *.py / *.sh 会同步（那是探针脚本）；日志和中间产物不带。
#
# 为什么不是 rsync：Windows 的 Git Bash 不带 rsync，而 git 两端都有。用 git
# 还顺带解决了三件事 —— 传输是增量的（只发变化的对象）、节点上 `git log`
# 能看到部署历史、以及不会被 Windows 的 CRLF 转换污染节点上的源码（之前用
# tar 打包上传就是这么把整棵树变成 CRLF 的）。
#
# 关键设计：同步的是**工作区快照**，不是某个提交。未提交的改动照样会传过去
# —— 否则每次改完代码都要先 commit 才能测，那是倒退。快照用临时索引生成，
# 不动 HEAD、不动暂存区、不动工作区。
set -euo pipefail

REMOTE_DIR="${QKD_REMOTE_DIR:-/opt/qkd/graph_mappo}"
REMOTE_REF="refs/heads/deploy"
LOCAL_REF="refs/qkd-deploy"          # 记录上次同步到哪，用于生成差异报告
SCRATCH_DIR=".tmp"

cd "$(dirname "$0")/.." || exit 1

# ---------------------------------------------------------------- ssh 选择
# 跑本脚本的 shell 可能是 WSL（Windows 盘在 /mnt/c）也可能是 Git Bash（在 /c）。
# 两种情况共同的问题是：shell 自带的那个 ssh（WSL 的，或 conda 的 MSYS 版）
# 找的是 Unix socket，连不上 Windows ssh-agent 服务 —— 只有 Windows 自带的
# OpenSSH 能连上。所以优先选它，并让 -x 测试自己挑出这个 shell 认得的写法。
pick_ssh() {
    for cand in "C:/Windows/System32/OpenSSH/ssh.exe" \
                "/mnt/c/Windows/System32/OpenSSH/ssh.exe" \
                "/c/Windows/System32/OpenSSH/ssh.exe" \
                "$(command -v ssh 2>/dev/null || true)"; do
        [ -n "$cand" ] && [ -x "$cand" ] && { printf '%s' "$cand"; return 0; }
    done
    echo "找不到可用的 ssh" >&2
    return 1
}
SSH_BIN="$(pick_ssh)"

# git 调用 ssh 时也要用同一个，否则它会用 Git for Windows 自带的 MSYS ssh，
# 那个同样连不上 agent。
export GIT_SSH_COMMAND="$SSH_BIN"
export GIT_SSH_VARIANT=ssh

# 目标解析顺序：环境变量 -> bootstrap 记下的目标 -> ssh 别名 qkd。
# 中间那一步是必需的：deployment/bootstrap.sh 用 user@host 直接连通（不需要
# ~/.ssh/config 里有条目），但它是个一次性命令；之后单独跑本脚本时那个
# user@host 就丢了，会回落到别名 qkd 然后报"连不上"。bootstrap 结束时把目标
# 写进 .tmp/qkd_host，这里读回来。
HOST_ALIAS="${QKD_HOST_ALIAS:-}"
if [ -z "$HOST_ALIAS" ] && [ -f "$SCRATCH_DIR/qkd_host" ]; then
    HOST_ALIAS="$(tr -d '\r\n' < "$SCRATCH_DIR/qkd_host")"
fi
HOST_ALIAS="${HOST_ALIAS:-qkd}"
REMOTE_URL="${HOST_ALIAS}:${REMOTE_DIR}"

# ---------------------------------------------------------------- 参数
MODE=sync
for arg in "$@"; do
    case "$arg" in
        --setup)     MODE=setup ;;
        --dry-run)   MODE=dry ;;
        # 保留旧参数不报错：探针脚本现在总是跟着走，这个开关已经没有作用了。
        --with-tmp)  echo "(--with-tmp 已废弃：.tmp/ 下的脚本现在总是同步)" >&2 ;;
        -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "未知参数: $arg" >&2; exit 2 ;;
    esac
done

die() { echo "错误: $*" >&2; exit 1; }

check_reachable() {
    "$SSH_BIN" -o BatchMode=yes -o ConnectTimeout=15 "$HOST_ALIAS" true 2>/dev/null \
        || die "连不上 $HOST_ALIAS（检查 ~/.ssh/config 里的别名和 ssh-agent）"
}

# ---------------------------------------------------------------- 一次性装配
if [ "$MODE" = setup ]; then
    check_reachable
    echo "== 在 $HOST_ALIAS:$REMOTE_DIR 上装配受管仓库 =="
    "$SSH_BIN" -o BatchMode=yes "$HOST_ALIAS" "bash -s" <<REMOTE
set -euo pipefail
# NOTE: this heredoc is UNQUOTED, so the LOCAL shell expands it before shipping.
# Variable expansion happening locally is the point. What must NOT appear in the
# prose below is a backtick or a dollar sign: both are live to the local shell.
# Each has already bitten this file once. A backtick-wrapped "git read-tree HEAD"
# in a comment ran against the local repo and silently unstaged a file, and a
# literal dollar-sign variable name in a comment tripped the parent's "set -u"
# guard. Keep the comments here free of both characters.
#
# --setup is what you run on a FRESH node, where this directory does not exist
# yet (deployment/setup.sh only creates the /opt/qkd parent). Without the mkdir the
# cd fails, and under "set -e" the whole setup dies right there.
mkdir -p "$REMOTE_DIR"
cd "$REMOTE_DIR"
# Guard on HEAD, not on the .git directory. A setup that died partway (e.g. the
# baseline commit failing on an empty tree before --allow-empty was added)
# leaves .git present but with no commit at all, and skipping on that state
# would leave the node with a repo the sync cannot use: the snapshot build
# starts from a "git read-tree HEAD" that has nothing to read.
if git rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "   已经是 git 仓库，跳过初始化"
else
    git init -q -b deploy .
    git config receive.denyCurrentBranch updateInstead   # push 直接更新工作区
    git config core.autocrlf false
    # 数据目录归节点所有：同步树里根本没有它，但 negation 规则会让
    # 'git add -A' 把 rate_stats.json 收进来 —— 那样下次同步就会把它删掉。
    printf 'dataset/\n%s\n' "/$SCRATCH_DIR/" > .git/info/exclude
    git add -A
    # --allow-empty matters on a FRESH node: the directory is empty, so there is
    # nothing to commit and a plain "git commit" returns 1 -- which under
    # "set -e" aborts the whole setup. The commit is not optional though, it is
    # a marker: the snapshot build starts from "git read-tree HEAD", so deploy
    # needs a HEAD to exist even when the node starts with no files at all.
    git -c user.email=sync@qkd -c user.name=sync commit -q --allow-empty \
        -m "baseline: 节点现有内容"
    echo "   初始化完成，基线提交 \$(git rev-parse --short HEAD)"
fi
git config receive.denyCurrentBranch updateInstead
REMOTE
    echo "== 装配完成。接下来直接跑: bash deployment/sync.sh"
    exit 0
fi

# ---------------------------------------------------------------- 造快照
# 临时索引：从 HEAD 起手（这样删除能被识别），再把工作区整体盖上去。
SNAP_INDEX=".git/qkd-sync-index"
rm -f "$SNAP_INDEX"
export GIT_INDEX_FILE="$SNAP_INDEX"
git read-tree HEAD
git add -A -- .
# .tmp 下的**脚本**永远跟着走（*.py / *.sh），日志、yaml、运行产物不带。
#
# 这里踩过两次坑，所以不再做成开关：
#   1. 一开始是整个 .tmp/ 强制加入，节点上跑的探针会改写自己那份被跟踪的日志
#      和 yaml，工作区变脏，而 receive.denyCurrentBranch=updateInstead 要求工作区
#      干净 —— 之后每次推送都被静默拒掉。
#   2. 改成"只加脚本、且要 --with-tmp 才加"之后，一次不带参数的同步就会把节点上
#      的探针脚本从树里移掉、checkout 顺手删掉它们，下次跑探针就是
#      "No such file or directory"。
# 脚本是节点从不改写的东西，所以永远带上最省事。
git add -f -- "${SCRATCH_DIR}/*.py" "${SCRATCH_DIR}/*.sh"
TREE="$(git write-tree)"
unset GIT_INDEX_FILE
rm -f "$SNAP_INDEX"

PARENT="$(git rev-parse -q --verify "$LOCAL_REF" 2>/dev/null || git rev-parse HEAD)"
if [ "$TREE" = "$(git rev-parse "${PARENT}^{tree}")" ]; then
    echo "工作区与上次同步的内容一致，无需同步。"
    exit 0
fi

STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# Identity is passed per-invocation rather than read from config: this workstation
# has no global git identity at all (and the sandboxed/non-sandboxed shells don't
# agree on which HOME to look in), so a bare `git commit-tree` dies with "empty
# ident name". Same reason the remote baseline commit below does it this way.
SNAP="$(git -c user.email=sync@qkd -c user.name=sync \
    commit-tree "$TREE" -p "$PARENT" -m "sync ${STAMP}")"

echo "== 本次同步内容 =="
git diff --stat "$PARENT" "$SNAP" | tail -n 30
CHANGED="$(git diff --name-only "$PARENT" "$SNAP" | wc -l | tr -d ' ')"
echo "   共 ${CHANGED} 个文件，快照 ${SNAP:0:10}"

if [ "$MODE" = dry ]; then
    echo "== dry-run，未推送 =="
    exit 0
fi

# ---------------------------------------------------------------- 推送
check_reachable
echo "== 推送到 $HOST_ALIAS:$REMOTE_DIR ($REMOTE_REF) =="
if ! git push --force "$REMOTE_URL" "$SNAP:$REMOTE_REF"; then
    cat >&2 <<'MSG'

推送被拒。最常见的原因是节点的工作区变脏了 ——
receive.denyCurrentBranch=updateInstead 要求工作区干净，否则拒绝更新。

先看脏在哪：

    ssh qkd 'cd /opt/qkd/graph_mappo && git status --porcelain'

如果脏的只是 .tmp/ 下的文件（那是节点自己生成的，不该被跟踪），丢弃即可：

    ssh qkd 'cd /opt/qkd/graph_mappo && git checkout -- .tmp && git clean -fd .tmp'

然后重跑本脚本。如果脏的是源码，先弄清楚它为什么变了再动手 —— 那说明节点上
有东西在改被跟踪的文件，直接 reset 会把它改的丢掉。
MSG
    exit 1
fi
git update-ref "$LOCAL_REF" "$SNAP"

echo "== 节点侧确认 =="
"$SSH_BIN" -o BatchMode=yes "$HOST_ALIAS" \
    "cd $REMOTE_DIR && git log --oneline -1 && git status --porcelain | head -5 && echo '(工作区干净)' "
echo "完成。"
