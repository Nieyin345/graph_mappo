#!/usr/bin/env bash
# PreToolUse(Bash) 守卫：本机不许跑这个项目的 Python。
#
# 2026-09-15 我两次"就本地验一下"，两次把这台 15.7 GB 的机器压满（实测单进程
# 6–8 GB），把你自己的桌面卡出去。口头答应不管用，所以改成强制。
#
# 走 ssh/scp 到实验节点的调用是**正确路径，永远放行** —— 现在所有东西都该那样跑。
#
# 退出码 2 = 拦截，stderr 会回给模型；0 = 放行。

set -uo pipefail

payload="$(cat)"
cmd="$payload"
if command -v jq >/dev/null 2>&1; then
    parsed="$(printf '%s' "$payload" | jq -r '.tool_input.command // empty' 2>/dev/null)"
    [ -n "$parsed" ] && cmd="$parsed"
else
    # 没有 jq 时的退路。贪婪匹配会把命令后面的字段一起带进来，但那些字段里
    # 通常没有 python，检测照样成立 —— 这个守卫宁可误拦，也不放过本地执行。
    parsed="$(printf '%s' "$payload" | sed -n 's/.*"command"[[:space:]]*:[[:space:]]*"\(.*\)/\1/p' | head -1)"
    [ -n "$parsed" ] && cmd="$parsed"
fi

# --- 放行：任何经 ssh / scp / rsync 发到节点上的执行 ------------------------
# 只要出现 ssh/scp/rsync（含 ssh.exe 形式、含被赋给变量再 "$SSH" 调用）就放行。
# 一开始写成"必须出现在命令位置且后面跟空格"，结果把自己最常用的写法
#   SSH=/c/Windows/System32/OpenSSH/ssh.exe; "$SSH" -o BatchMode=yes qkd '...'
# 给拦了 —— 那条命令里根本没有字面量 "ssh "。放行规则宁松勿紧：漏放的代价是
# 我可以多跑一条远程命令，误拦的代价是唯一正确的路径被堵死。
ALLOW_RE='(^|[^[:alnum:]_])(ssh|scp|rsync)(\.exe)?([^[:alnum:]_]|$)'
if printf '%s' "$cmd" | grep -Eq "$ALLOW_RE"; then
    exit 0
fi

# --- 拦截：本机 python ------------------------------------------------------
# 只认"命令位置"上的 python（行首，或 ; & | ( 之后）。这样
# `ps -W | grep python` 这类查看命令、以及 `git add foo.py` 这类只是**提到**
# 某个 .py 文件的命令都不会被误伤 —— 要拦的是执行，不是提及。
BLOCK_RE='(^|[;&|(])[[:space:]]*([^[:space:]]*/)?(python[0-9.]*|py)(\.exe)?([[:space:]]|$)'
PYTEST_RE='(^|[[:space:]/;&|(])pytest([[:space:]]|$)|-m[[:space:]]+pytest'

if printf '%s' "$cmd" | grep -Eq "$BLOCK_RE|$PYTEST_RE"; then
    cat >&2 <<'MSG'
拦截：本地不跑项目代码。

本机只有 15.7 GB 内存，一次 1920 步的基准进程就涨到 6–8 GB，已经两次把机器
压满、桌面卡死。训练 / 测试 / 探针一律在实验节点上跑：

    bash scripts/server_sync.sh --with-tmp      # 先同步代码和探针
    ssh qkd 'cd /opt/qkd/graph_mappo && /opt/qkd/venv/bin/python .tmp/xxx.py ...'

长任务必须 setsid 起，否则 ssh 通道会被挂住（而且 TaskStop 会留下孤儿进程）：

    ssh qkd 'cd /opt/qkd/graph_mappo && setsid nohup \
        /opt/qkd/venv/bin/python -u .tmp/xxx.py > .tmp/xxx.log 2>&1 < /dev/null &'

确实需要在本机跑（例如不 import torch 的纯文本处理）：把
.claude/settings.json 里的这条 hook 临时去掉。
MSG
    exit 2
fi

exit 0
