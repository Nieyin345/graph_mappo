#!/usr/bin/env bash
# PreToolUse(Bash) 守卫：本机不许跑**吃内存**的东西。
#
# 2026-09-15 我两次"就本地验一下"，两次把这台 15.7 GB 的机器压满（实测单进程
# 6–8 GB），把桌面卡出去。口头答应不管用，所以改成强制。
#
# ### 2026-09-19 修订：为什么放宽，判据怎么变的
#
# 原版拦的是「命令里出现 python」—— 太宽了。它把 `python 模板库/代码/rules.py`
# 这种只 import 标准库（几十 MB）的纯文本处理也拦了，而这类任务本机跑完全没事。
#
# **真正的内存来源是 torch 和本项目代码（qkd_rl 的 env：numpy/scipy 图运算），
# 不是「python」这个词。** 所以判据改成：
#
#     拦 = 这条命令会拉起 torch 或 qkd_rl
#     放 = 其余一切（含只 import 标准库的 python 脚本）
#
# 三条独立的判定信号，**任一命中即拦**（宁拦不放过）：
#   ① 命令行里直接写了重量模块 / 重入口名 / 重目录
#   ② `-m pytest`、`-m qkd_rl` 这类
#   ③ 命令行指向的**本地 .py 文件**，其内容 import 了 torch/qkd_rl（会真的读文件）
#
# 走 ssh/scp/rsync 到实验节点的调用是**正确路径，永远放行**，且优先于以上全部。
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
    # 通常没有 python，检测照样成立。
    parsed="$(printf '%s' "$payload" | sed -n 's/.*"command"[[:space:]]*:[[:space:]]*"\(.*\)/\1/p' | head -1)"
    [ -n "$parsed" ] && cmd="$parsed"
fi

# --- 放行（最高优先）：任何经 ssh / scp / rsync 发到节点上的执行 ------------------
# 只要出现 ssh/scp/rsync（含 ssh.exe 形式、含被赋给变量再 "$SSH" 调用）就放行。
# 一开始写成"必须出现在命令位置且后面跟空格"，结果把自己最常用的写法
#   SSH=/c/Windows/System32/OpenSSH/ssh.exe; "$SSH" -o BatchMode=yes qkd '...'
# 给拦了 —— 那条命令里根本没有字面量 "ssh "。放行规则宁松勿紧：漏放的代价是
# 我可以多跑一条远程命令，误拦的代价是唯一正确的路径被堵死。
ALLOW_RE='(^|[^[:alnum:]_])(ssh|scp|rsync)(\.exe)?([^[:alnum:]_]|$)'
if printf '%s' "$cmd" | grep -Eq "$ALLOW_RE"; then
    exit 0
fi

# --- 先判：这条命令到底在不在本机跑 python -------------------------------------
# 只认"命令位置"上的 python / pytest（行首，或 ; & | ( 之后）。这样
# `ps -W | grep python` 这类查看命令、以及 `git add foo.py` 这类只是**提到**
# 某个 .py 文件的命令都不会被误伤 —— 要拦的是执行，不是提及。
# ★ 备选顺序是 pytest 优先：写成 `python[0-9.]*|py` 时，"pytest" 会先匹配到 `py`，
#   再要求后面是空格/行尾，撞到 "t" 失败 —— 靠回溯虽能救，但显式排前面更稳。
PY_INVOKE_RE='(^|[;&|(])[[:space:]]*([^[:space:]]*/)?(pytest|python[0-9.]*|py)(\.exe)?([[:space:]]|$)'
if ! printf '%s' "$cmd" | grep -Eq "$PY_INVOKE_RE"; then
    exit 0                       # 本机不跑 python，随便用
fi

# --- 信号 ①：命令行里直接写了重量模块 / 重入口 / 重目录 --------------------------
# 命中即拦，不需要读文件。这些是"拉到 torch / 本项目 env"的确凿写法。
#
# ★ 尾边界必须用 `([^[:alnum:]_]|$)`，**不能**用 `([[:space:].]|$)`：
#   `python -c "import torch; print(1)"` 里 torch 后面跟的是 `;`，不在后者集合内，
#   会漏拦。实测就是这么漏掉的（见 .tmp/test_hook.sh 的用例）。
#   用"非标识符字符"做边界仍然精确：`import torchvision` 里 torch 后面是 `v`，
#   是标识符字符 → 不命中，不会把无关模块误判成 torch。
# ★ 裸 `pytest` 单独列一支：它没有 `-m`、命令行里也没有 torch，但 pytest 一旦
#   收集测试就会 import 本项目代码 → 必须拦。原版是显式拦的，别丢。
HEAVY_CMD_RE='(import|from)[[:space:]]+(torch|qkd_rl|numpy|scipy)([^[:alnum:]_]|$)|-m[[:space:]]+(pytest|qkd_rl|torch)|(^|[;&|(])[[:space:]]*([^[:space:]]*/)?pytest([[:space:]]|$)|(^|[^[:alnum:]_])(train_graph_mappo|supervised_train_pg|eval_expert|compare_policies|run_baselines|eval_long_horizon)[^[:alnum:]_]|(^|[^[:alnum:]_])(qkd_rl|scripts/train|scripts/eval|scripts/baselines)/'

# --- 信号 ③：命令行指向的本地 .py 文件，内容 import 了 torch / qkd_rl -------------
# 会真的读文件。这是"信号 ① 漏掉但照样吃内存"的那一类，例如
#   python .tmp/my_probe.py     ← 命令行里没有 torch，但文件里有
# 找不到文件 / 读不了 → 不算命中（那种命令本来也会立刻报错，无害）。
heavy_file_hit() {
    local f
    # 取命令行里所有以 .py 结尾的 token（去掉引号）
    for f in $(printf '%s' "$cmd" | tr ' \t' '\n\n' | tr -d "\"'" | grep -E '\.py$'); do
        [ -f "$f" ] || continue
        if grep -Eq '(^|[^[:alnum:]_])(import|from)[[:space:]]+(torch|qkd_rl)([[:space:].]|$)' "$f" 2>/dev/null; then
            printf '%s' "$f"
            return 0
        fi
    done
    return 1
}

HEAVY_HIT=""
if printf '%s' "$cmd" | grep -Eq "$HEAVY_CMD_RE"; then
    HEAVY_HIT="命令行里直接写了重量模块 / 重入口"
else
    hit="$(heavy_file_hit || true)"
    [ -n "$hit" ] && HEAVY_HIT="目标文件 import 了 torch/qkd_rl：$hit"
fi

if [ -z "$HEAVY_HIT" ]; then
    exit 0                       # 轻量 python（只 import 标准库等）→ 放行
fi

cat >&2 <<MSG
拦截：本机不跑**吃内存**的任务（$HEAVY_HIT）。

本机 15.7 GB 内存，一次 1920 步的基准进程涨到 6–8 GB，已两次把机器压满、
桌面卡死。判据不是「是不是 python」而是「会不会拉起 torch / 本项目代码」——
只 import 标准库的纯文本处理本机跑没问题，会被放行。

训练 / 测试 / 探针一律在实验节点上跑：

    bash scripts/diag/push.sh .tmp/xxx.py        # 推脚本 + 服务器上查语法
    ssh qkd 'cd /opt/qkd/graph_mappo && /opt/qkd/venv/bin/python .tmp/xxx.py ...'

长任务必须 setsid 起，否则 ssh 通道会被挂住（而且 TaskStop 会留下孤儿进程）：

    ssh qkd 'cd /opt/qkd/graph_mappo && setsid nohup \\
        /opt/qkd/venv/bin/python -u .tmp/xxx.py > .tmp/xxx.log 2>&1 < /dev/null &'

确实需要在本机跑（例如不 import torch 的纯文本处理）：把
.claude/settings.json 里的这条 hook 临时去掉。
MSG
exit 2
