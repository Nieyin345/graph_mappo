#!/usr/bin/env bash
# 等待型脚本的**唯一正确原语**。凡是要「跑一个命令然后等结果」的地方都走这里。
#
# ### 为什么需要它：2026-09-19 一晚上抓到四个**静默失效**的等待句柄
#
# 四个都是"看起来在等，实际立刻返回/永不返回"，而且**都不报错**：
#
#  1. `until [ A ] || [ B ] && [ ! C ]; do` —— **`||` 和 `&&` 同优先级、左结合**，
#     实际解析成 `([A] || [B]) && [!C]`。B 一旦为真就短路掉整个条件，**循环根本
#     没进**，脚本瞬间跑完。正确写法是 `{ B && ! C; }` 显式分组。
#     （踩到的：等待 mini512e1 那一波，条件恒真，等了个寂寞。）
#
#  2. `[ -f /tmp/x.go ] && ! pgrep -qf PAT` —— 本节点的 procps **不支持 `-q`**。
#     pgrep 直接报 "invalid option"，返回非零 → `! ...` 恒为真 → 同上的短路。
#     **`pgrep -q` 在别处能用不代表在这里能用，属环境事实，必须实测。**
#
#  3. `pgrep -f <模式>` 自匹配：远程 `bash -c` 的命令行**自身含该模式**，
#     pgrep 会匹配到自己 → 永远"进程还活着" → 永远等下去。
#     （项目已记录过 `pkill -f` 的同一问题，见 CLAUDE.md；pgrep 同理。）
#
#  4. 定时器写**绝对时间点** —— cron 走**本机**时区（CST/UTC+8），而实验结果
#     的时间和服务器日志走**节点**时区（实验节点是 MDT/UTC-6），差 14 小时。
#     按节点时间写的 23:52 一次性任务，在本机时区看是 14 小时前 → **永不触发**。
#     规则：**定时器一律用相对延迟（秒），不用绝对钟点。**
#
# ### 用法
#
#   source scripts/diag/wait_for.sh
#   wait_desc "mini512e1 收尾"                      # 打一行人可读的"我在等什么"
#   wait_pid_file /tmp/mini512e1.pid 10800 "mini512e1"   # 等 PID 退出，带超时
#   wait_file /tmp/thread_sweep.go 10800 "线程扫描"       # 等文件出现
#   wait_gone train_graph_mappo.py 600 "进程清空"        # 等进程消失
#
# **等待必须带超时**。没有超时的等待句柄在失败时不是"等得久"，是"永远不来"，
# 而"永远不来"和"还在跑"在外部完全无法区分 —— 这正是要消灭的东西。

# --- 存活判定：防自匹配 -------------------------------------------------------
# `pgrep -f "$PAT"` 在 `ssh host '...'` 里会匹配到承载命令的 `bash -c` 自身。
# 经典解法：把模式首字符写成 `[x]`，正则仍匹配目标串（run…），但**字面量**
# `[x]un…` 不出现在自己命令行里，于是不会自匹配。
# ★ 注意：仅当模式用于 pgrep 且自己命令行里也含该模式时才需要。收 PID 时不需要。
pgrep_alive() {   # pgrep_alive <模式>  -> 0=有存活进程, 1=无
    local pat="$1" bracketed
    bracketed="$(printf '%s' "$pat" | sed 's/^\(.\)\(.*\)$/[\1]\2/')"
    pgrep -f -- "$bracketed" >/dev/null 2>&1
}

pgrep_count() {   # pgrep_count <模式>  -> 打印数量（**永远打印数字，绝不空**）
    # ⚠ 不要写 `$(pgrep -cf X || echo 0)`：pgrep -c 无匹配时**打印 0 但退出码 1**，
    #   `||` 会再补一个 0，得到 "0\n0"（项目已记录）。这里显式兜底。
    local n
    n="$(pgrep -cf -- "$1" 2>/dev/null)"
    printf '%s' "${n:-0}"
}

# --- 三条等待原语（都带超时，超时返回 1 而不是死等）----------------------------

wait_file() {     # wait_file <路径> <超时秒> <描述>
    local f="$1" tmo="$2" desc="$3" waited=0 step=10
    while [ ! -e "$f" ]; do
        [ "$waited" -ge "$tmo" ] && { echo "  ✗ 超时 ${tmo}s：$desc（$f 未出现）" >&2; return 1; }
        sleep "$step"; waited=$((waited + step))
    done
    echo "  ✓ $desc：$f 已出现（等了 ${waited}s）"
}

wait_gone() {     # wait_gone <pgrep模式> <超时秒> <描述>
    local pat="$1" tmo="$2" desc="$3" waited=0 step=15
    while pgrep_alive "$pat"; do
        [ "$waited" -ge "$tmo" ] && { echo "  ✗ 超时 ${tmo}s：$desc（$pat 仍在）" >&2; return 1; }
        sleep "$step"; waited=$((waited + step))
    done
    echo "  ✓ $desc：$pat 已退出（等了 ${waited}s）"
}

wait_pid_file() {  # wait_pid_file <pid文件> <超时秒> <描述>
    # 最稳的一条：不靠模式匹配，直接看 PID 还在不在。启动时用 `echo $! > file` 记下来。
    local pf="$1" tmo="$2" desc="$3" waited=0 step=15 pid
    if [ ! -s "$pf" ]; then echo "  ✗ pid 文件不存在或为空：$pf" >&2; return 1; fi
    pid="$(cat "$pf")"
    while kill -0 "$pid" 2>/dev/null; do
        [ "$waited" -ge "$tmo" ] && { echo "  ✗ 超时 ${tmo}s：$desc（pid $pid 仍在）" >&2; return 1; }
        sleep "$step"; waited=$((waited + step))
    done
    echo "  ✓ $desc：pid $pid 已退出（等了 ${waited}s）"
}

wait_desc() { echo "== 等待：$1 =="; }

# --- 静态体检：扫一个脚本里的等待句柄，报出已知的坏形状 -----------------------
# 用在写完之后、跑之前。比"跑起来看看"便宜得多。
#   bash scripts/diag/wait_for.sh --lint <脚本>
#
# ★ 计数辅助：必须**捕获 stdout**，不能用 `|| echo 0`。
#   `grep -c` 无匹配时打印 "0" 但退出码 1，`|| echo 0` 会**再补一行**，
#   得到 "0\n0"，随后 `[ "$n" -eq 0 ]` 报 integer expression expected
#   → 条件短路 → **坏形状被静默漏报**。
#   这正是本文件要抓的形状 4，我自己在 lint 函数里先犯了一遍
#   （2026-09-19，test_wait_for.sh 的"形状3"用例抓出来的）。
_count() {   # _count <ERE> <文件> -> 永远打印一个干净整数
    local n
    n="$(grep -cE "$1" "$2" 2>/dev/null || true)"
    printf '%s' "${n:-0}"
}

# ★ **先剥掉整行注释再分析**。
#   否则一个**记录**了坏形状的文件（比如本文件、以及任何解释这些坑的脚本）
#   会被自己的文档举报。2026-09-19 实测：`.tmp/run_mode_de.sh` 的注释里写了
#   「无超时的 `while pgrep ...; do sleep; done` 是坏形状」→ 被形状 3 抓了。
#   一个会举报"描述反模式"的 linter 是误报机器，会被绕过。
#   只剥**整行**注释（行首可带空白后接 #），不动行内 #（那可能是字符串里的）。
_strip_comments() { sed 's/^[[:space:]]*#.*$//' "$1" 2>/dev/null; }

lint_waits() {
    local f="$1" bad=0
    [ -f "$f" ] || { echo "无此文件: $f" >&2; return 2; }

    # 形状 1：(until|while) 里同时有 || 和 && —— bash 里两者**同优先级、左结合**，
    # 实际解析成 `(A || B) && C`，与读代码的人以为的 `A || (B && C)` 不同。
    # 先把 `{ ... }` 分组剥掉再判，否则 `A || { B && C; }` 这种**正确**写法会被误报。
    local body stripped n1
    body="$(_strip_comments "$f")"
    stripped="$(printf '%s\n' "$body" | sed 's/{[^}]*}//g')"
    n1="$(printf '%s\n' "$stripped" | _count '(until|while) .*\|\|.*&&' /dev/stdin)"
    [ "$n1" -gt 0 ] && { echo "✗ $n1 处 (until|while) 里 || 与 && 混用且未分组（优先级陷阱，条件会短路）"; bad=1; }

    # 形状 2：pgrep 用了本节点不支持的 -q（procps 版本老，报 invalid option）
    local n2; n2="$(printf '%s\n' "$body" | _count 'pgrep +-[a-zA-Z]*q' /dev/stdin)"
    [ "$n2" -gt 0 ] && { echo "✗ $n2 处 pgrep -q —— 本节点 procps 不支持，会退化成恒真/恒假"; bad=1; }

    # 形状 3：有循环等待，但全文件找不到任何超时判据。
    # ★ 超时判据的写法很多（$tmo/$TMO/$TIMEOUT/-ge $x/timeout 命令），
    #   所以这里**大小写不敏感**且接受常见的命名，宁可漏报也不误报。
    local n3 n4
    n3="$(printf '%s\n' "$body" | _count '(while|until) .*; *do *sleep' /dev/stdin)"
    n4="$(printf '%s\n' "$body" | _count -i '\-ge *"?\$[a-z_]*tmo|\-ge *"?\$[a-z_]*timeout|^[[:space:]]*timeout |\-ge *"?\$\{?[A-Z_]*TMO' /dev/stdin)"
    if [ "$n3" -gt 0 ] && [ "$n4" -eq 0 ]; then
        echo "✗ $n3 处循环等待但全文件找不到超时判据（失败时不是等得久，是永远不来）"; bad=1
    fi

    # 形状 4：pgrep -c / grep -c 用了 `|| echo 0` 兜底（会得到 "0\n0"）
    local n5; n5="$(printf '%s\n' "$body" | _count '(pgrep|grep) +-c[^|]*\|\| *echo 0' /dev/stdin)"
    [ "$n5" -gt 0 ] && { echo "✗ $n5 处 \`<pgrep|grep> -c ... || echo 0\`（无匹配时打印 0 但退出 1，会得到 \"0\\n0\"）"; bad=1; }

    # 形状 5：`pgrep -f <模式>` 与自身命令行同模式 → 自匹配
    local n6; n6="$(printf '%s\n' "$body" | _count 'pgrep +-[a-zA-Z]*f[^|]*\$\{?[A-Za-z_]*(SCRIPT|SELF|BASH_SOURCE)' /dev/stdin)"
    [ "$n6" -gt 0 ] && { echo "✗ $n6 处 pgrep -f 用了 \$0/\$BASH_SOURCE 类模式（会匹配到自己）；需写 [x] 转义"; bad=1; }

    [ "$bad" -eq 0 ] && echo "✓ $f：等待句柄未见已知坏形状"
    [ "$bad" -ne 0 ] && echo "  ↑ 以上问题属于：$f"
    return $bad
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    case "${1:-}" in
        --lint) shift; rc=0; for f in "$@"; do lint_waits "$f" || rc=1; done; exit $rc ;;
        --demo) shift; exec "$(dirname "${BASH_SOURCE[0]}")/test_wait_for.sh" ;;
        --help|-h|"")
            sed -n '2,60p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            ;;
        *) echo "用法: bash scripts/diag/wait_for.sh --lint <脚本>... | --demo" >&2; exit 2 ;;
    esac
fi
