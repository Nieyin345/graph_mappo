#!/usr/bin/env bash
# wait_for.sh 的**证伪测试**：先证明它抓得住已知的坏形状，再拿它去守真东西。
#
# 为什么要有这个文件：`wait_for.sh --lint` 本身也是代码，一个抓不住东西的 linter
# 比没有更坏 —— 它给人"检查过了"的错觉。所以每个坏形状都必须有一个**故意写坏**
# 的样本把它逼出来。这与 `.tmp/test_hook.sh`（hook 的 24 用例）是同一个方法：
# 上生产前先证伪。
#
#   bash scripts/diag/test_wait_for.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WF="$HERE/wait_for.sh"
[ -f "$WF" ] || { echo "找不到 $WF" >&2; exit 2; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
ok()   { pass=$((pass+1)); printf '  ✓ %s\n' "$1"; }
no()   { fail=$((fail+1)); printf '  ✗ %s\n' "$1"; }

# 断言 lint 对某段代码返回什么
expect_lint() {  # expect_lint <rc,0|1> <用例名> <代码>
    local want="$1" name="$2" code="$3"
    printf '%s\n' "$code" > "$TMP/case.sh"
    bash "$WF" --lint "$TMP/case.sh" >/dev/null 2>&1
    local got=$?
    if [ "$got" -eq "$want" ]; then ok "$name"
    else no "$name（期望 rc=$want，实得 rc=$got）"; fi
}

echo "=== 1. 四个已实测踩到的坏形状，lint 必须全部抓住 ==="

expect_lint 1 "形状1 until 里 || 与 && 混用未分组（优先级陷阱）" \
'until [ -f /tmp/a.go ] || [ -f /tmp/b.go ] && [ ! -d /tmp/c ]; do sleep 10; done'

expect_lint 1 "形状2 pgrep -q（本节点 procps 不支持）" \
'until [ -f /tmp/a.go ] && ! pgrep -qf PAT; do sleep 10; done'

expect_lint 1 "形状3 无限循环且无超时判据" \
'while ! test -f /tmp/done; do sleep 30; done'

expect_lint 1 "形状4 pgrep -c 用 || echo 0 兜底（会得到 0\\n0）" \
'n=$(pgrep -cf train_graph_mappo.py 2>/dev/null || echo 0)'

expect_lint 1 "多种坏形状同时出现" \
'until [ -f /tmp/a ] || [ -f /tmp/b ] && [ ! -d /tmp/c ]; do
    pgrep -qf X && sleep 1
done'

# ---- 形状 6：完成标记 = 等进程退出（2026-09-19 实测的真事）----
# `probe_stop_sensitivity.py` 因 --values 收到负数被 argparse 当成选项标志而**秒退**，
# 看门狗立刻报「已完成」——一次**从未运行**的探针差点被当成跑完。
expect_lint 1 "形状6 等进程退出后 touch 完成标记（秒死也算完成）" \
'PID=$!
while kill -0 "$PID" 2>/dev/null; do sleep 10; done
touch /tmp/probe.done'

expect_lint 1 "形状6 变体：用 pgrep 等进程、同样 touch done" \
'while pgrep -f "[p]robe_x" > /dev/null; do sleep 10; done
touch /tmp/probe_complete'

echo
echo "=== 2. 正确写法不能被误报（否则 linter 会被绕过）==="

# ★ 这条用例本身改过一次：最初写的"正确写法"只有分组、**没有超时**，被形状 3
#   抓了。当时第一反应是 lint 误报，看清楚才发现是**用例不合格** —— 按我自己
#   定的标准，"没有超时的循环"就是坏形状，哪怕逻辑分组完全正确。
#   保留这段记录：写测试时最想当然的那条，往往就是标准被自己违反的地方。
expect_lint 0 "分组 + 超时（两条标准都满足的完整正确写法）" \
'waited=0; tmo=3600
until [ -f /tmp/a.go ] || { [ -f /tmp/b.go ] && [ ! -d /tmp/c ]; }; do
    [ "$waited" -ge "$tmo" ] && exit 1
    sleep 10; waited=$((waited+10))
done'

expect_lint 0 "带显式超时的循环" \
'waited=0
while [ ! -f /tmp/done ]; do
    [ "$waited" -ge $tmo ] && exit 1
    sleep 10; waited=$((waited+10))
done'

expect_lint 0 "source wait_for.sh 后调用原语（不自造循环）" \
'source scripts/diag/wait_for.sh
wait_file /tmp/x.go 3600 "描述"
wait_gone train_graph_mappo.py 600 "描述"'

expect_lint 0 "pgrep -cf 显式兜底（不用 || echo 0）" \
'n="$(pgrep -cf PAT 2>/dev/null)"; n=${n:-0}'

# ★ 形状 6 的对偶：**等输出产物**出现则不算坏形状。这是它要逼出来的正确写法。
expect_lint 0 "等输出 JSON 出现再 touch（不是等进程退出）" \
'J=/opt/qkd/graph_mappo/outputs/eval/r.json
for i in $(seq 1 360); do
    if [ -f "$J" ]; then touch /tmp/probe.done; exit 0; fi
    if ! pgrep -f "[p]robe_x" > /dev/null 2>&1; then touch /tmp/probe.failed; exit 1; fi
    sleep 20
done
touch /tmp/probe.timeout'

# ★ 这条是 2026-09-19 实测踩到的**误报**：我在 .tmp/run_mode_de.sh 的注释里
#   写了「无超时的 `while pgrep ...; do sleep; done` 是坏形状」，结果被形状 3
#   抓了 —— linter 举报了自己正在解释的那个反模式。
#   一个会举报"文档里描述反模式"的 linter 是误报机器，会被绕过。故先剥注释。
expect_lint 0 "注释里描述坏形状，不得被举报（否则文档即误报）" \
'# 坏形状示例（反面教材）：while pgrep -f X > /dev/null; do sleep 120; done
# 再举一例：until [ -f a ] || [ -f b ] && [ ! -d c ]; do sleep 1; done
echo ok'

expect_lint 0 "真实代码里带超时的 PID 轮询（本项目 run_mode_de.sh 的写法）" \
'TMO=$((12 * 3600)); waited=0
while :; do
    alive=0
    for p in $(cat "$PIDS"); do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
    [ "$alive" -eq 0 ] && break
    if [ "$waited" -ge "$TMO" ]; then exit 1; fi
    sleep 120; waited=$((waited+120))
done'

echo
echo "=== 3. 原语本身要真的会超时（不是死等，也不是立刻返回）==="
# 这是本文件存在的核心：坏的等待句柄的病症是**静默立刻返回**或**永不来**，
# 所以必须实测"等不到时会在 tmo 附近返回非零"。

source "$WF"

t0=$(date +%s)
wait_file "$TMP/never_appears" 3 "不可能出现的文件" >/dev/null 2>&1
rc=$?; dt=$(( $(date +%s) - t0 ))
if [ "$rc" -ne 0 ] && [ "$dt" -ge 3 ] && [ "$dt" -le 12 ]; then
    ok "wait_file 超时正确返回非零（rc=$rc，耗时 ${dt}s）"
else
    no "wait_file 超时行为异常（rc=$rc，耗时 ${dt}s；期望非零且 3-12s）"
fi

t0=$(date +%s)
wait_file "$TMP/never_appears_2" 3 "不可能出现的文件(2)" >/dev/null 2>&1
dt=$(( $(date +%s) - t0 ))
if [ "$dt" -ge 2 ]; then ok "wait_file 没有立刻返回（耗时 ${dt}s）"
else no "wait_file 立刻返回了（耗时 ${dt}s）—— 正是短路陷阱的病症"; fi

# 已存在的文件应立刻成功
touch "$TMP/present"
if wait_file "$TMP/present" 5 "已存在的文件" >/dev/null 2>&1; then
    ok "wait_file 对已存在文件立即成功"
else
    no "wait_file 对已存在文件判定失败"
fi

echo
echo "=== 4. 能力探测：本机有没有 pgrep ==="
if command -v pgrep >/dev/null 2>&1; then
    echo "  pgrep    存在 → 进程类原语可用"
    if pgrep -qf "bash" 2>/dev/null; then echo "  pgrep -q 支持"
    else echo "  pgrep -q **不支持**（本节点实测如此，wait_for.sh 已避开）"; fi
else
    echo "  pgrep    不存在 → 进程类原语在本机跑不了（服务器上有）"
fi

echo
echo "===== $pass 通过 / $fail 失败 ====="
[ "$fail" -eq 0 ] || exit 1
