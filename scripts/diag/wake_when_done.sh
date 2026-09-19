#!/usr/bin/env bash
# 跑完自动醒来。**这是"跑一个命令然后等结果"的标准收尾方式。**
#
# 用法（本机执行，长跑，配合 run_in_background 使用）：
#   bash scripts/diag/wake_when_done.sh [--host qkd] [--timeout 10800] [--label 名字] \
#        [--wait-pid-file /tmp/x.pid] [--wait-file /tmp/x.go] [--wait-gone 模式] [--no-fetch]
#
# 它做四件事，缺一不可：
#   1. 等（用 scripts/diag/wait_for.sh 的原语，**一定带超时**）
#   2. 抓结果到本地（默认开；只抓结果不抓权重）
#   3. **推送通知**（PushNotification 走不到这里 —— 那是我这边的工具，
#      脚本只能用终端的响铃 + 桌面 toast 作为回调信号）
#   4. 打印一份"醒来后要看的东西"清单，让人一眼知道下一步
#
# ### 为什么不用 cron / 绝对时间点
#
# 2026-09-19 实测：cron 走**本机**时区（CST/UTC+8），实验结果的时间戳走**节点**
# 时区（实验节点 MDT/UTC-6），差 14 小时。按节点时间写的一次性任务在本机看是
# 14 小时前 → **永不触发**，而且不报错。所以这里一律用**相对延迟**。
#
# 附：本机时区 CST，节点时区 MDT，换算是「本机 = 节点 + 14h」。
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

HOST=qkd
TIMEOUT=10800          # 默认 3 小时
LABEL="任务"
DO_FETCH=1
WAIT_PID_FILE=""
WAIT_FILE=""
WAIT_GONE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --host)          HOST="$2"; shift 2 ;;
        --timeout)       TIMEOUT="$2"; shift 2 ;;
        --label)         LABEL="$2"; shift 2 ;;
        --wait-pid-file) WAIT_PID_FILE="$2"; shift 2 ;;
        --wait-file)     WAIT_FILE="$2"; shift 2 ;;
        --wait-gone)     WAIT_GONE="$2"; shift 2 ;;
        --no-fetch)      DO_FETCH=0; shift ;;
        -h|--help)       sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "未知参数: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$WAIT_PID_FILE$WAIT_FILE$WAIT_GONE" ]; then
    echo "至少给一个等待目标：--wait-pid-file / --wait-file / --wait-gone" >&2
    exit 2
fi

echo "=============================================="
echo " 自动唤醒守护：$LABEL"
echo " 节点 $HOST ｜ 本机 $(date '+%F %T %Z') ｜ 节点 $(ssh -o ConnectTimeout=15 -o BatchMode=yes "$HOST" 'date "+%F %T %Z"' 2>/dev/null || echo '(取不到)')"
echo " 超时 ${TIMEOUT}s（到点返回，不无限等）"
echo "=============================================="

# 把原语拿过来用 —— 老节点上不能 source 本地文件，所以整段内联过去，
# 在**节点上**判定（进程和文件都在节点上，本机看不见）。
#
# ★ 必须**去掉 wait_for.sh 末尾的 CLI 分发块**。否则内联到远程、被
#   `bash /tmp/_wake_remote.sh` 执行时 `$0 == BASH_SOURCE` 成立 → 走它自己的
#   `--help|""` 分支 → 往日志里灌 40 行用法说明（2026-09-19 实测，bzre83kol）。
#   行为没错（那个分支不 exit），但日志噪声会淹没真正的等待信息。
R="$(sed '/^if \[ "\${BASH_SOURCE\[0\]}" = "\$0" \]; then$/,$d' "$HERE/wait_for.sh")"
RC=0
{
  printf '%s\n' "$R"
  # 注意：这里用 heredoc 直接展开变量，因为要传进远程的已经是字面量
  printf '\n'
  printf 'wait_desc "%s"\n' "$LABEL"
  pf=0
  [ -n "$WAIT_PID_FILE" ] && { printf 'wait_pid_file %q %s "%s" || exit 1\n' "$WAIT_PID_FILE" "$TIMEOUT" "$LABEL(pid)"; pf=1; }
  [ -n "$WAIT_FILE" ]     && printf 'wait_file %q %s "%s" || exit 1\n'    "$WAIT_FILE"     "$TIMEOUT" "$LABEL(file)"
  [ -n "$WAIT_GONE" ]     && printf 'wait_gone %q %s "%s" || exit 1\n'    "$WAIT_GONE"     "$TIMEOUT" "$LABEL(proc)"
  printf 'echo "%s"\n' "WAKE_TRIGGERED"
} > /tmp/_wake_remote.sh

# 把远程脚本送上去再执行（避免 `ssh host 'bash -s'` 里 heredoc 与引号打架）
if ! scp -q -o ConnectTimeout=20 -o BatchMode=yes /tmp/_wake_remote.sh "$HOST:/tmp/_wake_remote.sh" 2>/dev/null; then
    echo "✗ 送脚本失败，改用 stdin 方式" >&2
    ssh -o ConnectTimeout=20 -o BatchMode=yes "$HOST" 'cat > /tmp/_wake_remote.sh' < /tmp/_wake_remote.sh
fi

ssh -o ConnectTimeout=30 -o BatchMode=yes -o ServerAliveInterval=60 "$HOST" \
    'bash /tmp/_wake_remote.sh; echo "WAKE_RC=$?"'
RC=$?

echo
echo "=============================================="
echo " 醒来：$LABEL（rc=$RC）  $(date '+%F %T')"
echo "=============================================="

if [ "$DO_FETCH" -eq 1 ] && [ "$RC" -eq 0 ]; then
    echo "--- 抓结果到本地（只要结果与方法，不要权重）---"
    bash "$ROOT/scripts/diag/fetch_results.sh" 2>&1 | tail -15
fi

# 终端响铃 —— 这是脚本能给到的最强提示；真正的推送由我在这边补
printf '\a'
echo
echo "下一步要看的东西："
echo "  * outputs/<run>/metrics.jsonl —— 每轮一行；success_rate 在最后一列附近"
echo "  * outputs/<run>/rollout_debug.jsonl —— 每轮 rollout 分解（免费的高价值诊断）"
echo "  * outputs/<run>/resolved_config.yaml —— 生效配置（文档比代码旧，以它为准）"
echo
echo "WAKE_DONE"
