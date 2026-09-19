#!/usr/bin/env bash
# 本机侧唤醒器：等服务器上的一批实验跑完，自动把结果拉回来并汇总。
#
#   bash .tmp/exp/watch.sh [远端脚本名] [最长等待分钟]
#   例：bash .tmp/exp/watch.sh run.sh 180
#
# 为什么用 Monitor/后台任务而不是轮询：本脚本退出时 Claude 会被唤醒一次，
# 那时结果已经被拉回本地，可以直接接着干。远程侧用 nohup 保证 ssh 断开不影响训练。
set -uo pipefail

TARGET=${1:-run.sh}
MAXMIN=${2:-180}

SSH="C:/Windows/System32/OpenSSH/ssh.exe"
SCP="C:/Windows/System32/OpenSSH/scp.exe"
OPTS="-o BatchMode=yes -o ConnectTimeout=15"
REMOTE=/opt/qkd/graph_mappo
LOCAL_EXP=".tmp/exp"

cd "$(dirname "$0")/../.." || exit 1   # 回到工作区根（qkd_rl/）
mkdir -p "$LOCAL_EXP"

# pgrep 模式加中括号：远端 bash -c 自己的命令行也含这串字符，裸写会匹配到自己。
PAT="${TARGET%%.*}"; PAT="${PAT:0:6}[${TARGET:6:1}]${TARGET:7}"
deadline=$(( $(date +%s) + MAXMIN * 60 ))

echo "[watch] 等待远端 $TARGET 结束（最多 $MAXMIN 分钟）..."
while true; do
    state=$($SSH $OPTS qkd "if pgrep -f \"$PAT\" >/dev/null 2>&1; then echo RUNNING; else echo GONE; fi" 2>/dev/null || true)
    [ "$state" = "GONE" ] && { echo "[watch] 远端已结束"; break; }
    if [ "$(date +%s)" -gt "$deadline" ]; then
        echo "[watch] ！！超时（$MAXMIN 分钟），远端可能仍在跑"; break
    fi
    sleep 45
done

echo "[watch] 拉回结果..."
$SCP $OPTS "qkd:$REMOTE/.tmp/exp/results.csv" "$LOCAL_EXP/results.csv" 2>&1

echo "[watch] 远端日志末尾："
$SSH $OPTS qkd "tail -15 /tmp/exp_speed.log" 2>&1

echo
echo "================ 汇总 ================"
PY="D:/destop/work_space/learning_space/.venv/Scripts/python.exe"
[ -x "$PY" ] || PY=python
"$PY" "$LOCAL_EXP/harvest.py" ${WARMUP:-3}
