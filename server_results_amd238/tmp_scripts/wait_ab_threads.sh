#!/usr/bin/env bash
# 本地监视：等服务器上的 .tmp/ab_threads.sh 跑完，然后汇总各臂结果。
# 这个文件在本地 .tmp/ 下，只用于本机轮询，不参与服务器训练。
set -uo pipefail

SSH="C:/Windows/System32/OpenSSH/ssh.exe"
OPTS="-o BatchMode=yes -o ConnectTimeout=15"
REMOTE=/opt/qkd/graph_mappo
PY=$REMOTE/venv/bin/python

# 注意 pgrep 用 "ab_thread[s].sh" 的中括号技巧：远端执行这条命令的 bash -c
# 自己的命令行里也含这串字符，写成裸的 ab_threads.sh 会匹配到自己，永远 RUNNING。
probe() {
    $SSH $OPTS qkd \
        'if pgrep -f "ab_thread[s].sh" >/dev/null 2>&1; then echo RUNNING; else echo GONE; fi' \
        2>/dev/null || true
}

deadline=$(( $(date +%s) + 5400 ))   # 最多等 90 分钟
while true; do
    state=$(probe)
    # ssh 瞬时失败时 state 为空 —— 当作"还在跑"继续等，不要误判成结束
    if [ "$state" = "GONE" ]; then
        echo "=== 驱动脚本已退出 $(date +%H:%M:%S) ==="
        break
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
        echo "=== 等待超时（90 分钟），仍在运行 ==="
        break
    fi
    sleep 30
done

echo
echo "=== /tmp/ab_threads.log 末尾 ==="
$SSH $OPTS qkd 'tail -5 /tmp/ab_threads.log' 2>&1

echo
echo "=== 是否正常收尾 ==="
$SSH $OPTS qkd 'grep -c "^=== done" /tmp/ab_threads.log' 2>&1

echo
echo "=== 各臂 metrics（update / rollout_s / update_s / success） ==="
$SSH $OPTS qkd 'for r in thr2_a thr16_a thr16_b thr2_b; do f=/opt/qkd/graph_mappo/outputs/$r/metrics.jsonl; echo "### $r"; if [ -f "$f" ]; then /opt/qkd/venv/bin/python /opt/qkd/graph_mappo/scripts/show_metrics.py "$f" 2>/dev/null | tail -n +2 | grep -E "^ *[0-9]+ "; else echo "  (还没有 metrics.jsonl)"; fi; done' 2>&1
