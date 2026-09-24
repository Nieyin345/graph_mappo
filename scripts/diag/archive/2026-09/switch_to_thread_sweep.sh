#!/usr/bin/env bash
# 节点还剩 ~2 小时（用户 2026-09-18 22:11 告知）。
#
# ### 为什么必须现在换
#
# 排队的链条全都跑不完：
#   ep2e1       5/30，每轮 352s → 还需 2.45h → 约 00:38，**超过期限**
#   mini512e1   还没起
#   thread_sweep ~1.3h
#   g999         ~1.8h
#   ent03        ~1.9h
# 而且 ep2e1 是 epochs=2 的旋钮，平台窗口是 u25~u30，现在 u5 —— 就算
# 硬跑到期限也**没有可判读的点**，留着只是占机器。
#
# ### 换成什么，为什么是它
#
# 只保留 **thread_sweep**（8 线程 × 种子 42/43/44，约 1.3h，卡得进）：
#
#   · 它是**速度这条线上唯一剩下的、且能出结论的**实验。代码层已实测无油水
#     （update 段 98% 是模型计算），线程数是最后一个杠杆，4→8 整轮 1.34x。
#   · **对后续所有节点都有效**：判完就知道以后能不能默认 8 线程，
#     是给所有未来实验加一个 1.34x 的乘数。
#   · 两个方向都有用：测不出 → 白捡 1.34x；测得出 → 说明不能改，
#     省得以后踩坑。
#
# ent03（entropy 0.03）约 1.9h，**卡不进 2 小时**，这一轮放弃；
# 它是对已知有效方向的微调，优先级低于线程这个乘数。
#
# ### 做法
#
# 杀掉链条脚本 + ep2e1 三个 run。thread_sweep.sh **已经在服务器上等**着了
# （PID 见下），它的等待条件是
#     k=$(pgrep -c "bash /tmp/chain_knobs_ent01.sh")
#     r=$(pgrep -c "run-name (ep2e1|mini512e1)_s")
# 两者归零它就会**自动往下走**并起 8 线程三臂。所以这里只负责"清场"。
#
# ★ 杀法：先 pgrep 取 PID 再 kill，**不用 `pkill -f <模式>`**
#   —— 在 `ssh host '...'` 里远程 bash -c 的命令行自身含该模式，
#   pkill -f 会连同自己一起杀（CLAUDE.md 记过这个坑）。
#   且 ep2e1 是 setsid 起的，杀父进程组即可带走 8 个 spawn worker。
set -u

echo "=== 杀之前 ==="
date -Is
ps -eo pid,pgid,etime,args | grep "[t]rain_graph_mappo.py" \
  | grep -oE "^ *[0-9]+ +[0-9]+ +[0-9:]+.*run-name [a-z0-9_]+" | sed 's/^/  /'

# ---- 1. 链条脚本（按 PID，不用 pkill -f）----
echo
echo "=== 停链条脚本 ==="
for pat in chain_knobs_ent01 chain_g999_confirm chain_ent03; do
  pids=$(pgrep -f "bash /tmp/${pat}.sh" 2>/dev/null || true)
  for pid in $pids; do
    # 跳过自己（本脚本名与被杀模式不同，这里只是防御）
    [ "$pid" = "$$" ] && continue
    echo "  kill $pat pid=$pid"
    kill -TERM "$pid" 2>/dev/null || true
  done
done
sleep 3

# ---- 2. ep2e1 三个 run（按 pgid 杀，带走 spawn worker）----
echo
echo "=== 停 ep2e1 三个 run ==="
for s in 42 43 44; do
  pid=$(pgrep -f "run-name ep2e1_s${s}\$" 2>/dev/null || true)
  pid=$(printf '%s\n' "$pid" | head -1)
  if [ -n "${pid:-}" ]; then
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    echo "  ep2e1_s$s pid=$pid pgid=${pgid:-?}"
    if [ -n "${pgid:-}" ]; then
      kill -TERM -"$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    else
      kill -TERM "$pid" 2>/dev/null || true
    fi
  else
    echo "  ep2e1_s$s 没找到（可能已退出）"
  fi
done
sleep 8

# ---- 3. 收残留（只收 ep2e1 的；别的 run 不动）----
echo
echo "=== 复查残留 ==="
left=$(pgrep -f "run-name ep2e1_s" 2>/dev/null || true)
if [ -n "${left:-}" ]; then
  echo "  还有残留，升级 SIGKILL: $left"
  for pid in $left; do kill -KILL "$pid" 2>/dev/null || true; done
  sleep 4
else
  echo "  干净，无残留"
fi

echo
echo "=== 杀之后 ==="
date -Is
awk '/MemAvailable/{printf "  MemAvailable %.1f GB\n", $2/1048576}' /proc/meminfo
echo "  仍在跑的 run:"
ps -eo pid,etime,args | grep "[t]rain_graph_mappo.py" \
  | grep -oE "run-name [a-z0-9_]+" | sort | uniq -c | sed 's/^/    /' || echo "    （无）"

# ---- 4. 孤儿检查 ----
# OOM 会把 worker reparent 到 init；正常 SIGTERM 到进程组不该有。
n_orph=$(ps -eo ppid,args | grep "[m]ultiprocess" | awk '$1==1' | wc -l)
echo "  孤儿 worker（PPid==1 且含 multiprocess）: $n_orph"

echo
echo "=== thread_sweep 是否已自动往下走 ==="
if pgrep -f "bash /tmp/thread_sweep.sh" > /dev/null; then
  echo "  thread_sweep.sh 还在（等它的 60s 轮询）"
else
  echo "  !! thread_sweep.sh 不在了 —— 看 /tmp/thread_sweep.log"
fi
tail -6 /tmp/thread_sweep.log 2>/dev/null | sed 's/^/    /'
