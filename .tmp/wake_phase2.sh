#!/usr/bin/env bash
# 唤醒器（本地端）：轮询服务器，**只在有新信息时**输出一行。每行 = 一次唤醒。
#
# ── 为什么又写一个（而不是用 .tmp/wake_poll.sh + wake_parse.py）────────────
# wake_parse.py 里有一段**自动补起 ent01_g999_s43** 的逻辑。g999 属于已废弃的
# 工作线，那个 auto-respawn 会在无人察觉时**重起一条废臂**、白烧 25 GB。
# 记忆里的判据：「建在错误数量级上的自动保护就是破坏措施，比没有更糟」。
# ⟹ 本唤醒器**只报告、绝不动作**。要起什么由主代理看到唤醒后自己决定。
#   （旧的 wake_poll.sh/wake_parse.py 不再使用；不要删，作为历史保留。）
#
# ── 指纹规则（继承旧版，踩过坑）──────────────────────────────────────────
#   只把**稳定行**算进指纹。含 MemAvailable / 时间戳的行**必须排除**，
#   否则每轮都"有变化" ⟹ 每 2 分钟唤醒一次，正好是"只在有新信息时"的反面。
#
# ── 事件 ──────────────────────────────────────────────────────────────
#   PROBE   探针日志出现了新的结论行
#   ARM     某个 run 的 update 数 / 验证点变了
#   GONE    run 目录消失或进程没了（可能是被杀/OOM）
#   ALLDONE 所有目标都不再推进
#   LOST    ssh 连续 3 次不通
#
# ★ 所有 ssh 都带 timeout（记忆：等待句柄必须带超时）。
# ★ 不用 pgrep -f（会自匹配 + 数进 bash -c 包装器）；用 ps + awk 筛 comm。
# ★ 不用 bc（缺失时静默返回空串 ⟹ 门永不过）。
set -u

HOST=qkd
PROBE_LOG=/tmp/dlhead_fix.log      # 修正版 dl 扫描（当前主目标）
PROBE_LOG2=/tmp/oldprobe15.log     # 旧探针全 15 种子（交叉核对 86.45%）
RPOLL_EVERY=120          # 秒
PREV=""
FAILS=0

emit() { printf '%s\n' "$1"; }

snapshot() {
  # 输出：稳定行（探针结论 + 各 run 的 update/验证点数）。**不含内存、不含时刻。**
  timeout 45 ssh -o ConnectTimeout=20 -o BatchMode=yes "$HOST" '
    for L in '"$PROBE_LOG"' '"$PROBE_LOG2"'; do
      echo "== 探针 $L =="
      if [ -f "$L" ]; then
        grep -E "^( *[0-9]+ +[0-9]| *⚠|DECISION_|  ✗|  ✓|  拓扑|  并集|  \*\*|  ⟹)" "$L" 2>/dev/null | tail -14
        tail -3 "$L" 2>/dev/null | grep -q "DECISION_" && echo "DONE_$L"
      else
        echo "(不存在)"
      fi
    done
    echo "== 在跑的探针 =="
    ps -eo comm,args | awk "\$1 ~ /^python/ && /rlprobe/" \
      | sed -E "s|.*/rlprobe/([^ ]+).*|\1|" | sort | tr "\n" " "
    echo
    echo "== 在跑的 run =="
    ps -eo comm,args | awk "\$1 ~ /^python/ && /train_graph_mappo/" \
      | sed -E "s/.*--run-name +([^ ]+).*/\1/" | sort | tr "\n" " "
    echo
    echo "== run 进度 =="
    for d in /opt/qkd/graph_mappo/outputs/*/; do
      f="$d/metrics.jsonl"
      [ -f "$f" ] || continue
      echo "$(basename "$d") u=$(grep -c "\"update\"" "$f" 2>/dev/null || echo 0) v=$(grep -c eval_validation "$f" 2>/dev/null || echo 0)"
    done | sort
  ' 2>/dev/null
}

emit "WAKE 唤醒器启动（只报告、不动作）。目标：探针 $PROBE_LOG + 所有 run。"
emit "（每 ${RPOLL_EVERY}s 轮询；无变化时静默）"

while true; do
  OUT=$(snapshot)

  if [ -z "$OUT" ]; then
    FAILS=$((FAILS + 1))
    if [ "$FAILS" -eq 3 ]; then
      emit "LOST ssh 连续 3 次不通 —— 节点可能已到期或地址变了（CloudLab 地址是动态的）"
      FAILS=0
    fi
    sleep "$RPOLL_EVERY"
    continue
  fi
  FAILS=0

  # 指纹：整段快照（已是稳定行；内存与时刻从未进入）
  FP=$(printf '%s' "$OUT" | md5sum | cut -d' ' -f1)
  if [ "$FP" != "$PREV" ]; then
    PREV="$FP"
    if printf '%s' "$OUT" | grep -q "^DONE_"; then
      emit "PROBE 探针完成 —— 结论："
      printf '%s\n' "$OUT" | grep -vE "^== |^DONE_|^$" | tail -16
    fi
    RUNS=$(printf '%s' "$OUT" | sed -n '/== 在跑的 run ==/,/== run 进度 ==/p' | grep -v "^== " | tr -d '\n')
    PROBES=$(printf '%s' "$OUT" | sed -n '/== 在跑的探针 ==/,/== 在跑的 run ==/p' | grep -v "^== " | tr -d '\n')
    emit "ARM 状态变化。探针：${PROBES:-（无）} 训练：${RUNS:-（无）}"
    printf '%s\n' "$OUT" | sed -n '/== run 进度 ==/,$p' | tail -n +2 | grep -v "^== " | tail -12
  fi

  sleep "$RPOLL_EVERY"
done
