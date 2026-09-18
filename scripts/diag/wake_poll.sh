#!/usr/bin/env bash
# 唤醒链（本地端）：轮询服务器状态，**只在有新信息时输出**。每行 stdout = 一次唤醒。
#
# 用轮询而不是长连 tail -f：
#   1. ssh 断了下一轮自动恢复；
#   2. 服务器失联会输出 LOST，而不是静默假死（静默看起来就像"还在跑"）。
#
# 指纹只取**稳定行**（RUNNING/VAL/ALERT），**必须排除 MEM 与 RESPAWN**：
#   这两行里含 MemAvailable 这种每次都变的数，算进指纹会导致每一轮都"有变化"，
#   于是每 2 分钟唤醒一次 —— 正好是"只在有新信息时唤醒"的反面。
#
# 事件：RUNNING/VAL 变化、ALERT（某个 run 消失但别的还在跑 = 疑似被杀）、
#       LOST（连续 3 次不通）、ALLDONE。
set -u

PREV=""
FAILS=0
PREV_RUNS=""

while true; do
  OUT=$(ssh -o ConnectTimeout=20 -o BatchMode=yes qkd \
        '/opt/qkd/venv/bin/python /tmp/wake_parse.py 2>/dev/null' 2>/dev/null)

  if [ -z "$OUT" ]; then
    FAILS=$((FAILS + 1))
    if [ "$FAILS" -eq 3 ]; then
      echo "LOST ssh 连续 3 次不通，服务器可能已到期"
    fi
    sleep 120
    continue
  fi
  FAILS=0

  # 指纹：稳定行（去掉 MEM/RESPAWN 这类带易变数字的行）
  BODY=$(printf '%s\n' "$OUT")
  STABLE=$(printf '%s\n' "$OUT" | grep -Ev '^(MEM|RESPAWN) ')
  HASH=$(printf '%s' "$STABLE" | md5sum | cut -c1-32)

  RUNS=$(printf '%s\n' "$OUT" | grep '^RUNNING ' | sed 's/^RUNNING //')

  # 崩溃检测：上一轮有 N 个，这一轮数量变少而没到 0 —— 是"被杀"不是"跑完"
  if [ -n "$PREV_RUNS" ] && [ -n "$RUNS" ]; then
    n_prev=$(printf '%s' "$PREV_RUNS" | wc -w)
    n_now=$(printf '%s' "$RUNS" | wc -w)
    if [ "$n_now" -lt "$n_prev" ]; then
      gone=""
      for r in $PREV_RUNS; do
        case " $RUNS " in *" $r "*) ;; *) gone="$gone $r";; esac
      done
      if [ -n "$gone" ]; then
        echo "ALERT run 消失（疑似被杀，非正常结束）：$gone"
        echo "  仍存活：$RUNS"
        printf '%s\n' "$OUT"
        echo "---"
      fi
    fi
  fi
  PREV_RUNS="$RUNS"

  if [ -n "$PREV" ] && [ "$HASH" != "$PREV" ] && [ -n "$HASH" ]; then
    printf '%s\n' "$OUT"
    echo "---"
  fi
  PREV="$HASH"

  if ! printf '%s' "$RUNS" | grep -q '[^ ]'; then
    echo "ALLDONE 所有训练已结束"
    printf '%s\n' "$OUT"
    exit 0
  fi
  sleep 120
done
