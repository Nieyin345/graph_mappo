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
#
# 两种用法：
#   wake_poll.sh          一直轮询，每次变化打一行（配 Monitor 用，Monitor 上限 30 分钟）
#   wake_poll.sh --once   **看到第一次变化就退出**（配后台 bash 用）
#
# 为什么要 --once：后台 bash 只在**进程退出**时才唤醒主代理。一个永不退出的
# 轮询循环因此**永远叫不醒人**——它照样在跑，但那是"静默"，而静默看起来和
# "还在跑"完全一样（这正是本文件开头反对 tail -f 的同一个理由，只是换了一层）。
# 所以长等待要用 --once：条件满足 → 打印 → 退出 → 唤醒。
set -u

ONCE=0
[ "${1:-}" = "--once" ] && ONCE=1

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
      [ "$ONCE" -eq 1 ] && exit 0
    fi
    sleep 120
    continue
  fi
  FAILS=0

  # 指纹：稳定行（去掉 MEM/RESPAWN 这类带易变数字的行）
  BODY=$(printf '%s\n' "$OUT")
  # VAL 行里的 `u=<末轮>` 每轮都变，但它**不是新信息**——真正的新信息是
  # 验证点（`u5=`/`u10=`/… 这些只在 eval_interval 轮才多一个）。不归一化掉，
  # 指纹每轮都变 → 每 ~3.5 分钟唤醒一次，正好是"只在有新信息时唤醒"的反面。
  # 这与本文件开头排除 MEM 的理由是同一条，只是位置不同。
  # 正则 ` u=[0-9]+` 不会误伤验证点，因为验证点是 `u5=`（u 后直接跟数字再跟 =）。
  STABLE=$(printf '%s\n' "$OUT" \
           | grep -Ev '^(MEM|RESPAWN) ' \
           | sed -E 's/^(VAL [^ ]+) u=[0-9]+/\1/')
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
        [ "$ONCE" -eq 1 ] && exit 0
      fi
    fi
  fi
  PREV_RUNS="$RUNS"

  if [ -n "$PREV" ] && [ "$HASH" != "$PREV" ] && [ -n "$HASH" ]; then
    printf '%s\n' "$OUT"
    echo "---"
    [ "$ONCE" -eq 1 ] && exit 0
  fi
  PREV="$HASH"

  if ! printf '%s' "$RUNS" | grep -q '[^ ]'; then
    echo "ALLDONE 所有训练已结束"
    printf '%s\n' "$OUT"
    exit 0
  fi
  sleep 120
done
