#!/usr/bin/env bash
# 等待**一整个波**跑完，而不是每次验证点都叫一次。
#
# 为什么不用 wake_poll.sh：那个的指纹含验证点，于是每 20 分钟（u5/u10/…）
# 就变一次 —— 4 波 × 6 个点 ≈ 24 次唤醒，而中间点多数不可行动。
# 本条链只在三件事上值得叫醒人：
#   1. 本波三个 run 全部消失（跑完）→ 可以读判决了；
#   2. 某个 run 消失但另外的还在（疑似被杀 / OOM）；
#   3. ssh 连续不通（节点可能到期）。
#
# 用法：bash wake_wave.sh vcoef1
set -u

WAVE="${1:?用法: wake_wave.sh <波标签>}"
FAILS=0
PREV=""

while true; do
  OUT=$(ssh -o ConnectTimeout=20 -o BatchMode=yes qkd \
        '/opt/qkd/venv/bin/python /tmp/wake_parse.py 2>/dev/null' 2>/dev/null)

  if [ -z "$OUT" ]; then
    FAILS=$((FAILS + 1))
    if [ "$FAILS" -ge 3 ]; then
      echo "LOST ssh 连续 3 次不通，服务器可能已到期"
      exit 0
    fi
    sleep 60
    continue
  fi
  FAILS=0

  RUNS=$(printf '%s\n' "$OUT" | grep '^RUNNING ' | sed 's/^RUNNING //')
  # 本波还活着几个
  alive=0
  for s in 42 43 44; do
    case " $RUNS " in *" ${WAVE}_s${s} "*) alive=$((alive + 1));; esac
  done

  if [ "$alive" -eq 0 ]; then
    echo "WAVE_DONE ${WAVE} 三个 run 都不在了"
    printf '%s\n' "$OUT" | grep -E '^(PAIR|VERDICT) ' || true
    printf '%s\n' "$OUT" | grep -E "^VAL ${WAVE}_s" || true
    exit 0
  fi

  # 部分消失 = 异常（本波应当三个同时在跑或三个同时结束）
  if [ -n "$PREV" ] && [ "$alive" -lt "$PREV" ]; then
    echo "ALERT ${WAVE} 有 run 提前消失（$PREV → $alive），疑似被杀/OOM"
    echo "  仍存活：$RUNS"
    printf '%s\n' "$OUT" | grep -E "^(MEM|PAIR|VERDICT) " || true
    exit 0
  fi
  PREV="$alive"
  sleep 90
done
