#!/usr/bin/env bash
# 覆盖**整条剩余流水线**的唤醒链。
#
# 为什么重写：`wake_wave.sh vcoef1` 是一次性的——vcoef1 一跑完它就 exit，
# 而后面还排着预注册波（ent01_s45/s46）和两波旋钮重做（ep2e1/mini512e1），
# 那三件事同样需要被叫醒。用一次性唤醒器盯整条流水线，等于后三波跑完没人知道。
#
# 状态机：每波经历 pending → running → done。只在 **running→done** 时叫一次，
# 并直接打出判决所需的 PAIR/VERDICT 行。另报两类异常：
#   - 某波部分消失（疑似 OOM / 被杀）
#   - ssh 连续不通（节点可能到期）
#
# 用法：bash wake_pipeline.sh
set -u

# 还要盯的波（按预期发生顺序）。每波：标签 + 它的 run 名前缀（种子拼在后面）
WAVES="ent01 ep2e1 mini512e1"
SEEDS="42 43 44"

FAILS=0
declare -A SEEN=()        # 波 → 是否曾见过它在跑

echo "WATCH 开始盯：$WAVES（外加预注册波的 ent01_s45/s46）"

while true; do
  OUT=$(ssh -o ConnectTimeout=20 -o BatchMode=yes qkd \
        '/opt/qkd/venv/bin/python /tmp/wake_parse.py 2>/dev/null' 2>/dev/null)

  if [ -z "$OUT" ]; then
    FAILS=$((FAILS + 1))
    if [ "$FAILS" -ge 3 ]; then
      echo "LOST ssh 连续 3 次不通，服务器可能已到期 —— 停止盯守"
      exit 0
    fi
    sleep 60
    continue
  fi
  FAILS=0

  RUNNING=$(printf '%s\n' "$OUT" | grep '^RUNNING ' | sed 's/^RUNNING //')

  # ---- 1. 预注册波：ent01_s45/s46（2 个 run，不是 3 个）----
  alive45=0
  for s in 45 46; do
    case " $RUNNING " in *" ent01_s${s} "*) alive45=$((alive45 + 1));; esac
  done
  if [ "${SEEN[prereg]:-0}" -eq 1 ] && [ "$alive45" -eq 0 ]; then
    echo "WAVE_DONE prereg ent01_s45/s46 结束"
    printf '%s\n' "$OUT" | grep -E '^VAL ent01_s4[56] ' || true
    echo "  → 用 scripts/diag/ent01_vs_expert.py 合并 n=5 判定（df=4，临界值 2.776）"
    SEEN[prereg]=2
  elif [ "$alive45" -gt 0 ]; then
    SEEN[prereg]=1
  fi

  # ---- 2. 旋钮重做波 ----
  for w in $WAVES; do
    alive=0
    for s in $SEEDS; do
      case " $RUNNING " in *" ${w}_s${s} "*) alive=$((alive + 1));; esac
    done
    prev=${SEEN[$w]:-0}
    if [ "$prev" -eq 1 ] && [ "$alive" -eq 0 ]; then
      echo "WAVE_DONE ${w} 三个 run 都不在了"
      printf '%s\n' "$OUT" | grep -E "^(PAIR ${w}_s|VERDICT ${w} )" || true
      SEEN[$w]=2
    elif [ "$prev" -eq 3 ] && [ "$alive" -gt 0 ] && [ "$alive" -lt 3 ]; then
      # 曾见过 3 个在跑，现在少了但没归零 → 异常
      echo "ALERT ${w} 有 run 提前消失（3 → ${alive}），疑似被杀/OOM"
      echo "  仍存活：$RUNNING"
      printf '%s\n' "$OUT" | grep -E '^(MEM|PAIR|VERDICT) ' || true
    elif [ "$alive" -eq 3 ]; then
      SEEN[$w]=3
    elif [ "$alive" -gt 0 ]; then
      SEEN[$w]=1
    fi
  done

  # ---- 3. 全跑完就退出 ----
  if [ "${SEEN[prereg]:-0}" -eq 2 ]; then
    allw=1
    for w in $WAVES; do
      [ "${SEEN[$w]:-0}" -eq 2 ] || allw=0
    done
    if [ "$allw" -eq 1 ]; then
      echo "ALL_DONE 预注册波与两波旋钮重做都结束了"
      printf '%s\n' "$OUT" | grep -E '^(MEM|PAIR|VERDICT) ' || true
      exit 0
    fi
  fi

  sleep 90
done
