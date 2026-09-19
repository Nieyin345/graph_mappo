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
  # ⚠ 初版这里是 `prev==1 && alive==0` 就直接报 WAVE_DONE，**漏了"正常收尾 vs 被杀"的
  #   分辨**——这正是同一个文件第 67 行（旋钮波那条）已经修过的 bug，而这一条漏改了。
  #   两个 run 隔 ~12s 启动，结束也差 ~12s，所以 **"2 → 1" 在每次正常收尾时都会出现**；
  #   而"1 → 0"既可能是第 2 个正常跑完，也可能是第 1 个被杀。不分辨就会把 OOM/被杀
  #   报成"结束"，而这一波正是要拿来下结论的波——报错了会直接污染结论。
  #   判据同 wake_parse.py：正常结束有 checkpoint_final.pt + 日志 `Final: UpdateStats`。
  alive45=0
  for s in 45 46; do
    case " $RUNNING " in *" ent01_s${s} "*) alive45=$((alive45 + 1));; esac
  done
  if [ "${SEEN[prereg]:-0}" -ge 1 ] && [ "$alive45" -eq 0 ]; then
    killed=""
    for s in 45 46; do
      n="ent01_s${s}"
      case " $RUNNING " in *" $n "*) continue;; esac
      d="/opt/qkd/graph_mappo/outputs/$n"
      if [ -f "$d/checkpoint_final.pt" ] \
         && grep -q "Final: UpdateStats" "/tmp/$n.log" 2>/dev/null; then
        continue
      fi
      killed="$killed $n"
    done
    if [ -n "$killed" ]; then
      echo "ALERT prereg 有 run 提前消失且**非正常收尾**：$killed"
      echo "  ⚠ 这一波的读数**不可用**，先查 /tmp/<name>.log 与 dmesg 再谈结论"
      printf '%s\n' "$OUT" | grep -E '^(MEM|PAIR|VERDICT) ' || true
      SEEN[prereg]=3        # 标成"已报过"，避免每 90s 重复刷同一行
    else
      echo "WAVE_DONE prereg ent01_s45/s46 结束（两者都是**正常收尾**）"
      printf '%s\n' "$OUT" | grep -E '^VAL ent01_s4[56] ' || true
      echo "  → 用 scripts/diag/prereg_ent01_nseeds.py 合并 n=5 判定（df=4，临界值 2.776）"
      echo "  → ★ 必须**同时**报第 4 节的「只用干净种子 45/46」读数（窗口选择偏差）"
      SEEN[prereg]=2
    fi
  elif [ "$alive45" -gt 0 ]; then
    [ "${SEEN[prereg]:-0}" -lt 1 ] && SEEN[prereg]=1
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
      # 曾见过 3 个在跑，现在少了但没归零 → **先分辨正常收尾与被杀**。
      # 三个 run 隔 10s 依次启动，结束时刻也差 ~10s，所以"3 → 2"
      # 在每次正常收尾时都会出现。判据同 wake_parse.py：正常结束有
      # checkpoint_final.pt + 日志 `Final: UpdateStats`。
      # ⚠ wake_wave.sh 初版漏了这一层，vcoef1 正常跑完时报了"疑似被杀/OOM"。
      killed=""
      for s in $SEEDS; do
        n="${w}_s${s}"
        case " $RUNNING " in *" $n "*) continue;; esac
        d="/opt/qkd/graph_mappo/outputs/$n"
        if [ -f "$d/checkpoint_final.pt" ] \
           && grep -q "Final: UpdateStats" "/tmp/$n.log" 2>/dev/null; then
          continue
        fi
        killed="$killed $n"
      done
      if [ -z "$killed" ]; then
        echo "PARTIAL ${w}: 3 → ${alive}，消失的都是**正常收尾**，继续等"
        SEEN[$w]=1
      else
        echo "ALERT ${w} 有 run 提前消失且**非正常收尾**（3 → ${alive}）"
        echo "  疑似被杀/OOM：$killed"
        echo "  仍存活：$RUNNING"
        printf '%s\n' "$OUT" | grep -E '^(MEM|PAIR|VERDICT) ' || true
      fi
    elif [ "$alive" -eq 3 ]; then
      SEEN[$w]=3
    elif [ "$alive" -gt 0 ]; then
      SEEN[$w]=1
    fi
  done

  # ---- 3. 全跑完就退出 ----
  # `-ge 2` 而不是 `-eq 2`：预注册波若报过 ALERT，SEEN 会被设成 3，
  # 用 `-eq 2` 就永远不满足，本循环会白转到 ssh 断掉为止。
  if [ "${SEEN[prereg]:-0}" -ge 2 ]; then
    allw=1
    for w in $WAVES; do
      [ "${SEEN[$w]:-0}" -ge 2 ] || allw=0
    done
    if [ "$allw" -eq 1 ]; then
      echo "ALL_DONE 预注册波与两波旋钮重做都结束了"
      printf '%s\n' "$OUT" | grep -E '^(MEM|PAIR|VERDICT) ' || true
      exit 0
    fi
  fi

  sleep 90
done
