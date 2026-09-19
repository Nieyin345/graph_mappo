#!/usr/bin/env bash
# =============================================================================
# gae90（λ=0.90）的配对 A/B + 跑到 u50 的延长 —— 接在 wave263 之后
# =============================================================================
#
# ## 为什么是这套臂名、这个线程数（不是我想选的，是**被在跑的波定死的**）
#
# 节点恢复后，**我自己之前挂的"节点复活即重启"监视器**自动放出了 wave263
# （`.tmp/launch_wave263.py`，即 hist32 判读波）。它已经在节点上跑着：
#
#     ent01_rerun_s{42,43,44}   对照（base，25 GB）
#     hist32_s{42,43,44}        hist 臂（63 GB，MAX_HIST=2 ⟹ s44 在排队等）
#
# 于是本波**必须**服从它的三条既有约定，否则产出的数与它对不上：
#
#   1. **臂名**：对照是 `ent01_rerun_s*`（第一代名），**不是** `_v2`。
#      我原计划的 `ent01_v2_*` 是另一套对照 ⟹ 与 wave263 的对照**不可混用**。
#   2. **线程数 8**：`launch_wave263.py` 的 `THREADS = "8"`。线程数会**确定性**
#      改变训练结果（差 0.018，记忆 `thread-count-changes-training`）
#      ⟹ 本波**必须**同为 8，我原计划的 16 线程作废。
#   3. **同一时刻只有一个启动器**：wave263 的 driver（`launch_wave263.py`）还在
#      决策。两个启动器各自看内存门会**同时通过**（记忆 `respawn-guard-two-views`）
#      ⟹ 本链**先等 driver 进程退出**（无歧义、无竞态），再接棒当唯一决策者。
#
# ## ★★ 为什么必须**分批**（2026-09-20 实测纠正）
#
# 本链要放的臂：对照 s45/s46（25G each）+ gae90 s42–s46（25G each）= **7 条 = 175G 稳态**。
#
# 第一版把 A 阶段当成"一批 7 条"，指望内存门等到 192G。
# **实测这会让机器闲置 5.5 小时、然后仍然可能失败**：
#
#   - driver 卡在 hist 门 `2<2`，**要等 hist32_s42/s43 之一跑完（~3h）**才会放 s44
#   - 放了 s44 之后再等 ~2.5h 它跑完，driver 才退出
#   - 那时空档 = 251 − (s44 63G 若仍在 + 其它) ≈ **119G < 192G** ⟹ 门还过不去
#   - 只能再等 ~2.5h 到 s44 也跑完 ⟹ 7 条才放得进去
#
# ⟹ **5.5 小时里机器只用到 126/251 GB（50%），而我在等一个自己的批次大小造成的门。**
#   这直接违反用户"把电脑性能吃满"的指示。分小批就没有这个问题。
#
# 分批**不破坏配对**：配对是在**分析**时按种子做的，不是执行时。
# 分批只把墙钟拉长（每批串行），换来的是**机器一直是满的**。
#
# 批次表（每批 ≤5 条 = 142G，正好落在任何空档里）：
#   A: gae90_s42..s46            （5 条 = 142G）
#   B: ent01_rerun_s45/s46       （2 条 =  67G）→ 补到 n=5 的对照
#   C: gae90_s{42..46}_u30to50   （5 条，续跑，需父臂 ckpt）
#   D: ent01_rerun_s{42..46}_u30to50（5 条，续跑）
#
# ## 预注册（跑之前写死，防事后编故事）
#
# **待检验的唯一机制**：λ=0.95 → 0.90（更少依赖 bootstrap）。
# 旧节点上：窗口 u25/u30、n=5、Δ=+0.0146、t=3.61、df=4、临界 2.776 ⟹ **过线**；
# **但换窗口就翻转**（单轮 u5 是 −0.0244、t=−3.23；全程 u5–u30 只有 +0.0005）。
#
# ⟹ 真问题不是"有没有效应"，而是：**那个正号是稳定平台，还是窗口挑出来的涨落？**
#
# 判据（C/D 批才用得上，现在写死）：
#   - u35–u50 仍给正号 ⟹ **不是「多训就涨」**（对照自己后半段没涨，见
#     `scripts/diag/check_control_plateau.py`）⟹ 支持「稳定平台」。
#   - u35–u50 正号消失 ⟹ 就是窗口效应 ⟹ **gae90 这条线关闭**。
#   - 观测单位 = **训练种子**（n=5，df=4）；p 与临界值一律**现算**，不手抄。
#
# ## 用法（服务器上）
#   setsid nohup bash /tmp/chain_g2_gae90.sh A > /tmp/chain_g2_gae90.A.log 2>&1 < /dev/null &
#   每一批跑完会写 /tmp/g2_gae90_<批>.go，再起下一批。
# =============================================================================
set -u
cd /opt/qkd/graph_mappo || exit 1

PHASE="${1:-A}"
case "$PHASE" in A|B|C|D) ;; *) echo "!! 用法: $0 A|B|C|D"; exit 2 ;; esac

PY=/opt/qkd/venv/bin/python
RC_GET=/tmp/rc_get.py
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt

SEEDS="42 43 44 45 46"
NEW_CTRL_SEEDS="45 46"       # wave263 已经放了 42/43/44 的对照，这两个要自己补
THREADS=8                    # ★ 必须与 launch_wave263.py 一致
STEADY=25.0                  # GB/run（base 配置，minibatch 256）
FLOOR=17.0
UPDATES_NEW=30               # 从头训到 u30
UPDATES_EXT=20               # 续跑 u30 → u50

GO="/tmp/g2_gae90_${PHASE}.go"
PIDS="/tmp/g2_gae90_${PHASE}.pids"
RUNS="/tmp/g2_gae90_${PHASE}.runs"

BASE_CFGS="rl_algorithm.yaml train_full_rl.yaml train_ent01.yaml"
GAE_CFGS="$BASE_CFGS train_gae90.yaml"

log() { echo "[$(date -Is)] $*"; }

if [ -f "$GO" ]; then log "已有标记 $GO，跳过"; exit 0; fi

# ---------- 预检：**失败必须吵**（记忆 failed-launch-must-be-loud）----------
log "=== 预检（phase=$PHASE）==="
bad=0
[ -f "$CKPT" ] || { log "!! 缺 BC 起点 $CKPT"; bad=1; }
if [ ! -f "$RC_GET" ]; then
  log "!! 缺 $RC_GET —— 配置自检**无法进行**。不要在这一步假装通过。"
  bad=1
fi
for c in $BASE_CFGS train_gae90.yaml; do
  case "$c" in configs/*|/*) log "!! --configs 要裸文件名，收到 '$c'"; bad=1 ;; esac
  [ -f "configs/$c" ] || { log "!! 缺 configs/$c"; bad=1; }
done
[ "$bad" -eq 1 ] && { log "预检失败，**不启动**"; exit 2; }
log "预检通过：BC 起点 ✓ ｜ rc_get ✓ ｜ 配置 ✓"

# ---------- 批次定义 ----------
# 每批 ≤5 条（=142G），这样**任何一个空档**都放得下，不用把机器晾着。
case "$PHASE" in
  A) FAMS="gae90new";  UPDATES=$UPDATES_NEW ;;   # gae90_s42..s46
  B) FAMS="ctrlnew";   UPDATES=$UPDATES_NEW ;;   # ent01_rerun_s45/s46
  C) FAMS="gae90ext";  UPDATES=$UPDATES_EXT ;;   # 续跑 u30→u50
  D) FAMS="ctrlext";   UPDATES=$UPDATES_EXT ;;   # 续跑 u30→u50
esac

build_runs() {   # $1 = 族名
  local fam="$1"
  : > "$RUNS"
  local s c cfgs name parent
  case "$fam" in
    gae90new)
      for s in $SEEDS; do
        echo "gae90_s${s}|${GAE_CFGS}|${s}|${CKPT}|${UPDATES_NEW}" >> "$RUNS"
      done ;;
    ctrlnew)
      for s in $NEW_CTRL_SEEDS; do
        echo "ent01_rerun_s${s}|${BASE_CFGS}|${s}|${CKPT}|${UPDATES_NEW}" >> "$RUNS"
      done ;;
    gae90ext)
      for s in $SEEDS; do
        parent="gae90_s${s}"
        c="outputs/${parent}/checkpoint_update_000030.pt"
        [ -f "$c" ] || { log "!! 缺父臂检查点 $c —— **不启动**（宁可漏跑，不拿半截数据充数）"; return 2; }
        echo "${parent}_u30to50|${GAE_CFGS}|${s}|${c}|${UPDATES_EXT}" >> "$RUNS"
      done ;;
    ctrlext)
      for s in $SEEDS; do
        parent="ent01_rerun_s${s}"
        c="outputs/${parent}/checkpoint_update_000030.pt"
        [ -f "$c" ] || { log "!! 缺父臂检查点 $c —— **不启动**"; return 2; }
        echo "${parent}_u30to50|${BASE_CFGS}|${s}|${c}|${UPDATES_EXT}" >> "$RUNS"
      done ;;
    *) log "!! 未知族 $fam"; return 2 ;;
  esac
  return 0
}

# ---------- 等 wave263 的 driver 退出 —— 无歧义、无竞态 ----------
if [ "$PHASE" = "A" ]; then
  # ⚠ 用**字符类**避免 pgrep 匹配到本脚本自己的命令行（`pkill -f` 自匹配的教训）。
  DRV='launch_wave263\.p[y]'
  if pgrep -f "$DRV" >/dev/null 2>&1; then
    log "等 wave263 的 driver 退出（它卡在 hist 门 2<2，还要放 hist32_s44）…"
    for _ in $(seq 1 720); do        # 最多等 6h
      pgrep -f "$DRV" >/dev/null 2>&1 || { log "driver 已退出"; break; }
      sleep 30
    done
    if pgrep -f "$DRV" >/dev/null 2>&1; then
      log "⚠ 等 driver 超时（6h），它还在跑 —— **不启动**（宁可漏跑，不冒险 OOM）"
      exit 1
    fi
  else
    log "wave263 的 driver 已不在（可能早已退出）"
  fi
fi

# ---------- 启动前：已在跑的不要重起（问**进程表**，不问自己的账本）----------
# 记忆 chain-must-ask-proc-not-its-own-ledger：`launched` 是局部账本，链一重启
# 就从空开始 ⟹ 把在跑的臂再起一遍，两份进程写同一 outputs ⟹ 读数作废。
# ★ 判据是 **outputs 目录已存在**，而不是 PID 还活着 —— 这才是"跑过没跑过"的
#   世界事实。下面启动循环里已按此跳过。
SKIP=0
if [ -s "$PIDS" ]; then
  alive=0
  for p in $(cat "$PIDS"); do kill -0 "$p" 2>/dev/null && alive=$((alive + 1)); done
  [ "$alive" -gt 0 ] && { log "已有 ${alive} 个本波臂在跑 —— 不重复启动"; SKIP=1; }
fi

wait_mem() {   # $1 = 本批臂数；门**打印它自己的输入**并断言已知答案
  local n="$1"
  local need
  need=$(awk -v n="$n" -v s="$STEADY" -v f="$FLOOR" 'BEGIN{printf "%.0f", n*s+f}')
  log "内存门（三量）：可用 − Σ待涨 − 本批稳态 ≥ ${FLOOR}G"
  log "  输入: 本批臂数=${n} ｜ 稳态/run=${STEADY}G ｜ 下限=${FLOOR}G ⟹ 需要 ${need}G"
  # 断言**已知答案**，防止公式被改坏（恒真的门不报错，直到 OOM 才现形）。
  case "$n" in
    2) [ "$need" = "67" ]  || { log "!! 公式自检失败（${need}G，期望 67G=2×25+17）"; return 2; } ;;
    5) [ "$need" = "142" ] || { log "!! 公式自检失败（${need}G，期望 142G=5×25+17）"; return 2; } ;;
    *) log "!! 未知批大小 ${n}，没有已知答案可断言 —— **不猜，停下**"; return 2 ;;
  esac
  local i avail
  for i in $(seq 1 240); do
    avail=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
    [ "$avail" -ge "$need" ] && { log "  实测可用 ${avail}G ≥ ${need}G ✓ 放行"; return 0; }
    log "  实测可用 ${avail}G < ${need}G，等 60s（$i/240）"
    [ "$i" -eq 240 ] && { log "!! 等内存超时（4h）—— **不启动**"; return 2; }
    sleep 60
  done
}

if [ "$SKIP" = "0" ]; then
  : > "$PIDS"
  for fam in $FAMS; do
    build_runs "$fam" || exit 2
    n=$(wc -l < "$RUNS")
    log "本批：族=${fam} ｜ ${n} 臂：$(cut -d'|' -f1 "$RUNS" | tr '\n' ' ')"
    wait_mem "$n" || exit 2
    log "=== 启动本批 ${n} 臂（${THREADS} 线程，每臂 ${UPDATES} 轮）==="
    while IFS='|' read -r name cfgs seed ckpt nupd; do
      if [ -d "outputs/$name" ] && [ -f "outputs/$name/metrics.jsonl" ]; then
        log "  !! outputs/$name 已存在 —— 跳过（outputs 只增不删，不覆盖）"
        continue
      fi
      setsid nohup env OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
        $PY -u scripts/train/train_graph_mappo.py \
          --configs $cfgs --checkpoint "$ckpt" \
          --seed "$seed" --num-updates "$nupd" --run-name "$name" \
          > "/tmp/${name}.log" 2>&1 < /dev/null &
      echo "$!" >> "$PIDS"
      log "  启动 ${name} pid=$! （seed ${seed}，起点 $(basename "$ckpt")，${nupd} 轮）"
      sleep 10
    done < "$RUNS"
    log "本批 PID 已记入 $PIDS"
  done
fi

# ---------- 启动后验证：**进程在 ≠ 跑对了** ----------
log "等 150s 让各臂写出 resolved_config.yaml 并跑过第一轮…"
sleep 150
log "=== 启动后验证 ==="
bad=0
VRUNS="$RUNS"                 # 每批只有一个族，RUNS 就是本批全部
while IFS='|' read -r name cfgs seed ckpt nupd; do
  lg="/tmp/${name}.log"
  if [ ! -f "$lg" ]; then log "  !! ${name}: 没有日志"; bad=1; continue; fi
  if grep -qiE "traceback|error|exception|valueerror|keyerror" "$lg"; then
    log "  !! ${name} 日志里有错误："
    grep -iE "traceback|error|exception|valueerror|keyerror" "$lg" | head -3 | sed 's/^/       /'
    bad=1; continue
  fi
  rc="outputs/${name}/resolved_config.yaml"
  if [ ! -f "$rc" ]; then
    log "  ?? ${name}: 还没有 resolved_config.yaml（进程可能已死）"
    tail -2 "$lg" | sed 's/^/       /'
    bad=1; continue
  fi
  got=$($PY "$RC_GET" "$rc" train.gae_lambda train.ppo.entropy_coef 2>&1)
  log "  ${name}: $(echo "$got" | tr '\n' ' ')"
  case "$name" in
    gae90_*) echo "$got" | grep -qxF "train.gae_lambda=0.9" || {
      log "       ✗ 期望 train.gae_lambda=0.9 —— **结果不可用**"; bad=1; } ;;
    ent01_*) echo "$got" | grep -qxF "train.gae_lambda=0.95" || {
      log "       ✗ 期望 train.gae_lambda=0.95 —— **结果不可用**"; bad=1; } ;;
  esac
  echo "$got" | grep -qxF "train.ppo.entropy_coef=0.01" || {
    log "       ✗ 期望 train.ppo.entropy_coef=0.01 —— **结果不可用**"; bad=1; }
done < "$VRUNS"
[ "$bad" -ne 0 ] && { log "!! 有臂异常 —— **不写标记**，修好后可重跑"; exit 1; }
log "启动后验证全部通过 ✓"

# ---------- 等跑完（PID + 超时）----------
TMO=$((10 * 3600))
log "等本批臂跑完（${UPDATES} 轮，${THREADS} 线程约 $((UPDATES * 165 / 60)) 分钟 + 余量）"
waited=0
while :; do
  alive=0
  for p in $(cat "$PIDS" 2>/dev/null); do kill -0 "$p" 2>/dev/null && alive=$((alive + 1)); done
  [ "$alive" -eq 0 ] && { log "✓ 全部 PID 已退出（等了 ${waited}s）"; break; }
  if [ "$waited" -ge "$TMO" ]; then
    log "✗ 超时 ${TMO}s：仍有 ${alive} 个 PID 存活 —— **不写标记**"; exit 1
  fi
  sleep 120; waited=$((waited + 120))
done

touch "$GO"
log "=== phase ${PHASE} 结束，可以判读 ==="
case "$PHASE" in
  A) log "→ 起 B：setsid nohup bash /tmp/chain_g2_gae90.sh B > /tmp/chain_g2_gae90.B.log 2>&1 < /dev/null &" ;;
  B) log "→ 起 C（续跑，需 gae90 的 u30 ckpt）" ;;
  C) log "→ 起 D（续跑对照）" ;;
  D) log "→ 判读：scripts/diag/check_gae90_window.py --outputs <dir> --pairs ..." ;;
esac
