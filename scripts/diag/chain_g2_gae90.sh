#!/usr/bin/env bash
# =============================================================================
# 第二代基线（新节点重装后）：gae90 的配对 A/B + 跑到 u50 的延长
# =============================================================================
#
# ## 为什么整批重起，而不是续跑
#
# 节点重装（clnode316 → node0.qinglong-317045）后 `outputs/` 里**只剩 BC 起点**，
# 历史臂的检查点全没了 ⟹ **u50 延长无法续跑**。而且按
# 记忆 `arms-must-not-be-carried-across-nodes`：跨节点的臂**不能配对**
# （u1 rollout 指纹是节点相关的；专家锚是纯启发式所以它那一侧无所谓，
# 但"臂 vs 臂"必须同节点）。⟹ 本地镜像里那份旧节点的 `ent01_*_u30to50`
# **不能**用来回答新节点上的任何问题。
#
# 结论：这是一次**从零重建**。旧数据只当"方向线索"，不当对照。
#
# ## 预注册（跑之前写死，防事后编故事）
#
# 待检验的**唯一**机制 = 「训练得更久」（gae90 的 λ=0.90）。
# 它在旧节点上：窗口 u25/u30、n=5、Δ=+0.0146、t=3.61、df=4、临界 2.776 ⟹ 过线；
# **但换窗口就翻转**（早期 u5 是 −0.0244、t=−3.23），全程 u5–u30 只有 +0.0005。
# 所以真问题是：**那个正号是稳定平台，还是窗口挑出来的涨落？**
#
# 判据（Wave B 才用得上，这里先写死）：
#   - 若 u35–u50 仍给正号 ⟹ **不是"多训就涨"**（对照自己后半段没涨，
#     见 `scripts/diag/check_control_plateau.py`）⟹ 支持"稳定平台"。
#   - 若正号消失 ⟹ 就是窗口效应，gae90 这条线关闭。
#   - 观测单位 = **训练种子**（n=4，df=3，临界值**现算**，不手抄）。
#
# ## 为什么是 16 线程（不吃历史协议的亏）
#
# 历史协议一律 8 线程，理由是"线程数会确定性改变训练结果（差 0.018）"，
# 所以要跟旧基线保持一致。**但节点重装后旧基线已经不存在了**
# ⟹ 那个理由此刻不成立，这正是可以换线程数的**唯一窗口**。
#
# 实测（`probe_thread_scaling.py`，128 核节点）：4→8 是 1.47x、8→16 是 1.70x、
# 16→32 是 2.09x，**但 64 反而慢**。整轮按 update 占 65.3% 折算，8→16 约 1.21x。
#
# 并发上限由**内存**定（25 GB/run，251 GB ⟹ 9 个），也就是**核数不再是约束到 9 以上**
# ⟹ 用 8 线程只能占 72/144 核（半台机器闲着），用 16 线程占 128/144。
# 两者并发数上限**同为 9**（内存定）⟹ **16 线程严格更优**。
#
# ★ 硬约束：**同一节点内所有臂必须同线程数**，且 Wave B 的续跑臂必须与
#   它的父臂同线程数（否则一条 run 中途换了 regime）。
#
# ## 用法（服务器上）
#   setsid nohup bash /tmp/chain_g2_gae90.sh A > /tmp/chain_g2_gae90.A.log 2>&1 < /dev/null &
#   setsid nohup bash /tmp/chain_g2_gae90.sh B > /tmp/chain_g2_gae90.B.log 2>&1 < /dev/null &
# 完成后写 /tmp/g2_gae90_<phase>.go（唤醒句柄盯这个文件）。
# =============================================================================
set -u
cd /opt/qkd/graph_mappo || exit 1

PHASE="${1:-A}"
case "$PHASE" in A|B) ;; *) echo "!! 用法: $0 A|B"; exit 2 ;; esac

PY=/opt/qkd/venv/bin/python
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
RC_GET=/tmp/rc_get.py

SEEDS="42 43 44 45"
THREADS=16
STEADY=25.0        # GB/run，实测（minibatch 256）；见记忆 run-memory-23gb-and-growing
FLOOR=17.0         # GB，绝对余量下限（记忆 respawn-guard-two-views：只看增量会放行）
UPDATES_A=30
UPDATES_B=20
GO="/tmp/g2_gae90_${PHASE}.go"
PIDS="/tmp/g2_gae90_${PHASE}.pids"

log() { echo "[$(date -Is)] $*"; }

# ---------- 幂等 ----------
if [ -f "$GO" ]; then log "已有标记 $GO，跳过"; exit 0; fi

# ---------- 预检：**失败必须吵**（记忆 failed-launch-must-be-loud）----------
# 「跳过核对」不是「核对通过」，但它长得像通过。所以缺东西一律 exit 2。
log "=== 预检（phase=$PHASE）==="
bad=0
[ -f "$CKPT" ] || { log "!! 缺 BC 起点 $CKPT"; bad=1; }
if [ ! -f "$RC_GET" ]; then
  log "!! 缺 $RC_GET —— 配置自检**无法进行**。不要在这一步假装通过："
  log "   先 `bash scripts/diag/push.sh` 把它推上去。配置核对失败的后果是*结果不可用*。"
  bad=1
fi
# `--configs` 每项按 <ROOT>/configs/<name> 解析，所以必须是**裸文件名**
BASE_CFGS="rl_algorithm.yaml train_full_rl.yaml train_ent01.yaml"
for c in $BASE_CFGS; do
  case "$c" in configs/*|/*) log "!! --configs 要裸文件名，收到 '$c'"; bad=1 ;; esac
  [ -f "configs/$c" ] || { log "!! 缺 configs/$c"; bad=1; }
done
[ "$PHASE" = "A" ] && for c in train_gae90.yaml; do
  [ -f "configs/$c" ] || { log "!! 缺 configs/$c"; bad=1; }
done
[ "$bad" -eq 1 ] && { log "预检失败，**不启动**"; exit 2; }
log "预检通过：BC 起点 ✓ ｜ rc_get ✓ ｜ 配置 ✓"

# ---------- 这一波要起哪些臂 ----------
# 命名：`_v2` = 节点重装后的第二代基线。**不要复用旧名字**（本地镜像里有同名旧节点
# 产物，混起来就是跨节点配对，见记忆 arms-must-not-be-carried-across-nodes）。
#
# ⚠ 臂清单写成**文件**（`|` 分隔、一行一臂），不写成 shell 变量：
#   配置列表里有空格，`for job in $JOBS` 会把它按词拆坏，`N_RUNS` 也会算错。
#   第一版就是写成变量的，`grep -c .` 数出来的是**词数**不是臂数。
RUNS=/tmp/g2_gae90_${PHASE}.runs
: > "$RUNS"
BASE_CFGS="rl_algorithm.yaml train_full_rl.yaml train_ent01.yaml"

if [ "$PHASE" = "B" ]; then
  # 延长：从各自的 u30 检查点续跑，**配置一律不动**（保持 u1..u50 是一条曲线）
  UPDATES=$UPDATES_B
  for s in $SEEDS; do
    for p in ent01_v2 gae90_v2; do
      c="outputs/${p}_s${s}/checkpoint_update_000030.pt"
      if [ ! -f "$c" ]; then
        log "!! phase B 缺父臂检查点 $c —— **不启动**（宁可漏跑，不拿半截数据充数）"
        exit 2
      fi
      cfgs="$BASE_CFGS"
      case "$p" in gae90_*) cfgs="$BASE_CFGS train_gae90.yaml" ;; esac
      echo "${p}_s${s}_u30to50|${cfgs}|${s}|${c}" >> "$RUNS"
    done
  done
  log "phase B 父臂检查点全部就位（$(wc -l < "$RUNS") 条）"
else
  UPDATES=$UPDATES_A
  for s in $SEEDS; do
    echo "ent01_v2_s${s}|${BASE_CFGS}|${s}|${CKPT}" >> "$RUNS"
    echo "gae90_v2_s${s}|${BASE_CFGS} train_gae90.yaml|${s}|${CKPT}" >> "$RUNS"
  done
fi

N_RUNS=$(wc -l < "$RUNS")

# ---------- 启动前：已在跑的不要重起（问**进程表**，不问自己的账本）----------
# 记忆 chain-must-ask-proc-not-its-own-ledger：`launched` 是局部账本，链一重启就从空
# 开始 ⟹ 把在跑的臂再起一遍，两份进程写同一 outputs ⟹ 读数作废。
if [ -s "$PIDS" ]; then
  alive=0
  for p in $(cat "$PIDS"); do kill -0 "$p" 2>/dev/null && alive=$((alive + 1)); done
  [ "$alive" -gt 0 ] && { log "已有 ${alive} 个臂在跑（$PIDS）—— 不重复启动"; SKIP=1; }
fi
SKIP=${SKIP:-0}

if [ "$SKIP" = "0" ]; then
  # ---------- 内存门：**打印它自己的输入**（记忆 gate-must-print-its-inputs）----------
  # 恒真的门不报错，直到 OOM 才现形。所以这里把三个量都打印出来，并断言已知答案。
  NEED=$(awk -v n="$N_RUNS" -v s="$STEADY" -v f="$FLOOR" 'BEGIN{printf "%.0f", n*s+f}')
  log "内存门：可用 − Σ待涨 − 本波稳态 ≥ 余量下限"
  log "  输入: 臂数=$N_RUNS  稳态/run=${STEADY}G  下限=${FLOOR}G  ⟹ 需要 ${NEED}G"
  # 断言：8 臂 × 25 + 17 = 217。写死一个已知答案，防止公式被改坏。
  [ "$NEED" = "217" ] || { log "!! 内存公式自检失败（算出 ${NEED}G，期望 217G）—— 停下"; exit 2; }
  for i in $(seq 1 60); do
    avail=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
    [ "$avail" -ge "$NEED" ] && { log "  实测可用 ${avail}G ≥ ${NEED}G ✓ 放行"; break; }
    log "  实测可用 ${avail}G < ${NEED}G，等 60s（$i/60）"
    [ "$i" -eq 60 ] && { log "!! 等内存超时 —— **不启动**"; exit 2; }
    sleep 60
  done

  # ---------- 启动 ----------
  log "=== 启动 ${N_RUNS} 臂（${THREADS} 线程，${UPDATES} 轮）==="
  : > "$PIDS"
  while IFS='|' read -r name cfgs seed ckpt; do
    if [ -d "outputs/$name" ] && [ -f "outputs/$name/metrics.jsonl" ]; then
      log "  !! outputs/$name 已存在 —— 跳过（outputs 只增不删，不覆盖）"
      continue
    fi
    setsid nohup env OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
      $PY -u scripts/train/train_graph_mappo.py \
        --configs $cfgs --checkpoint "$ckpt" \
        --seed "$seed" --num-updates "$UPDATES" --run-name "$name" \
        > "/tmp/${name}.log" 2>&1 < /dev/null &
    echo "$!" >> "$PIDS"
    log "  启动 ${name} pid=$! （seed ${seed}，起点 $(basename "$ckpt")）"
    sleep 8     # 错开，避免同时抢内存峰值
  done < "$RUNS"
  log "PID 已记入 $PIDS"
fi

# ---------- 启动后验证：**进程在 ≠ 跑对了**（记忆 failed-launch-must-be-loud）----------
log "等 120s 让各臂写出 resolved_config.yaml 并跑过第一轮…"
sleep 120
log "=== 启动后验证 ==="
bad=0
while IFS='|' read -r name cfgs seed ckpt; do
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
    grep -E "update|success" "$lg" | tail -2 | sed 's/^/       /'
    bad=1; continue
  fi
  # 配置自检：**按点分路径**，不按行（rc_get.py 的 docstring 记了这个坑）
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
done < "$RUNS"
[ "$bad" -ne 0 ] && { log "!! 有臂异常 —— **不写标记**，修好后可重跑"; exit 1; }
log "启动后验证全部通过 ✓"

# ---------- 等跑完（PID + 超时，不用 pgrep 自匹配）----------
TMO=$((8 * 3600))
log "等 ${N_RUNS} 个跑完（${UPDATES} 轮，16 线程约 $((UPDATES * 140 / 60)) 分钟 + 余量）"
waited=0
while :; do
  alive=0
  for p in $(cat "$PIDS" 2>/dev/null); do kill -0 "$p" 2>/dev/null && alive=$((alive + 1)); done
  [ "$alive" -eq 0 ] && { log "✓ 全部 PID 已退出（等了 ${waited}s）"; break; }
  if [ "$waited" -ge "$TMO" ]; then
    log "✗ 超时 ${TMO}s：仍有 ${alive} 个 PID 存活 —— **不写标记**"
    exit 1
  fi
  sleep 120; waited=$((waited + 120))
done

touch "$GO"
log "=== phase ${PHASE} 结束，可以判读 ==="
log "→ 抓结果：bash scripts/diag/fetch_results.sh（或由主代理抓）"
