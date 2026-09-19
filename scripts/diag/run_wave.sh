#!/usr/bin/env bash
# 通用「启动一波配对实验 + 等它跑完 + 出判据」启动器。
#
# ### 为什么要有它（而不是每波手抄一份）
#
# `.tmp/run_mode_de.sh` 那 170 行里，**四个坑全是与实验内容无关的通用坑**：
#
#   1. 自检用 `grep -E "^  mode:" | head -1` 抓错字段（抓到 action_resolver.mode），
#      把三个健康的 run 判成"结果不可用" → exit 1 → `.go` 永不生成 → 唤醒句柄悬空
#   2. 脚本不幂等：修好后重跑会**再起 3 个**，6×24G 直接打爆内存
#   3. 内存判据按 `N_RUNS=6` 算，实际只起 3 个 → 数字对不上
#   4. 等待用 `while pgrep ...; do sleep; done`：无超时、靠模式匹配（会自匹配）
#
# 每起一波就重抄一遍 = 每起一波就重踩一遍。所以抽出来，**坑修一次管全部**。
#
# ### 用法（服务器上）
#
#   setsid nohup bash /tmp/run_wave.sh \
#       --label ent03 \
#       --name-fmt 'ent03_s{}' --seeds '42 43 44' \
#       --threads 8 --updates 30 \
#       --configs 'rl_algorithm.yaml train_full_rl.yaml train_ent01.yaml train_ent03.yaml' \
#       --expect 'train.ppo.entropy_coef=0.03 model.mode=mixed' \
#       --verdict scripts/diag/ent03_verdict.py \
#       > /tmp/ent03_sup.log 2>&1 < /dev/null &
#
# 要按种子改不同字段（如 mode 实验）时用：
#   --per-seed-config 'model.mode=demand_edge'
# 它会为每个种子生成 `configs/train_<label>_s<seed>.yaml` 并追加到 --configs 之后。
#
# 完成后写 `/tmp/<label>.go`（唤醒句柄盯这个文件）。
set -u
cd /opt/qkd/graph_mappo || exit 1

# ---------- 参数 ----------
LABEL=""; NAME_FMT=""; SEEDS=""; CONFIGS=""; EXPECT=""; VERDICT=""
PER_SEED=""; THREADS=8; UPDATES=30
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
MEM_PER_RUN=24          # 实测 23.3 GB 且每轮还涨 ~0.21 GB，按 24 算

while [ $# -gt 0 ]; do
  case "$1" in
    --label)          LABEL="$2"; shift 2 ;;
    --name-fmt)       NAME_FMT="$2"; shift 2 ;;
    --seeds)          SEEDS="$2"; shift 2 ;;
    --configs)        CONFIGS="$2"; shift 2 ;;
    --expect)         EXPECT="$2"; shift 2 ;;
    --verdict)        VERDICT="$2"; shift 2 ;;
    --per-seed-config) PER_SEED="$2"; shift 2 ;;
    --threads)        THREADS="$2"; shift 2 ;;
    --updates)        UPDATES="$2"; shift 2 ;;
    --checkpoint)     CKPT="$2"; shift 2 ;;
    *) echo "!! 未知参数: $1" >&2; exit 2 ;;
  esac
done

[ -n "$LABEL" ] && [ -n "$NAME_FMT" ] && [ -n "$SEEDS" ] && [ -n "$CONFIGS" ] || {
  echo "!! 必填: --label --name-fmt --seeds --configs" >&2; exit 2; }

GO="/tmp/${LABEL}.go"
PIDS="/tmp/${LABEL}.pids"
N_RUNS=$(echo $SEEDS | wc -w)
NEED=$((N_RUNS * MEM_PER_RUN + 8))

echo "[$(date -Is)] === ${LABEL}：${N_RUNS} 臂，${THREADS} 线程，${UPDATES} 轮 ==="

if [ -f "$GO" ]; then echo "[$(date -Is)] 已有标记 $GO，跳过"; exit 0; fi

# ---------- 幂等：已在跑就不要再起一遍（坑 2）----------
SKIP_LAUNCH=0
if [ -s "$PIDS" ]; then
  alive=0
  for p in $(cat "$PIDS"); do kill -0 "$p" 2>/dev/null && alive=$((alive + 1)); done
  if [ "$alive" -gt 0 ]; then
    echo "[$(date -Is)] 已有 ${alive} 个臂在跑（$PIDS）—— 不重复启动，直接进入等待"
    SKIP_LAUNCH=1
  fi
fi

# ---------- 每种子独立 yaml（可选）----------
if [ -n "$PER_SEED" ] && [ "$SKIP_LAUNCH" = "0" ]; then
  mkdir -p configs
  for s in $SEEDS; do
    f="configs/train_${LABEL}_s${s}.yaml"
    : > "$f"
    for kv in $PER_SEED; do
      k="${kv%%=*}"; v="${kv#*=}"
      # 只支持两层以内的点分路径（够用；再深就该写 yaml 文件了）
      case "$k" in
        *.*) printf '%s:\n  %s: %s\n' "${k%%.*}" "${k#*.}" "$v" >> "$f" ;;
        *)   printf '%s: %s\n' "$k" "$v" >> "$f" ;;
      esac
    done
    echo "  [$(date -Is)] 写 $f: $(tr '\n' ' ' < "$f")"
  done
fi

wait_mem() {   # 坑 3：按**真实**臂数算，不按想跑的算
  local want="$1" a
  for _ in $(seq 1 480); do
    a=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
    if [ "$a" -ge "$want" ]; then
      echo "  [$(date -Is)] 内存够: ${a}G >= ${want}G"; return 0
    fi
    echo "  [$(date -Is)] 内存不足: ${a}G < ${want}G，等 60s"; sleep 60
  done
  echo "  [$(date -Is)] ⚠ 等内存超时（需 ${want}G）"; return 1
}

if [ "$SKIP_LAUNCH" = "0" ]; then
  echo "[$(date -Is)] 启动前等内存 —— 需要 ${NEED}G（${N_RUNS} × ${MEM_PER_RUN}G + 8G）"
  wait_mem "$NEED" || exit 1
  echo; echo "=== 启动 ${N_RUNS} 臂 ==="
  : > "$PIDS"
  for s in $SEEDS; do
    name=$(printf '%s' "$NAME_FMT" | sed "s/{}/$s/g")
    cfgs="$CONFIGS"
    [ -n "$PER_SEED" ] && cfgs="$CONFIGS configs/train_${LABEL}_s${s}.yaml"
    setsid nohup env OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
        /opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py \
        --configs $cfgs --checkpoint "$CKPT" \
        --seed "$s" --num-updates "$UPDATES" --run-name "$name" \
        > "/tmp/${name}.log" 2>&1 < /dev/null &
    echo "$!" >> "$PIDS"
    echo "  [$(date -Is)] 启动 ${name} pid=$! (${THREADS} 线程)"
    sleep 10
  done
  echo "  PID 已记入 $PIDS"
else
  echo; echo "=== 跳过启动，直接到等待 ==="
fi

# ---------- 自检：按**点分路径**核对，不按行（坑 1）----------
# ★ rc_get.py 用**绝对路径 /tmp/rc_get.py**，不是 scripts/diag/。
#   这两个位置都对：`push.sh` 默认推到 `/tmp/`，而本脚本被 scp 到 `/tmp/` 跑。
#   第一版写成 `scripts/diag/rc_get.py`（相对 /opt/qkd/graph_mappo）——
#   文件不在那儿（它在 /tmp/，且没同步 git），于是自检**静默跳过**：
#   「跳过核对」不是「核对通过」，但它长得像通过。宁可去 /tmp 找不到就报错，
#   也不要在检查不到的时候假装查过。
RC_GET=/tmp/rc_get.py
sleep 60
echo; echo "=== 自检（按路径核对配置，不只看『进程在』）==="
if [ ! -f "$RC_GET" ]; then
  echo "  !! 缺 $RC_GET —— 自检**无法进行**。不要在这一步假装通过："
  echo "     先把 scripts/diag/rc_get.py 推上去（bash scripts/diag/push.sh）。"
  echo "     配置核对失败的后果是**结果不可用**，比晚点开跑贵得多。"
  bad=1
fi
bad=${bad:-0}
for s in $SEEDS; do
  name=$(printf '%s' "$NAME_FMT" | sed "s/{}/$s/g")
  lg="/tmp/${name}.log"
  if grep -qiE "error|traceback|parsererror|valueerror" "$lg" 2>/dev/null; then
    echo "  !! ${name} 报错:"; grep -iE "error|traceback|valueerror" "$lg" | head -3 | sed 's/^/      /'
    bad=1; continue
  fi
  rc="outputs/${name}/resolved_config.yaml"
  if [ ! -f "$rc" ]; then
    echo "  ?? ${name}: 还没写出 resolved_config.yaml（正常，约 1 分钟后再看）"; continue
  fi
  [ -f "$RC_GET" ] || continue
  got=$(/opt/qkd/venv/bin/python "$RC_GET" "$rc" $(for kv in $EXPECT; do echo "${kv%%=*}"; done) 2>&1)
  miss=0
  for kv in $EXPECT; do
    printf '%s\n' "$got" | grep -qxF "$kv" || { miss=1; echo "      ✗ 期望 $kv，实得 $(printf '%s\n' "$got" | grep -F "${kv%%=*}=" )"; }
  done
  if [ "$miss" -eq 0 ]; then
    echo "  ok ${name}: $(printf '%s' "$got" | tr '\n' ' ')"
  else
    echo "  !! ${name}: 配置不符 —— **结果不可用**"; bad=1
  fi
done

if [ "$bad" -ne 0 ]; then echo "  ⚠ 有臂异常 —— 不写标记，修好后可重跑"; exit 1; fi

# ---------- 等待：用 PID + 超时（坑 4）----------
echo; echo "  [$(date -Is)] 等 ${N_RUNS} 个跑完（${UPDATES} 轮，约 $((UPDATES*2/60+1))h 内）"
TMO=$((12 * 3600)); waited=0
while :; do
  alive=0
  for p in $(cat "$PIDS" 2>/dev/null); do kill -0 "$p" 2>/dev/null && alive=$((alive + 1)); done
  [ "$alive" -eq 0 ] && { echo "  ✓ 全部 PID 已退出（等了 ${waited}s）"; break; }
  if [ "$waited" -ge "$TMO" ]; then
    echo "  ✗ 超时 ${TMO}s：仍有 ${alive} 个 PID 存活 —— 不写标记" >&2; exit 1
  fi
  sleep 120; waited=$((waited + 120))
done
touch "$GO"
echo "  [$(date -Is)] === ${LABEL} 结束，可以判读 ==="

if [ -n "$VERDICT" ] && [ -f "$VERDICT" ]; then
  echo; echo "=== 判据 ==="
  /opt/qkd/venv/bin/python "$VERDICT" 2>&1 | tail -45
fi
