#!/usr/bin/env bash
# 把 chain_safe_gain.sh 的三波（ep2 / mini512 / runt）**在正确的基线上重跑**。
#
# 为什么重跑：那三波跑在 entropy_coef=0.001 上（`resolved_config.yaml` 实测），
# 而 0.001 是本项目两次同种子配对判定为错的值。在错基线上加旋钮的读数，
# 无论正负都要在 0.01 上重测一遍才知道有没有用——等于白跑。
#   证据与数据见 docs/训练诊断记录.md「启动链的三波全跑在已判定为错的基线上」。
#
# 本脚本跑两波，都叠在 **train_ent01.yaml（entropy_coef=0.01）之上**：
#   波 ep2e1    = ent01 + epochs 1→2
#   波 mini512e1 = ent01 + minibatch_size 256→512
# 于是「波 X − ent01」就是该旋钮在**现用基线**上的净效应，归因干净。
#
# runt（纯基线时间锚点）**不重跑**：它的信息量最低（不加任何旋钮），
# 而这台机器的时间要留给能回答问题的那两波。
#
# 用法（服务器上）：
#   setsid nohup bash /tmp/chain_knobs_ent01.sh > /tmp/chain_knobs_ent01.log 2>&1 < /dev/null &
set -u
cd /opt/qkd/graph_mappo || exit 1

CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
THREADS=4
UPDATES=30
# ★ 与 chain_safe_gain.sh 的唯一实质差别：BASE 末尾加了 train_ent01.yaml
BASE="rl_algorithm.yaml train_full_rl.yaml train_ent01.yaml"

wait_mem() {   # $1 = 需要多少 GB
  for _ in $(seq 1 240); do
    a=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
    [ "$a" -ge "$1" ] && { echo "  [$(date -Is)] 内存够: ${a}G >= ${1}G"; return 0; }
    sleep 30
  done
  echo "  [$(date -Is)] ⚠ 等内存超时（需 ${1}G）"; return 1
}

launch() {     # $1=run-name  $2=seed  $3...=额外 config（裸文件名）
  local name="$1"; local seed="$2"; shift 2
  for c in "$@"; do
    case "$c" in
      configs/*|/*) echo "  !! --configs 要裸文件名，收到 '$c'"; return 1;;
    esac
    [ -f "configs/$c" ] || { echo "  !! 缺 configs/$c"; return 1; }
  done
  setsid nohup env OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
    /opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py \
      --configs $BASE "$@" \
      --checkpoint "$CKPT" \
      --seed "$seed" --num-updates "$UPDATES" --run-name "$name" \
      > "/tmp/${name}.log" 2>&1 < /dev/null &
  echo "  [$(date -Is)] 启动 ${name} (seed $seed)"
  sleep 10
}

wave() {       # $1=波标签 $2=额外 config（可为空）
  local label="$1"; local cfg="${2:-}"
  local marker="/tmp/wave_${label}.go"
  if [ -f "$marker" ]; then
    echo "=== 波 ${label}: 已有标记，跳过 ==="
    return 0
  fi
  wait_mem 100 || return 1
  echo "=== 波 ${label}: 启动（BASE 含 train_ent01.yaml）==="
  for s in 42 43 44; do
    if [ -n "$cfg" ]; then launch "${label}_s${s}" "$s" "$cfg"
    else launch "${label}_s${s}" "$s"; fi
  done
  sleep 45
  echo "  --- 启动后自检（含基线核对）---"
  local bad=0
  for s in 42 43 44; do
    local name="${label}_s${s}"
    local f="/tmp/${name}.log"
    if grep -qiE "error|traceback|parsererror" "$f" 2>/dev/null; then
      echo "  !! ${name} 报错:"; grep -iE "error|traceback" "$f" | head -3 | sed 's/^/      /'
      bad=1
      continue
    fi
    # ★ 核对 resolved_config：确认真的跑在 ent=0.01 上（不是只看"进程在"）
    local rc="outputs/${name}/resolved_config.yaml"
    if [ -f "$rc" ]; then
      local ec; ec=$(grep -A40 "^train:" "$rc" | grep "entropy_coef" | head -1 | awk '{print $2}')
      if [ "$ec" = "0.01" ]; then
        echo "  ok ${name}: entropy_coef=$ec ✓"
      else
        echo "  !! ${name}: entropy_coef=$ec（期望 0.01）—— **结果不可用**"
        bad=1
      fi
    else
      echo "  ?? ${name}: 还没写出 resolved_config.yaml"
    fi
  done
  if [ "$bad" -ne 0 ]; then
    echo "  ⚠ 有臂异常 —— 不写标记，修好后可重跑本波"
    return 1
  fi
  echo "  [$(date -Is)] 等 ${label} 三个跑完（每 run ~2.0h）"
  while pgrep -f "run-name ${label}_s" > /dev/null; do sleep 120; done
  touch "$marker"
  echo "  [$(date -Is)] 波 ${label} 结束"
}

# 等**预注册波**（ent01_s45/s46）跑完再开始。两个条件都要等：
#   1. chain_prereg.sh 还在（说明它还没启动 s45/s46）—— 若不等，本脚本会在
#      它启动之前就开始抢内存，那不是我们要的顺序；
#   2. ent01_s45/s46 正在跑。chain_prereg.sh 启动完就退出，所以必须靠这条
#      继续等它们真跑完。
# 两条合起来，无论本脚本是现在启动还是之后启动，顺序都正确。
echo "[$(date -Is)] 等预注册波（ent01_s45/s46）跑完再开始"
for _ in $(seq 1 1440); do        # 最多等 12h
  p=$(pgrep -cf "bash /tmp/chain_prereg.sh" 2>/dev/null || echo 0)
  r=$(pgrep -cf "run-name ent01_s4[56]" 2>/dev/null || echo 0)
  [ "$p" -eq 0 ] && [ "$r" -eq 0 ] && break
  sleep 30
done
echo "[$(date -Is)] 预注册波已结束（prereg=$p run=$r）"

wave ep2e1     train_safe_ep2.yaml
wave mini512e1 train_safe_mini512.yaml

echo
echo "[$(date -Is)] === knobs_on_ent01 链条结束 ==="
