#!/usr/bin/env bash
# =============================================================================
# v7 论文实验批量流水线（2026-10-07）—— **版本控制内的**启动器。
#
# 为什么要有它：论文那一批实验过去是靠 /tmp/launch_*.sh 一串即席脚本跑出来的，
# 它们**不在 git 里**、节点一换就全丢、每起一波重踩一遍通用坑（幂等、内存门、
# 线程墙、等待超时）。本脚本是那串链的**受控替代**，配方写死在文件里。
#
# -----------------------------------------------------------------------------
# 前置（本脚本不生成数据）：
#   1) 数据集：dataset/global_v7，由
#        /opt/qkd/venv/bin/python tools/dataset_v2/build_dataset_v2.py \
#          --out dataset/global_v7 --days 365 --workers 12 \
#          --constellation walker60 --cloud-model coverage_layers \
#          --capture-gamma 0.20 --intra-slot-samples 4 --allow-era5-fallback
#      生成。（`--channel-model v7` 是 2026-10-07 起的默认，可省略；v2 仅显式回退时用。）
#      ★ γ=0.20（规范 §5.1 捕获因子）必须在这里给；env 侧 rate_decay_factor
#      对含 k_max_first 的 v7 数据集是惰性的。
#      ★ --intra-slot-samples 4（2026-10-07 外部审查第 4 条）：几何每 60 s 槽移动可达
#      ~8° 仰角，单点采样会高估 SAT-GS 速率 14–135%。n=4 已收敛（n=4→16 变化 <1%）。
#   2) 专家轨迹：outputs/trajs_n45/... （由 collect_pg_phased_trajectories.py 采）；
#      或直接用 --bc 让 BC 阶段现场采（慢）。
#
# -----------------------------------------------------------------------------
# 用法（服务器，根目录 /opt/qkd/graph_mappo）：
#   setsid nohup bash scripts/batch/run_paper_experiments.sh --phase all \
#       > /tmp/paper_pipeline.log 2>&1 < /dev/null &
#
#   bash scripts/batch/run_paper_experiments.sh --dry-run          # 只打印将起的臂
#   bash scripts/batch/run_paper_experiments.sh --phase bc         # 只跑 BC 起点
#   bash scripts/batch/run_paper_experiments.sh --phase main       # 主表（Table 2）
#   bash scripts/batch/run_paper_experiments.sh --phase ablation   # 消融（Table 3）
#   bash scripts/batch/run_paper_experiments.sh --phase eval       # 评测守护（含泛化）
#
# -----------------------------------------------------------------------------
# 论文数据 ←→ 脚本臂的对应（docs/训练与实验规划及论文展示设计_20261006.md）：
#
#   Table 2 主表（1x/2x/3x 负载 × 30 留出种子 1000–1029）：
#     Random / Greedy-*      → scripts/baselines/run_baselines.py + eval_expert.py（阶段 eval）
#     Expert-Heuristic       → scripts/eval/eval_expert.py
#     gnnppo_scratch         → 阶段 main，无 --resume
#     bc_expert              → BC 起点的 BC-only 评测
#     full_ro                → v8 + BCV3 + v8_bias
#     selfnew_full_w04       → v8 + BC + self_new(horizon 3, w=0.4)   ← 本文方法
#
#   Table 3 消融（3x 负载）：
#     nograph                → + no_attention + train_nograph
#     selfshuffle_full_w04   → + self_shuffle（时序打乱对照）
#     full_ro (no commit)    → 主表 full_ro
#     gnnppo_scratch (no BC) → 主表 scratch
#     bc_expert (no RL)      → BC-only
#   Figure 6 w 敏感性：self_new w ∈ {0.0,0.2,0.4,0.6,0.8,1.0}
#   Figure 8 泛化：scenario_v7_size_small / _large 零样本；fail_* 节点失效
#
# 并发纪律（CLAUDE.md / 实验产物规范）：内存是墙不是核数；每 run 240 步约 9.4 GB
#   PSS，跑满自测（见 --probe-mem）。线程墙：并发 × cpu_threads ≤ nproc。
# =============================================================================
set -u

# ---------------------------------------------------------------- 冻结配方
STAMP="${STAMP:-$(date -u +%Y%m%d)}"
PY="${PY:-/opt/qkd/venv/bin/python}"
E="${E:-outputs/experiments}"
O="${O:-outputs/eval}"
T="${T:-outputs/trajs_v7}"

# 主场景 100 节点：60 SAT / 45 HAP / 45 GS（dataset_v7.yaml 定数据集，scenario_v7.yaml 定节点）
D="--config env_full_demand_v2.yaml --config dataset_v7.yaml --config scenario_v7.yaml"
NA="--config no_attention.yaml"                       # 关 pair-path 注意力（MLP 打分）
RO="--config v8_bias.yaml --config diag_head_readout.yaml"   # 预测头读出式偏置（本文方法用）
FO="--config forecast_off.yaml"                       # 关预测头 ⟹ v8 退化为 v3

# BC 起点（每个模型一个，跨种子复用）
BCV3="$E/v3/v7_bc/bc/supervised_pg_phased.pt"
BCN="$E/v8/v7_bc_noatt/bc/supervised_pg_phased.pt"
BCV2="$E/v2/v7_bc/bc/supervised_pg_phased.pt"

UPDATES="${UPDATES:-60}"
MAX="${MAX:-9}"          # 并发上限；起跑前用 --probe-mem 复核
THREADS="${THREADS:-8}"
SEEDS="${SEEDS:-42 43 44}"
SEEDS5="${SEEDS5:-45 46}"      # 补到 n=5 的种子
HOLDOUT="${HOLDOUT:-1000-1029}"

PHASE="all"; DRY=0; PROBE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --phase)    PHASE="$2"; shift 2 ;;
    --dry-run)  DRY=1; shift ;;
    --probe-mem) PROBE=1; shift ;;
    --max)      MAX="$2"; shift 2 ;;
    --updates)  UPDATES="$2"; shift 2 ;;
    --seeds)    SEEDS="$2"; shift 2 ;;
    -h|--help)  sed -n '2,60p' "$0"; exit 0 ;;
    *) echo "!! 未知参数: $1" >&2; exit 2 ;;
  esac
done

cd /opt/qkd/graph_mappo 2>/dev/null || { echo "!! 需在 /opt/qkd/graph_mappo 下运行" >&2; exit 1; }
ulimit -n 65535 || true

log() { echo "[$(date -Is)] $*"; }
# 幂等闩：同一 STAMP+phase 只跑一次
LOCK="/tmp/paper_${STAMP}_${PHASE}.lock"
exec 9>"$LOCK"; flock -n 9 || { log "已有同 STAMP/phase 在跑（$LOCK），退出"; exit 0; }

running() { pgrep -f "experiment.py train.*--name v7_" 2>/dev/null | wc -l; }
ev_running() { pgrep -f "experiment.py evaluate.*v7_" 2>/dev/null | wc -l; }

# ---------------------------------------------------------------- 环境自检
require_dataset() {
  [ "$DRY" = 1 ] && return 0            # dry-run 只预览计划，不要求数据/BC 就绪
  if [ ! -f dataset/global_v7/link_data.h5 ]; then
    log "!! 缺少 dataset/global_v7/link_data.h5 —— 先按本脚本头部命令生成 v7 数据集"
    exit 3
  fi
  [ -f dataset/global_v7/rate_stats.json ] || log "⚠ dataset/global_v7/rate_stats.json 缺失：归一化 p99 会回退到 10.0（特征饱和）"
}
require_bc() { [ "$DRY" = 1 ] && return 0; local f="$1"; [ -f "$f" ] || { log "!! 缺 BC 起点 $f —— 先跑 --phase bc"; exit 3; }; }

# ---------------------------------------------------------------- 内存门
# ★ 门/判据类代码必须打印它自己的输入（CLAUDE.md：恒真的门不报错，伪装成"资源不够"）
probe_mem() {
  local avail nproc per need tw
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  nproc=$(nproc)
  # 240 步 regime 实测 PSS ≈ 9.4 GB/run（CLAUDE.md）；留 4 GB 地板。
  # 注意：`local per=10 need=$((MAX*per+4))` 在 set -u 下会因 RHS 先求值而报 unbound，
  # 必须分开赋值。
  per=10
  need=$(( MAX * per + 4 ))
  log "内存门：MemAvailable=${avail}GB nproc=${nproc} MAX=${MAX} 每run≈${per}GB 需≈${need}GB"
  if [ "$avail" -lt "$need" ]; then
    log "  ⚠ 余量不足：把 MAX 降到 $(( (avail - 4) / per )) 再跑"; return 1
  fi
  tw=$(( MAX * THREADS ))
  [ "$tw" -le "$nproc" ] || log "  ⚠ 线程墙：MAX×THREADS=${tw} > nproc=${nproc}（内存够但线程超）"
  return 0
}

# ---------------------------------------------------------------- 起臂
# launch <model> <seed> <name> <bc|-> <overlay...>
launch() {
  local model="$1" seed="$2" name="$3" bc="$4"; shift 4
  [ -d "$E/$model/$name" ] && { log "  skip $name（目录已存在）"; return; }
  while [ "$(running)" -ge "$MAX" ]; do sleep 30; done
  local RES=""; [ "$bc" != "-" ] && RES="--resume $bc"
  if [ "$DRY" = 1 ]; then
    log "  [dry] $model/$name seed=$seed bc=$bc $*"; return
  fi
  setsid nohup env OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
    "$PY" -u scripts/experiment.py train --model "$model" --seed "$seed" \
      --updates "$UPDATES" --name "$name" $D "$@" --config cpu8.yaml $RES --device cpu --evaluate \
      > "/tmp/v7_${name}.log" 2>&1 < /dev/null &
  log "  launched $name pid=$! $(date +%T)"
  sleep 3
}

lcfg() { case "$1" in 1x) echo "";; 2x) echo "--config scenario_v7_x20.yaml";; 3x) echo "--config scenario_v7_x30.yaml";; esac; }
load_name() { case "$1" in 1x) echo "1x";; 2x) echo "2x";; 3x) echo "3x";; esac; }

# 主表四臂 + 本文方法（arm <arm> <load> <seed>）
arm() {
  local a="$1" L="$2" s="$3" n; local C; C=$(lcfg "$L"); n="v7_${L}_${a}_s${s}"
  case "$a" in
    full_ro)   launch v8 "$s" "$n" "$BCV3" $C $RO ;;
    lite_ro)   launch v8 "$s" "$n" "$BCN"  $C $NA $RO ;;
    gnnppo)    launch v8 "$s" "$n" "$BCN"  $C $NA $FO ;;
    mappo)     launch v2 "$s" "$n" "$BCV2" $C ;;
    scratch)   launch v8 "$s" "$n" -       $C $NA $FO ;;      # 无 BC
    selfnew)   launch v8 "$s" "$n" "$BCV3" $C $RO --config diag_selfnew_h3_w04.yaml ;;  # 本文方法
  esac
}

# 3x 负载消融臂（ablation <arm> <seed>）
ablation() {
  local a="$1" s="$2" n="v7_3x_${a}_s${s}"
  case "$a" in
    nograph)   launch v8 "$s" "$n" "$BCN"  --config scenario_v7_x30.yaml $NA --config train_nograph.yaml ;;
    shuffle)   launch v8 "$s" "$n" "$BCV3" --config scenario_v7_x30.yaml $RO --config diag_selfshuffle_h3_w04.yaml ;;
    gat)       launch v8 "$s" "$n" "$BCN"  --config scenario_v7_x30.yaml $NA --config gat_encoder.yaml ;;
    rec_head)  launch v8 "$s" "$n" "$BCN"  --config scenario_v7_x30.yaml $NA ;;
  esac
}

# ---------------------------------------------------------------- 各阶段
phase_bc() {
  require_dataset
  log "=== 阶段 0：BC 起点（每模型一个，跨种子复用）==="
  # 专家轨迹：若 T 下有 day_*.pkl.gz 就用 --data-dir，否则 --bc 现场采
  local DATA=""; [ -d "$T/agg3" ] && DATA="--bc-data $T/agg3"
  local BCS=( "$E/v3/v7_bc:v3:-" "$E/v8/v7_bc_noatt:v8:$NA" "$E/v2/v7_bc:v2:-" )
  local path rest model extra runname
  for spec in "${BCS[@]}"; do
    path="${spec%%:*}"; rest="${spec#*:}"; model="${rest%%:*}"; extra="${rest#*:}"
    [ -f "$path/bc/supervised_pg_phased.pt" ] && { log "  skip BC $model"; continue; }
    runname=$(basename "$path")     # 末段即 run 名（v7_bc / v7_bc_noatt / v7_bc）
    if [ "$DRY" = 1 ]; then log "  [dry] BC $model/$runname"; continue; fi
    mkdir -p "$path"
    setsid nohup env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 "$PY" -u \
      scripts/experiment.py train --model "$model" --seed 42 --updates 1 --name "$runname" \
      --bc --bc-only --bc-episodes 16 $D $DATA $extra --config cpu8.yaml --device cpu \
      > "/tmp/v7_bc_${model}.log" 2>&1 < /dev/null &
    log "  BC launched $model/$runname pid=$!"
  done
  while [ "$(running)" -gt 0 ] || pgrep -f "experiment.py train.*--bc" >/dev/null; do sleep 15; done
  log "阶段 0 完成"
}

phase_main() {
  require_dataset; require_bc "$BCV3"; require_bc "$BCN"; require_bc "$BCV2"
  log "=== 阶段 1：主表（Table 2）—— 6 臂 × 3 负载 × ${SEEDS} ==="
  for s in $SEEDS; do for L in 1x 2x 3x; do
    for a in full_ro lite_ro gnnppo mappo selfnew; do arm "$a" "$L" "$s"; done
  done; done
  # 从零 RL 对照（仅 1x/3x，省机时）
  for s in $SEEDS; do arm scratch 1x "$s"; arm scratch 3x "$s"; done
  wait_train
}

phase_ablation() {
  require_dataset; require_bc "$BCV3"; require_bc "$BCN"
  log "=== 阶段 2：消融（Table 3）—— 3x 负载 ==="
  for s in $SEEDS; do for a in nograph shuffle gat rec_head; do ablation "$a" "$s"; done; done
  # Figure 6：w ∈ {0.2,0.4,0.6,0.8,1.0}（3x；0.0 即 full_ro）
  for w in 02 06 08 10; do
    printf 'model:\n  actor:\n    usage_forecast:\n      label_mode: self_new\n      horizon: 3\n      bias_weight: %s\n' "$(if [ "$w" = 10 ]; then printf '1.0'; else printf '0.%s' "${w#0}"; fi)" \
      > "configs/diag_selfnew_h3_w${w}.yaml"
  done
  for s in $SEEDS; do
    for w in 02 06 08 10; do
      launch v8 "$s" "v7_3x_selfnew_w${w}_s${s}" "$BCV3" --config scenario_v7_x30.yaml $RO --config "diag_selfnew_h3_w${w}.yaml"
    done
  done
  wait_train
  # 5 种子补齐（Table 2 要求 n≥3；关键臂补到 n=5）
  log "=== 阶段 2b：关键臂补 5 种子 ==="
  for s in $SEEDS5; do for L in 1x 3x; do
    arm selfnew "$L" "$s"; arm full_ro "$L" "$s"
  done; done
  wait_train
}

wait_train() {
  log "  等待训练结束…"
  while [ "$(running)" -gt 0 ]; do sleep 60; done
  log "  训练阶段结束"
}

# 评测守护：训练完成的 run → 留出评测（30 种子）；并跑基线/专家/MPC 上界
phase_eval() {
  require_dataset
  log "=== 阶段 3：评测（含泛化/鲁棒/上界）==="
  local MAXE="${MAXE:-8}"
  ev() { local R=$1 tag=$2; shift 2
    [ -f "$R/checkpoint_final.pt" ] || return 1
    [ -e "$R/eval_$tag.json" ] && return 0
    while [ "$(ev_running)" -ge "$MAXE" ]; do sleep 20; done
    (OMP_NUM_THREADS=2 setsid nohup "$PY" -u scripts/experiment.py evaluate --run "$R" \
        --seeds "$HOLDOUT" "$@" --tag "$tag" --threads 2 > "/tmp/ev_$(basename "$R")_$tag.log" 2>&1 < /dev/null &)
    log "  eval $R $tag"; sleep 2
  }
  # 主表：每 run 的 1x/2x/3x 场景跟随其 experiment_config（--run 已含场景），
  # 这里统一跑一次留出；跨负载对比由各 run 自己的配置决定。
  for d in "$E"/v8/v7_* "$E"/v2/v7_*; do case "$d" in *_bc*) continue;; esac
    [ -d "$d" ] && ev "$d" heldout_v7_s1000
  done
  R0="$E/v8/v7_3x_full_ro_s42"
  # 基线（随机/贪心）+ 专家锚 + 流体 UB + MPC（用同一 run 的任务指纹）
  [ -d "$R0" ] && {
    mkdir -p "$O"
    [ -e "$O/v7_expert_3x.json" ] || (setsid nohup env OMP_NUM_THREADS=2 "$PY" -u scripts/eval/eval_expert.py \
        --run "$R0" --seeds "$HOLDOUT" --out "$O/v7_expert_3x.json" > /tmp/ev_v7_expert.log 2>&1 < /dev/null &)
    [ -e "$O/v7_fluid_ub_3x.json" ] || (setsid nohup env OMP_NUM_THREADS=4 "$PY" -u scripts/milp/fluid_upper_bound.py \
        --run "$R0" --seeds "$HOLDOUT" --threads 4 --out "$O/v7_fluid_ub_3x.json" > /tmp/ev_v7_ub.log 2>&1 < /dev/null &)
    [ -e "$O/v7_mpc_3x.json" ] || (setsid nohup env OMP_NUM_THREADS=2 "$PY" -u scripts/milp/mpc_online.py \
        --run "$R0" --seeds "$HOLDOUT" --fill --forecast persist --out "$O/v7_mpc_3x.json" > /tmp/ev_v7_mpc.log 2>&1 < /dev/null &)
  }
  # 零样本跨规模（Figure 8 左）：在 small/large 场景叠加下评测
  for SZ in small large; do for L in 1x 3x; do
    R="$E/v8/v7_${L}_selfnew_s42"; [ -f "$R/checkpoint_final.pt" ] || continue
    for s in 42 43 44; do
      Rs="$E/v8/v7_${L}_selfnew_s$s"; [ -f "$Rs/checkpoint_final.pt" ] || continue
      ev "$Rs" "heldout_v7_size_${SZ}" --overlay "scenario_v7_size_${SZ}.yaml"
    done
  done; done
  # 节点失效鲁棒（Figure 8 右）：fail_* 名单
  for F in sat5 sat10 hap5 mix10; do for s in 42 43 44; do
    R="$E/v8/v7_3x_selfnew_s$s"; [ -f "$R/checkpoint_final.pt" ] || continue
    ev "$R" "heldout_v7_fail_$F" --overlay "fail_$F.yaml"
  done; done
  log "阶段 3 提交完成（评测在后台跑）"
}

# ---------------------------------------------------------------- main
if [ "$PROBE" = 1 ]; then probe_mem; exit $?; fi
log "v7 论文流水线：phase=$PHASE STAMP=$STAMP DRY=$DRY MAX=$MAX UPDATES=$UPDATES"
case "$PHASE" in
  bc)       phase_bc ;;
  main)     phase_main ;;
  ablation) phase_ablation ;;
  eval)     phase_eval ;;
  all)      probe_mem || exit 4; phase_bc; phase_main; phase_ablation; phase_eval ;;
  *) echo "!! 未知 phase: $PHASE" >&2; exit 2 ;;
esac
log "phase=$PHASE 完成：$(date -Is)"
