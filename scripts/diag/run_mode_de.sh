#!/usr/bin/env bash
# 结构假设的第一刀：`model.mode: mixed → demand_edge`。
#
# ### 为什么是这一刀，而不是别的结构改动
#
# 见 docs/当前状态与优化方向.md §2 P1。三个理由：
#
#  1. **它不改任何权重形状**。`graph_mappo.py:254` 那一处
#     `if self.fuse_physical_to_node and n_phys > 0:` 是个**布尔开关**，
#     只决定「物理边要不要往节点上聚合」。encoder / actor / critic 的
#     参数形状全不变 ⟹ **BC 暖启动照常加载**（否则要动
#     `mappo_trainer.py:1226` 的无 strict 加载器）。
#  2. **合法且被校验**：`config.py:168` 明写支持值就是 ('mixed','demand_edge')。
#  3. **可与 ent01 配对**：配置链只多一个字段，起点/种子/轮数/线程数全同。
#
# 假设内容：`mixed` 下节点聚合同时吃物理边和需求边两种消息；`demand_edge`
# 下节点**只听需求边**，物理链路信息不再直接进节点，只能经由边嵌入间接传递。
# 若"信息缺失"假设成立（见 §1.3），拿掉一路输入应当**可测地**变差。
#
# ### 配对与判据（**跑之前写死**）
#
#   Δ_s = sr(mode_de_s) − sr(ent01_s)，s ∈ {42,43,44}
#   平台窗口 u25/u30（与 ent01/t8 判读一致），单样本 t，df=2，临界 **4.303**
#
#   |t| ≥ 4.303 ⟹ 物理边消息**确实被用到了**（两方向都算结论）
#   |t| < 4.303 ⟹ 以 n=3 分辨率（~0.035）测不出 ⟹ **说不清**，
#                  必须报「测不出」，**不能**报「没影响」
#
# ⚠ 双线程数记账：ent01_s42/43/44 是 **4 线程**，而本项目 2026-09-19 已采纳
#   全项目 8 线程。若要与 ent01 **干净配对**，本波也必须用 4 线程。
#   **本脚本用 8 线程**，所以严格说与 ent01 的配对**多了一个线程数的自由变量**。
#   两种做法各有代价，选 8 的理由：与今后所有新臂同制式（8 线程），
#   且线程数效应已实测"以 n=3 测不出"（Δ=−0.0090, t=−0.903）。
#   **记账在此，结论里必须带上这一条。**
#
# 用法（服务器上）：
#   setsid nohup bash /tmp/run_mode_de.sh > /tmp/mode_de.log 2>&1 < /dev/null &
set -u
cd /opt/qkd/graph_mappo || exit 1

CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
UPDATES=30
THREADS=8
MEM_PER_RUN=24                        # 实测 23.3 GB 且每轮还涨 ~0.21 GB，按 24 算
N_RUNS=3                              # ★ 实际起 3 个（不是 6）—— 内存判据必须按真实数算
NEED=$((N_RUNS * MEM_PER_RUN + 8))    # 80 GB：3×24 + 8G 余量
GO=/tmp/mode_de.go
PIDS=/tmp/mode_de.pids

if [ -f "$GO" ]; then echo "[$(date -Is)] 已有标记 $GO，跳过"; exit 0; fi

# ★ 幂等：三臂已在跑时**不要**再启动一遍。
#   起因：本脚本第一版的自检有 bug（grep 抓错了 mode 字段），把三个健康的 run
#   全判成"结果不可用"并 exit 1 —— 监管进程因此退出，.go 标记永不生成。
#   修好后重跑时若不检查，"启动前先起 3 个新进程"会变成起 6 个，直接打爆内存。
#   判据用 PID 文件 + 进程存活，不靠模式匹配（见 scripts/diag/wait_for.sh 的教训）。
if [ -s "$PIDS" ]; then
  alive=0
  for p in $(cat "$PIDS"); do kill -0 "$p" 2>/dev/null && alive=$((alive + 1)); done
  if [ "$alive" -gt 0 ]; then
    echo "[$(date -Is)] 已有 ${alive} 个臂在跑（PID 文件 $PIDS）—— 不重复启动，直接进入等待"
    SKIP_LAUNCH=1
  fi
fi
SKIP_LAUNCH="${SKIP_LAUNCH:-0}"

# 每臂一个独立 yaml，内容只差 mode —— 免得手写命令行时漏掉一个字段
mkdir -p configs
for s in 42 43 44; do
  printf 'model:\n  mode: demand_edge\n' > "configs/train_mode_de_s${s}.yaml"
done
echo "[$(date -Is)] 已写 3 个 mode=demand_edge 叠加配置"

wait_mem() {   # $1 = 需要多少 GB
  local want="$1" a
  for _ in $(seq 1 480); do
    a=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
    if [ "$a" -ge "$want" ]; then
      echo "  [$(date -Is)] 内存够: ${a}G >= ${want}G"; return 0
    fi
    echo "  [$(date -Is)] 内存不足: ${a}G < ${want}G，等 60s"
    sleep 60
  done
  echo "  [$(date -Is)] ⚠ 等内存超时（需 ${want}G）"; return 1
}

echo "[$(date -Is)] 启动前先等内存 —— 需要 ${NEED}G（${N_RUNS} × ${MEM_PER_RUN}G）"
wait_mem "$NEED" || exit 1

echo
if [ "$SKIP_LAUNCH" = "1" ]; then
  echo "=== 跳过启动（三臂已在跑）——直接到等待 ==="
else
  echo "=== 启动 ${N_RUNS} 臂：seed 42/43/44，mode=demand_edge ==="
  echo "    对照 = 已有的 ent01_t8_s42/43/44（同 8 线程、同配置链，只差 mode）"
  : > "$PIDS"
  for s in 42 43 44; do
    name="mode_de_s${s}"
    setsid nohup env OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
        /opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py \
        --configs rl_algorithm.yaml train_full_rl.yaml train_ent01.yaml "train_mode_de_s${s}.yaml" \
        --checkpoint "$CKPT" \
        --seed "$s" --num-updates "$UPDATES" --run-name "$name" \
        > "/tmp/${name}.log" 2>&1 < /dev/null &
    echo "$!" >> "$PIDS"
    echo "  [$(date -Is)] 启动 ${name} (seed $s, ${THREADS} 线程, mode=demand_edge) pid=$!"
    sleep 10
  done
  echo "  PID 已记入 $PIDS（等待时用它，不靠模式匹配）"
fi

sleep 60
echo
echo "=== 启动后自检（核对线程数与 mode，不只看『进程在』）==="
bad=0
for s in 42 43 44; do
  name="mode_de_s${s}"
  f="/tmp/${name}.log"
  if grep -qiE "error|traceback|parsererror|ValueError" "$f" 2>/dev/null; then
    echo "  !! ${name} 报错:"; grep -iE "error|traceback|ValueError" "$f" | head -3 | sed 's/^/      /'
    bad=1; continue
  fi
  rc="outputs/${name}/resolved_config.yaml"
  if [ -f "$rc" ]; then
    # ★ 必须**锚定在 model: 段内**取 mode。`grep -E "^  mode:" | head -1` 会抓到
    #   文件里第一个两空格缩进的 mode —— 那是 `env.action_resolver.mode`
    #   （值 mutual_choice），不是 `model.mode`。而且 `model:` 段在
    #   resolved_config.yaml 的**最后**（第 281 行附近），所以"取最后一个"
    #   这类取巧也不可靠。2026-09-19 实测：自检把三个健康的 run 全判成
    #   "结果不可用"。**一个会把成功报成失败的判据，会把真失败一起淹掉。**
    m=$(awk '/^model:/{f=1;next} /^[a-z]/{f=0} f && /^  mode:/{print $2; exit}' "$rc")
    ec=$(grep -A40 "^train:" "$rc" | grep "entropy_coef" | head -1 | awk '{print $2}')
    if [ "$m" = "demand_edge" ] && [ "$ec" = "0.01" ]; then
      echo "  ok ${name}: model.mode=$m entropy_coef=$ec ✓"
    else
      echo "  !! ${name}: model.mode=$m entropy_coef=$ec（期望 demand_edge / 0.01）—— **结果不可用**"
      bad=1
    fi
  else
    echo "  ?? ${name}: 还没写出 resolved_config.yaml（正常，约 1 分钟后再看）"
  fi
done

if [ "$bad" -ne 0 ]; then
  echo "  ⚠ 有臂异常 —— 不写标记，修好后可重跑"
  exit 1
fi

echo
echo "  [$(date -Is)] 等三个跑完（30 轮 × ${THREADS} 线程，约 1.2-1.6h）"
# ★ 等待必须带超时，且不靠模式匹配 —— 用启动时记下的 PID。
#   无超时的 `while pgrep ...; do sleep; done` 是本项目 2026-09-19 抓到的
#   坏形状之一：失败时不是"等得久"而是"永远不来"，且与"还在跑"无法区分。
TMO=$((12 * 3600))                    # 12 小时上限
waited=0
while :; do
  alive=0
  for p in $(cat "$PIDS" 2>/dev/null); do
    kill -0 "$p" 2>/dev/null && alive=$((alive + 1))
  done
  [ "$alive" -eq 0 ] && { echo "  ✓ 三个 PID 均已退出（等了 ${waited}s）"; break; }
  if [ "$waited" -ge "$TMO" ]; then
    echo "  ✗ 超时 ${TMO}s：仍有 ${alive} 个 PID 存活 —— 不写标记，等人工看" >&2
    exit 1
  fi
  sleep 120; waited=$((waited + 120))
done
touch "$GO"
echo "  [$(date -Is)] === mode=demand_edge 三臂结束，可以跑配对判据 ==="

# 自动出结论：有 verdict 脚本就跑，没有就只报原始平台均值
V=scripts/diag/mode_de_verdict.py
if [ -f "$V" ]; then
  echo; echo "=== 判据 ==="
  /opt/qkd/venv/bin/python "$V" 2>&1 | tail -40
fi
