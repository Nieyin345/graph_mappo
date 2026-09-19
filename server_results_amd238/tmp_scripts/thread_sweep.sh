#!/usr/bin/env bash
# ★ 决定"能不能把线程数从 4 改成 8"的那个实验。
#
# ### 为什么是这个形状
#
# 已测：update 段 4→8 线程快 1.64x，折合整轮 **1.34x**（`.tmp/probe_thread_scaling.py`，
# 同一 buffer 交错三rep）。但 `OMP_NUM_THREADS` **确定性地改变训练结果**
# （2 vs 4 线程差 +0.0176，t=+7.68；两次重复给出同一个数）。
#
# 所以要问的不是"快多少"（已知），而是：
#
#     **4 线程与 8 线程训练出的策略，差多少？**
#
#   · 差 ≈ 0（在种子间方差内）⟹ 白捡 1.34x，全项目改 8。
#   · 差 ≈ 0.018（照 2 vs 4 外推）⟹ 现有 arm 全要重建基线，**不值得**。
#
# ### 为什么这样最省
#
# 不做 4×4 = 8 个 run 的"双向扫描"，只跑**单侧**：
#
#     ent01（entropy_coef=0.01）在 **8 线程**下的 s42/s43/s44
#     对照 = **已有的** ent01_s42/s43/s44（4 线程、同配置、同 BC、同 30 轮）
#
# 按种子配对，n=3（df=2，临界值 **4.303**）。**3 个 run ≈ 1.5h**，
# 而不是 8 个 run ≈ 4h。
#
# 为什么这样是合法的配对：线程数是唯一的差别，配置链完全一致
# （`rl_algorithm.yaml train_full_rl.yaml train_ent01.yaml`），
# 起点是同一个 BC checkpoint，种子相同。**但必须记账**：对照臂是几小时前
# 跑的，机器负载与现在不同 —— 这正是 `.tmp/probe_thread_repro.sh` 排除过的
# 变量（当时确认"背景负载改变结果"不成立，真正的变量只有 OMP_NUM_THREADS）。
#
# ### 判据（跑之前写死）
#
#   Δ = mean_s(8线程_s − 4线程_s)，s ∈ {42,43,44}
#   t = Δ / (SD(Δ)/√3)，df=2，临界 **4.303**（不是 2！本项目已经踩过这个坑）
#
#   |t| ≥ 4.303 ⟹ 线程数**确实改变结果**，不值得为 1.34x 重建基线
#   |t| < 4.303 ⟹ 测不出差异 ⟹ **建议改 8 线程**
#
#   ★ 无论哪个方向都照实报。n=3 的功效有限，所以"测不出"要带上
#     "以 n=3 的分辨率 ~0.035 测不出"这半句，不能写成"两者相同"。
#
# 用法（服务器上）：
#   setsid nohup bash /tmp/thread_sweep.sh > /tmp/thread_sweep.log 2>&1 < /dev/null &
set -u
cd /opt/qkd/graph_mappo || exit 1

CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
UPDATES=30
THREADS=8                     # ★ 被测的那个值；对照臂是已存在的 4 线程读数
BASE="rl_algorithm.yaml train_full_rl.yaml train_ent01.yaml"

wait_mem() {   # $1 = 需要多少 GB
  for _ in $(seq 1 480); do
    a=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
    [ "$a" -ge "$1" ] && { echo "  [$(date -Is)] 内存够: ${a}G >= ${1}G"; return 0; }
    sleep 30
  done
  echo "  [$(date -Is)] ⚠ 等内存超时（需 ${1}G）"; return 1
}

echo "[$(date -Is)] 等前一条链条（knobs）跑完再开始，避免抢内存"
for _ in $(seq 1 960); do
  k=$(pgrep -cf "bash /tmp/chain_knobs_ent01.sh" 2>/dev/null); k=${k:-0}
  r=$(pgrep -cf "run-name (ep2e1|mini512e1)_s" 2>/dev/null); r=${r:-0}
  [ "$k" -eq 0 ] && [ "$r" -eq 0 ] && break
  sleep 60
done
echo "[$(date -Is)] 前序链条已结束（knobs=$k run=$r）"

if [ -f /tmp/thread_sweep.go ]; then
  echo "[$(date -Is)] 已有标记，跳过"; exit 0
fi

wait_mem 100 || exit 1

echo
echo "=== 启动 8 线程三臂（对照 = 已有 ent01_s42/43/44 的 4 线程读数）==="
for s in 42 43 44; do
  name="ent01_t8_s${s}"
  setsid nohup env OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
    /opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py \
      --configs $BASE \
      --checkpoint "$CKPT" \
      --seed "$s" --num-updates "$UPDATES" --run-name "$name" \
      > "/tmp/${name}.log" 2>&1 < /dev/null &
  echo "  [$(date -Is)] 启动 ${name} (seed $s, ${THREADS} 线程)"
  sleep 10
done

sleep 60
echo
echo "=== 启动后自检（核对线程数与配置，不只看'进程在'）==="
bad=0
for s in 42 43 44; do
  name="ent01_t8_s${s}"
  f="/tmp/${name}.log"
  if grep -qiE "error|traceback|parsererror" "$f" 2>/dev/null; then
    echo "  !! ${name} 报错:"; grep -iE "error|traceback" "$f" | head -3 | sed 's/^/      /'
    bad=1; continue
  fi
  rc="outputs/${name}/resolved_config.yaml"
  if [ -f "$rc" ]; then
    ec=$(grep -A40 "^train:" "$rc" | grep "entropy_coef" | head -1 | awk '{print $2}')
    if [ "$ec" = "0.01" ]; then
      echo "  ok ${name}: entropy_coef=$ec ✓"
    else
      echo "  !! ${name}: entropy_coef=$ec（期望 0.01）—— **结果不可用**"; bad=1
    fi
  else
    echo "  ?? ${name}: 还没写出 resolved_config.yaml"
  fi
done

if [ "$bad" -ne 0 ]; then
  echo "  ⚠ 有臂异常 —— 不写标记，修好后可重跑"
  exit 1
fi

echo
echo "  [$(date -Is)] 等三个跑完（8 线程约 1.5h）"
while pgrep -f "run-name ent01_t8_s" > /dev/null; do sleep 120; done
touch /tmp/thread_sweep.go
echo "  [$(date -Is)] === 8 线程三臂结束，可以跑配对判据 ==="
