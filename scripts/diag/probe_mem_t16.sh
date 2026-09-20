#!/usr/bin/env bash
# 单臂内存探针：在**真实启动路径**上先跑一条，量 16 线程下的每-run PSS。
#
# 为什么不能省这一步：
#   `STEADY=25.0` 是**4 线程**时实测的（见记忆 run-memory-23gb-and-growing）。
#   本波改用 16 线程 —— OMP 线程数不改变 rollout buffer 与 log_probs 张量
#   （那是父进程内存的主体），但会加每线程 scratch 与 MKL buffer。
#   改线程数 ⟹ 每-run 内存**必须重量**，不能沿用旧数。
#
#   8 臂 × 25G + 17G = 217G，而可用 247G ⟹ 只剩 30G 余量（约 1.2 个 run）。
#   若 16 线程使每 run 涨到 ≥30G，8 臂就是 240G+ ⟹ 顶格。**先量再铺。**
#
# 它同时验证：BC 检查点能加载、16 线程配置生效、resolved_config 落在预期上
# —— 即整条启动路径。记忆 test-harness-must-use-real-launch-path。
set -u
cd /opt/qkd/graph_mappo || exit 1

PY=/opt/qkd/venv/bin/python
NAME=memprobe_t16
LOG=/tmp/${NAME}.log

if [ -d "outputs/$NAME" ]; then
  echo "已存在 outputs/$NAME —— 先删掉它再跑（这是探针，不是臂）"
  exit 2
fi

setsid nohup env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
  $PY -u scripts/train/train_graph_mappo.py \
    --configs rl_algorithm.yaml train_full_rl.yaml train_ent01.yaml \
    --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
    --seed 42 --num-updates 3 --run-name "$NAME" \
    > "$LOG" 2>&1 < /dev/null &
echo "启动 $NAME pid=$!，日志 $LOG"
echo "等 600s 让它跑过第一轮（含 rollout 与一次 update）…"
sleep 600
echo "--- 日志尾部 ---"
tail -6 "$LOG"
echo
echo "--- resolved_config 核对（16 线程是否生效看配置，不看环境变量）---"
$PY /tmp/rc_get.py "outputs/$NAME/resolved_config.yaml" \
    train.ppo.entropy_coef train.gae_lambda 2>&1
