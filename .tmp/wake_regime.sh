#!/usr/bin/env bash
# 唤醒脚本：等机制扫描 -> 按预注册规则选键位 -> 起训练 -> 等训练 -> 打印判读材料。
#
# 纪律（每条对应一次踩过的坑）：
#  - 节点会间歇性 reset（实测 3 次探测掉 1 次）-> 每次 ssh 带重试
#  - setsid nohup + timeout ssh：后台子 shell 持有 ssh stdout 管道，必须 timeout 掐
#  - python -u：重定向到文件时块缓冲，空日志与"还在跑"无法区分
#  - 不用 pgrep -f（在 ssh 里自匹配）；用 grep '^/opt/...' 只匹配解释器路径
#  - 数值判据不 shell out 到 bc（bc 缺失得空串 -> 恒假且不报错）
#  - 启动要预检 + 启动后验证，否则失败被当成"还在跑"静默空转
set -u
HOST=qkd
REPO=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
THREADS=8
SEEDS="42 43 44"
SWEEP=/tmp/regime.log
PREFIX=il_warm

rssh() {
  local n=0
  while [ "$n" -lt 5 ]; do
    if out=$(timeout 30 ssh -o ConnectTimeout=10 "$HOST" "$1" 2>/dev/null); then
      printf '%s' "$out"; return 0
    fi
    n=$((n + 1)); sleep 6
  done
  printf ''; return 1
}

echo "=== 阶段 1：等机制扫描（上限 60 分钟）==="
DEADLINE=$(( $(date +%s) + 3600 ))
L=""
while true; do
  L=$(rssh "cat $SWEEP 2>/dev/null")
  if printf '%s' "$L" | grep -q 'DECISION_INITIAL_LEVEL='; then
    echo "  扫描完成 $(date +%H:%M:%S)"; break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "  ！等待超时，当前日志尾部："
    printf '%s\n' "$L" | tail -20
    echo "  => 不自动起训练（宁可不动，也不基于不完整读数起臂）"
    exit 2
  fi
  echo "  [$(date +%H:%M:%S)] 在等…"
  sleep 90
done

echo
echo "=== 阶段 1 结果 ==="
printf '%s\n' "$L" | grep -vE '^  \[[0-9]{2}:' | tail -50

LEVEL=$(printf '%s' "$L" | awk -F= '/^DECISION_INITIAL_LEVEL=/{print $2}' | tail -1)
EXP_SR=$(printf '%s' "$L" | awk -F= '/^DECISION_EXPERT_SR=/{print $2}' | tail -1)
BASE_SR=$(printf '%s' "$L" | awk -F= '/^BASELINE_EXPERT_SR=/{print $2}' | tail -1)
echo
echo "  解析：level=$LEVEL  expert_sr=$EXP_SR  base_sr=$BASE_SR"

# 仅在初始有效时起训练；起日 0（基线）则跳过（已饱和，无情报价值）
if [ -z "$LEVEL" ]; then
  echo "  ！level 解析失败 => 不起训练"; exit 3
fi
IS_ZERO=$(awk -v v="$LEVEL" 'BEGIN{print (v+0 == 0) ? 1 : 0}')
if [ "$IS_ZERO" = "1" ]; then
  echo "  level=0（退回基线）=> 说明冷启动档没有判别力，跳过训练"
  echo "  => 下一步应转向奖励/机制方向（不再动 initial_level）"
  echo "WAKE_DONE" ; exit 0
fi

echo
echo "=== 阶段 2：预检 + 起训练（level=$LEVEL）==="
rssh "cd $REPO && [ -f $CKPT ] && ls -la $CKPT" || { echo "  ！ckpt 缺失"; exit 4; }
# 去重：问进程表，不问自己的账本
for s in $SEEDS; do
  n=$(rssh "ps -eo args 2>/dev/null | grep -c '^/opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py.*${PREFIX}_s${s}'")
  echo "  已有 ${PREFIX}_s${s} 进程 = ${n:-?}"
done

# 机制配置：写一个显式 yaml（唯一自变量 = initial_level），不动其它
echo "  写入 configs/train_il_warm.yaml（唯一自变量 initial_level=$LEVEL）"
rssh "cd $REPO && printf 'qkp:\n  initial_level: %s\n' '$LEVEL' > configs/train_il_warm.yaml && cat configs/train_il_warm.yaml && ls configs/train_il_warm.yaml"

CFGS="rl_algorithm.yaml train_full_rl.yaml train_ent01.yaml train_il_warm.yaml"
for s in $SEEDS; do
  RUN="${PREFIX}_s${s}"
  rssh "cd $REPO && OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS setsid nohup $PY -u scripts/train/train_graph_mappo.py --configs $CFGS --checkpoint $CKPT --seed $s --num-updates 30 --run-name $RUN > outputs/$RUN.log 2>&1 < /dev/null & echo started"
  echo "  启动 $RUN"
done

echo "=== 阶段 2 启动后验证 ==="
sleep 35
rssh "cd $REPO && for s in $SEEDS; do f=outputs/${PREFIX}_s\$s.log; echo \"# \$f\"; echo -n '  Threads: '; grep -h '^Threads:' \$f 2>/dev/null | head -1; tail -3 \$f 2>/dev/null | sed 's/^/    /'; done; echo '--- 进程 ---'; ps -eo args | grep -c '^/opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py'"

echo
echo "=== 阶段 3：等训练（上限 150 分钟）==="
DEADLINE=$(( $(date +%s) + 9000 ))
while true; do
  N=$(rssh "ps -eo args 2>/dev/null | grep -c '^/opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py'")
  U=$(rssh "cd $REPO && for s in $SEEDS; do printf '%s:%s ' \$s \$(grep -c '\"update\"' outputs/${PREFIX}_s\$s/metrics.jsonl 2>/dev/null || echo 0); done")
  echo "  [$(date +%H:%M:%S)] 在跑 ${N:-?} 个   update 数: ${U:-?}"
  if [ "${N:-x}" = "0" ]; then echo "  训练全部退出"; break; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "  ！训练等待超时"; break; fi
  sleep 180
done

echo
echo "=== 阶段 4：结果材料（供主代理判读）==="
rssh "cd $REPO && for s in $SEEDS; do echo \"### ${PREFIX}_s\$s\"; echo -n '  .pt='; ls outputs/${PREFIX}_s\$s/*.pt 2>/dev/null | wc -l; echo -n '  update 号: '; grep -o '\"update\": *[0-9]*' outputs/${PREFIX}_s\$s/metrics.jsonl 2>/dev/null | grep -o '[0-9]*' | tr '\n' ' '; echo; echo -n '  重复号: '; grep -o '\"update\": *[0-9]*' outputs/${PREFIX}_s\$s/metrics.jsonl 2>/dev/null | grep -o '[0-9]*' | sort -n | uniq -d | tr '\n' ' '; echo ' (空=健康)'; done; echo '--- 内存 ---'; awk '/^MemAvailable/{printf \"  %.1f GiB\n\", \$2/1048576}' /proc/meminfo"
echo
echo "  判读命令（主代理执行）："
echo "    python scripts/diag/paired_verdict.py --label 'il_warm' --ctrl-fmt 'ent01_rerun_s{}' --exp-fmt '${PREFIX}_s{}' --seeds 42,43,44 --plateau 25,30"
echo "WAKE_DONE"
