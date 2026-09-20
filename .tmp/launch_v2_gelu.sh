#!/usr/bin/env bash
# 启动 v2_gelu 批次：`model.encoder.activation: relu -> gelu`（**唯一自变量**）。
#
# 对照 = 已有的 `ent01_rerun_s{42,43,44}`（relu），不必另起对照臂。
# 预注册判据见 configs/train_v2_gelu.yaml：Δ≥+0.035 有用 / |Δ|<0.035 测不出。
#
# ★★ 本次启动修掉的上一版致命错误：**没设 OMP_NUM_THREADS**。
#    实测上一版 v2_gelu 日志写 `Threads: torch=72 OMP=None MKL=None`，
#    而全部 7 条其它臂写 `torch=8 OMP=8 MKL=8`。
#    线程数会确定性影响训练结果（差约 0.018）⟹ 不设的话连配对资格都没有。
#    修法：`OMP_NUM_THREADS=8 MKL_NUM_THREADS=8` 显式前置。
#
# ★ 上一版还**静默死亡**：日志 11 行、停在 "Resumed ... at update 0"、
#   无 traceback、无 metrics、无 .pt，只留一条 resource_tracker 泄漏告警。
#    ⟹ 必须有**启动后验证**，否则失败会被当成「还在跑」。
set -u
HOST=qkd
REPO=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
CFGS="rl_algorithm.yaml train_full_rl.yaml train_ent01.yaml train_v2_gelu.yaml"
UPDATES=30
THREADS=8
SEEDS="42 43 44"
PREFIX=v2_gelu

echo "=============================================================="
echo " 1. 预检（**起任何一条之前**全员预检）"
echo "=============================================================="
timeout 25 ssh -o ConnectTimeout=10 "$HOST" "
  cd $REPO || exit 1
  ok=1
  echo '--- 起点 checkpoint ---'
  if [ -f $CKPT ]; then ls -la $CKPT; else echo '  ✗ 缺失'; ok=0; fi
  echo '--- 配置链各文件 ---'
  for c in $CFGS; do
    if [ -f configs/\$c ]; then echo \"  ✓ configs/\$c\"; else echo \"  ✗ configs/\$c 缺失\"; ok=0; fi
  done
  echo '--- 唯一自变量确认（gelu 配置实际内容）---'
  cat configs/train_v2_gelu.yaml | grep -A2 '^model:'
  echo '--- 去重：进程表里有没有同名臂在跑（问世界，不问账本）---'
  for s in $SEEDS; do
    n=\$(ps -eo args | awk -v r=\"${PREFIX}_s\$s\" '\$0 ~ /train_graph_mappo/ && index(\$0, r) > 0' | wc -l)
    echo \"  ${PREFIX}_s\$s 在跑进程数 = \$n\"
  done
  echo \"PREFLIGHT_OK=\$ok\"
" 2>&1

echo
echo "=============================================================="
echo " 2. 内存门（**必须打印自己的输入**）"
echo "=============================================================="
timeout 25 ssh -o ConnectTimeout=10 "$HOST" "
  avail_kb=\$(awk '/^MemAvailable/{print \$2}' /proc/meminfo)
  echo \"  输入① MemAvailable = \$avail_kb kB\"
  awk -v kb=\$avail_kb 'BEGIN{printf \"  输入② 换算 = %.2f GiB（kB 按 KiB 读，不是十进制 GB）\\n\", kb/1048576}'
  awk -v kb=\$avail_kb 'BEGIN{
     avail = kb/1048576; n = 3; per = 25; floor = 17;
     need = n*per; left = avail - need;
     printf \"  输入③ 本批 n = %d，每 run 稳态 = %d GiB，FLOOR = %d GiB\\n\", n, per, floor;
     printf \"  算式：  余量 = %.2f − %d×%d = %.2f GiB\\n\", avail, n, per, left;
     printf \"  判决：  %.2f >= %d ?\n\", left, floor;
     printf \"  结论：  **%s**\n\", (left >= floor) ? \"放行\" : \"拒绝\";
     printf \"GATE_VERDICT=%s\\n\", (left >= floor) ? \"放行\" : \"拒绝\";
  }'
" 2>&1

echo
echo "=============================================================="
echo " 3. 保留上一版失败启动的证据（**只增不删**，改名而不是覆盖）"
echo "=============================================================="
timeout 25 ssh -o ConnectTimeout=10 "$HOST" "
  cd $REPO
  for s in $SEEDS; do
    if [ -f outputs/${PREFIX}_s\$s.log ]; then
      cp -n outputs/${PREFIX}_s\$s.log outputs/${PREFIX}_s\$s.failed-t72.log 2>/dev/null \
        && echo \"  已另存 outputs/${PREFIX}_s\$s.failed-t72.log（原文件保留）\"
    fi
  done
" 2>&1

echo
echo "=============================================================="
echo " 4. 启动（OMP_NUM_THREADS=$THREADS 必须显式）"
echo "=============================================================="
for s in $SEEDS; do
  RUN="${PREFIX}_s${s}"
  echo "--- 启动 $RUN ---"
  timeout 25 ssh -o ConnectTimeout=10 "$HOST" \
    "cd $REPO && OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS setsid nohup $PY -u \
       scripts/train/train_graph_mappo.py \
       --configs $CFGS \
       --checkpoint $CKPT \
       --seed $s --num-updates $UPDATES --run-name $RUN \
       > outputs/$RUN.log 2>&1 < /dev/null & echo started"
  echo "  （ssh 被 timeout 掐掉是预期的）"
done

echo
echo "=============================================================="
echo " 5. 启动后验证：**确认真的在跑**，不是静默空转"
echo "=============================================================="
sleep 30
timeout 25 ssh -o ConnectTimeout=10 "$HOST" "
  cd $REPO
  echo '--- 进程（python 解释器，不数 bash -c 包装器）---'
  ps -eo pid,etime,args | awk '\$3 ~ /^\\/opt/ && /train_graph_mappo/' | sed 's/--checkpoint[^ ]* [^ ]*//'
  echo
  echo '--- 每臂日志：行数 / Threads 行 / 尾部 ---'
  for s in $SEEDS; do
    f=outputs/${PREFIX}_s\$s.log
    echo \"### \$f\"
    echo -n '  行数: '; wc -l < \$f 2>/dev/null || echo 'MISSING'
    echo -n '  '; grep -h '^Threads:' \$f 2>/dev/null || echo '  (还没有 Threads 行)'
    echo -n '  metrics: '; wc -l < outputs/${PREFIX}_s\$s/metrics.jsonl 2>/dev/null || echo 0
    echo '  尾部:'; tail -4 \$f 2>/dev/null | sed 's/^/    /'
  done
" 2>&1
