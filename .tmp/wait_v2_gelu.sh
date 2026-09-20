#!/usr/bin/env bash
# 等 v2_gelu 三臂跑完，并打印判读所需的关键行。
#
# ★ 三条纪律（都踩过）：
#  · 判活在**进程所在机器**上判，不在本机
#  · 不用 `pgrep -f` —— 在 ssh 里远程 bash 命令行自身含该模式 ⟹ 自匹配；
#    连 `awk -v r=...` 也会因为 awk 自己的命令行含 r 而自匹配
#    ⟹ 改成**只匹配解释器路径**（`/^\/opt\//`），臂名靠后面的 index 判断，
#    且用 `ps -eo args` 时把 awk 自身排除。
#  · 三态显式：在跑 / 已退出 / ssh 探测失败。**失败不能当成完成**。
set -u
HOST=qkd
REPO=/opt/qkd/graph_mappo
DEADLINE=$(( $(date +%s) + 7200 ))   # 上限 2 小时
SEEDS="42 43 44"

echo "等待 v2_gelu 三臂（上限 2 小时）..."
echo

while true; do
  # 只数真正的训练进程：第 1 字段是以 /opt/ 开头的解释器路径
  N=$(timeout 25 ssh -o ConnectTimeout=10 "$HOST" \
      "ps -eo args | grep -c '^/opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py'" 2>/dev/null)
  # grep -c 失败会返回非零且输出 0/空，用显式判空区分「探测失败」
  if [ -z "$N" ]; then
    echo "  [$(date +%H:%M:%S)] ⚠ ssh 探测失败（**不是**跑完了），继续等"
  elif [ "$N" -eq 0 ]; then
    echo "  [$(date +%H:%M:%S)] 三臂均已退出"; break
  else
    U=$(timeout 25 ssh -o ConnectTimeout=10 "$HOST" \
        "cd $REPO && for s in $SEEDS; do printf '%s:%s ' \$s \$(wc -l < outputs/v2_gelu_s\$s/metrics.jsonl 2>/dev/null || echo 0); done" 2>/dev/null)
    echo "  [$(date +%H:%M:%S)] 在跑 $N 个   metrics 行数（update 数+eval 行）: ${U:-探测失败}"
  fi
  [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "  ⚠ 本地等待超时"; break; }
  sleep 120
done

echo
echo "=================== 结果 ==================="
timeout 30 ssh -o ConnectTimeout=10 "$HOST" "
  cd $REPO
  for s in $SEEDS; do
    echo \"### v2_gelu_s\$s\"
    echo -n '  .pt 数: '; ls outputs/v2_gelu_s\$s/*.pt 2>/dev/null | wc -l
    echo -n '  update 号序列: '; grep -o '\"update\": *[0-9]*' outputs/v2_gelu_s\$s/metrics.jsonl 2>/dev/null | grep -o '[0-9]*' | tr '\n' ' '
    echo
    echo -n '  重复的 update 号: '; grep -o '\"update\": *[0-9]*' outputs/v2_gelu_s\$s/metrics.jsonl 2>/dev/null | grep -o '[0-9]*' | sort -n | uniq -d | tr '\n' ' '
    echo '（空 = 无重复 = 目录健康）'
    echo -n '  Threads: '; grep -h '^Threads:' outputs/v2_gelu_s\$s.log 2>/dev/null | head -1
    echo '  尾部:'; tail -3 outputs/v2_gelu_s\$s.log 2>/dev/null | sed 's/^/    /'
    echo
  done
  echo '--- 内存 ---'
  awk '/^MemAvailable/{printf \"  MemAvailable = %.1f GiB\\n\", \$2/1048576}' /proc/meminfo
  echo '--- 仍在跑的 run ---'
  ps -eo args | grep -c '^/opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py'
"
