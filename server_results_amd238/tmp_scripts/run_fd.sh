#!/usr/bin/env bash
# 跑失败分解探针（专家 + BC），诊断场景协议（第 0 天、240 步、12 种子 7-18）。
#
# 为什么单独一个脚本：这条命令里有 `env VAR=x python ...`，从 PowerShell 经 ssh
# 内联下发时会被拆坏（实测远端只跑了 `env`，把整份环境变量打印进日志，
# 而真正的探针一声不响地没跑）。放进脚本里用 bash 执行，引号就只剩一层。
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
CFG=configs/var2_diag.yaml
SEEDS=7-18
STEPS=240

cd "$MAIN" || exit 1

for spec in "expert:" "rl:$CKPT"; do
    policy="${spec%%:*}"
    ckpt="${spec#*:}"
    log="/tmp/fd_${policy}.log"
    echo "== 策略=$policy  日志=$log =="
    if [ "$policy" = "rl" ]; then
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -u .tmp/probe_failure_decomp.py \
            --policy rl --checkpoint "$ckpt" --config "$CFG" \
            --seeds "$SEEDS" --steps "$STEPS" 2>&1 | tee "$log"
    else
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -u .tmp/probe_failure_decomp.py \
            --policy expert --config "$CFG" \
            --seeds "$SEEDS" --steps "$STEPS" 2>&1 | tee "$log"
    fi
    echo
done
echo FAILURE_DECOMP_DONE