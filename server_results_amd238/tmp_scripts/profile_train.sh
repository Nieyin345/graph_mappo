#!/usr/bin/env bash
# 对**真实训练脚本**做 cProfile —— 不猜 API，直接profiler套在入口上，稳。
#
# 为什么不用 .tmp/prof_rollout.py：那个脚本要自己复现 env+policy 的构造，
# 而构造 API 几经变动（build_env_from_config 的参数、reset/step 的返回签名
# 都改过），一猜就错。套在 scripts/train/train_graph_mappo.py 上，
# 走的就是训练真正走的那条路。
#
# 跑法（节点上）：nohup bash .tmp/profile_train.sh > /tmp/profile_train.out 2>&1 &
# 产物：/tmp/prof_full.out（pstats 用），再由 .tmp/analyze_profile.py 分析。
set -uo pipefail
cd /opt/qkd/graph_mappo || exit 1

PY=/opt/qkd/venv/bin/python
THREADS=${THREADS:-4}
# 2 轮：第 1 轮含懒加载与首次分配，第 2 轮才是稳态。profile 只留第 2 轮。
UPDATES=${UPDATES:-2}

rm -rf outputs/prof_run
OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS timeout 3600 \
    $PY -u -m cProfile -o /tmp/prof_full.out \
        scripts/train/train_graph_mappo.py \
            --configs rl_algorithm.yaml train_full_rl.yaml \
            --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
            --num-updates "$UPDATES" --run-name prof_run \
    > /tmp/prof_run.log 2>&1

echo "退出码=$?"
ls -lh /tmp/prof_full.out 2>/dev/null
echo "=== PROFILE_DONE ==="
