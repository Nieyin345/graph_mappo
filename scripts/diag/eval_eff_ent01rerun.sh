#!/usr/bin/env bash
# 把「RL 比专家省密钥」放到**同一个验证 regime** 上实测（三训练种子）。
#
# ### 为什么必须重跑，而不是复用现成产物
#
# 走到这一步之前有两次「写好了没跑成」：
#   1. `eff_3seeds.py` / `eval_eff_3seeds.sh` 引用
#      `outputs/ent01_s${s}/checkpoint_update_000030.pt` —— 实测该目录
#      **`.pt` 数为 0**（权重已丢）⟹ 这脚本从来没成功过。
#   2. `gen_compare.py` / `expert_gen.py` 读
#      `outputs/eval/rl_ent01_s42_u30/rl_ent01_s42_u30/steps.csv` —— 实测
#      `outputs/eval/` 下**只有 3 个 json**，没有这个目录。
#   3. `expert_gen.py` 的文件头声称「旧 json 里已经存了 generated_keys」，
#      **实测是假的**（旧 json 键里没有 generated）⟹ 我 2026-09-19 18:39
#      重跑专家才把这个键补上（逐位复现 61/61，确定性坐实）。
#
# ⟹ 所以「RL 省 36.5%」这条**在本节点、验证 regime 上从来没有被实测过**。
#    本脚本补的就是这个缺口。
#
# ### 为什么用 ent01_rerun 而不是 ent01
#
# `ent01_s{42,43,44}` 的 `.pt` 已丢（见上）。而 `ent01_rerun_s{42,43,44}`
# 是**同一配置在本节点重跑的干净对照**（wave5 判读：平台 0.6819/0.6876/0.6981，
# 均值 0.6892；u1 指纹与预期逐位一致），u30 checkpoint 六个都在。
#
# ### 有效性前提（已逐条查过代码，不是假设）
#
#   · `run_baselines.py` 与 `eval_expert.py` 都走
#     `load_validation_profile` + `build_validation_env_config`，
#     且 reset 都是 `start_seed = profile.start_seed + seed`
#     ⟹ **同 regime、同环境构造**（`run_baselines.py:63/75-92`）
#   · `collect_steps=True` ⟹ 落 `steps.csv`，其中有逐步 `generated_keys`
#     （`evaluator.py:132`）
#   · `--policies __none__` ⟹ 只跑 RL，不跑任何基线（省时间）
#
# ### 本脚本只增不删
#
#   输出到新目录 `outputs/eval/rl_ent01rerun_s${s}_u30`，
#   `[ -f summary.json ] && skip` 保证可重入、不覆盖。
set -u
cd /opt/qkd/graph_mappo || exit 1

SEEDS_STR="100,101,102,103,104,105,106,107,108,109,110,111,112,113,114"
RC=0

for s in 42 43 44; do
  ck="outputs/ent01_rerun_s${s}/checkpoint_update_000030.pt"
  out="outputs/eval/rl_ent01rerun_s${s}_u30"
  if [ ! -f "$ck" ]; then
    echo "[$(date -Is)] !! 缺 $ck"
    RC=2
    continue
  fi
  if [ -f "$out/summary.json" ]; then
    echo "[$(date -Is)] 已有 $out/summary.json，跳过"
    continue
  fi
  echo "[$(date -Is)] 评估 s${s}：$ck"
  # OMP_NUM_THREADS 固定为 8（与这些臂训练时一致）。记忆 thread-count-changes-training
  # 说评测不受线程数影响，这里固定只是为了少一个可变量。
  env OMP_NUM_THREADS=8 /opt/qkd/venv/bin/python -u scripts/baselines/run_baselines.py \
    --config configs/global.yaml \
    --episodes 15 \
    --seeds "$SEEDS_STR" \
    --out "$out" \
    --policies __none__ \
    --rl-checkpoint "$ck" \
    --rl-name "rl_ent01rerun_s${s}_u30" \
    > "/tmp/eval_eff_rerun_s${s}.log" 2>&1
  rc=$?
  echo "[$(date -Is)] s${s} 结束 exit=$rc"
  [ $rc -ne 0 ] && RC=$rc
done

echo "[$(date -Is)] 三个种子评测结束 总 rc=$RC"
touch /tmp/eval_eff_rerun.done
exit $RC
