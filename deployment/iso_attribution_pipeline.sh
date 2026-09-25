#!/usr/bin/env bash
# 50 天专家轨迹 + v3 / v3+隔离 独立 BC 对比实验（2026-09-25）。
#
# 目的：隔离反转归因。v3 与隔离版用同一批预采集专家轨迹（独立 BC 各自出权重），
# 每臂在 BC 后、PPO 后各评一次（冻结协议窗口 A），定位退化发生在哪个阶段。
#
# 固定协议：
#   轨迹 50 天（day 0-49）× 1440 步，v3 图构建器（obs 自带 pair_path masks）
#   BC   每臂独立：50 个 day 文件，batch 64
#   PPO  10 轮 × 8 rollout × 240 步，actor_lr 1e-5，种子 42/43/44
#   评测 frozen_eval.py 窗口 A（330-365 / 种子 100-114）
#
# 用法: bash deployment/iso_attribution_pipeline.sh qkd
set -uo pipefail

pick_ssh() {
    for cand in "C:/Windows/System32/OpenSSH/ssh.exe" \
                "/mnt/c/Windows/System32/OpenSSH/ssh.exe" \
                "/c/Windows/System32/OpenSSH/ssh.exe" \
                "$(command -v ssh 2>/dev/null || true)"; do
        [ -n "$cand" ] && [ -x "$cand" ] && { printf '%s' "$cand"; return 0; }
    done
    echo "no usable ssh found" >&2
    return 1
}
SSH_BIN="$(pick_ssh)"
TARGET="${1:?usage: deployment/iso_attribution_pipeline.sh <ssh-alias>}"
TARGET="$(tr -d '\r\n' <<<"$TARGET")"
REMOTE_DIR="/opt/qkd/graph_mappo"
STAMP="$(date -u +%Y%m%d)"
RUN_ROOT="outputs/iso_attr_${STAMP}"
TRAJ_DIR="outputs/trajs_v3_50d"
LOCAL_DIR="server_results_node342/iso_attr_${STAMP}"

FETCH_ONLY=0
[ "${2:-}" = "--fetch" ] && FETCH_ONLY=1

if [ "$FETCH_ONLY" = "0" ]; then
echo "== 1/4 启动：采集 50 天轨迹 -> 独立 BC x2 臂 x3 种子 -> PPO -> 评测 =="
"$SSH_BIN" "$TARGET" "cd $REMOTE_DIR && mkdir -p $RUN_ROOT && cat > $RUN_ROOT/pipeline.sh" <<EOS
#!/usr/bin/env bash
set -uo pipefail
cd $REMOTE_DIR
ulimit -n 65535
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
LOG=\$RUN_ROOT_LOG/pipeline.log
echo "[iso-attr] start \$(date -u)" >> \$LOG

# ---- 1. 采集 50 天专家轨迹（v3 graph builder，day 0-49 × 1440 步）----
if [ ! -f "$TRAJ_DIR/day_0049.pkl.gz" ]; then
  echo "[iso-attr] collecting 50d trajs \$(date -u +%H:%M)" >> \$LOG
  /opt/qkd/venv/bin/python scripts/train/collect_pg_phased_trajectories.py \\
    --out-dir $TRAJ_DIR --start-day 0 --end-day 50 --workers 16 \\
    --extra-config qkd_rl/model_zoo/v3/config.yaml >> \$LOG 2>&1 \\
    && echo "[iso-attr] collect rc=0" >> \$LOG || { echo "[iso-attr] collect FAILED" >> \$LOG; exit 1; }
fi

# ---- 2. 每臂独立 BC（同一批轨迹）+ BC 后评测 + PPO + PPO 后评测 ----
run_arm () {
  local model=\$1 name=\$2; shift 2
  for seed in 42 43 44; do
    local rd=outputs/experiments/\$model/\${name}_s\${seed}
    echo "[iso-attr] \$name s\$seed BC start \$(date -u +%H:%M)" >> \$LOG
    /opt/qkd/venv/bin/python scripts/experiment.py train \\
      --model \$model --seed \$seed --name \${name}_s\${seed} "\$@" \\
      --bc --bc-only --bc-data $TRAJ_DIR --evaluate >> \$LOG 2>&1 \\
      && echo "[iso-attr] \$name s\$seed bc rc=0" >> \$LOG \\
      || { echo "[iso-attr] \$name s\$seed bc FAILED" >> \$LOG; continue; }
    # BC 权重另存，PPO 从它续起
    cp \$rd/bc/supervised_pg_phased.pt \$rd/bc_weights.pt
    # ---- PPO（从 BC 续）----
    echo "[iso-attr] \$name s\$seed PPO start \$(date -u +%H:%M)" >> \$LOG
    /opt/qkd/venv/bin/python scripts/experiment.py train \\
      --model \$model --seed \$seed --name \${name}_s\${seed} "\$@" \\
      --updates 10 --resume \$rd/bc_weights.pt --evaluate >> \$LOG 2>&1 \\
      && echo "[iso-attr] \$name s\$seed ppo rc=0" >> \$LOG \\
      || echo "[iso-attr] \$name s\$seed ppo FAILED" >> \$LOG
    # BC 后评测（PPO 前的基线读数，定位退化阶段用）
    echo "[iso-attr] \$name s\$seed bc-eval start" >> \$LOG
    /opt/qkd/venv/bin/python scripts/eval/frozen_eval.py "rl:\$rd" \\
      --window A --out outputs/frozen_eval_bc_stage --checkpoint \$rd/bc_weights.pt >> \$LOG 2>&1 \\
      || echo "[iso-attr] \$name s\$seed bc-eval FAILED" >> \$LOG
  done
}

run_arm v3 v3full
run_arm v3 iso

echo "[iso-attr] done \$(date -u)" >> \$LOG
touch \$RUN_ROOT_LOG/PIPELINE_DONE
EOS
"$SSH_BIN" "$TARGET" "cd $REMOTE_DIR && sed -i \"s|\\\$RUN_ROOT_LOG|$RUN_ROOT|g\" $RUN_ROOT/pipeline.sh && chmod +x $RUN_ROOT/pipeline.sh && setsid nohup bash $RUN_ROOT/pipeline.sh >/dev/null 2>&1 & echo LAUNCHED"

fi

echo "== 2/4 轮询等待完成 =="
while :; do
    DONE="$("$SSH_BIN" "$TARGET" "test -f $REMOTE_DIR/$RUN_ROOT/PIPELINE_DONE && echo yes || echo no" 2>/dev/null || echo conn)"
    echo "  $(date +%H:%M:%S) PIPELINE_DONE=$DONE"
    [ "$DONE" = "yes" ] && break
    [ "$DONE" = "conn" ] && echo "  (连接失败，5 分钟后重试)"
    sleep 300
done

echo "== 3/4 回传 =="
SCP_BIN="${SSH_BIN%ssh.exe}scp.exe"
mkdir -p "$LOCAL_DIR"
"$SCP_BIN" -q "$TARGET:$REMOTE_DIR/$RUN_ROOT/pipeline.log" "$LOCAL_DIR/" || true
"$SCP_BIN" -q -r "$TARGET:$REMOTE_DIR/outputs/frozen_eval" "$LOCAL_DIR/frozen_eval_ppo_stage" || true
"$SCP_BIN" -q -r "$TARGET:$REMOTE_DIR/outputs/frozen_eval_bc_stage" "$LOCAL_DIR/" || true
"$SCP_BIN" -q -r "$TARGET:$REMOTE_DIR/outputs/experiments/v3" "$LOCAL_DIR/experiments_v3" || true
echo "回传完成: $LOCAL_DIR"
