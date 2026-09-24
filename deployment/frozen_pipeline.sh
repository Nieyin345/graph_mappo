#!/usr/bin/env bash
# 冻结流水线（2026-09-24 定稿）：重训三臂 → 冻结全局评测 → 结果自动回传本地。
#
# 本地用法:
#   bash deployment/frozen_pipeline.sh <user@host>          # 全程（训练+评测+回传）
#   bash deployment/frozen_pipeline.sh <user@host> --fetch  # 只回传已完成的结果
#
# 固定配方（勿改；改配方 = 新流水线新名字）:
#   BC   live 专家（PathScoreGreedy phased，同采集器权重），8 个专家日，batch 64
#   PPO  10 次更新 × 8 rollout × 240 步，actor_lr 1e-5（experiment_base + model config）
#   臂   v2 / v3 / v3+需求隔离（model_zoo/v3/config_demand_isolated.yaml）× 种子 42/43/44
#   评测 scripts/eval/frozen_eval.py：窗口 A 330-365（种子 100-114）+ B 300-329（200-214），
#        15 集 × 240 步，含专家与全部启发式 baseline 调用方式见该脚本 docstring
#
# 纪律（2026-09-24 clnode263 事故）: 节点随时释放。训练结束立即回传
# checkpoint+日志+评测 JSON 到本地 server_results_node342/，本地确认后才算完成。
set -uo pipefail

# Windows/GitBash 坑（见 deployment/bootstrap.sh 同款注释）：裸 user@host 会用
# MSYS ssh，连不上 Windows ssh-agent/IdentityFile。全部走 ssh config 别名 qkd。
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
TARGET="${1:?usage: deployment/frozen_pipeline.sh <ssh-alias> [--fetch]}"
TARGET="$(tr -d '\r\n' <<<"$TARGET")"
REMOTE_DIR="${QKD_REMOTE_DIR:-/opt/qkd/graph_mappo}"
STAMP="$(date -u +%Y%m%d)"
RUN_ROOT="outputs/frozen_${STAMP}"
LOCAL_DIR="server_results_node342/frozen_${STAMP}"

FETCH_ONLY=0
[ "${2:-}" = "--fetch" ] && FETCH_ONLY=1

if [ "$FETCH_ONLY" = "0" ]; then
    echo "== 1/3 同步代码到 $TARGET =="
    bash deployment/sync.sh
    "$SSH_BIN" "$TARGET" "cd $REMOTE_DIR && git log --oneline -1 && test -f scripts/experiment.py && test -f scripts/eval/frozen_eval.py && echo CODE_OK" || exit 1

    echo "== 2/3 服务器上启动流水线（setsid nohup 后台）=="
    "$SSH_BIN" "$TARGET" "cd $REMOTE_DIR && mkdir -p $RUN_ROOT && cat > $RUN_ROOT/pipeline.sh" <<EOS
#!/usr/bin/env bash
set -uo pipefail
cd $REMOTE_DIR
ulimit -n 65535
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
LOG=\$RUN_ROOT_LOG/pipeline.log
echo "[pipeline] start \$(date -u)" >> \$LOG

# ---- 三臂 × 三种子：BC(live 专家 8 天) + PPO(10 轮) + 即时评测窗口 A/B ----
run_one () {
  local model=\$1 name=\$2; shift 2
  echo "[pipeline] \$name start \$(date -u +%H:%M)" >> \$LOG
  /opt/qkd/venv/bin/python scripts/experiment.py train \\
    --model \$model --name \$name "\$@" \\
    --bc --bc-episodes 8 --updates 10 --evaluate >> \$LOG 2>&1 \\
    && echo "[pipeline] \$name rc=0 \$(date -u +%H:%M)" >> \$LOG \\
    || echo "[pipeline] \$name FAILED" >> \$LOG
  local rd=outputs/experiments/\$model/\$name
  # 训练完立即评测第二窗口 B（--evaluate 只跑配置里的窗口 A）
  if [ -f "\$rd/checkpoint_final.pt" ]; then
    /opt/qkd/venv/bin/python scripts/eval/frozen_eval.py "rl:\$rd" \\
      --window B --out outputs/frozen_eval >> \$LOG 2>&1 \\
      && echo "[pipeline] \$name windowB rc=0" >> \$LOG \\
      || echo "[pipeline] \$name windowB FAILED" >> \$LOG
  fi
}

for seed in 42 43 44; do run_one v2  "frozen_s\${seed}"           --seed \$seed; done
for seed in 42 43 44; do run_one v3  "frozen_s\${seed}"           --seed \$seed; done
for seed in 42 43 44; do run_one v3  "frozen_isolated_s\${seed}"  --seed \$seed \
    --config model_zoo/v3/config_demand_isolated.yaml; done

# ---- 专家锚：两窗口 ----
/opt/qkd/venv/bin/python scripts/eval/frozen_eval.py expert \\
  --window A B --out outputs/frozen_eval >> \$LOG 2>&1 \\
  && echo "[pipeline] expert rc=0" >> \$LOG || echo "[pipeline] expert FAILED" >> \$LOG

echo "[pipeline] done \$(date -u)" >> \$LOG
touch \$RUN_ROOT_LOG/PIPELINE_DONE
EOS
    "$SSH_BIN" "$TARGET" "cd $REMOTE_DIR && sed -i \"s|\\\$RUN_ROOT_LOG|$RUN_ROOT|\" $RUN_ROOT/pipeline.sh && chmod +x $RUN_ROOT/pipeline.sh && setsid nohup bash $RUN_ROOT/pipeline.sh >/dev/null 2>&1 & echo LAUNCHED"
    echo "流水线已启动。进度: ssh $TARGET 'tail -5 $REMOTE_DIR/$RUN_ROOT/pipeline.log'"
fi

echo "== 3/3 等待完成并回传（fetch 轮询，Ctrl-C 可中断后随时重跑）=="
while :; do
    DONE="$("$SSH_BIN" "$TARGET" "test -f $REMOTE_DIR/$RUN_ROOT/PIPELINE_DONE && echo yes || echo no" 2>/dev/null || echo conn)"
    echo "  $(date +%H:%M:%S) PIPELINE_DONE=$DONE"
    [ "$DONE" = "yes" ] && break
    [ "$DONE" = "conn" ] && echo "  (连接失败，5 分钟后重试——节点可能正在释放，尽快 fetch)"
    sleep 300
done

SCP_BIN="${SSH_BIN%ssh.exe}scp.exe"
mkdir -p "$LOCAL_DIR"
"$SCP_BIN" -q "$TARGET:$REMOTE_DIR/$RUN_ROOT/pipeline.log" "$LOCAL_DIR/" || true
"$SCP_BIN" -q -r "$TARGET:$REMOTE_DIR/outputs/frozen_eval" "$LOCAL_DIR/" || true
"$SCP_BIN" -q -r "$TARGET:$REMOTE_DIR/outputs/experiments" "$LOCAL_DIR/experiments" || true
echo "回传完成: $LOCAL_DIR"
ls -R "$LOCAL_DIR" | head -30
