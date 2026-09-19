#!/usr/bin/env bash
# 第二轮变体筛选。学习率 bug 修好之后，这是**第一次真正跑在配置的 actor_lr 上**。
#
# 为什么一次开这么多：见 docs/并行与吞吐.md。这台机器 56 核，单进程吃不满
# （rollout 是 Python 串行的），但多进程接近线性 —— 24 进程 × 2 线程是吞吐拐点。
# 于是实验的排法从"一次一个"改成"一次一浪"。
#
# **浪** = 一组成本相当的变体。墙钟由最慢的那个决定，所以同一浪里混进一个
# 4 倍成本的变体，其余的全都在白等。epochs / episodes_per_update 会成倍抬高
# update 成本，所以它们单独放第二浪。
#
# 判据：metrics.jsonl 里的 eval_validation（固定场景、12 个种子、确定性）才是
# 低方差读数；训练侧的 mean_success_rate 是采样值，噪声大得多，只作参考。
# 逐种子值也记在 eval_validation.per_seed_success 里 —— 配对比较才分辨得出
# 0.02 量级的差异（见 docs/测试规范.md ⑦）。汇总用 .tmp/summarize_screen.py。
#
# 用法（在节点上）：
#     bash .tmp/screen_r2.sh wave1
#     bash .tmp/screen_r2.sh wave2
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
WAVE="${1:-wave1}"
UPDATES="${UPDATES:-30}"
JOBS="${JOBS:-24}"
THREADS="${THREADS:-2}"
CKPT="outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
BASE_CFGS="rl_algorithm.yaml train_diag_fast.yaml"

case "$WAVE" in
    wave1) NAMES=(base lr1e4 lr1e3 lr3e3 mini128 mini512 ent0 ent1e2 clip02 kl001 vcoef1 mini128lr1e3) ;;
    wave2) NAMES=(ep2 ep4 ep4lr1e3 ep4mini128 epi16 epi16lr1e3) ;;
    *) echo "用法: $0 {wave1|wave2}" >&2; exit 2 ;;
esac

write_yaml() {
    # 变体配置必须落在 configs/ 下：build_config 拼的是 ROOT/configs/<name>，
    # 写 .tmp/var_x.yaml 会被解析成 configs/.tmp/var_x.yaml。
    # 同时加进节点的 .git/info/exclude（只影响本节点），否则工作区变脏，
    # 下一次 server_sync.sh 的推送会被 receive.denyCurrentBranch=updateInstead 拒掉。
    if ! grep -qxF 'configs/var2_*.yaml' .git/info/exclude 2>/dev/null; then
        printf 'configs/var2_*.yaml\n' >> .git/info/exclude
    fi
    case "$1" in
        base)         cat > configs/var2_base.yaml <<'YAML'
# 对照。**首次真正跑在 train.optimizer.actor_lr = 0.0003 上** —— 学习率 bug
# 修好之前，load_checkpoint 会把 BC 权重里存的 0.001 恢复回来盖掉它，
# 所以全部历史结果实际都是 0.001。与它们不可直接比。
#
# 显式写一个等于默认值的键，而不是只留注释：全注释的 YAML 解析出来是 None，
# deep_merge(config, None) 不一定是空操作。写死默认值既保证非空，又确认了
# "对照就是默认"，不会因为以后改了默认值而悄悄变成另一个变体。
train:
  ppo:
    epochs: 1
YAML
                      ;;
        lr1e4)        printf 'train:\n  optimizer:\n    actor_lr: 0.0001\n' > configs/var2_lr1e4.yaml ;;
        lr1e3)        printf 'train:\n  optimizer:\n    actor_lr: 0.001\n' > configs/var2_lr1e3.yaml ;;
        lr3e3)        printf 'train:\n  optimizer:\n    actor_lr: 0.003\n' > configs/var2_lr3e3.yaml ;;
        mini128)      cat > configs/var2_mini128.yaml <<'YAML'
# 每轮梯度步数 8 -> 16。样本总量不变，只是切成更小的批。
train:
  ppo:
    minibatch_size: 128
YAML
                      ;;
        mini512)      cat > configs/var2_mini512.yaml <<'YAML'
# 每轮梯度步数 8 -> 4。批更大、BLAS 更吃得满，但单步方差更大。
train:
  ppo:
    minibatch_size: 512
YAML
                      ;;
        ent0)         printf 'train:\n  ppo:\n    entropy_coef: 0.0\n' > configs/var2_ent0.yaml ;;
        ent1e2)       printf 'train:\n  ppo:\n    entropy_coef: 0.01\n' > configs/var2_ent1e2.yaml ;;
        clip02)       printf 'train:\n  ppo:\n    clip_eps: 0.2\n' > configs/var2_clip02.yaml ;;
        kl001)        printf 'train:\n  ppo:\n    target_kl: 0.01\n' > configs/var2_kl001.yaml ;;
        vcoef1)       printf 'train:\n  ppo:\n    value_coef: 1.0\n' > configs/var2_vcoef1.yaml ;;
        mini128lr1e3) cat > configs/var2_mini128lr1e3.yaml <<'YAML'
# 两个最直接"多走几步"的杠杆叠加。
train:
  ppo:
    minibatch_size: 128
  optimizer:
    actor_lr: 0.001
YAML
                      ;;
        ep2)          printf 'train:\n  ppo:\n    epochs: 2\n' > configs/var2_ep2.yaml ;;
        ep4)          printf 'train:\n  ppo:\n    epochs: 4\n' > configs/var2_ep4.yaml ;;
        ep4lr1e3)     printf 'train:\n  ppo:\n    epochs: 4\n  optimizer:\n    actor_lr: 0.001\n' > configs/var2_ep4lr1e3.yaml ;;
        ep4mini128)   printf 'train:\n  ppo:\n    epochs: 4\n    minibatch_size: 128\n' > configs/var2_ep4mini128.yaml ;;
        epi16)        printf 'train:\n  episodes_per_update: 16\n' > configs/var2_epi16.yaml ;;
        epi16lr1e3)   printf 'train:\n  episodes_per_update: 16\n  optimizer:\n    actor_lr: 0.001\n' > configs/var2_epi16lr1e3.yaml ;;
    esac
}

run_one() {
    local name="$1"
    (
        cd "$MAIN" || exit 1
        rm -rf "outputs/r2_${name}"
        timeout 21600 env OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
            "$PY" -u scripts/train/train_graph_mappo.py \
                --configs $BASE_CFGS "var2_${name}.yaml" \
                --checkpoint "$CKPT" \
                --num-updates "$UPDATES" --run-name "r2_${name}" \
                > "/tmp/r2_${name}.log" 2>&1
    )
    echo "  [$name] 结束（退出码 $?）"
}

cd "$MAIN" || exit 1
for n in "${NAMES[@]}"; do write_yaml "$n"; done
echo "浪 = $WAVE   变体 = ${NAMES[*]}"
echo "轮数=$UPDATES  并发=$JOBS  每进程线程=$THREADS"
echo

n=0
for name in "${NAMES[@]}"; do
    run_one "$name" &
    n=$((n + 1))
    if [ $((n % JOBS)) -eq 0 ]; then wait; fi
done
wait

echo
"$PY" .tmp/summarize_screen.py "${NAMES[@]}"
echo "SCREEN_R2_DONE"
