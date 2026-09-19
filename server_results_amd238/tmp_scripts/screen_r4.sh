#!/usr/bin/env bash
# 第四浪：把轮数从 30 拉到 90，围绕唯一的真增益（entropy_coef 0.01）做单变量延伸。
#
# 为什么必须拉长：第三浪 9 个变体的验证序列**到最后一轮还在涨**（m512ent
# 0.809→0.811→0.813→0.820→0.825→0.828 单调），30 轮是**被截断**的读数，
# 不是收敛点。在没收敛的位置上比变体，比出来的是"谁跑得快"，不是"谁能到更高"。
#
# 为什么只带这几个：第三浪里对 base 的配对差，除了 entropy 之外全部 |t|<2 或为负；
# 所以这一浪不再扫旋钮，只把 entropy 这一个真增益往下接三步，外加一个 base 对照。
#
#   base       对照（无 entropy）
#   ent        唯一真增益本身 —— 90 轮下它到底到哪
#   ent_ep2    entropy + epochs 2（多走梯度步；第三浪没测过 epochs）
#   ent_c128   entropy + minibatch 128（第三浪没测过 128 与 entropy 的组合）
#   ent_lr2e4  entropy + actor_lr 2e-4（3e-4 是峰值，但峰值旁边还有没有空间）
#   c128       minibatch 128 单独（把 batch 从 entropy 里隔离出来）
#
# 全部 2 线程（换线程数会确定性改变结果，见 docs/测试规范.md §6）。
#
# 用法（在节点上）：
#     bash .tmp/screen_r4.sh
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
UPDATES="${UPDATES:-90}"
JOBS="${JOBS:-6}"
THREADS=2
CKPT=outputs/supervised_pg_phased/supervised_pg_phased_latest.pt
BASE_CFGS="rl_algorithm.yaml train_diag_fast.yaml"

NAMES=(base ent ent_ep2 ent_c128 ent_lr2e4 c128)

write_yaml() {
    if ! grep -qxF 'configs/var2_*.yaml' .git/info/exclude 2>/dev/null; then
        printf 'configs/var2_*.yaml\n' >> .git/info/exclude
    fi
    case "$1" in
        base)      printf 'train:\n  ppo:\n    epochs: 1\n' > configs/var2_r4_base.yaml ;;
        ent)       printf 'train:\n  ppo:\n    entropy_coef: 0.01\n' > configs/var2_r4_ent.yaml ;;
        ent_ep2)   printf 'train:\n  ppo:\n    entropy_coef: 0.01\n    epochs: 2\n' > configs/var2_r4_ent_ep2.yaml ;;
        ent_c128)  printf 'train:\n  ppo:\n    entropy_coef: 0.01\n    minibatch_size: 128\n' > configs/var2_r4_ent_c128.yaml ;;
        ent_lr2e4) printf 'train:\n  ppo:\n    entropy_coef: 0.01\n  optimizer:\n    actor_lr: 0.0002\n' > configs/var2_r4_ent_lr2e4.yaml ;;
        c128)      printf 'train:\n  ppo:\n    minibatch_size: 128\n' > configs/var2_r4_c128.yaml ;;
    esac
}

run_one() {
    local name="$1"
    (
        cd "$MAIN" || exit 1
        rm -rf "outputs/r4_${name}"
        timeout 21600 env OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
            "$PY" -u scripts/train/train_graph_mappo.py \
                --configs $BASE_CFGS "var2_r4_${name}.yaml" \
                --checkpoint "$CKPT" \
                --num-updates "$UPDATES" --run-name "r4_${name}" \
                > "/tmp/r4_${name}.log" 2>&1
    )
    echo "  [$name] 结束（退出码 $?）"
}

cd "$MAIN" || exit 1
for n in "${NAMES[@]}"; do write_yaml "$n"; done
echo "第四浪：${NAMES[*]}"
echo "轮数=$UPDATES  并发=$JOBS  线程=$THREADS（钉死）"
echo

n=0
for name in "${NAMES[@]}"; do
    run_one "$name" &
    n=$((n + 1))
    if [ $((n % JOBS)) -eq 0 ]; then wait; fi
done
wait

echo
echo "======== 第四浪结果（对 base 逐种子配对；参照 BC 0.7680 / 专家 0.7828）========"
"$PY" .tmp/sum_group.py r4
echo SCREEN_R4_DONE