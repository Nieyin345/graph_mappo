#!/usr/bin/env bash
# 快速筛选：哪一组 PPO 配置能让训练真正涨起来。
#
# 诊断已排除的：PPO 比率（探针 A，正确）、奖励设计（探针 B/C，主项 served 对动作
# 强敏感）、critic（corr(V,R)→0.99）、模型前向等价的实现细节。剩下的是优化过程
# 本身 —— 每轮 KL 只有 ~0.001（目标 0.02），45 个 minibatch 梯度的合矢量只有它们
# 长度之和的 10-13%（低于"互相独立"的 0.149）。
#
# 逐个试，看哪一组能让"训练成功率"真的往上走：
#   base       现状（对照组）
#   noadvnorm  关掉按 minibatch 归一化优势
#   noent      关掉熵奖励（熵一路涨而成功率平，怀疑熵项是唯一方向一致的分量）
#   ep4        每轮 4 个 epoch（现在只有 1 个）
#   lr1e3      actor 学习率 3e-4 -> 1e-3
#   ep4noadv   ep4 + noadvnorm
#
# 用 train_diag_fast（240 步 rollout）跑，比全窗口快得多，而且它本身也是平的
# （基线 20 轮 0.7627 不动），"能不能涨"在这里同样可判。
#
# 两个坑（都踩过）：
#   1. 覆盖配置必须落在 **configs/** 下 —— build_config 是拿 ROOT/configs/<name>
#      拼路径的，写 .tmp/var_x.yaml 会被解析成 configs/.tmp/var_x.yaml。
#   2. 学习率在 **train.optimizer** 下，不是 train.ppo。写错层级不会报错，
#      只会被静默忽略（第一版 lr1e3 就是这么白跑的）。所以每个变体直接写完整
#      YAML，不做键路径拼接。
#
# 用法（在节点上）：
#     bash .tmp/screen_ppo_variants.sh
#     UPDATES=20 JOBS=8 THREADS=4 bash .tmp/screen_ppo_variants.sh
#
# 并行度按实测定的（见 docs/并行与吞吐.md）：这台机器 56 核 / 251 GB，**单进程
# 吃不满**（rollout 是 Python 串行的，16 线程也只有 1.33×），但**多进程接近线性**。
# 同样 48 线程的预算，切得越薄吞吐越高：
#     8×6 线程 0.102 轮/s  <  12×4 0.129  <  24×2 0.172  <  48×1 0.223
# 选 24×2 而不是 48×1：后者只多 30% 吞吐，却要 154 GB 内存（每个进程 ~3.2 GB），
# 把 251 GB 的余量压到三分之一。24×2 约 77 GB，留足余量。
set -u

MAIN=/opt/qkd/graph_mappo
PY=/opt/qkd/venv/bin/python
UPDATES="${UPDATES:-30}"
JOBS="${JOBS:-24}"
THREADS="${THREADS:-2}"
CKPT="outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
BASE_CFGS="rl_algorithm.yaml train_diag_fast.yaml"

NAMES=(base lr1e3 lr1e4 ep4 ep4lr1e3 ep8lr1e3)

write_yaml() {
    local name="$1"
    # 变体配置写在受跟踪的 configs/ 里，会让工作区多出未跟踪文件，下一次
    # server_sync.sh 的推送就会被 receive.denyCurrentBranch=updateInstead 拒掉。
    # 加进 .git/info/exclude（只影响本节点，不进版本库）挡住这一点。
    if ! grep -qxF 'configs/var_*.yaml' .git/info/exclude 2>/dev/null; then
        printf 'configs/var_*.yaml\n' >> .git/info/exclude
    fi
    case "$name" in
        base)      cat > configs/var_base.yaml <<'YAML'
# 对照。注意：**修好 load_checkpoint 的学习率覆盖 bug 之后**，本组才第一次
# 真正跑在配置值 actor_lr=0.0003 上；此前所有历史结果实际都是 0.001。
train:
  ppo:
    epochs: 1
YAML
                   ;;
        lr1e3)     cat > configs/var_lr1e3.yaml <<'YAML'
# actor/encoder 学习率 0.001 —— 等价于**此前全部历史实验**的实际值，
# 用作和旧结果对接的锚点。
train:
  optimizer:
    actor_lr: 0.001
YAML
                   ;;
        lr1e4)     cat > configs/var_lr1e4.yaml <<'YAML'
train:
  optimizer:
    actor_lr: 0.0001
YAML
                   ;;
        lr3e3)     cat > configs/var_lr3e3.yaml <<'YAML'
train:
  optimizer:
    actor_lr: 0.003
YAML
                   ;;
        ep4)       cat > configs/var_ep4.yaml <<'YAML'
# 4 个 epoch，跑在配置值 0.0003 上（修好 bug 之后才第一次如此）。
train:
  ppo:
    epochs: 4
YAML
                   ;;
        ep4lr1e3)  cat > configs/var_ep4lr1e3.yaml <<'YAML'
# 第一轮里唯一出现上升趋势的那组（ep4），但保持历史上的实际学习率 0.001。
# 用来判断"上升"是 epoch 的功劳还是 lr 的功劳。
train:
  ppo:
    epochs: 4
  optimizer:
    actor_lr: 0.001
YAML
                   ;;
        ep8lr1e3)  cat > configs/var_ep8lr1e3.yaml <<'YAML'
train:
  ppo:
    epochs: 8
  optimizer:
    actor_lr: 0.001
YAML
                   ;;
        ep4noadv)  cat > configs/var_ep4noadv.yaml <<'YAML'
train:
  ppo:
    epochs: 4
    normalize_advantages: false
YAML
                   ;;
    esac
}

run_one() {
    local name="$1"
    (
        cd "$MAIN" || exit 1
        rm -rf "outputs/screen_${name}"
        timeout 7200 env OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
            "$PY" -u scripts/train/train_graph_mappo.py \
                --configs $BASE_CFGS "var_${name}.yaml" \
                --checkpoint "$CKPT" \
                --num-updates "$UPDATES" --run-name "screen_${name}" \
                > "/tmp/screen_${name}.log" 2>&1
    )
    echo "  [$name] 结束（退出码 $?）"
}

cd "$MAIN" || exit 1
for n in "${NAMES[@]}"; do write_yaml "$n"; done
echo "已生成变体配置：$(ls configs/var_*.yaml 2>/dev/null | tr '\n' ' ')"
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
echo "======== 结果 ========"
for name in "${NAMES[@]}"; do
    f="$MAIN/outputs/screen_${name}/metrics.jsonl"
    printf '%-12s ' "$name"
    if [ -f "$f" ]; then
        "$PY" - "$f" <<'PY'
import json, sys
rows, evals = [], []
for line in open(sys.argv[1], encoding="utf-8"):
    try: d = json.loads(line)
    except Exception: continue
    if "eval_validation" in d:
        evals.append(d["eval_validation"]["mean_success_rate"])
    elif "update" in d:
        rows.append(d)
if not rows:
    print("(没有训练记录)")
else:
    half = len(rows) // 2
    a = sum(r["mean_success_rate"] for r in rows[:half]) / max(1, half)
    b = sum(r["mean_success_rate"] for r in rows[half:]) / max(1, len(rows) - half)
    ea = sum(r.get("entropy", 0) for r in rows[:half]) / max(1, half)
    eb = sum(r.get("entropy", 0) for r in rows[half:]) / max(1, len(rows) - half)
    kl = sum(r.get("kl", 0) for r in rows) / len(rows)
    ev = ("验证 %.4f (%d 次)" % (evals[-1], len(evals))) if evals else "无验证"
    print(f"训练 {a:.4f} -> {b:.4f}   熵 {ea:.3f}->{eb:.3f}   平均kl {kl:.5f}   {ev}")
PY
    else
        echo "(没有 metrics —— 看 /tmp/screen_${name}.log)"
    fi
done
echo
echo "参照：train_full_rl 30 轮基线：训练 0.8626 -> 0.8538，验证平在 0.63-0.68"
echo "SCREEN_DONE"
