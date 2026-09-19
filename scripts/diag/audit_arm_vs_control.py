"""起臂**之后**的核对：把实验臂的 resolved_config 与对照臂逐字段对。

### 为什么需要它（2026-09-19 实测事故）

gae99 波的自检是 `selfcheck_gae99.py`，它在**一个进程里解析两条配置链**
再 diff，报「271 字段、差异 1」—— 看着无懈可击。

但它**结构上不可能**发现真正出的事：两条链都是同一个进程
`build_config()` 出来的，而线程数是**进程级**的（`torch.set_num_threads`
读 `OMP_NUM_THREADS` 环境变量，没有就退到 `cpu_count()//2`）。
自检进程里两边一样，**不等于两个训练进程里一样**。

结果：gae99 三臂以 `num_threads=64` 起跑（脚本没设 OMP_NUM_THREADS，
节点 128 核 ⟹ `cpu_count()//2 = 64`），而对照臂 `ent01_t8_*` 是
`OMP_NUM_THREADS=8`。**线程数会确定性改变训练结果（差 0.018，与效应量同量级）**
⟹ 这个 A/B 被污染，跑了 21 分钟、0 轮，全部作废。

### 本脚本的判据

不是「配置文件的文本 diff」—— `resolved_config.yaml` 是**运行时**写下来的，
它含 `runtime.torch_num_threads` / `omp_num_threads_env`，所以能直接看见
**进程真正用的是多少线程**。这一点是 `build_config()` 做不到的。

用法：
    python scripts/diag/audit_arm_vs_control.py \
        --arms gae99_s42 gae99_s43 gae99_s44 \
        --ctrl  ent01_t8_s42 ent01_t8_s43 ent01_t8_s44
退出码：0 = 干净；1 = 有差异（逐条打印）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

OUT = Path("/opt/qkd/graph_mappo/outputs")

# 这些字段**允许**不同（本来就该随臂变）；其余任何不同都要报出来。
ALLOWED_DIFF = {
    "train.gae_lambda", "train.entropy_coef", "train.clip_eps",
    "train.learning_rate", "train.critic_learning_rate", "train.epochs",
    "train.minibatch_size", "train.value_coef", "train.max_grad_norm",
    "train.target_kl", "train.normalize_advantages", "train.num_updates",
    "activation_window.start_day", "activation_window.end_day",
    "runtime.run_name", "runtime.seed", "train.seed", "seed",
    "project.run_name", "run_name",
    # 种子字段：正常的配对用法里 arms 与 ctrl 同种子，这些不会差；
    # 但拿它跨种子比（例如核对两条不同种子的臂）时该放行。
    "seed.env_seed", "seed.global_seed", "seed.rollout_seed",
}

# 这些字段**看起来**像差异其实不是（同一配置的两种写法）；但也别静默放过，
# 列出来让人看一眼就好。
COSMETIC = {"project.run_name", "run_name"}

# ★ 线程相关的字段**永远**必须一致：它们不改变"算法"，但确定性改变结果
THREAD_KEYS = {
    "runtime.num_threads", "runtime.torch_num_threads",
    "runtime.omp_num_threads_env", "runtime.mkl_num_threads_env",
}


def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{p}.{k}" if p else str(k)))
    elif isinstance(d, list):
        out[p] = str(d)
    else:
        out[p] = d
    return out


def load(run: str) -> dict | None:
    p = OUT / run / "resolved_config.yaml"
    if not p.exists():
        print(f"  ✗ 缺 {p}")
        return None
    return flat(yaml.safe_load(p.read_text(encoding="utf-8")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--ctrl", nargs="+", required=True)
    ap.add_argument("--strict-threads", action="store_true", default=True,
                    help="线程字段不一致直接判失败（默认开）")
    a = ap.parse_args()

    if len(a.arms) != len(a.ctrl):
        print("  ✗ --arms 与 --ctrl 数量必须相同（逐臂配对）")
        return 1

    bad_total = 0
    for arm, ctrl in zip(a.arms, a.ctrl):
        print("=" * 78)
        print(f"{arm}  vs  {ctrl}")
        print("=" * 78)
        fa, fc = load(arm), load(ctrl)
        if fa is None or fc is None:
            bad_total += 1
            continue

        keys = sorted(set(fa) | set(fc))
        diffs = [(k, fc.get(k), fa.get(k)) for k in keys if fc.get(k) != fa.get(k)]

        if not diffs:
            print("  ✓ 无任何差异")
            print()
            continue

        thread_diffs = [(k, x, y) for k, x, y in diffs if k in THREAD_KEYS]
        other = [(k, x, y) for k, x, y in diffs if k not in THREAD_KEYS]

        # ---- 1. 线程：最严重，单列 ----
        if thread_diffs:
            print("  ✗✗ **线程设置不同 ⟹ 这个 A/B 被污染，结果不可用**")
            for k, x, y in thread_diffs:
                print(f"       {k}")
                print(f"           对照 {ctrl}: {x}")
                print(f"           实验 {arm}: {y}")
            print("       线程数确定性改变训练结果（实测差 0.018，与效应量同量级）。")
            print("       修法：起臂时显式 `env OMP_NUM_THREADS=<与对照相同>` "
                  "（或 MKL_NUM_THREADS），别依赖脚本的 cpu_count()//2 兜底。")
            bad_total += 1

        # ---- 2. 其余差异：区分"预期内的"与"意外的" ----
        unexpected = [(k, x, y) for k, x, y in other if k not in ALLOWED_DIFF]
        expected = [(k, x, y) for k, x, y in other if k in ALLOWED_DIFF]

        if expected:
            print(f"  · 预期内的差异 {len(expected)} 条（在白名单里）：")
            for k, x, y in expected:
                print(f"       {k}: {x} → {y}")
        if unexpected:
            print(f"  ⚠ **不在白名单里的差异 {len(unexpected)} 条** —— 逐条确认再收下：")
            for k, x, y in unexpected:
                print(f"       {k}: {x} → {y}")
            bad_total += 1
        if not thread_diffs and not unexpected:
            print("  ✓ 差异全部在白名单内、且线程一致 ⟹ 是干净的单参数 A/B")
        print()

    print("=" * 78)
    if bad_total:
        print(f"✗ {bad_total}/ {len(a.arms)} 臂有问题 ⟹ **不要用这组读数下结论**")
        return 1
    print(f"✓ {len(a.arms)} 臂全部干净")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
