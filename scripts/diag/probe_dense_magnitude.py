# -*- coding: utf-8 -*-
"""`dense_enabled: false` 打开后**实际值是多少**？—— 先量幅度，再谈要不要开。

### 为什么先量这个

`dense_generation_importance_weight: 0.02` 是配置里唯一的"给生成定价"的旋钮
（`dense_reward` 正是喂给 `generated_reward` 的那一项，而它被 `dense_enabled:
false` 关着）。但**配置里的数不等于生效的幅度**：

    dense = weight × mean_over_edges(1.01×importance − 0.01)      [份额型]
    （1.01 = 1 + penalty 0.01；importance ∈ [0,1]）

`importance` 是 `relay_importance`，而探针 P 实测"需求要 6.5 条边、专家激活
60.8 条、重合只有 3.4 条"（覆盖率 31.71%）—— 也就是说**货大多加在重要度很低的
边上**，那么这个均值可能非常小。若实际幅度只有 served 项的 1%，开它就是白开；
若是 10%，那它是个真旋钮。**先量，再决定。**

同一条式子还给出一个**负值区间**：importance < 0.0099 时 value 为负 ⟹
低重要度上加货会被**惩罚**。这正是"别把货堆在错的边上"的信号 ——
探针 P 指出的病。

用法（服务器上，隔离副本）：
    /opt/qkd/venv/bin/python /tmp/probe_dense_magnitude.py /tmp/metrics_check
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/metrics_check")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

from qkd_rl.env.factory import build_env_from_config
from qkd_rl.baselines.path_greedy import PathScoreGreedy
from qkd_rl.baselines.serve_probe import ServeProbe

STEPS = 40
SEEDS = [100, 101]


def run_one(seed: int, dense_on: bool):
    profile = _tp.load_validation_profile(ROOT / "configs" / "global.yaml")
    config = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=STEPS,
        start_mode=profile["start_mode"])
    # 只改这一个开关，其余不动 —— 好让读数就是这一项。
    config["reward"]["dense_enabled"] = dense_on
    env = build_env_from_config(config)
    obs = env.reset(seed=seed, start_seed=int(profile.get("start_seed", 0)) + seed)
    expert = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True,
                             principles=False, router=ServeProbe(env))
    agg = {"dense": 0.0, "served": 0.0, "storage": 0.0, "keep": 0.0,
           "failed": 0.0, "total": 0.0}
    n = 0
    done = False
    while not done and n < STEPS:
        actions, scores = expert.act(obs)
        obs, _r, terminated, truncated, info = env.step(actions, scores)
        d = info.get("reward_detail")
        if d is not None:
            agg["dense"] += float(getattr(d, "dense_reward", 0.0))
            agg["served"] += float(getattr(d, "served_reward", 0.0))
            agg["storage"] += float(getattr(d, "storage_reward", 0.0))
            agg["keep"] += float(getattr(d, "keep_active_reward", 0.0))
            agg["failed"] += float(getattr(d, "failed_penalty", 0.0))
            agg["total"] += float(getattr(d, "total", 0.0))
        n += 1
        done = terminated or truncated
    return {k: v / max(1, n) for k, v in agg.items()}, n


print("=" * 72)
print("dense 关闭 vs 打开：每一步的奖励分项（实测，非配置推算）")
print("=" * 72)

for seed in SEEDS:
    off, n = run_one(seed, dense_on=False)
    on, _ = run_one(seed, dense_on=True)
    print(f"\nseed {seed}  ({n} 步)")
    print(f"  {'分项':<12}{'dense关':>14}{'dense开':>14}{'差':>14}")
    for k in ("served", "dense", "storage", "keep", "failed", "total"):
        print(f"  {k:<12}{off[k]:>14.6f}{on[k]:>14.6f}{on[k]-off[k]:>14.6f}")
    ratio_off = off["dense"] / off["served"] if off["served"] else 0.0
    ratio_on = on["dense"] / on["served"] if on["served"] else 0.0
    print(f"  dense/served = {ratio_off*100:.2f}%  →  {ratio_on*100:.2f}%")

print("\n" + "=" * 72)
print("判读口径：dense 打开后占 served 的比例")
print("  < 2%   ⟹ 太弱，开它等于没开（不值得占一个旋钮位）")
print("  2~10%  ⟹ 真旋钮，但需要 ≥3 训练种子才测得出来")
print("  > 10%  ⟹ 强信号，可能显著改变训练行为（要先小步验证）")
print("=" * 72)
