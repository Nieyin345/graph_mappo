#!/usr/bin/env python
"""对 rollout 单步做 cProfile，找**当前**的热点。

背景：docs/deployment/server.md §5.2 那份 profile 是 chunk=512 时代测的，
列出的热点是"逐 1978 条链路走 Python / dict.get 每步 5000 次"。但
graph_builder.py 在 2026-09-18 已经改过（物理边特征向量化、req_hop 直接
产数组），那份数据已经不能作为优化依据。这个脚本重新测一遍。

跑法（服务器上）：
    /opt/qkd/venv/bin/python .tmp/prof_rollout.py 300

输出：top 25 按 self 时间排序的函数，以及按模块聚合的占比。
"""
import cProfile
import io
import pstats
import sys
import time
from collections import defaultdict

sys.path.insert(0, "/opt/qkd/graph_mappo")

from qkd_rl.core.config import deep_merge, load_config  # noqa: E402
from qkd_rl.env.factory import build_env_from_config, load_default_config  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path("/opt/qkd/graph_mappo")
N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def build_env():
    cfg = load_default_config(ROOT)
    cfg = deep_merge(cfg, load_config([ROOT / "configs" / "env_full.yaml"]))
    profiles = load_config([ROOT / "configs" / "train_profiles.yaml"])["train_profiles"]
    cfg = deep_merge(cfg, profiles["random_episode"])
    cfg = deep_merge(cfg, load_config([ROOT / "configs" / "rl_algorithm.yaml"]))
    if cfg["env"].get("continuous"):
        pass
    # 与训练一致：full 场景、1440 步
    cfg["train"]["rollout_steps"] = 1440
    # build_env_from_config(config) 只收 1 个参数（factory.py:42），
    # 路径从 config 里取，不要再传 ROOT。
    return build_env_from_config(cfg)


def main():
    env = build_env()
    obs, info = env.reset()

    # 预热若干步，避免把首次的懒加载算进去
    for _ in range(20):
        mask = info.get("action_mask") if isinstance(info, dict) else None
        action = _pick(env, mask)
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            obs, info = env.reset()

    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    for i in range(N_STEPS):
        mask = info.get("action_mask") if isinstance(info, dict) else None
        action = _pick(env, mask)
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            obs, info = env.reset()
    prof.disable()
    wall = time.perf_counter() - t0

    print(f"\n{N_STEPS} 步，墙钟 {wall:.2f} s，{wall / N_STEPS * 1000:.2f} ms/步\n")

    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(25)
    print(s.getvalue())

    # 按模块聚合（只看有源码的帧）
    agg = defaultdict(float)
    total = 0.0
    for (fn, _, name), (cc, nc, tt, ct, _) in prof.getstats().items():
        total += tt
        if "qkd_rl/env" in fn:
            key = "env"
        elif "qkd_rl/rl/models" in fn:
            key = "models"
        elif "qkd_rl/rl/algos" in fn:
            key = "algos"
        elif "qkd_rl/link" in fn:
            key = "link"
        elif "numpy" in fn or "site-packages/numpy" in fn:
            key = "numpy"
        elif "torch" in fn:
            key = "torch"
        else:
            key = "other"
        agg[key] += tt
    print("\n按模块聚合（self 时间占比）：")
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<10} {v:7.2f}s  {v / total * 100:5.1f}%")


def _pick(env, mask):
    """随便选一个合法动作 —— 只要能让 env 往前走，不关心策略质量。"""
    import numpy as np
    if mask is not None:
        legal = np.flatnonzero(np.asarray(mask).ravel())
        if legal.size:
            return int(legal[0])
    return 0


if __name__ == "__main__":
    main()
