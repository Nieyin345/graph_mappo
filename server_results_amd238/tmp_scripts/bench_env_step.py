"""env.step 的端到端耗时基准（多次重复取中位数，避免单次抖动）。

用来对比优化前后的每步耗时；同一进程内连续测两轮，第二轮是主读数。
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qkd_rl.core.config import ConfigValidator, deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config


def build(steps: int, diag: bool):
    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    extra = {"env": {"episode_steps": steps}, "runtime": {"device": "cpu"}}
    if diag:
        extra["env"].update({"episode_start_mode": "fixed", "episode_start_day": 0})
    else:
        extra["env"].update({"episode_start_mode": "random_day"})
    config = deep_merge(config, extra)
    ConfigValidator().validate(config)
    return build_env_from_config(config)


def timed(env, n: int) -> float:
    """返回每步中位耗时（ms）。"""
    obs = env.reset(seed=7)
    per_step = []
    for _ in range(n):
        actions = {nid: (None, None) for nid in obs.node_ids}
        t0 = time.perf_counter()
        obs, *_ = env.step(actions, {}, edge_scores=None)  # type: ignore[arg-type]
        per_step.append((time.perf_counter() - t0) * 1000)
    return statistics.median(per_step)


def main() -> int:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    for diag in (True, False):
        env = build(steps, diag)
        timed(build(steps, diag), 10)  # 预热
        med = timed(env, steps)
        tag = "诊断(fixed day0)" if diag else "真实(random_day)"
        print(f"  {tag:<20} 每步中位 {med:7.3f} ms   ({steps} 步)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
