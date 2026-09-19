"""对 env.step 做 cProfile，找出真实热点（而非凭代码观感猜）。

用诊断场景规模（80 步）跑一个 episode，打印累计耗时前 25 名函数。
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qkd_rl.core.config import ConfigValidator, deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config


def build(steps: int):
    config = load_default_config(ROOT)
    for name in ("env_full.yaml",):
        config = deep_merge(config, load_config([ROOT / "configs" / name]))
    config = deep_merge(
        config,
        {
            "env": {"episode_steps": steps, "episode_start_mode": "fixed", "episode_start_day": 0},
            "runtime": {"device": "cpu"},
        },
    )
    ConfigValidator().validate(config)
    return build_env_from_config(config)


def run(env, n: int, greedy_zero: bool = True):
    """用全零动作跑 n 步（不依赖策略，纯测环境本身的开销）。"""
    obs = env.reset(seed=7)
    for _ in range(n):
        # 让环境自己走：所有节点选 STOP（最省动作解析，聚焦环境内部开销）
        actions = {nid: (None, None) for nid in obs.node_ids}
        obs, reward, terminated, truncated, info = env.step(actions, {}, edge_scores=None)  # type: ignore[arg-type]
        if terminated or truncated:
            break
    return info


def main() -> int:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    env = build(steps)

    # 预热一次，避免首次 import / 缓存构建混进 profile
    run(build(steps), 5)

    pr = cProfile.Profile()
    pr.enable()
    run(env, steps)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(25)
    print(s.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
