"""Probe whether the observed ~50% success rate is a hard ceiling.

Runs deterministic greedy baselines on the same fixed first day while scaling
the offered request load. If lowering the load raises the success rate well
above 50%, the current plateau is load/topology pressure rather than a
capacity ceiling of the QKD network itself.

Usage:
    python scripts/probe_success_ceiling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qkd_rl.baselines.greedy_rate import GreedyRatePolicy
from qkd_rl.baselines.greedy_relay import GreedyRelayPolicy
from qkd_rl.core.config import deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config


def main() -> None:
    base = load_default_config(ROOT)
    base = deep_merge(base, load_config([ROOT / "configs" / "env_full.yaml"]))
    base["rate_provider"]["provider"] = "h5"
    base["env"]["episode_start_mode"] = "fixed"
    base["env"]["episode_steps"] = int(10**9)
    base["scenario"]["time_limit"]["days"] = 1

    policies = {
        "greedy_rate": GreedyRatePolicy(),
        "greedy_relay": GreedyRelayPolicy(),
    }
    for load in (1.0, 0.5, 0.25):
        config = deep_merge(base, {"requests": {"arrival_rate": 0.8 * load}})
        env = build_env_from_config(config)
        row = []
        for name, policy in policies.items():
            obs = env.reset(seed=1234)
            done = False
            while not done:
                actions, scores = policy.act(obs)
                obs, _reward, terminated, truncated, _info = env.step(actions, scores)
                done = terminated or truncated
            summary = env.metrics.episode_summary()
            row.append(f"{name}={summary['success_rate']:.3f}")
        print(f"load={load:.2f} arrived={env.metrics.arrived_keys:.0f} " + " ".join(row))


if __name__ == "__main__":
    main()
