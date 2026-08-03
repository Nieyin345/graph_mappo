from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qkd_rl.baselines.greedy_rate import GreedyRatePolicy
from qkd_rl.core.config import ConfigValidator, deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config


def main() -> None:
    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    config["rate_provider"]["provider"] = "h5"
    print(f"Resolved feature dims: {ConfigValidator().resolve_feature_dims(config)}")
    env = build_env_from_config(config)
    obs = env.reset()
    policy = GreedyRatePolicy()
    total_reward = 0.0
    done = False
    steps = 0
    while not done:
        actions, scores = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(actions, scores)
        total_reward += reward
        done = terminated or truncated
        steps += 1
    summary = env.metrics.episode_summary()
    print(
        {
            "steps": steps,
            "total_reward": round(total_reward, 3),
            "summary": summary,
            "last_info": {k: v for k, v in info.items() if k != "reward_detail"},
        }
    )


if __name__ == "__main__":
    main()
