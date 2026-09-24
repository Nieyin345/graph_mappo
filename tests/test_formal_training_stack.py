"""Regression contract for the currently deployed formal RL preset stack."""

from pathlib import Path

from qkd_rl.core.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs"
FORMAL_STACK = [
    "rl_algorithm.yaml",
    "train_full_rl.yaml",
    "train_ent01.yaml",
    "train_joint_ppo_fix.yaml",
    "train_stocked_graph.yaml",
]


def test_formal_joint_ppo_stack_keeps_calibrated_overrides():
    cfg = load_config([CONFIG / name for name in FORMAL_STACK])

    assert cfg["train"]["episodes_per_update"] == 8
    assert cfg["train"]["rollout_steps"] == 1440
    assert cfg["train"]["optimizer"]["actor_lr"] == 5e-5
    assert cfg["train"]["ppo"]["entropy_coef"] == 0.01
    assert cfg["train"]["ppo"]["target_kl"] == 0.02
    assert cfg["features"]["edge"]["include_stocked_edges"] is True
    assert cfg["validation"]["episodes"] == 15
