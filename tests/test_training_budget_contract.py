"""All models must resolve to the SAME training budget, and it must actually train.

Why this exists (the failure it prevents):

The training budget (rollout_steps / episodes_per_update / minibatch_size)
determines `nb`, the number of gradient steps per update:

    nb = rollout_steps * episodes_per_update / minibatch_size

Measured on this repo before the fix (2026-09-26), for EVERY model v2/v3/v4:

    nb = 240 * 8 / 1024 = 1.875

At nb < 2 PPO degenerates to ~one parameter step per update. The observable
signature is `clip_frac == 0` and `kl ~ 1e-5` on every update for the whole run
-- not "worse results", but *no training at all*. Several archived batches
(frozen, retrain8, v4) are BC starting points for this reason, and the
architecture comparison everyone wanted was never actually run.

A second leak the same layering bug produced: `actor_lr` was 3e-4 for v2 and
1e-5 for v3/v4 (30x apart), because v3/v4 each patched it in their own config
while v2 inherited the profile default. Models trained at different learning
rates cannot be compared either.

So this test pins three things:
  1. all models resolve to the SAME budget keys
  2. nb >= the floor declared in config (training_budget.min_nb)
  3. model configs do not set training hyperparameters (structure only)
"""

from __future__ import annotations

import math

import yaml

from scripts.experiment import resolve_experiment

MODELS = ["v2", "v3", "v4"]

#: Keys that MUST be identical across models. A model may differ in structure
#: (that is the point of having models); it may not differ in how hard it is
#: trained, or the readings are not comparable.
BUDGET_KEYS = [
    ("train", "rollout_steps"),
    ("train", "episodes_per_update"),
    ("train", "ppo", "minibatch_size"),
    ("train", "ppo", "epochs"),
    ("train", "ppo", "clip_eps"),
    ("train", "ppo", "entropy_coef"),
    ("train", "ppo", "value_coef"),
    ("train", "ppo", "max_grad_norm"),
    ("train", "optimizer", "actor_lr"),
    ("train", "optimizer", "critic_lr"),
    ("train", "gamma"),
    ("train", "gae_lambda"),
]

#: Training hyperparameters a model config must NOT contain (see
#: docs/实验产物规范.md §3.6: model configs carry structure only).
FORBIDDEN_IN_MODEL_CONFIG = {
    "rollout_steps", "episodes_per_update", "minibatch_size", "epochs",
    "clip_eps", "entropy_coef", "value_coef", "max_grad_norm",
    "actor_lr", "critic_lr", "gamma", "gae_lambda", "batch_chunk",
    "n_rollout_workers", "target_kl", "normalize_advantages",
}


def dig(cfg: dict, path: tuple[str, ...]):
    cur = cfg
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def resolved(model: str) -> dict:
    return resolve_experiment(model, 42, 10, [], f"budget_contract_{model}")


def test_all_models_share_one_training_budget():
    values = {}
    for model in MODELS:
        cfg = resolved(model)
        values[model] = {path: dig(cfg, path) for path in BUDGET_KEYS}

    ref_model = MODELS[0]
    ref = values[ref_model]
    diffs = []
    for model in MODELS[1:]:
        for path, ref_val in ref.items():
            got = values[model][path]
            if got != ref_val:
                diffs.append(f"{model}.{'.'.join(path)}: {got!r} != {ref_model}'s {ref_val!r}")

    assert not diffs, (
        "模型间的训练超参不一致 —— 这样的模型读数不可比：\n  " + "\n  ".join(diffs)
    )


def test_budget_yields_enough_gradient_steps():
    for model in MODELS:
        cfg = resolved(model)
        rs = dig(cfg, ("train", "rollout_steps"))
        ep = dig(cfg, ("train", "episodes_per_update"))
        mb = dig(cfg, ("train", "ppo", "minibatch_size"))
        floor = dig(cfg, ("training_budget", "min_nb")) or 8
        # The trainer chunks the buffer with `range(0, N, minibatch_size)`
        # (mappo_trainer.update), so a partial trailing batch still costs one
        # gradient step: steps = ceil(N / minibatch_size), NOT N / minibatch_size.
        n_samples = (rs or 0) * (ep or 0)
        nb = math.ceil(n_samples / mb) if (n_samples and mb) else 0
        assert nb >= floor, (
            f"{model}: nb = ceil({rs} * {ep} / {mb}) = {nb} < {floor} —— "
            "PPO 会退化成每轮一次参数更新（实测 clip_frac=0、kl~1e-5），"
            "即「没训练」。调小 minibatch_size 或调大 episodes_per_update。"
        )


def test_entropy_coef_stays_at_the_measured_best_value():
    """entropy_coef must be 0.01, the project's largest measured effect.

    0.001 -> 0.01 measured +0.0954 (t=+9.96), the biggest single-knob effect in
    the project (docs/训练诊断记录.md:2535); the n=15 arm `ent01_rerun` (0.01)
    is the current baseline at 0.6989 vs expert 0.6979. Several historical runs
    actually used 0.001 -- not by design, but because rl_algorithm.yaml
    overrode the tuned preset (record 2576-2593). At 0.001 entropy collapses and
    validation success is ~0.653, BELOW the 0.7674 expert, which is the origin
    of the "RL loses to the expert" claim. Pinned here because a silent
    regression to 0.001 costs the project its single biggest known gain.
    """
    for model in MODELS:
        cfg = resolved(model)
        got = dig(cfg, ("train", "ppo", "entropy_coef"))
        assert got == 0.01, (
            f"{model}: entropy_coef = {got!r}, expected 0.01. 0.001 is the known "
            "bad value (entropy collapse -> validation ~0.653, below the expert). "
            "见 docs/实验产物规范.md §3.6。"
        )


def test_model_configs_carry_structure_only():
    """A model config must not set training hyperparameters."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for model in MODELS:
        cfg_path = root / "qkd_rl" / "model_zoo" / model / "config.yaml"
        if not cfg_path.is_file():
            continue
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        train = cfg.get("train") or {}

        def walk(node, prefix=""):
            if not isinstance(node, dict):
                return
            for key, val in node.items():
                if key in FORBIDDEN_IN_MODEL_CONFIG:
                    offenders.append(f"{model}/config.yaml: train.{prefix}{key}")
                walk(val, f"{prefix}{key}.")

        walk(train)

    assert not offenders, (
        "模型 config 里不应写训练超参（只写模型结构，见 实验产物规范.md §3.6）：\n  "
        + "\n  ".join(offenders)
    )
