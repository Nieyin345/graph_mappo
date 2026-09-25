"""The shared short-run preset must collect the advertised rollout budget.

`minibatch_size` is 256, not 1024. At 1024 the collected 1920 samples
(240 steps x 8 episodes) chunked into 2 gradient steps per update
(ceil(1920/1024)) -- PPO barely moved: measured clip_frac == 0 and kl ~ 1e-5
for whole runs, so the artifact was just the BC starting point. At 256 the same
buffer yields 8 steps. The value lives in configs/experiment_base.yaml
(see 实验产物规范.md §3.6); this test pins that the preset actually gets it.
"""

import math

from scripts.experiment import resolve_experiment


def test_default_short_run_collects_eight_episodes_per_update():
    config = resolve_experiment("v3", 42, 10, [], "budget_contract")

    train = config["train"]
    assert train["episodes_per_update"] == 8
    assert train["rollout_steps"] == 240
    assert train["n_rollout_workers"] == 8

    mb = train["ppo"]["minibatch_size"]
    nb = math.ceil(train["rollout_steps"] * train["episodes_per_update"] / mb)
    assert nb >= config["training_budget"]["min_nb"], (
        f"minibatch_size={mb} gives {nb} gradient steps per update, below the "
        f"floor {config['training_budget']['min_nb']} -- this is the "
        "never-trained regime (clip_frac==0, kl~1e-5)."
    )
