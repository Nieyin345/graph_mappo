"""The shared short-run preset must collect the advertised rollout budget."""

from scripts.experiment import resolve_experiment


def test_default_short_run_collects_eight_episodes_per_update():
    config = resolve_experiment("v3", 42, 10, [], "budget_contract")

    assert config["train"]["episodes_per_update"] == 8
    assert config["train"]["rollout_steps"] == 240
    assert config["train"]["n_rollout_workers"] == 8
    assert config["train"]["ppo"]["minibatch_size"] == 1024
