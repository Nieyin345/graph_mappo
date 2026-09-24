def test_trainer_reexports_checkpoint_compat_helpers() -> None:
    from qkd_rl.rl.algos import checkpoint_compat
    from qkd_rl.rl.algos import mappo_trainer

    names = (
        "_reset_module",
        "_upgrade_state_dict_for_model",
        "_upgrade_optimizer_state_for_model",
        "build_param_groups",
    )
    for name in names:
        assert getattr(mappo_trainer, name) is getattr(checkpoint_compat, name)
