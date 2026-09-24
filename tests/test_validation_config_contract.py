from qkd_rl.evaluation.test_protocol import checkpoint_validation_config


def _validation():
    return {
        "features": {"edge": {"include_stocked_edges": False, "safe_default": 17}},
        "model": {"mode": "canonical", "new_default": True},
        "action_resolver": {"mode": "canonical", "tie_break": "stable"},
        "env": {"episode_steps": 720, "continuous": True},
        "qkp": {"capacity": 100.0},
        "requests": {"deadline_steps": 30},
        "routing": {"mode": "canonical"},
        "reward": {"served_weight": 1.0},
        "rate_provider": {"provider": "h5", "h5": {"dataset_dir": "canonical"}},
        "scenario": {"time_limit": {"days": 31}},
    }


def test_checkpoint_validation_config_preserves_only_policy_contract():
    validation = _validation()
    checkpoint = {
        "features": {"edge": {"include_stocked_edges": True}},
        "model": {"mode": "checkpoint"},
        "action_resolver": {"mode": "checkpoint"},
        "env": {"episode_steps": 9999, "continuous": True},
        "qkp": {"capacity": 7.0},
        "requests": {"deadline_steps": 999},
        "routing": {"mode": "checkpoint"},
        "reward": {"served_weight": 99.0},
        "rate_provider": {"provider": "wrong"},
        "scenario": {"time_limit": {"days": 999}},
    }
    merged = checkpoint_validation_config(checkpoint, validation)
    assert merged["features"]["edge"]["include_stocked_edges"] is True
    assert merged["features"]["edge"]["safe_default"] == 17
    assert merged["model"]["mode"] == "checkpoint"
    assert merged["model"]["new_default"] is True
    assert merged["action_resolver"]["mode"] == "checkpoint"
    assert merged["action_resolver"]["tie_break"] == "stable"
    for section in ("qkp", "requests", "routing", "reward", "rate_provider", "scenario"):
        assert merged[section] == validation[section]
    assert merged["env"]["episode_steps"] == 720
    assert merged["env"]["continuous"] is False
    merged["features"]["edge"]["include_stocked_edges"] = False
    assert checkpoint["features"]["edge"]["include_stocked_edges"] is True


def test_checkpoint_validation_config_without_saved_config_uses_copy():
    validation = _validation()
    merged = checkpoint_validation_config(None, validation)
    assert merged == {**validation, "env": {"episode_steps": 720, "continuous": False}}
    merged["qkp"]["capacity"] = 1.0
    assert validation["qkp"]["capacity"] == 100.0
