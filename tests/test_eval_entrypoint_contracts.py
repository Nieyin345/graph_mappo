import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_scenario_eval_is_deterministic():
    path = ROOT / "scripts" / "eval" / "eval_fixed_scenario.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    policy_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "policy"
        and node.func.attr == "act"
    ]
    assert policy_calls, "fixed-scenario evaluator must call policy.act"
    for call in policy_calls:
        deterministic = next((kw.value for kw in call.keywords if kw.arg == "deterministic"), None)
        assert isinstance(deterministic, ast.Constant) and deterministic.value is True


def test_long_horizon_eval_auto_selects_device_when_unspecified():
    path = ROOT / "scripts" / "baselines" / "eval_long_horizon.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    load_settings = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_settings"
    )
    settings_assign = next(
        node for node in load_settings.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "settings" for t in node.targets)
    )
    assert isinstance(settings_assign.value, ast.Dict)
    defaults = {
        key.value: value
        for key, value in zip(settings_assign.value.keys, settings_assign.value.values)
        if isinstance(key, ast.Constant)
    }
    assert isinstance(defaults["device"], ast.Constant)
    assert defaults["device"].value is None
    assert 'str(ev.get("device"' not in source
    assert "torch.cuda.is_available()" in source


def test_run_baselines_env_builders_do_not_reset_twice():
    path = ROOT / "scripts" / "baselines" / "run_baselines.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    builders = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in {"env_builder", "_rl_env_builder"}
    ]
    assert {node.name for node in builders} == {"env_builder", "_rl_env_builder"}
    for builder in builders:
        resets = [
            node for node in ast.walk(builder)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "reset"
        ]
        assert not resets, f"{builder.name} must leave reset/seed ownership to Evaluator"


def test_long_horizon_incremental_rows_identify_checkpoint():
    path = ROOT / "scripts" / "baselines" / "eval_long_horizon.py"
    source = path.read_text(encoding="utf-8")
    assert '"checkpoint": str(ckpt)' in source
    assert '"update": data.update' in source
