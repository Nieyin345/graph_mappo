"""Guard version selection across cross-model baseline and long-horizon evaluators."""
import ast
from pathlib import Path
import pytest
from qkd_rl.evaluation.test_protocol import checkpoint_validation_config

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("path",[
    "scripts/baselines/run_baselines.py",
    "scripts/baselines/eval_long_horizon.py",
])
def test_checkpoint_evaluator_constructs_versioned_model_via_registry(path):
    tree = ast.parse((ROOT/path).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and
             isinstance(n.func,ast.Name) and n.func.id=="build_model"]
    assert calls, f"{path} must construct the checkpoint model via model_zoo.build_model"
    old = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and
           isinstance(n.func,ast.Name) and n.func.id=="GraphMAPPOActorCritic"]
    assert not old, f"{path} must not force the v2 model"

def test_checkpoint_validation_preserves_version_only_not_training_identity():
    checkpoint={"experiment":{"model":"v8","cpu_threads":4,"run_name":"training_run"},
                "features":{"edge":{"include_stocked_edges":True}},
                "model":{"actor":{"attention_only":True}},
                "env":{"episode_steps":999},
                "requests":{"deadline_steps":999}}
    canonical={"experiment":{"model":"v2","cpu_threads":16,"run_name":"test"},
               "features":{"edge":{"include_stocked_edges":False}},
               "model":{"actor":{"attention_only":False}},
               "env":{"episode_steps":240,"continuous":True},
               "requests":{"deadline_steps":30}}
    result=checkpoint_validation_config(checkpoint,canonical)
    assert result["experiment"]["model"]=="v8"
    assert result["experiment"]["cpu_threads"]==16
    assert result["experiment"]["run_name"]=="test"
    assert result["features"]["edge"]["include_stocked_edges"] is True
    assert result["model"]["actor"]["attention_only"] is True
    assert result["env"]["episode_steps"]==240
    assert result["env"]["continuous"] is False
    assert result["requests"]["deadline_steps"]==30
    assert canonical["experiment"]["model"]=="v2"

def test_legacy_checkpoint_without_identity_keeps_canonical_fallback():
    canonical={"experiment":{"model":"v2"},"env":{"continuous":True}}
    result=checkpoint_validation_config({"model":{"mode":"old"}},canonical)
    assert result["experiment"]["model"]=="v2"

def test_paper_sensitivity_w10_is_weight_one_not_one_tenth():
    script=(ROOT/"scripts/batch/run_paper_experiments.sh").read_text(encoding="utf-8")
    assert 'for w in 02 06 08 10; do' in script
    assert 'if [ "$w" = 10 ]; then printf \'1.0\'' in script
    assert 'bias_weight: %s\\n' in script

def test_long_horizon_imports_validation_profile_resolver():
    tree=ast.parse((ROOT/"scripts/baselines/eval_long_horizon.py").read_text(encoding="utf-8"))
    matches=[n for n in tree.body if isinstance(n,ast.ImportFrom)
             and n.module=="qkd_rl.evaluation.test_protocol"]
    assert len(matches)==1
    assert "load_validation_profile" in {a.name for a in matches[0].names}
