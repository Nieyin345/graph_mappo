"""BC must rebuild observations with the class that MATCHES THE MODEL.

The failure this prevents (measured 2026-09-26):

`supervised_train_pg_phased._obs_class()` hardcoded

    if model_name == "v3": return v3.graph_builder.GraphObservation
    else:                  return env.graph_builder.GraphObservation

but EVERY model_zoo/<m>/graph_builder.py defines its own GraphObservation, and
only the v3/v4/v6 ones carry pair_path_edge_masks / pair_path_weights. So on the
`--data-dir` path (rebuild_obs), v4 and v6 fell through to the base class, which
has neither field:

  * v4/v6 silently received empty masks -> the pair_path scorer took its
    zero-context branch (the same branch BC previously measured as producing NO
    gradient into the attention on 2026-09-24), and
  * had rebuild_obs passed the fields anyway, the base dataclass would have
    raised TypeError instead.

The env's own online path never hit this (it builds the env via
model_zoo.graph_builder_class, which already dispatches on experiment.model), so
a run launched without --bc-data looked fine while a trajectory-replay run did
not. Two paths, two answers -- this test pins them equal.

Source of truth (must agree): model_zoo.graph_builder_class(config)(...) emits
observations; _obs_class(model) rebuilds them.
"""

from __future__ import annotations

import dataclasses
import importlib

import pytest

from qkd_rl.model_zoo import build_model, graph_builder_class, model_name
from scripts.experiment import resolve_experiment
from scripts.train.supervised_train_pg_phased import _obs_class, rebuild_obs

MODELS = ["v2", "v3", "v4", "v6", "v7", "v8"]


def _resolved(model: str) -> dict:
    return resolve_experiment(model, 42, 1, [], f"obs_class_{model}")


@pytest.mark.parametrize("model", MODELS)
def test_obs_class_matches_the_model_s_own_builder(model):
    """_obs_class(model) must be the class model_zoo's builder actually emits.

    Not "a class with the right fields" -- the SAME class, so the two BC paths
    (online env.reset vs rebuild_obs replay) produce indistinguishable input.
    """
    cfg = _resolved(model)
    builder_module = importlib.import_module(
        f"qkd_rl.model_zoo.{model_name(cfg)}.graph_builder"
    )
    emitted = builder_module.GraphObservation
    assert _obs_class(model) is emitted, (
        f"{model}: _obs_class returns {_obs_class(model).__qualname__} but the "
        f"env emits {emitted.__qualname__}. The BC replay path (rebuild_obs) "
        "would feed the model a different observation type than training does."
    )


@pytest.mark.parametrize("model", MODELS)
def test_rebuild_obs_round_trips_for_every_model(model):
    """rebuild_obs must construct the model's class without dropping/adding fields."""
    obs_cls = _obs_class(model)
    fields = getattr(obs_cls, "__dataclass_fields__", {})
    # Minimal record carrying exactly the fields rebuild_obs reads.
    n, e = 2, 3
    rec = {
        "node_features": [[0.0] for _ in range(n)],
        "edge_index": [(0, 1), (1, 0), (0, 1)],
        "edge_features": [[0.0] for _ in range(e)],
        "node_ids": [f"n{i}" for i in range(n)],
        "edge_ids": [f"e{i}" for i in range(e)],
        "physical_edge_ids": [f"e{i}" for i in range(e)],
        "demand_edge_ids": [],
        "action_candidates": {},
        "action_masks": {},
    }
    obs = rebuild_obs(rec, obs_cls)
    assert isinstance(obs, obs_cls)

    # Models whose class declares the pair_path fields must receive them (not
    # silently absent); models without the fields must not be handed them.
    if "pair_path_edge_masks" in fields:
        assert hasattr(obs, "pair_path_edge_masks")
        assert hasattr(obs, "pair_path_weights")
    else:
        assert "pair_path_edge_masks" not in fields


def test_pair_path_models_are_exactly_the_ones_that_need_masks():
    """Guard the set: a model that enables pair_path_attention needs the fields.

    If a future model turns on pair_path_attention but its builder does not
    declare pair_path_edge_masks, BC replay would hand the scorer nothing while
    the config promises attention -- the exact silent-zero-context bug.
    """
    for model in MODELS:
        cfg = _resolved(model)
        enabled = bool(
            (cfg.get("model", {}).get("actor", {}).get("pair_path_attention", {}) or {})
            .get("enabled", False)
        )
        obs_cls = _obs_class(model)
        has_fields = "pair_path_edge_masks" in getattr(
            obs_cls, "__dataclass_fields__", {}
        )
        assert enabled == has_fields, (
            f"{model}: pair_path_attention.enabled={enabled} but its observation "
            f"class ({obs_cls.__qualname__}) declares pair_path fields={has_fields}"
        )
