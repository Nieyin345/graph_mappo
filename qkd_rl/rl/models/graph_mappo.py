"""Compatibility re-export of the v2 model.

The single source of truth for the v2 implementation is
``qkd_rl/model_zoo/v2/model_impl.py``; this module exists only so that
older entry points and diagnostic scripts keep importing under the
historical ``qkd_rl.rl.models.graph_mappo`` path. New code should import
from ``qkd_rl.model_zoo.v2`` directly.
"""

from qkd_rl.model_zoo.v2.model_impl import (  # noqa: F401
    ActorCriticOutput,
    BatchedActorCriticOutput,
    DemandResidual,
    EdgeConditionedGraphLayer,
    GlobalCritic,
    GraphEncoder,
    GraphMAPPOActorCritic,
    GraphTensors,
    SharedNodeActor,
    observation_to_tensors,
)
