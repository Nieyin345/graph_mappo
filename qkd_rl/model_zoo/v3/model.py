"""v3 model and observation builder owned by this model directory."""

from .graph_builder import GraphBuilder
from .model_impl import GraphMAPPOActorCritic


def build_model(action_space, config):
    return GraphMAPPOActorCritic(action_space, config)


def graph_builder_class():
    return GraphBuilder
