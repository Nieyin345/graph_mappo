"""Standalone v4 snapshot; the common BC/PPO/evaluation pipeline imports this plugin."""

from .graph_builder import GraphBuilder
from .model_impl import GraphMAPPOActorCritic


def build_model(action_space, config):
    return GraphMAPPOActorCritic(action_space, config)


def graph_builder_class():
    return GraphBuilder
