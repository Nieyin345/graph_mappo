"""v8: v3 plus a usage-forecast auxiliary head (see model_impl.UsageForecastHead).

v3 is the base; the only additions are the forecast head, its BCE training
signal, and an optional score bias. Nothing in the v3 path is changed, so a v8
run with ``usage_forecast.enabled: false`` is bit-identical to v3.
"""

from .graph_builder import GraphBuilder
from .model_impl import GraphMAPPOActorCritic


def build_model(action_space, config):
    return GraphMAPPOActorCritic(action_space, config)


def graph_builder_class():
    return GraphBuilder
