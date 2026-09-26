"""v6 model plugin: v3 plus exactly one change -- demand edges never enter the GNN.

Built from v3 (not copied from v4) so v4's other bundled changes cannot leak in.
Note v4 *also* rewrote PairPathAttentionScorer (vectorized -> per-pair loop, and
link attention global -> pair-local); that rewrite is deliberately NOT included
here, so this model isolates a single variable. See ./config.yaml.
"""

from .graph_builder import GraphBuilder
from .model_impl import GraphMAPPOActorCritic


def build_model(action_space, config):
    return GraphMAPPOActorCritic(action_space, config)


def graph_builder_class():
    return GraphBuilder
