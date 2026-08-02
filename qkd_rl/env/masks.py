from __future__ import annotations

from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.qkp import LinkQKPPool
from qkd_rl.env.request import RequestQueue
from qkd_rl.env.state import EnvState


class ActionMaskBuilder:
    def __init__(self, action_space: NodeActionSpace, config: dict):
        self.action_space = action_space
        self.config = config

    def build(self, env_state: EnvState, qkp: LinkQKPPool, requests: RequestQueue) -> dict[str, list[bool]]:
        return {
            node_id: [
                self.is_action_legal(node_id, action, env_state, qkp, requests)
                for action in self.action_space.candidates_for_node(node_id)
            ]
            for node_id in self.action_space.node_ids
        }

    def is_action_legal(
        self,
        node_id: str,
        action: str,
        env_state: EnvState,
        qkp: LinkQKPPool,
        requests: RequestQueue,
    ) -> bool:
        if action == NodeActionSpace.IDLE:
            return True
        edge_id = self.action_space.action_to_edge(node_id, action)
        if edge_id is None:
            return False
        window = env_state.edge_windows[edge_id]
        if self.config.get("mask_unavailable_edges", True) and not window.available[0]:
            return False
        if self.config.get("mask_full_qkp_edges", False) and qkp.get_capacity_left(edge_id) <= 0:
            return False
        return True

