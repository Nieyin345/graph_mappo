from __future__ import annotations

from dataclasses import dataclass

from qkd_rl.core.types import KeyRequest
from qkd_rl.link.rate_provider import EdgeWindow


@dataclass
class EnvState:
    t: int
    qkp_snapshot: dict[str, float]
    pending_requests: list[KeyRequest]
    edge_windows: dict[str, EdgeWindow]
    last_activated_edges: list[str]

