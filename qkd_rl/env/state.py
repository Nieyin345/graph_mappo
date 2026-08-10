from __future__ import annotations

from dataclasses import dataclass, field

from qkd_rl.core.types import KeyRequest
from qkd_rl.link.rate_provider import EdgeWindow


@dataclass
class EnvState:
    t: int
    qkp_snapshot: dict[str, float]
    pending_requests: list[KeyRequest]
    edge_windows: dict[str, EdgeWindow]
    last_activated_edges: list[str]
    # Edges activated in the previous step; used to compute the switch flag
    # (activated now but not before) for the switch-cost feature.
    prev_activated_edges: list[str] = field(default_factory=list)
    # Per-link QKP capacity cap (edge_id -> bits). None when the observation
    # does not expose it (e.g. hand-built test fixtures); optimal baselines
    # use it to bound how much a link can serve this slot.
    qkp_capacity: dict[str, float] | None = None

