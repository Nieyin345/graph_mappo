from __future__ import annotations

"""Capacity providers turn physical context into a per-time-slot key-rate profile.

Every provider returns ``{time_slot: key_rate_bps}``. This is the single seam between
the (heavy, optional) physics/reference engines and the framework: the scenario builders
only ever ask a provider for a capacity profile, never touch NetSquid/SatQuMA directly.

Unit convention (see README ยง4): providers emit **bps**; MILP/QKP convert to bits per
slot via ``key_rate_bps * slot_duration_seconds``.
"""

from typing import Dict, List, Protocol


class CapacityProvider(Protocol):
    """Anything that can produce a key-rate profile for a link."""

    def capacity_by_time(self, link_key: str, time_slots: List[int]) -> Dict[int, float]:
        ...


def resample_trace(trace: List[float], time_slots: List[int]) -> Dict[int, float]:
    """Map an arbitrary-length trace onto ``time_slots`` by proportional index scaling.

    Lets a 48-slot reference trace populate a scenario with any number of slots.
    """

    if not trace:
        return {int(t): 0.0 for t in time_slots}
    n = len(trace)
    out: Dict[int, float] = {}
    m = len(time_slots)
    for i, t in enumerate(time_slots):
        # even coverage of the source trace across the target slots
        src_idx = int(round(i * (n - 1) / max(m - 1, 1))) if m > 1 else 0
        out[int(t)] = float(trace[min(src_idx, n - 1)])
    return out
