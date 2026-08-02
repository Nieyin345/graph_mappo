from __future__ import annotations

"""Explicit link failure / degradation events, injected on top of capacity-driven availability.

Capacity-driven availability (``is_available`` = capacity>0) already models a satellite
being out of its visibility window. These functions add failures that are *independent* of
capacity, via the ``availability_by_time`` hook (hard outage) or by scaling
``capacity_by_time`` (soft degradation). They model the StratoQ dynamic scenario and the
motivation ("satellite link unavailable OR degraded -> HAP backs up"):

  - ``inject_sat_gs_outage``       : satellite-ground link fails (pointing/tracking/battery).
  - ``inject_hap_gs_misalignment`` : HAP-GS link dropped with a small probability (StratoQ 5%).
  - ``inject_degradation``         : link capacity scaled down (degraded but not dead).
  - ``disable_relay_layer``        : remove a whole relay layer (e.g. HAP) for a counterfactual.

All are deterministic given ``seed`` and mutate the scenario in place — deep-copy the base
scenario first if you want to keep it clean.

**Switch / cancel:** every injector takes ``enabled=True``; pass ``enabled=False`` to make it
a no-op (a one-flag switch you can leave in place without deleting call sites). To undo
events already applied to a scenario, use ``clear_link_events`` (cancels hard outages /
misalignments) or ``snapshot_link_state`` + ``restore_link_state`` (full revert, incl.
capacity degradation).
"""

import random
from typing import List, Optional

from entities.links.link_type import LinkType
from entities.nodes.node_type import NodeType
from scenario.network_scenario import Scenario


def _set_unavailable(link, slots: List[int]) -> None:
    for t in slots:
        link.availability_by_time[t] = False


def inject_sat_gs_outage(
    scenario: Scenario, slots: Optional[List[int]] = None, prob: float = 1.0,
    seed: int = 0, enabled: bool = True,
) -> int:
    """Fail SAT-GS links (independent of visibility). Returns #(link,slot) knocked out.

    ``enabled=False`` -> no-op (the cancel switch)."""

    if not enabled:
        return 0
    rng = random.Random(seed)
    target_slots = slots if slots is not None else list(scenario.time_slots)
    n = 0
    for link in scenario.links.values():
        if link.link_type != LinkType.SAT_GS:
            continue
        for t in target_slots:
            if rng.random() < prob:
                link.availability_by_time[t] = False
                n += 1
    return n


def inject_hap_gs_misalignment(
    scenario: Scenario, prob: float = 0.05, seed: int = 0, enabled: bool = True
) -> int:
    """Each HAP-GS link, each slot, drops with probability ``prob`` (StratoQ dynamic: 5%).

    ``enabled=False`` -> no-op (the cancel switch)."""

    if not enabled:
        return 0
    rng = random.Random(seed)
    n = 0
    for link in scenario.links.values():
        if link.link_type != LinkType.HAP_GS:
            continue
        for t in scenario.time_slots:
            if rng.random() < prob:
                link.availability_by_time[t] = False
                n += 1
    return n


def inject_degradation(
    scenario: Scenario, link_type: LinkType, prob: float, factor: float,
    seed: int = 0, enabled: bool = True,
) -> int:
    """Scale capacity by ``factor`` (<1) on matched (link, slot) with probability ``prob``.

    ``enabled=False`` -> no-op (the cancel switch)."""

    if not enabled:
        return 0
    rng = random.Random(seed)
    n = 0
    for link in scenario.links.values():
        if link.link_type != link_type:
            continue
        for t in scenario.time_slots:
            if rng.random() < prob and link.capacity_by_time.get(t, 0.0) > 0:
                link.capacity_by_time[t] *= factor
                n += 1
    return n


def clear_link_events(scenario: Scenario) -> None:
    """Cancel injected hard outages / misalignments: reset availability to capacity-driven.

    (Capacity degradation is not restored by this — use snapshot/restore for that.)"""

    for link in scenario.links.values():
        link.availability_by_time = {
            t: link.capacity_by_time.get(t, 0.0) > 0.0 for t in scenario.time_slots
        }


def snapshot_link_state(scenario: Scenario) -> dict:
    """Capture per-link availability + capacity so failures can be fully undone later."""

    return {
        lid: (dict(l.availability_by_time), dict(l.capacity_by_time))
        for lid, l in scenario.links.items()
    }


def restore_link_state(scenario: Scenario, snapshot: dict) -> None:
    """Fully revert to a ``snapshot_link_state`` result (cancels outages AND degradation)."""

    for lid, link in scenario.links.items():
        if lid in snapshot:
            avail, cap = snapshot[lid]
            link.availability_by_time = dict(avail)
            link.capacity_by_time = dict(cap)


def disable_relay_layer(scenario: Scenario, node_type: NodeType) -> None:
    """Make every link touching a node of ``node_type`` unavailable (counterfactual view)."""

    targets = {n.node_id for n in scenario.nodes.values() if n.node_type == node_type}
    for link in scenario.links.values():
        if link.source in targets or link.target in targets:
            for t in scenario.time_slots:
                link.availability_by_time[t] = False
