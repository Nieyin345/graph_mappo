"""v8 usage-forecast: the LABEL must be forward-looking and policy-independent.

The whole reason v8 exists is that v7's preparation stage had no feedback. If
the label is computed wrongly the head trains on noise -- and it fails
silently, exactly like v7 did (entropy up, gradient down, no error). So the
label's defining properties are pinned here:

  1. FORWARD-looking: an edge that is on a request path only in a FUTURE step
     must be labelled 1 NOW (within the horizon). A current-step-only mask
     (which is what the env provides) would fail this.
  2. POLICY-independent: the label must not depend on which arcs were matched.
     It comes from the request stream's path set only.
  3. Horizon-bounded: an edge that only becomes relevant BEYOND the horizon
     must stay 0.
  4. Off by default: a buffer with usage_horizon == 0 must leave every
     usage_label as None, so v3/v2/v4/v6 pay nothing and are unchanged.
"""

from __future__ import annotations

import torch

from qkd_rl.rl.algos.rollout_buffer import RolloutBuffer, RolloutStep


class _Obs:
    """Minimal observation stand-in: the label code reads only these two."""

    def __init__(self, physical_edge_ids, mask):
        self.physical_edge_ids = physical_edge_ids
        self.request_path_edge_mask = mask


def _step(obs):
    return RolloutStep(
        obs=obs,
        actions={},
        log_probs={},
        entropies={},
        value=torch.zeros(()),
        reward=0.0,
        terminated=False,
        truncated=False,
    )


def _buffer(horizon):
    buf = RolloutBuffer(0.99, 0.95, torch.device("cpu"))
    buf.usage_horizon = horizon
    return buf


def test_label_is_forward_looking():
    """A request path appearing only at t+2 must mark the edge at t."""
    edges = ["e0", "e1", "e2"]
    steps = [
        _step(_Obs(edges, [False, False, False])),   # t=0   nothing
        _step(_Obs(edges, [False, False, False])),   # t=1   nothing
        _step(_Obs(edges, [True, False, False])),    # t=2   e0 on a path
    ]
    buf = _buffer(horizon=4)
    buf.steps.extend(steps)
    buf.finish_episode(0.0)

    # t=0 sees t=0..3, so e0 is labelled 1 even though its own mask is all-False.
    assert steps[0].usage_label.tolist() == [1.0, 0.0, 0.0]
    assert steps[2].usage_label.tolist() == [1.0, 0.0, 0.0]


def test_label_respects_horizon():
    """An edge that matters only beyond the horizon stays 0."""
    edges = ["e0", "e1"]
    steps = [
        _step(_Obs(edges, [False, False])),
        _step(_Obs(edges, [False, False])),
        _step(_Obs(edges, [False, False])),
        _step(_Obs(edges, [False, True])),   # e1 relevant at t=3
    ]
    buf = _buffer(horizon=2)   # t=0 window is t=0..1 -> e1 NOT visible
    buf.steps.extend(steps)
    buf.finish_episode(0.0)

    assert steps[0].usage_label.tolist() == [0.0, 0.0]
    # t=3's own window covers t=3..4 -> sees itself
    assert steps[3].usage_label.tolist() == [0.0, 1.0]


def test_label_uses_ids_not_positions():
    """Edge lists may be reordered between steps; matching is by edge id."""
    steps = [
        _step(_Obs(["a", "b"], [False, False])),
        _step(_Obs(["b", "a"], [True, False])),   # 'b' on a path, different order
    ]
    buf = _buffer(horizon=2)
    buf.steps.extend(steps)
    buf.finish_episode(0.0)
    # Step 0's list is [a, b]; only 'b' will be used -> [0, 1].
    assert steps[0].usage_label.tolist() == [0.0, 1.0]


def test_disabled_when_horizon_zero():
    """Default (v3/v2/v4/v6) must not compute labels at all."""
    steps = [_step(_Obs(["e0"], [True]))]
    buf = _buffer(horizon=0)
    buf.steps.extend(steps)
    buf.finish_episode(0.0)
    assert steps[0].usage_label is None


def test_new_mode_excludes_currently_used():
    """'new' must drop edges that are already in use -- that is the whole point.

    The batch-1 label ("used") was dominated by persistence: 100% of edges in
    use now stay in use across the horizon, so the head just relearned "keep
    serving what you already serve" and scored worse than a copy of the current
    mask. 'new' subtracts exactly that term.
    """
    edges = ["e0", "e1", "e2"]
    steps = [
        _step(_Obs(edges, [True, False, False])),    # t=0: e0 in use now
        _step(_Obs(edges, [True, True, False])),     # t=1: e0 persists, e1 new
        _step(_Obs(edges, [False, True, False])),    # t=2: e1 persists
    ]
    buf = _buffer(horizon=4)
    buf.usage_label_mode = "new"
    buf.steps.extend(steps)
    buf.finish_episode(0.0)

    # t=0: e0 is in use now -> excluded; e1 becomes used in the window -> 1.
    assert steps[0].usage_label.tolist() == [0.0, 1.0, 0.0]
    # t=1: e0 in use now (excluded), e1 in use now (excluded), e2 never -> all 0.
    assert steps[1].usage_label.tolist() == [0.0, 0.0, 0.0]


def test_new_mode_differs_from_used_mode():
    """The two modes must NOT agree on the persistence case (else 'new' is a
    no-op rename and the batch-2 arm would repeat batch 1)."""
    edges = ["e0"]
    steps = [_step(_Obs(edges, [True])), _step(_Obs(edges, [True]))]
    used = _buffer(horizon=2); used.steps.extend(steps); used.finish_episode(0.0)
    assert steps[0].usage_label.tolist() == [1.0]     # 'used': in use now counts

    steps2 = [_step(_Obs(edges, [True])), _step(_Obs(edges, [True]))]
    new = _buffer(horizon=2); new.usage_label_mode = "new"
    new.steps.extend(steps2); new.finish_episode(0.0)
    assert steps2[0].usage_label.tolist() == [0.0]    # 'new': already in use, so not new
