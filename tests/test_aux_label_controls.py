"""Control labels for the usage-forecast head (journal experiment (a)).

Our claim is that the *QKD-specific* label -- "edge e will newly lie on a
pending request's path within K slots" ("new") -- is what makes the auxiliary
head useful. Two controls share the same head, loss and bias pipeline and only
swap the label target:

  persist -- a generic, demand-agnostic channel/availability forecast:
             1 iff edge e is still in the observation graph (visible and
             rate >= min_link_rate) at EVERY step of (t, t+K]. Needs no
             request information at all -- the kind of generic dynamic-graph
             auxiliary task a reviewer would propose.
  shuffle -- the "new" label permuted within each step: identical per-step
             positive count (same class balance, same extra-gradient budget)
             but no information about WHICH edges. Rules out "any extra
             supervised signal regularises the encoder".

Pinned here:
  1. persist is an ALL-steps condition over the window, matched by edge id,
     and ignores request_path_edge_mask entirely (policy- and demand-free);
     an empty window at the episode end is vacuously 1.
  2. shuffle keeps each step's positive count, is deterministic, actually
     moves the positives, and never touches the global torch RNG (training
     reproducibility / identical rollouts vs. the "new" arm).
  3. an unknown mode is an error (it used to fall through silently to a
     wrong window).
"""

from __future__ import annotations

import pytest
import torch

from qkd_rl.rl.algos.rollout_buffer import RolloutBuffer, RolloutStep


class _Obs:
    def __init__(self, physical_edge_ids, mask):
        self.physical_edge_ids = physical_edge_ids
        self.request_path_edge_mask = mask


def _step(obs):
    return RolloutStep(
        obs=obs, actions={}, log_probs={}, entropies={}, value=torch.zeros(()),
        reward=0.0, terminated=False, truncated=False,
    )


def _run(obs_list, horizon, mode):
    steps = [_step(o) for o in obs_list]
    buf = RolloutBuffer(0.99, 0.95, torch.device("cpu"))
    buf.usage_horizon = horizon
    buf.usage_label_mode = mode
    buf.steps.extend(steps)
    buf.finish_episode(0.0)
    return [s.usage_label.tolist() for s in steps]


# ---------------------------------------------------------------- persist

def _persist_episode(masks):
    ids = [["a", "b", "c"], ["a", "b"], ["a", "c"], ["a"]]
    return [_Obs(i, m) for i, m in zip(ids, masks)]


def test_persist_requires_presence_at_every_future_step():
    masks = [[False] * 3, [False] * 2, [False] * 2, [False]]
    labels = _run(_persist_episode(masks), horizon=2, mode="persist")
    # t=0 window (1,2]: a in both; b only at 1; c only at 2  -> only a persists
    assert labels[0] == [1.0, 0.0, 0.0]
    # t=1 window (2,3]: a in both; b in neither
    assert labels[1] == [1.0, 0.0]
    # t=2 window (3]: a yes, c no
    assert labels[2] == [1.0, 0.0]
    # t=3: empty window at episode end -> vacuously persists
    assert labels[3] == [1.0]


def test_persist_ignores_request_paths():
    quiet = [[False] * 3, [False] * 2, [False] * 2, [False]]
    busy = [[True, False, True], [True, True], [False, True], [True]]
    assert _run(_persist_episode(quiet), 2, "persist") == _run(_persist_episode(busy), 2, "persist")


def test_persist_matches_by_id_not_position():
    obs = [_Obs(["a", "b"], [False, False]), _Obs(["b"], [False])]
    assert _run(obs, horizon=1, mode="persist")[0] == [0.0, 1.0]


# ---------------------------------------------------------------- shuffle

def _shuffle_episode(n_steps=6, n_edges=40):
    ids = [f"e{i}" for i in range(n_edges)]
    obs = []
    for t in range(n_steps):
        # a block of edges switches on at t+1 -> "new" positives cluster at
        # the low indices of step t, which a permutation must scatter
        on = [i < 3 * (t + 1) for i in range(n_edges)]
        obs.append(_Obs(ids, on))
    return obs


def test_shuffle_keeps_positive_count_per_step():
    new = _run(_shuffle_episode(), horizon=2, mode="new")
    shuf = _run(_shuffle_episode(), horizon=2, mode="shuffle")
    assert [sum(x) for x in shuf] == [sum(x) for x in new]
    assert sum(sum(x) for x in new) > 0


def test_shuffle_moves_positives_and_is_deterministic():
    new = _run(_shuffle_episode(), horizon=2, mode="new")
    a = _run(_shuffle_episode(), horizon=2, mode="shuffle")
    b = _run(_shuffle_episode(), horizon=2, mode="shuffle")
    assert a == b
    moved = [x != y for x, y in zip(a, new) if sum(y) > 0]
    assert moved and all(moved)


def test_shuffle_does_not_touch_global_rng():
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    _run(_shuffle_episode(), horizon=2, mode="shuffle")
    assert torch.equal(torch.random.get_rng_state(), before)


# ---------------------------------------------------------------- validation

def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        _run([_Obs(["a"], [True]), _Obs(["a"], [True])], horizon=1, mode="nwe")
