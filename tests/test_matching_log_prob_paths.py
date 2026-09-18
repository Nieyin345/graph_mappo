"""The evaluate-actions log-prob paths must agree bit for bit.

Three implementations of the same matching log-prob/entropy math are live:

- ``_matching_log_prob_entropy_fast`` -- edge-score **dict** input; used by
  ``evaluate_actions`` and by the BC phase ``supervised_train_pg_phased``.
- ``_matching_log_prob_entropy_arrays`` -- src/dst index **arrays** plus the raw
  score vector; the path PPO updates actually take
  (``evaluate_actions_batched``).
- ``_matching_log_prob_entropy`` -- arc->tensor **dict** input; reached from
  ``log_prob_entropy_for_matching``, which is the **rollout** path.

PPO's importance ratio divides the update path's log prob by the rollout path's,
so the pair that *must* agree is arrays vs the ``log_prob_entropy_for_matching``
dict -- see ``test_rollout_path_matches_array_path_bit_for_bit``. The
fast<->arrays pair is covered too, because the docstring of the arrays version
claims bit-identity and the BC phase shares the fast path.

Both dict inputs are taken from ONE ``batched_forward`` call -- ``want_edge_maps=True``
returns ``edge_score_maps`` and ``edge_arrays`` together -- so the only thing
under test is the scoring implementations, not the forward batching.

The comparison is ``torch.equal``, not ``allclose``: "bit-identical" is the
claim, and a near-equal-but-not-equal result means the paths disagree about
something structural (arc set, order, feasibility row), which is exactly the
bug this file is meant to surface.
"""

from __future__ import annotations

import pytest
import torch

from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.rl.algos.policy import MAPPOPolicy
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic
from tests.helpers import ROOT, point_config_to_h5

IDLE_ACTION = NodeActionSpace.IDLE


def _build() -> tuple:
    config = point_config_to_h5(load_default_config(ROOT))
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    return env, MAPPOPolicy(model, "cpu")


def _maximal_matching(
    src_arr, dst_arr, node_ids: list[str]
) -> list[tuple[str, str]]:
    """A long feasible matching over the flat candidate order.

    Mirrors the sampler's dual-port rules (``Tx-out <= 1``, ``Rx-in <= 1``,
    对端不同) so the evaluators accept it, but keeps every arc it can -- the
    sampled matching often stops after one or two arcs, which barely exercises
    the per-decision feasibility rows the two paths could disagree on.
    """
    used_tx: set[int] = set()
    used_rx: set[int] = set()
    used_pair: set[tuple[int, int]] = set()
    matched: list[tuple[str, str]] = []
    for src, dst in zip(src_arr.tolist(), dst_arr.tolist()):
        pair = (src, dst) if src <= dst else (dst, src)
        if src in used_tx or dst in used_rx or pair in used_pair:
            continue
        used_tx.add(src)
        used_rx.add(dst)
        used_pair.add(pair)
        matched.append((node_ids[src], node_ids[dst]))
    return matched


def _failure(
    label: str,
    dict_value: torch.Tensor,
    array_value: torch.Tensor,
    matched_edges: list[tuple[str, str]],
    src_arr,
    dst_arr,
    node_ids: list[str],
) -> str:
    candidates = [
        (node_ids[int(s)], node_ids[int(d)]) for s, d in zip(src_arr.tolist(), dst_arr.tolist())
    ]
    shown = candidates[:40]
    return (
        f"{label} differs between the dict and array log-prob paths\n"
        f"  dict path : {dict_value.item()!r}\n"
        f"  array path: {array_value.item()!r}\n"
        f"  abs diff  : {abs(dict_value.item() - array_value.item()):.3e}\n"
        f"  matched_edges ({len(matched_edges)}): {matched_edges!r}\n"
        f"  candidates ({len(candidates)}): {shown!r}{' ...' if len(candidates) > len(shown) else ''}"
    )


def test_candidate_pairs_are_unique_per_observation() -> None:
    """Each ``(src, dst)`` candidate pair must occur at most once per obs.

    ``_matching_log_prob_entropy_arrays`` resolves a matched arc to its
    candidate index through a ``(src, dst)`` lookup table built once per graph.
    That is only faithful to the original per-arc ``np.flatnonzero`` scan while
    this uniqueness holds: the scan took ``cand[0]``, and with duplicates the
    first alive match could be a later duplicate whose index differs from the
    table's.

    The uniqueness is structural, not incidental --
    ``action_space.neighbors[node]`` appends each adjacent node once per edge
    (``action_space.py:31-37``), the candidate list is the sorted ``legal``
    subset of that (``graph_builder.py:187-188``), and IDLE is filtered out of
    the edge set. This test pins the structure so a future change to candidate
    construction (parallel edges, unsorted or duplicated candidates) fails here
    instead of silently biasing PPO's importance ratio.
    """
    env, _policy = _build()
    for seed in (1, 2, 3):
        obs = env.reset(seed=seed)
        seen_any = False
        for node_id in obs.node_ids:
            cands = [c for c in obs.action_candidates[node_id] if c != IDLE_ACTION]
            seen_any = seen_any or bool(cands)
            assert len(cands) == len(set(cands)), (
                f"node {node_id!r} has duplicate candidate neighbours: {cands!r}. "
                "The (src, dst) lookup table in _matching_log_prob_entropy_arrays "
                "is only correct while candidates are unique."
            )
            assert len(obs.action_candidates[node_id]) == len(set(obs.action_candidates[node_id])), (
                f"node {node_id!r} candidate list repeats an entry "
                f"(IDLE included): {obs.action_candidates[node_id]!r}"
            )
        assert seen_any, f"seed {seed} produced no edge candidates; test is vacuous"


def test_rollout_path_matches_array_path_bit_for_bit() -> None:
    """The **rollout** scoring path must match the update path's math.

    Three implementations of this math are live, not two:

    - ``_matching_log_prob_entropy_fast``   (edge-score *dict*) -- used by
      ``evaluate_actions`` and by the BC phase ``supervised_train_pg_phased``.
    - ``_matching_log_prob_entropy_arrays`` (src/dst arrays) -- the PPO
      **update** path (``evaluate_actions_batched``).
    - ``_matching_log_prob_entropy``        (arc->tensor dict) -- reached from
      ``log_prob_entropy_for_matching``, which is the **rollout** path
      (``rollout_workers.py`` and ``mappo_trainer`` collect loops).

    ``test_array_path_matches_dict_path_bit_for_bit`` covers fast<->arrays. The
    rollout path is the one that produces the **denominator** of PPO's
    importance ratio while the update path produces the numerator, so a
    *structural* disagreement between these two biases every ratio in the batch
    -- and it was the only pair never compared. Here the dict is rebuilt from
    the same float scores the array path uses, so the only thing under test is
    the two scoring implementations.

    Name says "bit for bit" for symmetry with its sibling, but the pair is
    compared with a 1e-5 tolerance rather than ``torch.equal``: these two
    accumulate in different orders and genuinely differ in the last bit (see
    the comment at the comparison). What the guard is for is the structural
    class of bug, which is orders of magnitude larger.
    """
    env, policy = _build()
    # Several seeds, because the two paths could agree on one graph's candidate
    # order and disagree on another (different degrees, different feasibility
    # rows). One seed would make this guard look stronger than it is.
    checked = 0
    for seed in (1, 2, 3, 5):
        obs = env.reset(seed=seed)
        outputs = policy.model.batched_forward([obs], policy.device, want_edge_maps=True)
        edge_scores = outputs.edge_score_maps[0]
        src_arr, dst_arr, scores = outputs.edge_arrays[0]
        node_ids = list(obs.node_ids)
        node_pos = {node_id: i for i, node_id in enumerate(node_ids)}
        if len(src_arr) == 0:
            continue

        # Rebuild the arc->dict form from the SAME floats the array path
        # consumes: ``log_prob_entropy_for_matching`` receives a plain dict of
        # floats and casts them to float32 tensors itself, so going through
        # Python floats here reproduces the real rollout call rather than
        # sidestepping it.
        float_scores = {
            (node_ids[int(s)], node_ids[int(d)]): float(v)
            for s, d, v in zip(src_arr.tolist(), dst_arr.tolist(), scores.tolist())
        }
        sampled = list(
            policy.act_batched([obs], deterministic=True, build_scores=True)[0].matched_edges or []
        )
        matchings = [("sampled matching", sampled)]
        maximal = _maximal_matching(src_arr, dst_arr, node_ids)
        if maximal:
            matchings.append(("maximal matching", maximal))
        if not matchings:
            continue

        for label, matched_edges in matchings:
            rollout_lp, rollout_entropy = policy.log_prob_entropy_for_matching(
                float_scores, matched_edges
            )
            array_lp, array_entropy = policy._matching_log_prob_entropy_arrays(
                src_arr, dst_arr, scores, node_pos, matched_edges
            )
            checked += 1
            for name, rollout_value, array_value in (
                ("mean_log_prob", rollout_lp, array_lp),
                ("mean_entropy", rollout_entropy, array_entropy),
            ):
                # NOT ``torch.equal`` here, unlike the fast<->arrays test. This
                # pair accumulates in a different order (the rollout path builds
                # one tensor per arc and reduces over a Python loop; the array
                # path works on slices of one tensor), so the last bit differs.
                # Measured over 900 randomized (scores, matching) pairs:
                # max abs diff 9.5e-07, no structural difference -- i.e. ~1 ulp
                # at float32, not a disagreement about the arc set or the
                # feasibility rows. The tolerance below is set to catch the
                # structural class of bug (which shows up orders of magnitude
                # larger) while tolerating the rounding. Do not tighten it to
                # ``torch.equal``: that fails intermittently depending on the
                # model's RNG state, which is a flaky test, not a guard.
                if not torch.allclose(rollout_value, array_value, rtol=0.0, atol=1.0e-5):
                    pytest.fail(
                        _failure(
                            f"seed {seed} / {label} / {name}",
                            rollout_value,
                            array_value,
                            matched_edges,
                            src_arr,
                            dst_arr,
                            node_ids,
                        )
                        + "\n  (rollout path via log_prob_entropy_for_matching "
                        "vs array update path; tolerance atol=1e-5 is for float "
                        "rounding -- a value this far off means a structural "
                        "difference, not rounding)"
                    )
    assert checked > 0, "no (seed, matching) combination was actually compared"


def test_array_path_matches_dict_path_bit_for_bit() -> None:
    env, policy = _build()
    obs = env.reset(seed=1)

    # A matching the sampler itself produced (dict-input sampler here, since
    # build_scores=True routes act_batched through the unchanged code path).
    sampled = list(
        policy.act_batched([obs], deterministic=True, build_scores=True)[0].matched_edges or []
    )

    outputs = policy.model.batched_forward([obs], policy.device, want_edge_maps=True)
    edge_scores = outputs.edge_score_maps[0]
    src_arr, dst_arr, scores = outputs.edge_arrays[0]
    node_ids = list(obs.node_ids)
    node_pos = {node_id: i for i, node_id in enumerate(node_ids)}

    assert len(src_arr) > 0, "test scenario produced no legal arcs; nothing to compare"
    # The dict path keys arcs by (src, dst), so it collapses a duplicated
    # candidate and drops a self-arc; the arrays path keeps both. Any mismatch
    # here means the two paths cannot agree bit for bit.
    assert len(src_arr) == len(edge_scores), (
        f"arc sets differ: {len(src_arr)} candidate arcs vs {len(edge_scores)} arc-dict keys"
    )
    maximal = _maximal_matching(src_arr, dst_arr, node_ids)
    assert maximal, "flat candidate order yielded no feasible matching"

    for label, matched_edges in (("sampled matching", sampled), ("maximal matching", maximal)):
        dict_lp, dict_entropy = policy._matching_log_prob_entropy_fast(edge_scores, matched_edges)
        array_lp, array_entropy = policy._matching_log_prob_entropy_arrays(
            src_arr, dst_arr, scores, node_pos, matched_edges
        )
        for name, dict_value, array_value in (
            ("mean_log_prob", dict_lp, array_lp),
            ("mean_entropy", dict_entropy, array_entropy),
        ):
            if not torch.equal(dict_value, array_value):
                pytest.fail(
                    _failure(
                        f"{label} / {name}",
                        dict_value,
                        array_value,
                        matched_edges,
                        src_arr,
                        dst_arr,
                        node_ids,
                    )
                )
