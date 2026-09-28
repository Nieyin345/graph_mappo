"""Exact max-weight decoding of the matching policy (improvement 1a, opt-in).

The actor scores every legal directed arc ``src -> dst`` and a STOP option. The
default deterministic decoder is the greedy argmax of the sequential sampler:
repeatedly take the highest-scoring feasible arc until STOP scores highest.
Greedy can block two good arcs with one slightly better arc that shares their
ports -- exactly the "port conflict" loss class of the failure anatomy.

This module instead returns the feasible matching that maximises

    sum_{a in M} (s_a - s_stop)        (only arcs with s_a > s_stop can help)

subject to the dual-port rules: Tx-out <= 1 per node, Rx-in <= 1 per node, and
a pair never carries both ``u->v`` and ``v->u`` (对端不同). The first two rules
make it a bipartite assignment (Tx side x Rx side), solved exactly by
``scipy.optimize.linear_sum_assignment``. The third rule is enforced by
branch-and-bound: a violated pair is split into "forbid u->v" / "forbid v->u",
and the assignment relaxation is an upper bound used for pruning. Two-cycles
are rare, so the search is almost always a single assignment solve.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["max_weight_matching_indices", "matching_weight"]


def matching_weight(indices, scores, stop_score, beta: float = 0.0) -> float:
    m = [float(scores[i]) - float(stop_score) for i in indices]
    if beta > 0.0:
        return float(sum(np.expm1(beta * x) / beta for x in m))
    return float(sum(m))


def max_weight_matching_indices(
    src_code: np.ndarray,
    dst_code: np.ndarray,
    pair_code: np.ndarray,
    scores: np.ndarray,
    stop_score: float,
    *,
    beta: float = 0.0,
    max_nodes: int = 256,
) -> list[int]:
    """Indices of the arcs in the max-weight feasible matching.

    Returned in descending score order (ties by index), which is the order the
    sequential sampler replays them in when computing the joint log-probability.

    ``beta`` selects the arc weight: ``0`` gives the margin ``s - s_stop``;
    ``beta > 0`` gives ``expm1(beta * (s - s_stop)) / beta`` -- a convex
    transform of the margin that is monotone and positive on the same arcs,
    tends to the linear margin as ``beta -> 0`` and to the lexicographic
    (greedy-like) order as ``beta -> inf``. It interpolates between "sum of
    logits" and "protect the top arc".

    ``max_nodes`` caps the branch-and-bound tree; if it is exhausted the best
    feasible solution found so far is returned (a lower-weight direction of
    each remaining two-cycle is dropped), so the result is always feasible.
    """
    src_code = np.asarray(src_code, dtype=np.int64)
    dst_code = np.asarray(dst_code, dtype=np.int64)
    pair_code = np.asarray(pair_code, dtype=np.int64)
    margin = np.asarray(scores, dtype=np.float64) - float(stop_score)
    if beta > 0.0:
        w_all = np.expm1(np.minimum(beta * margin, 700.0)) / beta
    else:
        w_all = margin
    cand = np.flatnonzero(w_all > 0.0)
    if cand.size == 0:
        return []

    rows, r_inv = np.unique(src_code[cand], return_inverse=True)
    cols, c_inv = np.unique(dst_code[cand], return_inverse=True)
    base = np.zeros((rows.size, cols.size), dtype=np.float64)
    arc_at = np.full((rows.size, cols.size), -1, dtype=np.int64)
    for j, a in enumerate(cand.tolist()):
        r, c = r_inv[j], c_inv[j]
        if w_all[a] > base[r, c]:  # duplicate directed arcs: keep the best
            base[r, c] = w_all[a]
            arc_at[r, c] = a
    cell_of = {int(arc_at[r, c]): (int(r), int(c)) for r, c in zip(*np.nonzero(arc_at >= 0))}

    best = {"w": -1.0, "sel": []}
    budget = [max_nodes]

    def solve(forbidden: frozenset) -> None:
        if budget[0] <= 0:
            return
        budget[0] -= 1
        m = base.copy()
        for a in forbidden:
            r, c = cell_of[a]
            m[r, c] = 0.0
        ri, ci = linear_sum_assignment(m, maximize=True)
        sel = [int(arc_at[r, c]) for r, c in zip(ri, ci) if m[r, c] > 0.0]
        val = float(sum(w_all[a] for a in sel))
        if val <= best["w"] + 1e-12:
            return  # relaxation cannot beat the incumbent
        by_pair: dict[int, list[int]] = {}
        for a in sel:
            by_pair.setdefault(int(pair_code[a]), []).append(a)
        clash = next((v for v in by_pair.values() if len(v) > 1), None)
        if clash is None:
            best["w"], best["sel"] = val, sel
            return
        a, b = clash[0], clash[1]
        # Explore the branch keeping the stronger arc first (better incumbent sooner).
        first, second = (b, a) if w_all[a] >= w_all[b] else (a, b)
        solve(forbidden | {first})
        solve(forbidden | {second})
        if best["w"] < 0.0 and budget[0] <= 0:
            # Budget ran out before any feasible leaf: repair this relaxation.
            keep: dict[int, int] = {}
            for x in sel:
                p = int(pair_code[x])
                if p not in keep or w_all[x] > w_all[keep[p]]:
                    keep[p] = x
            fixed = list(keep.values())
            best["w"], best["sel"] = float(sum(w_all[x] for x in fixed)), fixed

    solve(frozenset())
    return sorted(best["sel"], key=lambda a: (-float(scores[a]), a))
