"""Improvement 1a: exact max-weight decoding of the matching policy (opt-in).

Pins three things:
1. ``max_weight_matching_indices`` is exact (brute force on small random
   instances, including two-cycles u->v / v->u) and always feasible;
2. the default ``greedy`` decode is untouched (same result as the original
   argmax loop, re-implemented here as the reference);
3. the single-graph and batched-array samplers agree under ``max_weight``, and
   the reported joint log-probability is the one of the executed sequence.
"""
from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from qkd_rl.rl.algos.matching_decode import matching_weight, max_weight_matching_indices
from qkd_rl.rl.algos.policy import MAPPOPolicy


def _feasible(sel, s, d, p) -> bool:
    return (len({s[i] for i in sel}) == len(sel) and len({d[i] for i in sel}) == len(sel)
            and len({p[i] for i in sel}) == len(sel))


def _brute(s, d, p, sc, stop, beta=0.0) -> float:
    idx = [i for i in range(len(sc)) if sc[i] > stop]
    best = 0.0
    for k in range(len(idx) + 1):
        for comb in itertools.combinations(idx, k):
            if _feasible(comb, s, d, p):
                best = max(best, matching_weight(comb, sc, stop, beta))
    return best


def _random_instance(rng, n_max=6, m_max=12):
    n = int(rng.integers(3, n_max + 1))
    pairs = [(u, v) for u in range(n) for v in range(n) if u != v]
    m = int(rng.integers(1, min(m_max, len(pairs)) + 1))
    arcs = [pairs[i] for i in rng.choice(len(pairs), m, replace=False)]
    s = np.array([a[0] for a in arcs])
    d = np.array([a[1] for a in arcs])
    p = np.minimum(s, d) * n + np.maximum(s, d)
    return n, arcs, s, d, p, rng.normal(size=m), float(rng.normal() * 0.5)


@pytest.mark.parametrize("beta", [0.0, 1.0, 3.0])
def test_max_weight_is_exact_and_feasible(beta):
    rng = np.random.default_rng(0)
    for _ in range(800):
        _n, _arcs, s, d, p, sc, stop = _random_instance(rng)
        sel = max_weight_matching_indices(s, d, p, sc, stop, beta=beta)
        assert _feasible(sel, s, d, p)
        assert all(sc[i] > stop for i in sel)
        assert matching_weight(sel, sc, stop, beta) == pytest.approx(
            _brute(s, d, p, sc, stop, beta), rel=1e-9, abs=1e-9)
        assert sel == sorted(sel, key=lambda a: (-sc[a], a))


def test_two_cycle_needs_branching():
    # u->v and v->u are both the best assignment entries; the pair rule forces a
    # choice, and the optimum keeps v->u plus the two arcs it no longer blocks.
    #   arcs: 0->1 (5), 1->0 (4), 0->2 (3.5), 2->1 (3.5)
    s = np.array([0, 1, 0, 2]); d = np.array([1, 0, 2, 1])
    p = np.minimum(s, d) * 3 + np.maximum(s, d)
    sc = np.array([5.0, 4.0, 3.5, 3.5])
    sel = max_weight_matching_indices(s, d, p, sc, 0.0)
    assert sorted(sel) == [1, 2, 3]
    assert matching_weight(sel, sc, 0.0) == pytest.approx(11.0)


class _Actor:
    def __init__(self, stop, temperature=1.0):
        self.stop_logit = torch.tensor(float(stop))
        self.temperature = temperature


class _Model:
    def __init__(self, node_ids, stop, temperature=1.0):
        self.actor = _Actor(stop, temperature)
        self.action_space = SimpleNamespace(IDLE="IDLE", node_ids=list(node_ids))

    def to(self, _device):
        return self


def _greedy_reference(arcs, scores, stop):
    used_tx, used_rx, used_pair, out = set(), set(), set(), []
    order = sorted(range(len(arcs)), key=lambda i: -scores[i])
    for i in order:
        if scores[i] <= stop:
            break
        u, v = arcs[i]
        if u in used_tx or v in used_rx or frozenset((u, v)) in used_pair:
            continue
        out.append((u, v)); used_tx.add(u); used_rx.add(v); used_pair.add(frozenset((u, v)))
    return out


@pytest.mark.parametrize("temperature", [1.0, 1.2])
def test_policy_decoders(temperature):
    rng = np.random.default_rng(1)
    for _ in range(200):
        n, arcs_i, s, d, p, sc, stop = _random_instance(rng, n_max=7, m_max=16)
        names = [f"N{i}" for i in range(n)]
        arcs = [(names[a], names[b]) for a, b in arcs_i]
        arc_scores = {a: torch.tensor(float(x)) for a, x in zip(arcs, sc)}

        greedy = MAPPOPolicy(_Model(names, stop, temperature))
        (_, _), m_greedy, _, _ = greedy._sample_matching(arc_scores, deterministic=True)
        assert m_greedy == _greedy_reference(arcs, sc, stop)

        mw = MAPPOPolicy(_Model(names, stop, temperature), deterministic_decode="max_weight")
        (acts, _), m_mw, lp, _ = mw._sample_matching(arc_scores, deterministic=True)
        sel = max_weight_matching_indices(s, d, p, sc, stop)
        assert m_mw == [arcs[i] for i in sel]
        assert matching_weight(sel, sc, stop) >= matching_weight(
            [arcs.index(a) for a in m_greedy], sc, stop) - 1e-9
        for u, v in m_mw:
            assert acts[u][0] == v and acts[v][1] == u
        # The replayed log-probability is the one the stochastic sampler assigns
        # to this ordered sequence (the policy's own evaluator agrees).
        lp_ref, _ = mw._matching_log_prob_entropy(arc_scores, m_mw)
        assert float(lp) == pytest.approx(float(lp_ref), abs=1e-4)

        # Batched arrays path gives the same matching and log-probability.
        (b_out,) = mw._sample_matching_arrays(
            [s], [d], [torch.tensor(sc, dtype=torch.float32)], [names], deterministic=True)
        assert b_out[1] == m_mw
        assert float(b_out[2]) == pytest.approx(float(lp), abs=1e-4)

        # Stochastic sampling ignores the decode mode.
        g1 = MAPPOPolicy(_Model(names, stop, temperature)); g1.set_sample_seed(7)
        g2 = MAPPOPolicy(_Model(names, stop, temperature), deterministic_decode="max_weight")
        g2.set_sample_seed(7)
        assert g1._sample_matching(arc_scores)[1] == g2._sample_matching(arc_scores)[1]


def test_unknown_decode_rejected():
    with pytest.raises(ValueError):
        MAPPOPolicy(_Model(["A"], 0.0), deterministic_decode="hungarian")
