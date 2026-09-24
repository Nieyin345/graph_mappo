"""Independent probability oracle: consistency between implementations is insufficient."""
from types import SimpleNamespace
import math

import numpy as np
import pytest
import torch

from qkd_rl.rl.algos.policy import MAPPOPolicy


def _policy():
    policy = object.__new__(MAPPOPolicy)
    policy.device = torch.device("cpu")
    policy.model = SimpleNamespace(
        actor=SimpleNamespace(temperature=1.0, stop_logit=torch.tensor(0.0, requires_grad=True)),
        action_space=SimpleNamespace(IDLE="IDLE", node_ids=list("abcd")))
    policy._matching_cache = None
    policy._sample_gen = None
    policy._sample_gens_batched = None
    return policy


@pytest.mark.parametrize("implementation", ["reference", "fast", "arrays"])
def test_ordered_matching_probabilities_sum_to_one(implementation):
    policy = _policy()
    a, b = ("a", "b"), ("c", "d")
    scores = {a: torch.tensor(0.0), b: torch.tensor(0.0)}
    sequences = [[], [a], [b], [a, b], [b, a]]
    probabilities = []
    for sequence in sequences:
        if implementation == "arrays":
            lp, _ = policy._matching_log_prob_entropy_arrays(
                np.array([0, 2]), np.array([1, 3]), torch.stack(list(scores.values())),
                {n: i for i, n in enumerate("abcd")}, sequence)
        else:
            method = (policy._matching_log_prob_entropy if implementation == "reference"
                      else policy._matching_log_prob_entropy_fast)
            lp, _ = method(scores, sequence)
        probabilities.append(lp.exp().item())
    # STOP: 1/3. Each one-arc+STOP / maximal two-arc sequence: (1/3)*(1/2).
    assert probabilities == pytest.approx([1/3, 1/6, 1/6, 1/6, 1/6])
    assert sum(probabilities) == pytest.approx(1.0)


def test_matching_ratio_and_gradient_use_product_not_geometric_mean():
    policy = _policy()
    a, b = ("a", "b"), ("c", "d")
    score = torch.tensor(0.4, requires_grad=True)
    lp, _ = policy._matching_log_prob_entropy_fast({a: score, b: score}, [a, b])
    old_probability = 1/6
    weight = score.exp()
    probability = weight / (2*weight + 1) * weight / (weight + 1)
    assert torch.exp(lp - math.log(old_probability)).item() == pytest.approx(
        (probability / old_probability).item())
    actual_grad = torch.autograd.grad(lp, score, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(probability.log(), score)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_bc_explicitly_preserves_per_decision_nll():
    policy = _policy()
    a, b = ("a", "b"), ("c", "d")
    scores = {a: torch.tensor(0.0), b: torch.tensor(0.0)}
    lp, _ = policy._matching_log_prob_entropy_fast(scores, [a, b], average_log_prob=True)
    assert lp.item() == pytest.approx(math.log(1/6) / 2)


@pytest.mark.parametrize("batched", [False, True])
def test_sampler_reports_probability_of_the_sampled_sequence(batched):
    policy = _policy()
    policy.set_sample_seed(42)
    seen = set()
    for _ in range(64):
        if batched:
            result = policy._sample_matching_arrays(
                [np.array([0, 2])], [np.array([1, 3])], [torch.zeros(2)], [list("abcd")])[0]
        else:
            result = policy._sample_matching({("a", "b"): torch.tensor(0.),
                                              ("c", "d"): torch.tensor(0.)})
        _, sequence, log_prob, _ = result
        seen.add(tuple(sequence))
        assert log_prob.exp().item() == pytest.approx(1/3 if not sequence else 1/6)
    assert len(seen) == 5
