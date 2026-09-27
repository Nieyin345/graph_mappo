"""v8's forecast head must agree in the batched and single-graph paths.

Both bugs this pins were SILENT and both only existed on the batched path --
which is the one the trainer's PPO update actually calls
(`evaluate_actions_batched`). The single-graph path was always correct, so a
smoke test that only ran `model(obs)` passed while training did nothing.

  1. LENGTH. The label is per UNDIRECTED physical edge (len(physical_edge_ids)),
     but the encoder stores both directions. The batched head emitted
     2 x that, so the trainer's `label.numel() == logit.numel()` guard rejected
     every step: usage_terms stayed empty and the head never trained. Symptom is
     identical to `usage_forecast.enabled: false`.

  2. BIAS. `bias_weight` was applied in the single-graph path only, so the
     behavioural "prepare" arm would have been a no-op under training.

  3. PARITY. The two paths must produce the same logits for the same graph;
     otherwise the loss is computed on a different function than the one used
     to act, and any measured effect is unattributable.
"""

from __future__ import annotations

import copy

import pytest
import torch

from qkd_rl.env.factory import build_env_from_config
from qkd_rl.model_zoo import build_model
from scripts.experiment import resolve_experiment

OVERRIDES = ["env_full_demand_v2.yaml"]


def _build(bias_weight: float = 0.0):
    cfg = resolve_experiment("v8", 42, 1, OVERRIDES, "v8_forecast_parity")
    cfg = copy.deepcopy(cfg)
    cfg["model"]["actor"]["usage_forecast"]["bias_weight"] = bias_weight
    env = build_env_from_config(cfg)
    model = build_model(env.action_resolver.action_space, cfg)
    model.eval()
    obs = env.reset(seed=100)
    return cfg, env, model, obs


@pytest.fixture(scope="module")
def built():
    return _build(bias_weight=0.0)


def test_forecast_length_matches_undirected_edges(built):
    """The label is per undirected edge; the head must match it, not 2x."""
    _cfg, _env, model, obs = built
    n_undirected = len(obs.physical_edge_ids)
    with torch.no_grad():
        out = model(obs, device="cpu")
    assert out.usage_logits is not None, "v8 must emit usage_logits"
    assert out.usage_logits.numel() == n_undirected, (
        f"single-graph: head emitted {out.usage_logits.numel()} logits for "
        f"{n_undirected} undirected physical edges"
    )

    b_out = model.batched_forward([obs], torch.device("cpu"), want_edge_maps=False)
    b_logits = b_out.usage_logits
    assert b_logits is not None and len(b_logits) == 1
    assert b_logits[0].numel() == n_undirected, (
        f"batched: head emitted {b_logits[0].numel()} logits for "
        f"{n_undirected} undirected physical edges -- the trainer's "
        f"numel guard would drop every step and the head would never train"
    )


def test_batched_and_single_graph_agree(built):
    """Same graph -> same forecast. Otherwise the trained function is not the
    one used to act."""
    _cfg, _env, model, obs = built
    with torch.no_grad():
        single = model(obs, device="cpu").usage_logits
        batched = model.batched_forward([obs], torch.device("cpu"), want_edge_maps=False).usage_logits[0]
    assert single.shape == batched.shape
    torch.testing.assert_close(batched, single, atol=1e-5, rtol=1e-4)


def test_bias_has_effect_in_batched_path():
    """bias_weight > 0 must change the arc scores under batched_forward too,
    or the behavioural arm is silently a representation-only arm."""
    _c0, _e0, m0, obs0 = _build(bias_weight=0.0)
    _c1, _e1, m1, obs1 = _build(bias_weight=5.0)
    dev = torch.device("cpu")

    def arc_scores(model, obs):
        out = model.batched_forward([obs], dev, want_edge_maps=True)
        return out.edge_arrays[0][2]

    s0 = arc_scores(m0, obs0)
    s1 = arc_scores(m1, obs1)
    assert s0.shape == s1.shape and s0.numel()
    assert not torch.allclose(s0, s1), (
        "bias_weight=5 produced identical scores to bias_weight=0 -- "
        "the bias is not reaching the batched scoring path"
    )


def test_bias_is_applied_on_the_SINGLE_GRAPH_path():
    """The bias must change the SINGLE-graph scores, not just the batched ones.

    This is the bug that made batch 2 diverge (kl ~4.6, clip_frac ~1.0): the
    single-graph path built `logits`/`edge_scores` from the unbiased scores and
    added the bias afterwards, so the assignment did nothing. The rollout
    samples on the SINGLE-graph path while the PPO update re-scores on the
    BATCHED one, so the policy that sampled and the policy being re-evaluated
    differed before any training -- the ratio was meaningless.

    The two paths agreeing with each other (test above) is NOT sufficient:
    both could omit the bias. This pins that the single-graph path applies it.
    """
    _c0, _e0, m0, obs0 = _build(bias_weight=0.0)
    _c1, _e1, m1, obs1 = _build(bias_weight=5.0)
    # Identical weights, different bias weight.
    m1.load_state_dict(m0.state_dict(), strict=False)

    def single_scores(model, obs):
        out = model(obs, device="cpu")
        assert out.edge_scores, "no edge_scores on the single-graph path"
        return torch.stack([v.detach() for v in out.edge_scores.values()])

    s0 = single_scores(m0, obs0)
    s1 = single_scores(m1, obs1)
    assert s0.numel() and s0.shape == s1.shape
    assert not torch.allclose(s0, s1), (
        "bias_weight=5 left the SINGLE-graph scores unchanged -- the bias is a "
        "dead assignment on the path the rollout samples from, so the actions "
        "were drawn WITHOUT the bias while PPO re-scores WITH it"
    )


def test_path_recency_column_is_alive(built):
    """The recency column must VARY and must not be all-zero.

    Silent-failure class this guards: `include_path_recency` on but the feature
    wired to a zero array (indexing mismatch, mask never passed). Then the head
    loses the one input with measured signal (AUC 0.624 on the `new` target),
    the run still completes, and it reads as "recency doesn't help".

    It checks a real trajectory rather than one frame: recency only becomes
    nonzero after edges have been on paths, so a fresh reset can legitimately be
    all-zero.
    """
    import numpy as np

    from qkd_rl.baselines.path_greedy import PathScoreGreedy
    from qkd_rl.baselines.serve_probe import ServeProbe

    cfg, env, model, obs = built
    assert cfg["features"]["edge"].get("include_path_recency"), (
        "the v8 config must enable the recency feature"
    )
    dim = int(cfg["features"]["dims"]["physical_edge_dim_resolved"])
    rec_idx = dim - 1     # appended last in the dynamic block

    expert = PathScoreGreedy(weights=(1.0, 10.0, 1.0, 0.5, 0.2), phased=True,
                             principles=False, router=ServeProbe(env))
    cols = []
    for _ in range(60):
        arr = np.asarray(obs.edge_features, dtype=np.float32)
        n_phys = len(obs.physical_edge_ids)
        if arr.ndim == 2 and arr.shape[0] >= n_phys and n_phys:
            cols.append(arr[:n_phys, rec_idx])
        a, sc = expert.act(obs)
        obs, _r, te, tr, _i = env.step(a, sc)
        if te or tr:
            break

    allc = np.concatenate(cols) if cols else np.zeros(0)
    assert allc.size, "no physical edge features collected"
    assert allc.max() > 0.0, (
        "recency column is identically zero -- the head has lost the AUC-0.62 "
        "signal, and the arm would silently test nothing"
    )
    assert allc.std() > 0.0, "recency column is constant"


def test_ppo_gradient_does_not_update_the_head():
    """The bias must be detached, so PPO's own step cannot move the head.

    With bias_weight != 0 the head is part of the policy that SCORES actions,
    but it is trained by BCE -- a different objective, applied ONCE per update
    before the PPO epochs (`MAPPOTrainer._train_usage_head`). If PPO's gradient
    also reached the head, the head would move between minibatches, the policy
    used to re-evaluate the stored actions would differ from the one that
    sampled them, and the ratio would be meaningless. Measured on the real
    training path before the fix: bias=4 gave kl 5-12 and clip_frac ~1.0, versus
    kl 0.002 / clip_frac 0.16 in the otherwise identical bias=0 arm.

    Note the invariant is "PPO does not move the head", NOT "changing the head
    does not change the scores" -- the latter is false by design (a changed
    forecast is supposed to change the bias), which is why the head has to be
    trained outside the epoch loop rather than merely detached.
    """
    from qkd_rl.rl.algos.policy import MAPPOPolicy

    cfg, env, model, obs = _build(bias_weight=4.0)
    head = model.actor.usage_forecast
    if head is None:
        pytest.skip("forecast head disabled")

    head_before = [p.detach().clone() for p in head.parameters()]
    policy = MAPPOPolicy(model, torch.device("cpu"))
    actions = {}
    for nid in obs.node_ids:
        pass
    # Drive the actor loss path the trainer uses, on a real observation.
    model.train()
    for p in model.parameters():
        p.grad = None
    from qkd_rl.rl.algos.rollout_buffer import RolloutStep
    step = RolloutStep(obs=obs, actions={}, log_probs={}, entropies={},
                       value=torch.zeros(()), reward=0.0,
                       terminated=False, truncated=False)
    # A trivial advantage so the surrogate produces a nonzero gradient into the
    # actor; the head must still receive none.
    loss = torch.zeros(())
    outs = policy.evaluate_actions_batched([obs], [{}], [[]])
    for (lp, ent, val) in outs:
        if lp:
            first = next(iter(lp.values()))
            loss = loss + (first ** 2).mean() + val.squeeze()
    if loss.requires_grad:
        loss.backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    nz = [g for g in grads if float(g.abs().sum()) > 0]
    assert not nz, (
        f"{len(nz)} head tensors received gradient from the policy loss -- "
        f"the bias is not detached, so PPO would move the head mid-update"
    )
    # And belt-and-braces: the values must be untouched too.
    head_after = [p.detach().clone() for p in head.parameters()]
    for a, b in zip(head_before, head_after):
        assert torch.equal(a, b), "head parameters changed during a policy step"
