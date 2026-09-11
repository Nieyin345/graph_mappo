from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

_tp_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp_ = importlib.util.module_from_spec(_tp_spec)
_tp_spec.loader.exec_module(_tp_)
build_validation_env_config = _tp_.build_validation_env_config
load_validation_profile = _tp_.load_validation_profile

from qkd_rl.core.config import ConfigValidator
from qkd_rl.env.factory import build_env_from_config
from qkd_rl.baselines.path_greedy import PathGreedyPolicy, edge_endpoints


def run(profile, seed, start_seed, steps, dense_fill, persist_kept=True, deadline_steps=30, amount_max=None):
    profile = dict(profile)
    profile["requests"] = dict(profile.get("requests", {}))
    profile["requests"]["deadline_steps"] = deadline_steps
    if amount_max is not None:
        profile["requests"]["amount_max"] = amount_max
    config = build_validation_env_config(profile, include_baselines=False,
                                         episode_steps=steps, start_mode=profile["start_mode"])
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    env.reset(seed=seed, start_seed=start_seed)
    obs = env._obs_cache[1] if env._obs_cache else None
    pol = PathGreedyPolicy(rate_weight=1.0, dense_fill=dense_fill, persist_kept=persist_kept)
    total_served = 0.0
    total_generated = 0.0
    total_expired_qkp = 0.0
    n_act = 0
    arrived_reqs = 0
    for _ in range(steps):
        actions, _ = pol.act(obs)
        obs, _, term, trunc, info = env.step(actions)
        total_served += info.get("served_keys", 0.0)
        total_generated += info.get("generated_keys", 0.0)
        n_act += len(getattr(env, "last_activated_edges", []) or [])
        # qkp stock level after this step = accumulated - consumed - expired; use
        # snapshot to detect how much stock is carried but never served.
        if term or trunc:
            break
    arrived = float(env.metrics.arrived_keys)
    sr = total_served / arrived if arrived > 0 else 0.0
    summary = env.metrics.episode_summary()
    return sr, total_served, n_act / max(1, steps), arrived, total_generated, summary


def trace_one_request(profile, seed, start_seed, steps):
    profile = dict(profile)
    config = build_validation_env_config(profile, include_baselines=False,
                                         episode_steps=steps, start_mode=profile["start_mode"])
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    env.reset(seed=seed, start_seed=start_seed)
    obs = env._obs_cache[1] if env._obs_cache else None
    pol = PathGreedyPolicy(rate_weight=1.0, dense_fill=True, persist_kept=True)

    # Track, for every request, whether its static shortest path was ever fully
    # physical (all hops present in physical_edge_ids) AND ever fully stocked
    # (all hops positive), over the whole window. This gives the feasibility cap.
    hop_cap = {"connected": 0, "disconnected": 0, "ever_phys": 0, "ever_stock": 0, "served": 0, "total": 0}
    stocks_tracked = {req.request_id: None for req in obs.state.pending_requests}
    activated_by_req_cap = None
    ran = 0
    for step in range(steps):
        active = set(obs.physical_edge_ids)
        def _satisfied(req):
            s = []
            midpoint = (req.src_gs, req.dst_gs)
            return s
        # handle arrivals that are new this step
        for rq in obs.state.pending_requests:
            rid = rq.request_id
            if stocks_tracked.setdefault(rid, {"done": False}).get("done"):
                continue
            rec = stocks_tracked[rid]
            p = env.routing.shortest_path(rq.src_gs, rq.dst_gs)
            rec["path"] = p
            hop_cap["total"] += 1
            if p is None:
                rec["done"] = True
                hop_cap["disconnected"] += 1
                continue
            rec["sorted"] = p
        actions, _ = pol.act(obs)
        obs, _, term, trunc, info = env.step(actions)
        # after serving stocks change; final tally at end
        if term or trunc:
            break
    run = 0
    # final tally on the last obs: check stock availability impossible w/o qkp;
    # use metrics for served counts
    sm = env.metrics.episode_summary()
    print(f"FINAL: arrived={sm['arrived_requests']} completed={sm['completed_requests']} "
          f"sr={sm['success_rate']:.4f}")
    # feasibility: how many requests were TOPOLOGICALLY connected (had a static
    # shortest path)
    conn = 0
    for r in stocks_tracked.values():
        if r and r.get("path") is not None:
            conn += 1
    total_tracked = sum(1 for r in stocks_tracked.values() if r)
    print(f"Requests tracked={total_tracked}, topologically connected (has static path)={conn}")
    return None


def main():
    profile = load_validation_profile(ROOT / "configs" / "global.yaml")
    import sys
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2160
    # Focused diagnosis: is a failing GS-pair's source/dst physically seeing any
    # available edge during its lifetime? If src has zero usable edges often,
    # it's an environment coverage cap, not a scheduler fault.
    cfg = build_validation_env_config(dict(profile), include_baselines=False,
                                      episode_steps=steps, start_mode=profile["start_mode"])
    ConfigValidator().validate(cfg)
    env = build_env_from_config(cfg)
    env.reset(seed=7, start_seed=7)
    obs = env._obs_cache[1] if env._obs_cache else None
    pol = PathGreedyPolicy(rate_weight=1.0, dense_fill=True, persist_kept=True)
    # map request_id -> [min neighbors seen over lifetime]
    src_min_nb: dict = {}
    req_mark: dict = {}
    completed_at = {}
    for step in range(steps):
        adj_n = {}
        for eid in obs.physical_edge_ids:
            u, v = edge_endpoints(eid)
            adj_n.setdefault(u, set()).add(v)
            adj_n.setdefault(v, set()).add(u)
        for rq in obs.state.pending_requests:
            rr = req_mark.setdefault(rq.request_id, {"src_n": None, "dst_n": None, "src": rq.src_gs, "dst": rq.dst_gs})
            sn = len(adj_n.get(rq.src_gs, ()))
            dn = len(adj_n.get(rq.dst_gs, ()))
            if rr["src_n"] is None or sn < rr["src_n"]:
                rr["src_n"] = sn
            if rr["dst_n"] is None or dn < rr["dst_n"]:
                rr["dst_n"] = dn
        actions, _ = pol.act(obs)
        obs, _, term, trunc, info = env.step(actions)
        if term or trunc:
            break
    sm = env.metrics.episode_summary()
    print(f"steps={steps} completed={sm['completed_requests']} sr={sm['success_rate']:.4f}")
    failed = [r for i, r in req_mark.items() if sm["completed_requests"]]
    # completed request ids
    # (re-derive roughly: requests that reached serve completion we can't see here;
    #  instead report distribution of min src/dst neighbors for ALL tracked reqs)
    import collections
    def hist(sel):
        vals = [r[sel] for r in req_mark.values() if r[sel] is not None]
        c = collections.Counter(vals)
        return dict(sorted(c.items()))
    print("min src neighbors histogram:", hist("src_n"))
    print("min dst neighbors histogram:", hist("dst_n"))
    return None


if __name__ == "__main__":
    main()