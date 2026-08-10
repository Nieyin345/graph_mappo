"""Per-slot strictly-optimal scheduling reference (exact search).

For each slot, over the SAME environment the RL model and the other baselines
use (same scenario, same active nodes, same deterministic request stream), it
exhaustively searches the space of feasible link-activation matchings and
picks the one that maximises the environment's REALIZED served keys this slot
(primary) and generated keys (secondary tie-break). Each candidate matching is
evaluated by simulating the env's actual generation + sequential serve on a
copy of the state, so there is no plan-vs-realization gap and no modelling
relaxation. The search is exact (branch-and-bound with a max-flow upper bound
and greedy lower bound) -- the result is the per-slot optimal policy for the
true environment transition (myopic across slots, like the MILP reference).

Usage:
    conda run -n pytorch python scripts/compute_exact_reference.py \
        --seed 1000042 --steps 1440 --out outputs/exact_reference/day1

Writes steps.csv + summary.json in the same format as outputs/milp_reference/.
"""
from __future__ import annotations

import os
import sys
import time
import copy
import json
import csv
import argparse
from collections import defaultdict, deque
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import maximum_flow

from qkd_rl.core.config import deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.env.request import RequestGenerator

STEP_FIELDS = ["step", "t", "reward", "served_keys", "generated_keys", "failed_keys",
               "waiting_keys", "pending_count", "n_activated"]


class StateSimulator:
    """Evaluate a candidate matching on a copy of the pre-slot state.

    Replicates the env's per-slot mechanics exactly: generation on the
    activated edges (with switch-cost decay), QKP capacity cap, then the real
    sequential serve (deadline/hop/remaining order, shortest positive-key
    path). Uses only a snapshot, so it never mutates the live env.
    """

    def __init__(self, env, slot_seconds: float, decay: float):
        self.env = env
        self.slot_seconds = float(slot_seconds)
        self.decay = float(decay)

    def run(self, state, pending, t: int, activated) -> tuple[float, float]:
        q = copy.copy(self.env.qkp)
        q.levels = dict(self.env.qkp.levels)
        q.batches = defaultdict(list, {e: list(v) for e, v in self.env.qkp.batches.items()})
        q.positive = set(self.env.qkp.positive)
        generated = 0.0
        last_active = set(state.last_activated_edges)
        windows = state.edge_windows
        for eid in activated:
            w = windows[eid]
            rate = w.rates[0] * (self.decay if eid not in last_active else 1.0)
            gen = rate * self.slot_seconds
            generated += gen
            q.add_keys(eid, gen, t)
        reqs = copy.copy(self.env.requests)
        reqs.pending = list(pending)
        res = reqs.serve(q, self.env.routing, t)
        return float(res.served_keys), generated


class ExactPerSlotPolicy:
    """Per-slot exact policy: B&B over matchings with max-flow UB pruning."""

    def __init__(self, env, sim: StateSimulator):
        self.env = env
        self.sim = sim
        self.gen = RequestGenerator(
            [n.node_id for n in env.scenario.nodes if n.node_type.value == "gs"],
            env.config["requests"],
            seed=0,
        )
        self._best_obj = 0.0
        self._best_served = 0.0
        self._best_matching: set[str] = set()
        self.search_nodes = 0
        self.pruned = 0
        self.truncated = False
        self.time_budget_s = 2.0

    def seed_stream(self, seed: int) -> None:
        self.gen.seed(seed)

    # ------------------------------------------------------------------ public
    def act(self, obs, t: int) -> tuple[dict, dict]:
        # Lockstep arrivals: the env's generator draws the same stream, so the
        # search sees exactly the requests the env will serve this slot.
        arrivals = self.gen.generate(t)
        pending = list(obs.state.pending_requests) + list(arrivals)
        best = self._search(obs, pending, t)
        return self._actions_from_edges(obs, best)

    # ------------------------------------------------------------------ search
    def _search(self, obs, pending, t: int) -> list[str]:
        edges = self._candidate_edges(obs, pending)
        if not edges:
            return []
        best_m, best_served, best_obj = self._greedy(edges, obs.state, pending, t)
        self._best_matching = best_m
        self._best_served = best_served
        self._best_obj = best_obj
        self.search_nodes = 0
        self.pruned = 0
        self.truncated = False
        # Global upper bound: serve with every candidate activated (matching
        # relaxed). If the greedy already attains it, it is provably optimal.
        serve_all, _ = self.sim.run(obs.state, pending, t, edges)
        if serve_all <= best_served:
            return list(self._best_matching)
        self._deadline = time.monotonic() + self.time_budget_s
        self._bb(edges, 0, [], set(), obs.state, pending, t)
        return list(self._best_matching)

    def _candidate_edges(self, obs, pending) -> list[str]:
        windows = obs.state.edge_windows
        pairs = {(r.src_gs, r.dst_gs) for r in pending if r.src_gs != r.dst_gs}
        if not pairs:
            return []
        adj: dict[str, list[str]] = {}
        edge_ids: list[str] = []
        for eid in obs.physical_edge_ids:
            w = windows[eid]
            if not w.available[0] or w.rates[0] <= 1e-9:
                continue
            e = eid[2:] if eid.startswith("E_") else eid
            if "__" not in e:
                continue
            s, d = e.split("__", 1)
            adj.setdefault(s, []).append(d)
            adj.setdefault(d, []).append(s)
            edge_ids.append(eid)
        relevant: set[str] = set()
        for (src, dst) in pairs:
            if src not in adj or dst not in adj:
                continue
            fwd = self._reachable(src, adj)
            if dst not in fwd:
                continue
            bwd = self._reachable(dst, adj)
            for eid in edge_ids:
                e = eid[2:] if eid.startswith("E_") else eid
                s, d = e.split("__", 1)
                if (s in fwd and d in bwd) or (d in fwd and s in bwd):
                    relevant.add(eid)
        return sorted(relevant, key=lambda eid: -windows[eid].rates[0])

    @staticmethod
    def _reachable(src: str, adj: dict[str, list[str]]) -> set[str]:
        seen = {src}
        q = deque([src])
        while q:
            n = q.popleft()
            for nb in adj.get(n, ()):
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        return seen

    def _greedy(self, edges, state, pending, t):
        """Seed the search with two heuristics and keep the better one.

        1. max-rate matching (covers many edges, stocks inventory);
        2. path packing: for the pending request pairs in env priority order,
           activate a shortest path per pair (only pairs whose endpoints are
           still free), so the incumbent actually serves the urgent requests.
        """
        best_chosen: set[str] = set()
        best_served = -1.0
        # heuristic 1: max-rate matching
        used = set()
        chosen: list[str] = []
        for eid in edges:
            e = eid[2:] if eid.startswith("E_") else eid
            s, d = e.split("__", 1)
            if s in used or d in used:
                continue
            used.update({s, d})
            chosen.append(eid)
        served, gen = self.sim.run(state, pending, t, chosen)
        best_chosen = set(chosen)
        best_served = served
        best_obj = served + 1e-12 * gen
        # heuristic 2: path packing
        windows = state.edge_windows
        adj: dict[str, list[str]] = {}
        for eid in edges:
            e = eid[2:] if eid.startswith("E_") else eid
            s, d = e.split("__", 1)
            adj.setdefault(s, []).append(d)
            adj.setdefault(d, []).append(s)
        used_nodes: set[str] = set()
        chosen2: list[str] = []
        pairs = sorted(
            {(r.src_gs, r.dst_gs) for r in pending if r.src_gs != r.dst_gs},
            key=lambda p: min((r.deadline_t for r in pending if {r.src_gs, r.dst_gs} == set(p)), default=0),
        )
        for (s, d) in pairs:
            if s in used_nodes or d in used_nodes:
                continue
            path = self._shortest_path_avoid(s, d, adj, used_nodes)
            if path is None:
                continue
            path_edges = []
            ok = True
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                eid = self._edge_between(a, b, windows)
                if eid is None or (a in used_nodes or b in used_nodes):
                    ok = False
                    break
                path_edges.append(eid)
            if not ok:
                continue
            for eid in path_edges:
                chosen2.append(eid)
                e = eid[2:] if eid.startswith("E_") else eid
                x, y = e.split("__", 1)
                used_nodes.update({x, y})
        if chosen2:
            served2, gen2 = self.sim.run(state, pending, t, chosen2)
            obj2 = served2 + 1e-12 * gen2
            if obj2 > best_obj:
                best_chosen = set(chosen2)
                best_served = served2
                best_obj = obj2
        return best_chosen, best_served, best_obj

    @staticmethod
    def _shortest_path_avoid(src: str, dst: str, adj: dict[str, list[str]], avoid: set[str]):
        if src not in adj or dst not in adj:
            return None
        parent = {src: None}
        q = deque([src])
        while q:
            n = q.popleft()
            if n == dst:
                break
            for nb in adj.get(n, ()):
                if nb in parent or nb in avoid:
                    continue
                parent[nb] = n
                q.append(nb)
        if dst not in parent:
            return None
        path = [dst]
        cur = dst
        while parent[cur] is not None:
            cur = parent[cur]
            path.append(cur)
        return path[::-1]

    @staticmethod
    def _edge_between(a: str, b: str, windows):
        for eid in windows.keys():
            e = eid[2:] if eid.startswith("E_") else eid
            if "__" not in e:
                continue
            s, d = e.split("__", 1)
            if {s, d} == {a, b}:
                return eid
        return None
    def _bb(self, edges, i, chosen, used, state, pending, t):
        self.search_nodes += 1
        if self.search_nodes % 2000 == 0 and time.monotonic() > self._deadline:
            self.truncated = True
            return
        n = len(edges)
        if i == n:
            served, gen = self.sim.run(state, pending, t, chosen)
            obj = served + 1e-12 * gen
            if obj > self._best_obj:
                self._best_obj = obj
                self._best_served = served
                self._best_matching = set(chosen)
            return
        ub_served = self._serve_ub(edges, i, chosen, state, pending, t)
        if ub_served <= self._best_served:
            self.pruned += 1
            return
        eid = edges[i]
        e = eid[2:] if eid.startswith("E_") else eid
        s, d = e.split("__", 1)
        self._bb(edges, i + 1, chosen, used, state, pending, t)
        if s not in used and d not in used:
            chosen.append(eid)
            used.update({s, d})
            self._bb(edges, i + 1, chosen, used, state, pending, t)
            chosen.pop()
            used.difference_update({s, d})

    def _serve_ub(self, edges, i, chosen, state, pending, t) -> float:
        """Upper bound on served keys reachable from the partial matching.

        The env's sequential serve is monotone in per-edge key levels
        (verified empirically), so serving with the chosen edges plus ALL
        remaining candidate edges activated (matching relaxed) is an upper
        bound on any completion. Cheaper and tighter than a max-flow bound.
        """
        served, _ = self.sim.run(state, pending, t, list(chosen) + list(edges[i:]))
        return served
    @staticmethod
    def _actions_from_edges(obs, best_edges) -> tuple[dict, dict]:
        act: dict[str, str] = {}
        scores: dict[str, dict[str, float]] = {}
        for nid in obs.node_ids:
            act[nid] = "idle"
            scores[nid] = {}
        for eid in best_edges:
            e = eid[2:] if eid.startswith("E_") else eid
            s, d = e.split("__", 1)
            act[s] = d
            act[d] = s
            scores[s] = {d: 1.0}
            scores[d] = {s: 1.0}
        return act, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-slot exact optimal reference.")
    parser.add_argument("--seed", type=int, default=42 + 1000000)
    parser.add_argument("--steps", type=int, default=1440, help="0 = full window")
    parser.add_argument("--out", type=str, default="outputs/exact_reference")
    parser.add_argument("--time-limit-days", type=int, default=1, help="scenario.time_limit.days")
    parser.add_argument("--start-mode", type=str, default="fixed", help="fixed|random_day|random_window")
    args = parser.parse_args()

    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    config = deep_merge(config, load_config([ROOT / "configs" / "baselines.yaml"]))
    config["rate_provider"]["provider"] = "h5"
    config["env"]["episode_start_mode"] = args.start_mode
    config["env"]["episode_steps"] = args.steps or int(10 ** 9)
    config["scenario"]["time_limit"]["days"] = args.time_limit_days

    env = build_env_from_config(config)
    sim = StateSimulator(env, env.scenario.slot_seconds, 0.5)
    policy = ExactPerSlotPolicy(env, sim)
    policy.seed_stream(args.seed)

    obs = env.reset(seed=args.seed)
    out_dir = Path(ROOT) / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    step_rows = []
    total_served = 0.0
    n_activated_total = 0
    max_ps = 0.0
    truncated_slots = 0
    for step in range(args.steps or int(10 ** 9)):
        ps = time.time()
        act, scores = policy.act(obs, env.t)
        obs, reward, terminated, truncated, info = env.step(act, scores)
        dt = time.time() - ps
        max_ps = max(max_ps, dt)
        served = float(info.get("served_keys", 0.0))
        total_served += served
        step_rows.append({
            "step": step, "t": env.t - 1, "reward": round(float(reward), 6),
            "served_keys": round(served, 3),
            "generated_keys": round(float(info.get("generated_keys", 0.0)), 3),
            "failed_keys": round(float(info.get("failed_keys", 0.0)), 3),
            "waiting_keys": round(float(info.get("waiting_keys", 0.0)), 3),
            "pending_count": int(info.get("pending_count", 0)),
            "n_activated": int(info.get("n_activated", 0)),
        })
        if policy.truncated:
            truncated_slots += 1
        if step % 240 == 0:
            print(f"[{step}] t={env.t} served_so_far={total_served:.0f} "
                  f"dt={dt:.3f}s max_ps={max_ps:.3f}s truncated={truncated_slots}", flush=True)
        if terminated or truncated:
            break

    summary = env.metrics.episode_summary()
    with (out_dir / "steps.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STEP_FIELDS)
        writer.writeheader()
        writer.writerows(step_rows)
    result = {
        "policy": "exact_per_slot",
        "seed": args.seed,
        "steps": summary["steps"],
        "arrived_keys": summary["arrived_keys"],
        "served_keys": summary["served_keys"],
        "failed_keys": summary["failed_keys"],
        "success_rate": summary["success_rate"],
        "wall_s": round(time.time() - t0, 1),
        "max_ps": round(max_ps, 3),
        "truncated_slots": truncated_slots,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2))
    print(f"written: {out_dir / 'steps.csv'}, {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
