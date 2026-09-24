"""Offline global-view MILP that produces an EXECUTABLE per-slot link plan.

This policy solves a *window* of W slots jointly with full future knowledge
(offline H5 rates/availability + request arrivals previewed from the env's
own RequestGenerator seed, i.e. the god's-eye view). It decides, per slot,
which directed arcs to activate (dual-port: each node has one Tx and one Rx,
Tx-out <= 1, Rx-in <= 1, 对端不同 af+ab <= 1) and how many keys each request
is served, and it maximises the total served keys.

The default model (``relax_multi_path=False``) is **executable in the real
env**: every request is routed along the SAME path the env's routing would
pick (``routing.partial_consume_for_request`` uses the BFS shortest path with
positive key stock), the env's switch-cost decay is modelled exactly
(``switch_decay=0.5`` halves the rate of a newly activated link), per-hop
pool capacity and inventory relay are exact, so replaying the plan in the env
reproduces the planned success rate (plan SR == executed SR). This is the
plan used to guide RL behaviour cloning.

``relax_multi_path=True`` switches to the loose flow model (a request may be
served over several enumerated paths per slot, no single-path constraint, no
switch decay by default) — a valid *upper bound* on the achievable SR, not an
executable plan. Used only by ``compute_milp_upper_bound.py``.

Interface matches the other baselines: ``act(obs) -> (actions, scores)``.
"""

from __future__ import annotations

import csv
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import lil_matrix

from qkd_rl.core.types import KeyRequest
from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.request import RequestGenerator


def _highs_status_to_scipy(status_enum) -> int:
    """Map a HiGHS model status to the scipy.optimize.milp convention.

    0 = optimal, 1 = terminated with a solution (limit reached), 2 = infeasible,
    3 = unbounded, 4 = other error.
    """
    import highspy

    statuses = highspy.HighsModelStatus
    if status_enum == statuses.kOptimal:
        return 0
    if status_enum in (
        statuses.kTimeLimit, statuses.kIterationLimit, statuses.kSolutionLimit,
        statuses.kObjectiveBound, statuses.kObjectiveTarget, statuses.kInterrupt,
        statuses.kHighsInterrupt,
    ):
        return 1
    if status_enum == statuses.kInfeasible:
        return 2
    if status_enum == statuses.kUnbounded:
        return 3
    return 4


def _solve_highs(
    c: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    integrality: np.ndarray,
    matrix_csc,
    upper: np.ndarray,
    time_limit_s: float,
    mip_rel_gap: float,
    lower: np.ndarray | None = None,
) -> tuple[int, np.ndarray | None, float | None, float | None]:
    """Solve the minimisation MILP min c'x s.t. lower <= A x <= upper with HiGHS.

    When *lower* is None, all rows are -inf (i.e. A x <= upper).

    Returns ``(status, x, dual_bound, mip_gap)`` where ``dual_bound`` is the
    solver's best bound on the *minimisation* objective (always <= the best
    feasible objective, so ``-dual_bound`` is a valid **upper bound** on the
    served amount even when the time limit is hit) and ``mip_gap`` is the
    relative optimality gap (0.0 means optimality was proven).
    """
    import highspy

    model = highspy.Highs()
    model.setOptionValue("output_flag", False)
    model.setOptionValue("time_limit", float(time_limit_s))
    model.setOptionValue("mip_rel_gap", float(mip_rel_gap))

    n_vars = int(c.shape[0])
    n_rows = int(matrix_csc.shape[0])
    lp = model.getLp()
    lp.num_col_ = n_vars
    lp.num_row_ = n_rows
    lp.col_cost_ = [float(v) for v in c]
    lp.col_lower_ = [float(v) for v in lb]
    lp.col_upper_ = [float(v) for v in ub]
    lp.row_lower_ = [float(v) for v in (lower if lower is not None else [-float("inf")] * n_rows)]
    lp.row_upper_ = [float(v) for v in upper]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = [int(v) for v in matrix_csc.indptr]
    lp.a_matrix_.index_ = [int(v) for v in matrix_csc.indices]
    lp.a_matrix_.value_ = [float(v) for v in matrix_csc.data]
    lp.integrality_ = [
        highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous
        for v in integrality
    ]
    model.passModel(lp)
    model.run()

    status = _highs_status_to_scipy(model.getModelStatus())
    info = model.getInfo()
    mip_gap = float(getattr(info, "mip_gap", float("nan")))
    dual_bound = float(getattr(info, "mip_dual_bound", float("nan")))
    if np.isnan(dual_bound) or not np.isfinite(dual_bound):
        dual_bound = None
    solution = model.getSolution()
    if solution.col_value is None:
        return status, None, dual_bound, mip_gap
    return status, np.asarray(solution.col_value, dtype=np.float64), dual_bound, mip_gap


def _require_finite_capacities(
    capacity_map: dict[str, float] | None, edge_ids: list[str]
) -> dict[str, float]:
    """Return finite per-edge capacities or fail loudly.

    Treating a missing capacity as ``inf`` silently loosens the MILP and can
    invalidate an advertised upper bound, so every modeled edge must carry a
    real finite capacity.
    """
    raw = capacity_map or {}
    missing: list[str] = []
    invalid: list[str] = []
    out: dict[str, float] = {}
    for edge_id in edge_ids:
        if edge_id not in raw:
            missing.append(edge_id)
            continue
        try:
            value = float(raw[edge_id])
        except (TypeError, ValueError):
            invalid.append(edge_id)
            continue
        if not np.isfinite(value):
            invalid.append(edge_id)
            continue
        out[edge_id] = value
    if missing or invalid:
        parts = []
        if missing:
            parts.append(f"missing={missing[:8]}")
        if invalid:
            parts.append(f"non_finite_or_invalid={invalid[:8]}")
        raise ValueError(
            "MILP requires a finite QKP capacity for every modeled edge: "
            + "; ".join(parts)
        )
    return out


def _enumerate_paths(
    src: str,
    dst: str,
    adj: dict[str, list[str]],
    max_hops: int,
    max_paths: int,
) -> list[list[str]]:
    """Hop-bounded BFS keeping up to max_paths simple paths per node."""
    best: dict[str, list[list[str]]] = {src: [[src]]}
    for _ in range(max_hops):
        additions: dict[str, list[list[str]]] = {}
        for node, paths in best.items():
            for nxt in adj.get(node, ()):
                for path in paths:
                    if nxt in path or len(path) > max_hops:
                        continue
                    new_path = path + [nxt]
                    bucket = additions.setdefault(nxt, [])
                    if len(bucket) < max_paths:
                        bucket.append(new_path)
        added_any = False
        for node, new_paths in additions.items():
            bucket = best.setdefault(node, [])
            for path in new_paths:
                if len(bucket) < max_paths:
                    bucket.append(path)
                    added_any = True
        if not added_any:
            break
    return best.get(dst, [])[:max_paths]


@dataclass
class WindowOutcome:
    activated_edges: list[tuple[str, str]]  # directed arcs (src, dst) at slot 0
    served_amount: float
    solve_time_s: float
    status: int
    flow_by_request: dict[str, float] = field(default_factory=dict)
    # True upper bound on served keys: -mip_dual_bound of the minimisation
    # problem (valid even when status == 1 / time limit was hit). Maps to
    # served_amount when status == 0 (optimality proven).
    upper_bound_amount: float = 0.0
    # Relative MIP optimality gap reported by HiGHS (0.0 == proven optimal).
    mip_gap: float = 0.0
    # Full-window activation plan: relative slot -> directed arcs (src, dst).
    # The policy executes this plan open-loop (no re-solve mid-window), because
    # a multi-hop path needs inventory stocked across slots; re-solving every
    # step would discard the future activations before they execute.
    activation_plan: dict[int, list[tuple[str, str]]] = field(default_factory=dict)
    # Exact per-(request, path, slot) flow for the ideal-execution upper bound:
    # (request_id, path edge indices, relative slot, amount).
    flow_detail: list[tuple[str, list[int], int, float]] = field(default_factory=list)
    # Per-edge inventory at the end of the window (last slot), for cross-window
    # carry-over: {edge_id: inventory_amount}
    ending_inventory: dict[str, float] = field(default_factory=dict)
    # Sum of remaining demand of the requests the MILP actually modelled after
    # dropping pathless requests (use_request filter). The upper-bound success
    # rate must divide by this — not the raw arrived demand — so the produced
    # bound matches the servable demand set the MILP can address.
    modeled_amount: float = 0.0


class RecedingHorizonMILPPolicy:
    """Offline upper bound: sliding-window MILP with future knowledge."""

    def __init__(
        self,
        config: dict,
        slot_seconds: float = 60.0,
        window_steps: int = 60,
        max_requests: int = 64,
        max_paths_per_request: int = 8,
        max_path_hops: int = 6,
        time_limit_s: float = 60.0,
        mip_rel_gap: float = 0.0,
        replan_every_steps: int | None = None,
        switch_decay: float = 1.0,
        single_path: bool = False,
        final_inventory_weight: float = 1.0e-4,
        max_edges: int = 2000,
        relax_multi_path: bool = False,
    ):
        self.config = config
        self.slot_seconds = float(slot_seconds)
        self.window_steps = int(window_steps)
        self.max_requests = int(max_requests)
        self.max_paths_per_request = int(max_paths_per_request)
        self.max_path_hops = int(max_path_hops)
        self.time_limit_s = float(time_limit_s)
        self.mip_rel_gap = float(mip_rel_gap)
        # Rolling-horizon cadence: re-solve the window every N executed steps
        # (None = only at window boundaries, i.e. open-loop plan execution).
        # For behavior-cloning demos use 1: the action is then the first step
        # of a fresh solve over the *actual* env state, so ideal-flow vs
        # env-execution mismatch never accumulates across steps.
        self.replan_every_steps = (int(replan_every_steps) if replan_every_steps else None)
        # env switch cost: a link activated this slot but not the previous one
        # generates at rate_decay_factor (0.5). The ideal upper bound ignores
        # it (1.0); for executable plans / demos pass 0.5 so the solver plans
        # with the reduced rate — otherwise executed SR falls far below the
        # ideal bound (measured ~0.4 vs >=1.0 on 120-step rollouts).
        self.switch_decay = float(switch_decay)
        # env serves each request along ONE path per slot (routing is
        # hard-wired single-path BFS). Only relevant in the relaxed upper-bound
        # model; the executable model always uses one fixed path per request.
        self.single_path = bool(single_path)
        # Loose flow model (upper bound, NOT executable): a request may be
        # served over several enumerated paths per slot. Default False = the
        # executable model (one fixed BFS path per request, switch decay exact).
        self.relax_multi_path = bool(relax_multi_path)
        self.final_inventory_weight = float(final_inventory_weight)
        self.max_edges = int(max_edges)

        self._gs_ids: list[str] | None = None
        self._node_type: dict[str, str] = {}
        self._req_cfg = dict(config.get("requests", {}))
        self._seed = int(config["seed"]["env_seed"])
        self._provider = None
        self._edge_ids: list[str] = []
        self._edge_link_ids: np.ndarray | None = None
        self.last_outcome: WindowOutcome | None = None
        self.window_solve_s = 0.0
        # Open-loop window plan: absolute t -> activated edges. Re-solved only
        # at window boundaries so multi-slot stocking actually executes.
        self._plan_t0: int | None = None
        self._plan: dict[int, list[tuple[str, str]]] = {}

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _parse_edge(edge_id: str) -> tuple[str | None, str | None]:
        """(src, dst) of an edge id like 'E_GS_001__SAT_001' (None if malformed)."""
        body = edge_id[2:] if edge_id.startswith("E_") else edge_id
        if "__" in body:
            u, v = body.split("__", 1)
            return u, v
        return None, None

    @staticmethod
    def _shortest_path(src: str, dst: str, adj: dict[str, list[str]]) -> list[str] | None:
        """BFS shortest node path — the SAME path the env's routing prefers
        (``routing._usable_cached_path`` / ``_find_positive_path`` both run a
        shortest-path BFS over usable edges). Returns the node list including
        both endpoints, or None when disconnected."""
        if src == dst:
            return []
        parent: dict[str, str | None] = {src: None}
        queue: deque[str] = deque([src])
        while queue:
            node = queue.popleft()
            if node == dst:
                path: list[str] = []
                cur: str | None = node
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for nxt in adj.get(node, ()):
                if nxt in parent:
                    continue
                parent[nxt] = node
                queue.append(nxt)
        return None

    def _load_node_types(self) -> None:
        if self._provider is None or self._node_type:
            return
        path = getattr(self._provider, "node_registry_path", None)
        if path is None or not path.exists():
            return
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                self._node_type[row["name"].strip()] = row["type"].strip().upper()

    def _ensure_init(self, obs, gen=None) -> None:
        if self._gs_ids is not None:
            return
        self._provider = getattr(getattr(obs.state, "edge_windows", None), "_provider", None)
        self._load_node_types()
        self._gs_ids = sorted(
            node_id for node_id in obs.node_ids if self._node_type.get(node_id) == "GS"
        )
        if not self._gs_ids:
            raise RuntimeError("RecedingHorizonMILPPolicy: no GS nodes found in observation")
        # Edge set = ALL scenario edges (from the provider registry), not the
        # current-slot legal subset: an offline upper bound must be able to
        # activate a link that becomes available inside the window. Per-slot
        # availability is enforced by the avail constraint on x, and
        # solve_window narrows to edges available at least once in the window.
        windows = getattr(obs.state, "edge_windows", None)
        all_edges = list(windows.keys()) if hasattr(windows, "keys") else []
        if not all_edges:
            all_edges = list(obs.generation_edge_ids)
        self._edge_ids = all_edges[: self.max_edges]
        if self._provider is not None:
            self._edge_link_ids = self._provider.edge_link_ids(self._edge_ids)
        # RequestGenerator RNG state depends on the call history from the
        # seed, and the env advances it one slot per step from t=0. The preview
        # generator must therefore fast-forward identically: generate 0..t0-1
        # (discard), then cache arrivals from t0 on.
        if gen is not None:
            # Reuse an external generator (e.g. the env's) whose RNG is already
            # at the right position — skips the O(t) fast-forward entirely.
            self._gen = gen
        else:
            self._gen = RequestGenerator(self._gs_ids, self._req_cfg, self._seed)
            for t in range(0, int(obs.state.t)):
                self._gen.generate(t)
        self._gen_t = int(obs.state.t)
        self._arrivals_cache: list[tuple[int, KeyRequest]] = []

    def _preview_requests(self, t0: int, horizon: int) -> list[tuple[int, KeyRequest]]:
        """Future arrivals in [t0, t0+horizon), previewed from the same seed
        the env uses (identical RNG call sequence), so the preview equals the
        requests the env will actually create. This is the offline "known
        future" oracle; arrivals are cached so repeated windows reuse them.
        """
        target = t0 + horizon
        while self._gen_t < target:
            for req in self._gen.generate(self._gen_t):
                self._arrivals_cache.append((self._gen_t, req))
            self._gen_t += 1
        return [item for item in self._arrivals_cache if t0 <= item[0] < target]

    # ------------------------------------------------------------------ solve
    def _window_rates_avail(self, t0: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """(rates, available) matrices of shape (horizon, n_edges) for [t0, t0+horizon)."""
        if self._provider is None or self._edge_link_ids is None:
            raise RuntimeError("RecedingHorizonMILPPolicy requires an H5 rate provider")
        n = len(self._edge_ids)
        rates = np.zeros((horizon, n), dtype=np.float64)
        avail = np.zeros((horizon, n), dtype=np.float64)
        idx = self._edge_link_ids
        for tau in range(horizon):
            blocks = self._provider.get_window_blocks(t0 + tau)
            rates[tau] = blocks[0][0][idx]
            avail[tau] = blocks[1][0][idx].astype(np.float64)
        return rates, avail

    def solve_window(self, obs) -> WindowOutcome:
        t0 = int(obs.state.t)
        W = self.window_steps
        t_start = time.perf_counter()

        # ---- edge set + future rates/availability ---------------------------
        # All scenario edges, narrowed to those available at least once inside
        # the window.  A link that never becomes available can never generate
        # keys, so it cannot help serve; keeping it only inflates the MILP.
        rates_all, avail_all = self._window_rates_avail(t0, W)
        window_avail = avail_all.max(axis=0) >= 0.5
        full_idx = {eid: i for i, eid in enumerate(self._edge_ids)}
        edges = [eid for eid, ok in zip(self._edge_ids, window_avail) if ok]
        if len(edges) == 0:
            edges = list(self._edge_ids)  # degenerate fallback
        sel = np.asarray([full_idx[e] for e in edges], dtype=np.int64)
        n_edges = len(edges)
        rates = rates_all[:, sel]  # (W, n_edges)
        avail = avail_all[:, sel]  # (W, n_edges)
        edge_endpoints = [self._parse_edge(eid) for eid in edges]
        cap = _require_finite_capacities(obs.state.qkp_capacity, edges)

        # ---- request set ------------------------------------------------------
        pending: list[KeyRequest] = [
            req
            for req in obs.state.pending_requests
            if req.src_gs in obs.node_ids and req.dst_gs in obs.node_ids
            and req.amount - req.served_amount > 1.0e-9
        ]
        future = self._preview_requests(t0, W)
        requests: list[KeyRequest] = list(pending) + [req for _t, req in future]
        requests.sort(key=lambda req: (-(req.amount - req.served_amount), req.deadline_t))
        requests = requests[: self.max_requests]
        remaining = [float(req.amount - req.served_amount) for req in requests]
        arrival_rel = [max(0, int(req.arrival_t) - t0) for req in requests]
        deadline_rel = [min(W, max(0, int(req.deadline_t) - t0)) for req in requests]

        # ---- adjacency --------------------------------------------------------
        # adj_all mirrors the env's static topology (RoutingPolicy.adj over ALL
        # scenario edges); adj_win is the subgraph whose edges can stock keys
        # inside the window (available at least once). pair_to_edge_* map the
        # sorted node pair back to the edge id.
        edge_idx = {eid: i for i, eid in enumerate(edges)}
        adj_all: dict[str, list[str]] = {node: [] for node in obs.node_ids}
        pair_to_edge_all: dict[tuple[str, str], str] = {}
        for eid in self._edge_ids:
            u, v = self._parse_edge(eid)
            if u is None:
                continue
            adj_all.setdefault(u, []).append(v)
            adj_all.setdefault(v, []).append(u)
            pair_to_edge_all[tuple(sorted((u, v)))] = eid
        adj_win: dict[str, list[str]] = {node: [] for node in obs.node_ids}
        pair_to_edge_win: dict[tuple[str, str], str] = {}
        for e, eid in enumerate(edges):
            u, v = edge_endpoints[e]
            if u is None:
                continue
            adj_win.setdefault(u, []).append(v)
            adj_win.setdefault(v, []).append(u)
            pair_to_edge_win[tuple(sorted((u, v)))] = eid

        # ---- paths per request ------------------------------------------------
        # Executable model (default): ONE fixed path per request, chosen exactly
        # like the env routes — the BFS shortest path. First try the static
        # shortest path (RoutingPolicy.shortest_path: the env's cached
        # preference) whenever every hop can stock keys inside the window;
        # otherwise fall back to the shortest path inside the window-available
        # subgraph (the env's positive-subgraph BFS). Routing over any other
        # path would NOT be reproducible by the env, so it is excluded.
        # Relaxed model (upper bound): enumerate up to max_paths_per_request
        # hop-bounded simple paths per request (multiple may be used per slot).
        max_plan_hops = max(self.max_path_hops, 6) + 6  # slack vs env's unbounded BFS
        request_paths: list[list[list[int]]] = []
        if not self.relax_multi_path:
            for req in requests:
                path: list[str] | None = None
                sp = self._shortest_path(req.src_gs, req.dst_gs, adj_all)
                if sp is not None and len(sp) - 1 <= max_plan_hops:
                    eids = [
                        pair_to_edge_all[tuple(sorted((sp[i], sp[i + 1])))]
                        for i in range(len(sp) - 1)
                    ]
                    if all(e in pair_to_edge_win for e in eids):
                        path = sp
                if path is None:
                    sp = self._shortest_path(req.src_gs, req.dst_gs, adj_win)
                    if sp is not None and len(sp) - 1 <= max_plan_hops:
                        path = sp
                if path is None:
                    request_paths.append([])
                    continue
                eids = [
                    pair_to_edge_win[tuple(sorted((path[i], path[i + 1])))]
                    for i in range(len(path) - 1)
                ]
                request_paths.append([[edge_idx[e] for e in eids]])
        else:
            for req in requests:
                paths = _enumerate_paths(req.src_gs, req.dst_gs, adj_win, self.max_path_hops, self.max_paths_per_request)
                kept: list[list[int]] = []
                for path in paths:
                    eids = [pair_to_edge_win[tuple(sorted((path[i], path[i + 1])))] for i in range(len(path) - 1)]
                    indices = [edge_idx.get(e, -1) for e in eids]
                    if all(ei >= 0 for ei in indices):
                        kept.append(indices)
                request_paths.append(kept)
        use_request = [bool(rp) for rp in request_paths]
        request_paths = [rp for rp, ok in zip(request_paths, use_request) if ok]
        requests = [r for r, ok in zip(requests, use_request) if ok]
        remaining = [r for r, ok in zip(remaining, use_request) if ok]
        arrival_rel = [a for a, ok in zip(arrival_rel, use_request) if ok]
        deadline_rel = [d for d, ok in zip(deadline_rel, use_request) if ok]
        # The env serves a request at t == deadline_t too (serve() runs before
        # expire() in env.step), so the last servable slot is deadline_rel + 1
        # (clamped to the window end).
        last_serve_rel = [min(d + 1, W) for d in deadline_rel]
        n_req = len(requests)
        if n_edges == 0 or n_req == 0:
            return WindowOutcome([], 0.0, time.perf_counter() - t_start, -1)

        # ---- variables ---------------------------------------------------------
        # A directed arc uses one node's Tx and another's Rx. Each activated
        # undirected pair can take at most one direction per slot (对端不同):
        #   af[e, tau] = 1 means edges[e].src -> edges[e].dst
        #   ab[e, tau] = 1 means edges[e].dst -> edges[e].src
        # Key-pool inventory is per undirected pair, so generation counts the
        # sum af+ab. Only arcs with the pair available get a binary variable.
        # s[e, tau] end-of-slot inventory (continuous); f[r, p, tau] keys
        # served to r over path p at slot tau.
        x_avail = avail >= 0.5  # (W, n_edges) bool
        sv: dict[tuple[int, int], int] = {}
        afv: dict[tuple[int, int], int] = {}
        abv: dict[tuple[int, int], int] = {}
        fv: dict[tuple[int, int, int], int] = {}
        var_count = 0
        for e in range(n_edges):
            for tau in range(W):
                sv[(e, tau)] = var_count
                var_count += 1
        for e in range(n_edges):
            for tau in range(W):
                if x_avail[tau, e]:
                    afv[(e, tau)] = var_count
                    var_count += 1
                    abv[(e, tau)] = var_count
                    var_count += 1
        # Switch-decay indicator z[e,tau]: 1 if the *pair* e is newly activated
        # at slot tau (either direction active now, none active last slot).
        # Continuous [0,1] with linearization constraints. A pair is "active"
        # when af+ab == 1 (对端不同 ensures the sum is 0 or 1).
        zv: dict[tuple[int, int], int] = {}
        if self.switch_decay < 1.0:
            for e in range(n_edges):
                for tau in range(W):
                    if (e, tau) in afv and (e, tau - 1) in afv:
                        zv[(e, tau)] = var_count
                        var_count += 1
        if self.relax_multi_path:
            # Loose upper-bound model: f[r,p,tau] keys served to r over path p
            # at slot tau (multiple paths per slot allowed unless single_path).
            fv: dict[tuple[int, int, int], int] = {}
            for r in range(n_req):
                for p in range(len(request_paths[r])):
                    for tau in range(arrival_rel[r], last_serve_rel[r]):
                        fv[(r, p, tau)] = var_count
                        var_count += 1
        else:
            # Executable model: f[r,tau] keys served to r at slot tau over its
            # ONE fixed path (chosen to match the env's routing exactly).
            fv: dict[tuple[int, int], int] = {}
            for r in range(n_req):
                for tau in range(arrival_rel[r], last_serve_rel[r]):
                    fv[(r, tau)] = var_count
                    var_count += 1
        # Single-path flag y[r,p,tau]: 1 if request r is served over path p at
        # slot tau (relaxed upper-bound model only). The executable model has a
        # single fixed path per request, so no y is needed.
        yv: dict[tuple[int, int, int], int] = {}
        if self.relax_multi_path and self.single_path:
            for (r, p, tau) in fv:
                yv[(r, p, tau)] = var_count
                var_count += 1
        n_vars = var_count

        integrality = np.zeros(n_vars, dtype=np.int8)
        lb = np.zeros(n_vars, dtype=np.float64)
        ub = np.full(n_vars, np.inf, dtype=np.float64)
        c = np.zeros(n_vars, dtype=np.float64)  # minimise negative objective
        # Both directed arcs are binary 0/1; per-arc availability < 0.999 is
        # enforced with an explicit row below (>= 0.999 already by ub).
        for var in afv.values():
            integrality[var] = 1
            ub[var] = 1.0
        for var in abv.values():
            integrality[var] = 1
            ub[var] = 1.0
        for var in zv.values():
            ub[var] = 1.0
        for var in yv.values():
            integrality[var] = 1
            ub[var] = 1.0
        if self.relax_multi_path:
            for (r, p, tau), var in fv.items():
                ub[var] = remaining[r]
                c[var] = -1.0  # maximise served keys
        else:
            for (r, tau), var in fv.items():
                ub[var] = remaining[r]
                c[var] = -1.0  # maximise served keys

        # ---- constraints --------------------------------------------------------
        constraints: list[tuple[np.ndarray, np.ndarray, float, float]] = []  # A_row x, lower <= A x <= upper
        gen = rates * self.slot_seconds  # (W, E); no switch decay (upper bound)
        init_level = obs.state.qkp_snapshot

        # per-arc availability: slots with avail < 0.5 get no arc variable (fixed
        # 0); avail >= 0.999 is enforced by ub; rare fractional [0.5, 0.999) get
        # an explicit row on both arc directions (they share the pair's avail).
        for (e, tau), var in afv.items():
            if avail[tau, e] < 0.999:
                constraints.append((np.array([1.0]), np.array([var]), -float("inf"), float(avail[tau, e])))
        for (e, tau), var in abv.items():
            if avail[tau, e] < 0.999:
                constraints.append((np.array([1.0]), np.array([var]), -float("inf"), float(avail[tau, e])))
        # 对端不同: the same undirected pair activates at most ONE direction per
        # slot (af + ab <= 1). This also guarantees generation count = af+ab.
        for (e, tau) in afv:
            constraints.append(
                (np.array([1.0, 1.0]),
                 np.array([afv[(e, tau)], abv[(e, tau)]], dtype=np.int64),
                 -float("inf"), 1.0)
            )
        # Dual-port directed matching:
        #   * Tx-out <= 1 : a node's transmitter emits at most one outgoing arc.
        #   * Rx-in  <= 1 : at most one arc points INTO a node (its receiver).
        #   * The two are independent, so a node may simultaneously transmit one
        #     arc and receive another (1 Tx-out + 1 Rx-in same slot).
        tx_out_vars: dict[tuple[str, int], list[int]] = {}
        rx_in_vars: dict[tuple[str, int], list[int]] = {}
        for e in range(n_edges):
            u, v = edge_endpoints[e]
            if u is None:
                continue
            for tau in range(W):
                if (e, tau) not in afv:
                    continue
                af_var, ab_var = afv[(e, tau)], abv[(e, tau)]
                # af: u --Tx--> v  (u emits, v receives)
                tx_out_vars.setdefault((u, tau), []).append(af_var)
                rx_in_vars.setdefault((v, tau), []).append(af_var)
                # ab: v --Tx--> u  (v emits, u receives)
                tx_out_vars.setdefault((v, tau), []).append(ab_var)
                rx_in_vars.setdefault((u, tau), []).append(ab_var)
        for lookup in (tx_out_vars, rx_in_vars):
            for cols in lookup.values():
                if cols:
                    constraints.append((np.ones(len(cols)), np.array(cols, dtype=np.int64), -float("inf"), 1.0))
        # switch-decay linearization: pair active = af+ab (a slot has at most one
        # direction).  z[e,tau] == (af+ab)[e,tau] * (1 - (af+ab)[e,tau-1]):
        #   z <= af_cur+ab_cur ; z + af_prev+ab_prev <= 1 ; z-af_cur-ab_cur+af_prev+ab_prev >= 0
        for (e, tau), var in zv.items():
            cur = [afv[(e, tau)], abv[(e, tau)]]
            prev = [afv[(e, tau - 1)], abv[(e, tau - 1)]]
            constraints.append((np.array([1.0, -1.0, -1.0]),
                                np.array([var, cur[0], cur[1]], dtype=np.int64),
                                -float("inf"), 0.0))
            constraints.append((np.array([1.0, 1.0, 1.0]),
                                np.array([var, prev[0], prev[1]], dtype=np.int64),
                                -float("inf"), 1.0))
            constraints.append((np.array([1.0, -1.0, -1.0, 1.0, 1.0]),
                                np.array([var, cur[0], cur[1], prev[0], prev[1]], dtype=np.int64),
                                0.0, float("inf")))
        # inventory dynamics: s[e,tau] = s[e,tau-1] + gen*x - serve, s >= 0
        # serve on edge e at tau = sum of f over (r,p) whose path uses e.
        serve_cols: dict[tuple[int, int], list[int]] = {(e, tau): [] for e in range(n_edges) for tau in range(W)}
        if self.relax_multi_path:
            for r in range(n_req):
                for p in range(len(request_paths[r])):
                    for tau in range(arrival_rel[r], last_serve_rel[r]):
                        for e in request_paths[r][p]:
                            serve_cols[(e, tau)].append(fv[(r, p, tau)])
        else:
            for r in range(n_req):
                for tau in range(arrival_rel[r], last_serve_rel[r]):
                    for e in request_paths[r][0]:
                        serve_cols[(e, tau)].append(fv[(r, tau)])
        # s[e,tau] - gen[e,tau]*(af+ab)[e,tau] + serve[e,tau] - s[e,tau-1] = init (tau=0) / 0 (tau>0)
        # NOTE: equality constraint (lower == upper) so inventory cannot be
        # artificially lost — keys must be accounted for exactly. Because 对端
        # 不同 bounds af+ab <= 1, both directions cannot generate in the same
        # slot, and generation = gen*(af+ab).
        for e in range(n_edges):
            init_e = float(init_level.get(edges[e], 0.0))
            for tau in range(W):
                row = {sv[(e, tau)]: 1.0}
                if (e, tau) in afv:
                    af_var, ab_var = afv[(e, tau)], abv[(e, tau)]
                    if self.switch_decay < 1.0:
                        if tau == 0:
                            # Relative to the outside world: keep-active links
                            # (activated in the previous env step) generate at
                            # full rate, newly activated ones at decay.
                            keep = set(getattr(obs.state, "last_activated_edges", None) or [])
                            gen_eff = gen[tau, e] if edges[e] in keep else gen[tau, e] * self.switch_decay
                            row[af_var] = -gen_eff
                            row[ab_var] = -gen_eff
                        elif (e, tau - 1) in afv:
                            # z models the switch inside the window
                            row[af_var] = -gen[tau, e]
                            row[ab_var] = -gen[tau, e]
                            row[zv[(e, tau)]] = gen[tau, e] * (1.0 - self.switch_decay)
                        else:
                            # previous slot unavailable -> definitely a switch
                            row[af_var] = -gen[tau, e] * self.switch_decay
                            row[ab_var] = -gen[tau, e] * self.switch_decay
                    else:
                        row[af_var] = -gen[tau, e]
                        row[ab_var] = -gen[tau, e]
                if tau > 0:
                    row[sv[(e, tau - 1)]] = -1.0
                for v in serve_cols[(e, tau)]:
                    row[v] = row.get(v, 0.0) + 1.0
                cols = np.array(list(row.keys()), dtype=np.int64)
                coeffs = np.array(list(row.values()), dtype=np.float64)
                rhs = init_e if tau == 0 else 0.0
                constraints.append((coeffs, cols, rhs, rhs))  # equality: lower == upper
        # inventory <= capacity
        for e in range(n_edges):
            cap_e = float(cap.get(edges[e], np.inf))
            if np.isfinite(cap_e):
                for tau in range(W):
                    constraints.append((np.array([1.0]), np.array([sv[(e, tau)]]), -float("inf"), cap_e))
        # per request: total served <= remaining
        for r in range(n_req):
            if self.relax_multi_path:
                cols = [
                    fv[(r, p, tau)]
                    for p in range(len(request_paths[r]))
                    for tau in range(arrival_rel[r], last_serve_rel[r])
                ]
            else:
                cols = [
                    fv[(r, tau)]
                    for tau in range(arrival_rel[r], last_serve_rel[r])
                ]
            constraints.append((np.ones(len(cols)), np.array(cols, dtype=np.int64), -float("inf"), remaining[r]))
        # single-path semantics (env serves one path per request per slot):
        #   f[r,p,tau] <= remaining[r] * y[r,p,tau]  and  sum_p y[r,p,tau] <= 1
        if self.single_path and yv:
            for (r, p, tau), fvar in fv.items():
                constraints.append(
                    (np.array([1.0, -remaining[r]]),
                     np.array([fvar, yv[(r, p, tau)]], dtype=np.int64),
                     -float("inf"), 0.0)
                )
            by_r_tau: dict[tuple[int, int], list[int]] = {}
            for (r, p, tau), var in yv.items():
                by_r_tau.setdefault((r, tau), []).append(var)
            for (r, tau), cols in by_r_tau.items():
                constraints.append((np.ones(len(cols)), np.array(cols, dtype=np.int64), -float("inf"), 1.0))
        # end-of-window inventory value in the objective
        for e in range(n_edges):
            c[sv[(e, W - 1)]] = -self.final_inventory_weight

        n_cons = len(constraints)
        matrix = lil_matrix((n_cons, n_vars), dtype=np.float64)
        upper = np.full(n_cons, np.inf, dtype=np.float64)
        lower = np.full(n_cons, -float("inf"), dtype=np.float64)
        for row_idx, (coeffs, cols, lb_row, ub_row) in enumerate(constraints):
            matrix[row_idx, cols] = coeffs
            lower[row_idx] = lb_row
            upper[row_idx] = ub_row

        status, x, dual_bound, mip_gap = _solve_highs(
            c=c, lb=lb, ub=ub, integrality=integrality,
            matrix_csc=matrix.tocsc(), upper=upper, lower=lower,
            time_limit_s=self.time_limit_s, mip_rel_gap=self.mip_rel_gap,
        )
        # The dual bound is a bound on the *minimisation* objective, so
        # -dual_bound is a valid upper bound on the served amount even when
        # the time limit was hit before optimality was proven.
        upper_bound = -float(dual_bound) if dual_bound is not None else 0.0
        import os as _os
        if _os.environ.get("RH_DEBUG"):
            n_act = sum(1 for var in list(afv.values()) + list(abv.values()) if x is not None and x[var] >= 0.5)
            f_sum = sum(float(x[var]) for var in fv.values()) if x is not None else -1.0
            print(f"[RH_DEBUG] t={t0} n_edges={n_edges} n_req={n_req} n_arcs={len(afv)} n_f={len(fv)} "
                  f"n_vars={n_vars} n_cons={n_cons} status={status} x_act={n_act} f_sum={f_sum:.1f} "
                  f"ub={upper_bound:.1f} gap={mip_gap:.4f}", flush=True)
        if x is None:
            return WindowOutcome([], 0.0, time.perf_counter() - t_start, status,
                                 upper_bound_amount=upper_bound, mip_gap=float(mip_gap))

        # Directed arcs activated in each slot (including slot 0). Both directions
        # of a pair are looked up; 对端不同 guarantees at most one is set.
        def _arcs_of(tau: int) -> list[tuple[str, str]]:
            out: list[tuple[str, str]] = []
            for e in range(n_edges):
                u, v = edge_endpoints[e]
                if u is None or (e, tau) not in afv:
                    continue
                if x[afv[(e, tau)]] >= 0.5:
                    out.append((u, v))
                if x[abv[(e, tau)]] >= 0.5:
                    out.append((v, u))
            return out

        activation_plan: dict[int, list[tuple[str, str]]] = {
            tau: _arcs_of(tau) for tau in range(W)
        }
        activated: list[tuple[str, str]] = _arcs_of(0) if W > 0 else []
        total_served = 0.0
        flow_by_request: dict[str, float] = {}
        flow_detail: list[tuple[str, list[int], int, float]] = []
        for r in range(n_req):
            total = 0.0
            if self.relax_multi_path:
                for p in range(len(request_paths[r])):
                    for tau in range(arrival_rel[r], last_serve_rel[r]):
                        amt = float(x[fv[(r, p, tau)]])
                        if amt > 1.0e-9:
                            # Path indices are reported against the FULL edge set
                            # (self._edge_ids) so ideal-execution consumers index
                            # inventory consistently.
                            path_full = [full_idx[edges[ei]] for ei in request_paths[r][p]]
                            flow_detail.append((requests[r].request_id, path_full, tau, amt))
                            total += amt
            else:
                for tau in range(arrival_rel[r], last_serve_rel[r]):
                    amt = float(x[fv[(r, tau)]])
                    if amt > 1.0e-9:
                        path_full = [full_idx[edges[ei]] for ei in request_paths[r][0]]
                        flow_detail.append((requests[r].request_id, path_full, tau, amt))
                        total += amt
            if total > 1.0e-6:
                flow_by_request[requests[r].request_id] = total
            total_served += total
        # Ending inventory per edge (last slot of the window) for cross-window carry-over
        ending_inventory = {
            edges[e]: float(x[sv[(e, W - 1)]])
            for e in range(n_edges) if x[sv[(e, W - 1)]] >= 1.0
        }
        outcome = WindowOutcome(
            activated_edges=activated,
            served_amount=float(total_served),
            solve_time_s=time.perf_counter() - t_start,
            status=status,
            flow_by_request=flow_by_request,
            upper_bound_amount=upper_bound,
            mip_gap=float(mip_gap),
            activation_plan=activation_plan,
            flow_detail=flow_detail,
            ending_inventory=ending_inventory,
            modeled_amount=float(sum(remaining)),
        )
        self.last_outcome = outcome
        self.window_solve_s = outcome.solve_time_s
        return outcome

    # ------------------------------------------------------------------ public
    def act(self, obs) -> tuple[dict[str, tuple[str, str]], dict[str, dict[str, float]]]:
        """Return slot-0 actions of the current window plan; re-solve on the
        configured cadence.

        Open-loop (``replan_every_steps=None``): re-solve only at window
        boundaries and execute the whole window plan. Rolling horizon
        (``replan_every_steps=N``): re-solve at least every N executed steps
        against the *actual* env state, so ideal-flow / env-execution mismatch
        does not accumulate — the emitted action is the first step of a fresh
        optimal plan for the current state.
        """
        self._ensure_init(obs)
        t = int(obs.state.t)
        replan = (
            self._plan_t0 is None
            or t < self._plan_t0
            or t >= self._plan_t0 + self.window_steps
            or (self.replan_every_steps is not None and t >= self._plan_t0 + self.replan_every_steps)
        )
        if replan:
            outcome = self.solve_window(obs)
            self._plan_t0 = t
            self._plan = {t + tau: arcs for tau, arcs in outcome.activation_plan.items()}
        # Directed arcs the plan activates at the current slot. Each arc uses
        # src's Tx and dst's Rx, so a node may simultaneously emit one arc and
        # receive another (dual port). Invalid/illegal arcs are dropped below.
        plan_arcs = list(self._plan.get(t, []))
        tx_target: dict[str, str] = {}
        rx_source: dict[str, str] = {}
        for src, dst in plan_arcs:
            tx_target.setdefault(src, dst)
            rx_source.setdefault(dst, src)
        obs_pair_to_edge = {
            tuple(sorted((u, v))): eid
            for eid in obs.generation_edge_ids
            for u, v in [self._parse_edge(eid)]
            if u is not None
        }
        actions: dict[str, tuple[str, str]] = {}
        scores: dict[str, dict[str, float]] = {}
        for node_id in obs.node_ids:
            tx = tx_target.get(node_id, NodeActionSpace.IDLE)
            rx = rx_source.get(node_id, NodeActionSpace.IDLE)
            # Drop illegal targets (an arc over a link not legal/visible now).
            legal = obs_pair_to_edge.get(tuple(sorted((node_id, tx)))) if tx != NodeActionSpace.IDLE else None
            if tx != NodeActionSpace.IDLE and legal is None:
                tx = NodeActionSpace.IDLE
            legal_rx = obs_pair_to_edge.get(tuple(sorted((node_id, rx)))) if rx != NodeActionSpace.IDLE else None
            if rx != NodeActionSpace.IDLE and legal_rx is None:
                rx = NodeActionSpace.IDLE
            actions[node_id] = (tx, rx)
            scores[node_id] = {NodeActionSpace.IDLE: 0.0}
        return actions, scores
