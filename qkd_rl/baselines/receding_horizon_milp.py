"""Receding-horizon offline MILP upper-bound baseline.

This policy solves a *window* of W slots jointly: it knows the future link
rates / availability (from the offline H5 dataset) and the future request
arrivals (previewed with a RequestGenerator seeded identically to the env), so
it can decide to activate links *now* to stock keys for later slots. The
window then slides forward and the solve repeats. This is the standard
engineering approximation of the offline global optimum for a problem whose
full-day MILP (millions of integer variables) is intractable.

The window MILP maximises total served keys over the window plus a small
weighted value of the end-of-window inventory (so stocking for the future is
rewarded). Intentional upper-bound relaxations (each makes the bound looser,
never tighter, so the result stays a valid ceiling):

- activation switch-cost rate decay is ignored (env halves the rate of a
  newly activated link);
- key TTL expiry inside the window is ignored (TTL=1920 >> window);
- a request may be served over several paths simultaneously (env serves along
  one path per request per slot);
- only currently-legal edges are in the edge set (links that only become
  visible inside the window are excluded; keep max_edges bounded).

Interface matches the other baselines: ``act(obs) -> (actions, scores)``.
"""

from __future__ import annotations

import csv
import time
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
) -> tuple[int, np.ndarray | None]:
    """Solve the minimisation MILP min c'x s.t. A x <= upper with HiGHS."""
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
    lp.row_lower_ = [-float("inf")] * n_rows
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
    solution = model.getSolution()
    if solution.col_value is None:
        return status, None
    return status, np.asarray(solution.col_value, dtype=np.float64)


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
    activated_edges: list[str]
    served_amount: float
    solve_time_s: float
    status: int
    flow_by_request: dict[str, float] = field(default_factory=dict)
    # Full-window activation plan: relative slot -> activated edge ids. The
    # policy executes this plan open-loop (no re-solve mid-window), because a
    # multi-hop path needs inventory stocked across slots; re-solving every
    # step would discard the future activations before they execute.
    activation_plan: dict[int, list[str]] = field(default_factory=dict)
    # Exact per-(request, path, slot) flow for the ideal-execution upper bound:
    # (request_id, path edge indices, relative slot, amount).
    flow_detail: list[tuple[str, list[int], int, float]] = field(default_factory=list)
    # Per-edge inventory at the end of the window (last slot), for cross-window
    # carry-over: {edge_id: inventory_amount}
    ending_inventory: dict[str, float] = field(default_factory=dict)


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
        final_inventory_weight: float = 1.0e-4,
        max_edges: int = 2000,
    ):
        self.config = config
        self.slot_seconds = float(slot_seconds)
        self.window_steps = int(window_steps)
        self.max_requests = int(max_requests)
        self.max_paths_per_request = int(max_paths_per_request)
        self.max_path_hops = int(max_path_hops)
        self.time_limit_s = float(time_limit_s)
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
        self._plan: dict[int, list[str]] = {}

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _parse_edge(edge_id: str) -> tuple[str | None, str | None]:
        """(src, dst) of an edge id like 'E_GS_001__SAT_001' (None if malformed)."""
        body = edge_id[2:] if edge_id.startswith("E_") else edge_id
        if "__" in body:
            u, v = body.split("__", 1)
            return u, v
        return None, None

    def _load_node_types(self) -> None:
        if self._provider is None or self._node_type:
            return
        path = getattr(self._provider, "node_registry_path", None)
        if path is None or not path.exists():
            return
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                self._node_type[row["name"].strip()] = row["type"].strip().upper()

    def _ensure_init(self, obs) -> None:
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
            all_edges = list(obs.physical_edge_ids)
        self._edge_ids = all_edges[: self.max_edges]
        if self._provider is not None:
            self._edge_link_ids = self._provider.edge_link_ids(self._edge_ids)
        # RequestGenerator RNG state depends on the call history from the
        # seed, and the env advances it one slot per step from t=0. The preview
        # generator must therefore fast-forward identically: generate 0..t0-1
        # (discard), then cache arrivals from t0 on.
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
        # the window (a link that never becomes available can never generate,
        # so it cannot help serve; keeping it only inflates the MILP).
        rates_all, avail_all = self._window_rates_avail(t0, W)
        window_avail = avail_all.max(axis=0) >= 0.5
        full_idx = {eid: i for i, eid in enumerate(self._edge_ids)}
        edges = [eid for eid, ok in zip(self._edge_ids, window_avail) if ok]
        if len(edges) == 0:
            edges = list(self._edge_ids)  # degenerate fallback
        sel = np.asarray([full_idx[e] for e in edges], dtype=np.int64)
        n_edges = len(edges)
        rates = rates_all[:, sel]
        avail = avail_all[:, sel]

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

        # ---- adjacency over the FULL edge set for path enumeration ----------
        # Paths are found through ALL scenario edges (self._edge_ids), not just
        # the window-available subset. The MILP can only activate window-available
        # edges, but a path might use edges that become available at different
        # slots. Paths that include unavailable edges are rejected by edge_idx.
        edge_idx = {eid: i for i, eid in enumerate(edges)}
        all_adj: dict[str, list[str]] = {node: [] for node in obs.node_ids}
        all_pair_to_edge: dict[tuple[str, str], str] = {}
        for eid in self._edge_ids:
            u, v = self._parse_edge(eid)
            if u is None:
                continue
            all_adj.setdefault(u, []).append(v)
            all_adj.setdefault(v, []).append(u)
            all_pair_to_edge[tuple(sorted((u, v)))] = eid

        request_paths: list[list[list[int]]] = []
        for req in requests:
            paths = _enumerate_paths(req.src_gs, req.dst_gs, all_adj, self.max_path_hops, self.max_paths_per_request)
            kept: list[list[int]] = []
            for path in paths:
                eids = [all_pair_to_edge[tuple(sorted((path[i], path[i + 1])))] for i in range(len(path) - 1)]
                if all(e in edge_idx for e in eids):
                    kept.append([edge_idx[e] for e in eids])
            request_paths.append(kept)
        use_request = [bool(rp) for rp in request_paths]
        request_paths = [rp for rp, ok in zip(request_paths, use_request) if ok]
        requests = [r for r, ok in zip(requests, use_request) if ok]
        remaining = [r for r, ok in zip(remaining, use_request) if ok]
        arrival_rel = [a for a, ok in zip(arrival_rel, use_request) if ok]
        deadline_rel = [d for d, ok in zip(deadline_rel, use_request) if ok]
        n_req = len(requests)
        if n_edges == 0 or n_req == 0:
            return WindowOutcome([], 0.0, time.perf_counter() - t_start, -1)

        # ---- variables ---------------------------------------------------------
        # x[e, tau] binary activation; s[e, tau] end-of-slot inventory (continuous);
        # f[r, p, tau] keys served to r over path p at slot tau (continuous).
        xv: dict[tuple[int, int], int] = {}
        sv: dict[tuple[int, int], int] = {}
        fv: dict[tuple[int, int, int], int] = {}
        var_count = 0
        for e in range(n_edges):
            for tau in range(W):
                xv[(e, tau)] = var_count
                sv[(e, tau)] = var_count + 1
                var_count += 2
        for r in range(n_req):
            for p in range(len(request_paths[r])):
                for tau in range(W):
                    fv[(r, p, tau)] = var_count
                    var_count += 1
        n_vars = var_count

        integrality = np.zeros(n_vars, dtype=np.int8)
        lb = np.zeros(n_vars, dtype=np.float64)
        ub = np.full(n_vars, np.inf, dtype=np.float64)
        c = np.zeros(n_vars, dtype=np.float64)  # minimise negative objective
        for e in range(n_edges):
            for tau in range(W):
                integrality[xv[(e, tau)]] = 1
                ub[xv[(e, tau)]] = 1.0  # activation <= availability enforced via constraint
        for r in range(n_req):
            for p in range(len(request_paths[r])):
                for tau in range(W):
                    ub[fv[(r, p, tau)]] = (
                        0.0
                        if tau < arrival_rel[r] or tau >= deadline_rel[r]
                        else remaining[r]
                    )
                    c[fv[(r, p, tau)]] = -1.0  # maximise served keys

        # ---- constraints --------------------------------------------------------
        constraints: list[tuple[np.ndarray, np.ndarray, float]] = []  # A_row x <= ub_row
        gen = rates * self.slot_seconds  # (W, E); no switch decay (upper bound)
        cap = obs.state.qkp_capacity or {}
        init_level = obs.state.qkp_snapshot

        # activation <= availability (x[e,tau] <= avail[e,tau])
        for e in range(n_edges):
            for tau in range(W):
                if avail[tau, e] < 0.5:
                    constraints.append((np.array([1.0]), np.array([xv[(e, tau)]]), 0.0))
        # matching: at most one activated edge per node per slot
        incident: dict[str, list[int]] = {node: [] for node in obs.node_ids}
        for e, eid in enumerate(edges):
            u, v = self._parse_edge(eid)
            if u is None:
                continue
            incident.setdefault(u, []).append(e)
            incident.setdefault(v, []).append(e)
        for node, node_edges in incident.items():
            for tau in range(W):
                if node_edges:
                    cols = np.array([xv[(e, tau)] for e in node_edges], dtype=np.int64)
                    constraints.append((np.ones(len(cols)), cols, 1.0))
        # inventory dynamics: s[e,tau] = s[e,tau-1] + gen*x - serve, s >= 0
        # serve on edge e at tau = sum of f over (r,p) whose path uses e.
        serve_cols: dict[tuple[int, int], list[int]] = {(e, tau): [] for e in range(n_edges) for tau in range(W)}
        for r in range(n_req):
            for p in range(len(request_paths[r])):
                for tau in range(W):
                    if tau < arrival_rel[r] or tau >= deadline_rel[r]:
                        continue
                    for e in request_paths[r][p]:
                        serve_cols[(e, tau)].append(fv[(r, p, tau)])
        # s[e,tau] - gen[e,tau]*x[e,tau] + serve[e,tau] - s[e,tau-1] = init (tau=0) / 0 (tau>0)
        for e in range(n_edges):
            init_e = float(init_level.get(edges[e], 0.0))
            for tau in range(W):
                row = {sv[(e, tau)]: 1.0, xv[(e, tau)]: -gen[tau, e]}
                if tau > 0:
                    row[sv[(e, tau - 1)]] = -1.0
                for v in serve_cols[(e, tau)]:
                    row[v] = row.get(v, 0.0) + 1.0
                cols = np.array(list(row.keys()), dtype=np.int64)
                coeffs = np.array(list(row.values()), dtype=np.float64)
                constraints.append((coeffs, cols, init_e if tau == 0 else 0.0))
        # inventory <= capacity
        for e in range(n_edges):
            cap_e = float(cap.get(edges[e], np.inf))
            if np.isfinite(cap_e):
                for tau in range(W):
                    constraints.append((np.array([1.0]), np.array([sv[(e, tau)]]), cap_e))
        # per request: total served <= remaining
        for r in range(n_req):
            cols = [fv[(r, p, tau)] for p in range(len(request_paths[r])) for tau in range(W)]
            constraints.append((np.ones(len(cols)), np.array(cols, dtype=np.int64), remaining[r]))
        # end-of-window inventory value in the objective
        for e in range(n_edges):
            c[sv[(e, W - 1)]] = -self.final_inventory_weight

        n_cons = len(constraints)
        matrix = lil_matrix((n_cons, n_vars), dtype=np.float64)
        upper = np.full(n_cons, np.inf, dtype=np.float64)
        for row_idx, (coeffs, cols, ub_row) in enumerate(constraints):
            matrix[row_idx, cols] = coeffs
            upper[row_idx] = ub_row

        status, x = _solve_highs(
            c=c, lb=lb, ub=ub, integrality=integrality,
            matrix_csc=matrix.tocsc(), upper=upper,
            time_limit_s=self.time_limit_s, mip_rel_gap=0.0,
        )
        import os as _os
        if _os.environ.get("RH_DEBUG"):
            n_act = sum(1 for e in range(n_edges) for tau in range(W) if x is not None and x[xv[(e, tau)]] >= 0.5)
            f_sum = sum(float(x[fv[(r, p, tau)]]) for r in range(n_req) for p in range(len(request_paths[r])) for tau in range(W)) if x is not None else -1.0
            print(f"[RH_DEBUG] t={t0} n_edges={n_edges} n_req={n_req} n_vars={n_vars} n_cons={n_cons} "
                  f"status={status} x_act={n_act} f_sum={f_sum:.1f} c_nnz={int((c != 0).sum())}", flush=True)
        if x is None:
            return WindowOutcome([], 0.0, time.perf_counter() - t_start, status)

        activated = [edges[e] for e in range(n_edges) if x[xv[(e, 0)]] >= 0.5]
        activation_plan: dict[int, list[str]] = {
            tau: [edges[e] for e in range(n_edges) if x[xv[(e, tau)]] >= 0.5]
            for tau in range(W)
        }
        total_served = 0.0
        flow_by_request: dict[str, float] = {}
        flow_detail: list[tuple[str, list[int], int, float]] = []
        for r in range(n_req):
            total = 0.0
            for p in range(len(request_paths[r])):
                for tau in range(W):
                    amt = float(x[fv[(r, p, tau)]])
                    if amt > 1.0e-9:
                        # Path indices are reported against the FULL edge set
                        # (self._edge_ids) so ideal-execution consumers index
                        # inventory consistently; the MILP internals use the
                        # window-narrowed edge subset.
                        path_full = [full_idx[edges[ei]] for ei in request_paths[r][p]]
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
            activation_plan=activation_plan,
            flow_detail=flow_detail,
            ending_inventory=ending_inventory,
        )
        self.last_outcome = outcome
        self.window_solve_s = outcome.solve_time_s
        return outcome

    # ------------------------------------------------------------------ public
    def act(self, obs) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        """Execute the current window plan; re-solve at window boundaries.

        Returns the slot-0 activation of a fresh solve, or the pre-planned
        activation for this absolute time step inside an active window (open
        loop). Re-solving every step would discard the future activations
        before they execute, breaking multi-hop inventory stocking.
        """
        self._ensure_init(obs)
        t = int(obs.state.t)
        if self._plan_t0 is None or t < self._plan_t0 or t >= self._plan_t0 + self.window_steps:
            outcome = self.solve_window(obs)
            self._plan_t0 = t
            self._plan = {t + tau: edges for tau, edges in outcome.activation_plan.items()}
        activated = set(self._plan.get(t, []))
        # Map activated edges to the actions the CURRENT observation can
        # execute (masked candidates). Plan edges that are not legal at this
        # slot are dropped (the env idles the node).
        obs_pair_to_edge = {
            tuple(sorted((u, v))): eid
            for eid in obs.physical_edge_ids
            for u, v in [self._parse_edge(eid)]
            if u is not None
        }
        actions: dict[str, str] = {}
        scores: dict[str, dict[str, float]] = {}
        for node_id in obs.node_ids:
            node_scores: dict[str, float] = {}
            best_action = NodeActionSpace.IDLE
            best_score = 0.0
            for action in obs.action_candidates[node_id]:
                if action == NodeActionSpace.IDLE:
                    score = 0.0
                else:
                    edge_id = obs_pair_to_edge.get(tuple(sorted((node_id, action))))
                    score = 1.0 if edge_id in activated else 0.0
                    if score > best_score:
                        best_score = score
                        best_action = action
                node_scores[action] = score
            actions[node_id] = best_action
            scores[node_id] = node_scores
        return actions, scores
