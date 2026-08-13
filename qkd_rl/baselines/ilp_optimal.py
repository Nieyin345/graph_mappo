"""MILP-optimal baseline: per-slot optimal link activation via mixed-integer programming.

Independent of the training stack: consumes only GraphObservation fields and
returns (actions, scores) exactly like the other baselines, so it can be
added/removed without touching the trainer or the RL model.

At each slot it solves a path-flow MILP that jointly decides

* which physical links to activate (a matching: every node activates at most
  one link),
* how many keys each pending GS->GS request is served this slot (partial
  service is allowed, exactly like the env), and
* which routing path each served amount uses.

This is a genuine MILP, not an ILP: the link-activation and path-selection
decisions are discrete, while the per-request served amount is continuous (a
request may be partially served up to its remaining demand
amount - served_amount).

It maximises the total amount of served keys given (a) the existing QKP
inventory on each link, (b) the keys generated this slot on activated links
rate * slot_seconds with the env switch-cost decay applied to links not
active in the previous slot (obs.state.last_activated_edges), and (c) the
QKP capacity cap on each link when the observation exposes it
(EnvState.qkp_capacity).

Documented limitations (not bugs): it is the single-slot optimum and does not
anticipate future demand, so activating links purely to stock keys for later
slots is not rewarded; and it relaxes the env sequential serve order by
choosing the best request-to-path assignment jointly.  It is solved with HiGHS
through the highspy package (scipy.optimize.milp bundled HiGHS is not used: on
some models it spuriously reports infeasibility for problems that are feasible
and optimally solvable).  Paths are enumerated with a hop-bounded BFS over the
currently legal links, which keeps the model small enough for the full
scenario.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import lil_matrix

try:
    import highspy
except ImportError:  # pragma: no cover
    highspy = None

from qkd_rl.core.types import KeyRequest
from qkd_rl.baselines.greedy_relay_diffusion import compute_dynamic_relay_importance
from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.graph_builder import GraphObservation


def _pair_to_edge(obs: GraphObservation) -> dict[tuple[str, str], str]:
    pair_to_edge: dict[tuple[str, str], str] = {}
    for edge_id in obs.physical_edge_ids:
        edge = edge_id[2:] if edge_id.startswith("E_") else edge_id
        if "__" in edge:
            src, dst = edge.split("__", 1)
            pair_to_edge[tuple(sorted((src, dst)))] = edge_id
    return pair_to_edge


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


def _highs_status_to_scipy(status_enum) -> int:
    """Map a HiGHS model status to the scipy.optimize.milp convention.

    0 = optimal, 1 = terminated with a solution (limit reached), 2 = infeasible,
    3 = unbounded, 4 = other error.
    """
    if highspy is None:
        raise ImportError("ILPOptimalPolicy requires the highspy package: pip install highspy")
    statuses = highspy.HighsModelStatus
    if status_enum == statuses.kOptimal:
        return 0
    if status_enum in (
        statuses.kTimeLimit,
        statuses.kIterationLimit,
        statuses.kSolutionLimit,
        statuses.kObjectiveBound,
        statuses.kObjectiveTarget,
        statuses.kInterrupt,
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
    """Solve the minimisation MILP min c'x s.t. A x <= upper with HiGHS.

    integrality is a 0/1 array marking integer variables.  Returns
    (status, x) where status follows the scipy convention above and x is None
    when the solver returned no solution.
    """
    if highspy is None:
        raise ImportError("ILPOptimalPolicy requires the highspy package: pip install highspy")
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


@dataclass
class ILPOutcome:
    activated_edges: list[str]
    served_request_ids: list[str]
    served_amount: float
    solve_time_s: float
    status: int
    # Total keys served per request this slot (partial service amounts).
    flow_by_request: dict[str, float] = field(default_factory=dict)


class ILPOptimalPolicy:
    """Per-slot optimal link-activation baseline solved as a path-flow MILP."""

    def __init__(
        self,
        slot_seconds: float = 60.0,
        max_requests: int = 64,
        max_paths_per_request: int = 8,
        max_path_hops: int = 6,
        time_limit_s: float = 5.0,
        mip_rel_gap: float = 0.0,
        switch_decay: float = 0.5,
        importance_weight: float = 2.0,
        max_path_links: int = 3,
        hop_decay_factor: float = 0.25,
        wait_urgency_tau_ratio: float = 0.8,
        service_slack_ratio: float = 0.0,
        ignore_consumption: bool = False,
    ):
        self.slot_seconds = float(slot_seconds)
        self.max_requests = int(max_requests)
        self.max_paths_per_request = int(max_paths_per_request)
        self.max_path_hops = int(max_path_hops)
        self.time_limit_s = float(time_limit_s)
        self.mip_rel_gap = float(mip_rel_gap)
        self.importance_weight = float(importance_weight)
        self.max_path_links = int(max_path_links)
        self.hop_decay_factor = float(hop_decay_factor)
        self.wait_urgency_tau_ratio = float(wait_urgency_tau_ratio)
        self.service_slack_ratio = float(service_slack_ratio)
        self.ignore_consumption = bool(ignore_consumption)
        # Matches env.switch_cost.rate_decay_factor: a link activated now but
        # not in the previous slot generates at this fraction of its rate.
        self.switch_decay = float(switch_decay)
        self.last_outcome: ILPOutcome | None = None
        if highspy is None:
            raise ImportError("ILPOptimalPolicy requires the highspy package: pip install highspy")

    # ------------------------------------------------------------------ public
    def act(self, obs: GraphObservation) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        outcome = self.solve(obs)
        self.last_outcome = outcome
        activated = set(outcome.activated_edges)
        pair_to_edge = _pair_to_edge(obs)
        edge_rate = {edge_id: obs.state.edge_windows[edge_id].rates[0] for edge_id in obs.physical_edge_ids}

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
                    edge_id = pair_to_edge.get(tuple(sorted((node_id, action))))
                    if edge_id in activated:
                        score = edge_rate.get(edge_id, 0.0)
                        if score > best_score:
                            best_score = score
                            best_action = action
                    else:
                        score = 0.0
                node_scores[action] = score
            actions[node_id] = best_action
            scores[node_id] = node_scores
        return actions, scores

    # ------------------------------------------------------------------- solve
    def solve(self, obs: GraphObservation) -> ILPOutcome:
        import time

        t0 = time.perf_counter()

        # ---- request set -----------------------------------------------------
        requests: list[KeyRequest] = [
            req
            for req in obs.state.pending_requests
            if req.src_gs != req.dst_gs
            and req.src_gs in obs.node_ids
            and req.dst_gs in obs.node_ids
            and req.amount - req.served_amount > 1.0e-9
        ]
        requests.sort(key=lambda req: (-(req.amount - req.served_amount), req.deadline_t))
        requests = requests[: self.max_requests]
        remaining = [float(req.amount - req.served_amount) for req in requests]

        pair_to_edge = _pair_to_edge(obs)
        # ---- legal-link adjacency --------------------------------------------
        adj: dict[str, list[str]] = {node: [] for node in obs.node_ids}
        for edge_id in obs.physical_edge_ids:
            edge = edge_id[2:] if edge_id.startswith("E_") else edge_id
            if "__" not in edge:
                continue
            src, dst = edge.split("__", 1)
            adj.setdefault(src, []).append(dst)
            adj.setdefault(dst, []).append(src)

        # ---- path enumeration -------------------------------------------------
        request_paths: list[list[list[str]]] = []  # per request: list of paths
        used_edges: set[str] = set()
        for req in requests:
            paths = _enumerate_paths(
                req.src_gs, req.dst_gs, adj, self.max_path_hops, self.max_paths_per_request
            )
            kept: list[list[str]] = []
            for path in paths:
                edges_on_path = [
                    pair_to_edge[tuple(sorted((path[i], path[i + 1])))] for i in range(len(path) - 1)
                ]
                kept.append(edges_on_path)
                used_edges.update(edges_on_path)
            request_paths.append(kept)

        edges: list[str] = sorted(used_edges)
        if not edges or not any(request_paths):
            return ILPOutcome([], [], 0.0, time.perf_counter() - t0, -1)

        # ---- variables --------------------------------------------------------
        # f_{r,i} >= 0      continuous: keys served to request r over path i.
        # y_{r,i} in {0,1}  path i is the single serving path of request r.
        # x_e   in {0,1}    physical link e is activated this slot.
        flow_var: dict[tuple[int, int], int] = {}
        path_choice_var: dict[tuple[int, int], int] = {}
        var_count = 0
        for r, paths in enumerate(request_paths):
            for i in range(len(paths)):
                flow_var[(r, i)] = var_count
                path_choice_var[(r, i)] = var_count + 1
                var_count += 2
        edge_var: dict[str, int] = {edge_id: var_count + idx for idx, edge_id in enumerate(edges)}
        n_vars = var_count + len(edges)

        integrality = np.zeros(n_vars, dtype=np.int8)
        c = np.zeros(n_vars, dtype=np.float64)  # minimise negative served keys
        lb = np.zeros(n_vars, dtype=np.float64)
        ub = np.full(n_vars, np.inf, dtype=np.float64)
        for r, paths in enumerate(request_paths):
            for i in range(len(paths)):
                integrality[flow_var[(r, i)]] = 0
                integrality[path_choice_var[(r, i)]] = 1
                ub[flow_var[(r, i)]] = remaining[r]
                ub[path_choice_var[(r, i)]] = 1.0
                c[flow_var[(r, i)]] = -1.0  # maximise total served keys
        for edge_id in edges:
            integrality[edge_var[edge_id]] = 1
            ub[edge_var[edge_id]] = 1.0

        window = {edge_id: obs.state.edge_windows[edge_id] for edge_id in edges}
        qkp = obs.state.qkp_snapshot
        existing = {edge_id: float(qkp.get(edge_id, 0.0)) for edge_id in edges}
        capacity = obs.state.qkp_capacity or {}
        last_active = set(obs.state.last_activated_edges)
        generated = {
            edge_id: float(window[edge_id].rates[0])
            * self.slot_seconds
            * (self.switch_decay if edge_id not in last_active else 1.0)
            for edge_id in edges
        }

        # ---- constraints -------------------------------------------------------
        constraints: list[tuple[np.ndarray, np.ndarray, float]] = []  # (coeffs, cols, ub)
        # per request: at most one serving path selected
        for r, paths in enumerate(request_paths):
            cols = np.array([path_choice_var[(r, i)] for i in range(len(paths))], dtype=np.int64)
            constraints.append((np.ones(len(cols)), cols, 1.0))
        # per (request, path): flow can only use the selected path, capped by
        # the remaining demand (big-M style linking constraint).
        for r, paths in enumerate(request_paths):
            for i in range(len(paths)):
                cols = np.array([flow_var[(r, i)], path_choice_var[(r, i)]], dtype=np.int64)
                constraints.append((np.array([1.0, -remaining[r]]), cols, 0.0))
        # per edge: total flow <= existing inventory + generated keys if activated
        edge_usage: dict[str, list[tuple[int, float]]] = {edge_id: [] for edge_id in edges}
        for r, paths in enumerate(request_paths):
            for i in range(len(paths)):
                var = flow_var[(r, i)]
                for edge_id in paths[i]:
                    if edge_id in edge_usage:
                        edge_usage[edge_id].append((var, 1.0))
        for edge_id in edges:
            cols = [v for v, _ in edge_usage[edge_id]] + [edge_var[edge_id]]
            coeffs = [a for _, a in edge_usage[edge_id]] + [-generated[edge_id]]
            constraints.append((np.array(coeffs, dtype=np.float64), np.array(cols, dtype=np.int64), existing[edge_id]))
            if edge_id in capacity:
                # Pool cap: even with generation the stock can never exceed the
                # link capacity, so service is bounded by the capacity too.
                cols_cap = [v for v, _ in edge_usage[edge_id]]
                constraints.append(
                    (np.ones(len(cols_cap)), np.array(cols_cap, dtype=np.int64), float(capacity[edge_id]))
                )
        # matching: every node activates at most one incident link
        incident: dict[str, list[str]] = {node: [] for node in obs.node_ids}
        for edge_id in edges:
            edge = edge_id[2:] if edge_id.startswith("E_") else edge_id
            src, dst = edge.split("__", 1)
            incident.setdefault(src, []).append(edge_id)
            incident.setdefault(dst, []).append(edge_id)
        for node, node_edges in incident.items():
            if node_edges:
                cols = np.array([edge_var[e] for e in node_edges], dtype=np.int64)
                constraints.append((np.ones(len(cols)), cols, 1.0))

        n_cons = len(constraints)
        matrix = lil_matrix((n_cons, n_vars), dtype=np.float64)
        upper = np.full(n_cons, np.inf, dtype=np.float64)
        for row_idx, (coeffs, cols, ub_row) in enumerate(constraints):
            matrix[row_idx, cols] = coeffs
            upper[row_idx] = ub_row

        # ---- lexicographic two-stage solve ------------------------------------
        # Stage 1: maximise the keys served this slot.  Stage 2: among all
        # solutions that reach the stage-1 optimum, maximise the keys generated
        # this slot so the reference trajectory also stocks inventory for future
        # slots (a single-slot greedy optimum is otherwise degenerate when no
        # request can be served: it would never activate a link and the episode
        # would collapse to zero service).
        c_served = c.copy()
        try:
            status, x = _solve_highs(
                c=c_served,
                lb=lb,
                ub=ub,
                integrality=integrality,
                matrix_csc=matrix.tocsc(),
                upper=upper,
                time_limit_s=self.time_limit_s,
                mip_rel_gap=self.mip_rel_gap,
            )
        except Exception:
            return ILPOutcome([], [], 0.0, time.perf_counter() - t0, -1)

        if x is None:
            return ILPOutcome([], [], 0.0, time.perf_counter() - t0, status)

        served_obj = float(c_served @ x)
        served_max = max(0.0, -served_obj)
        service_slack = (
            served_max * self.service_slack_ratio
            if served_max > 1.0e-9
            else 1.0e-4
        )
        importance = compute_dynamic_relay_importance(
            obs,
            max_path_links=self.max_path_links,
            hop_decay_factor=self.hop_decay_factor,
            wait_urgency_tau_ratio=self.wait_urgency_tau_ratio,
            ignore_consumption=self.ignore_consumption,
        )
        c_gen = np.zeros(n_vars, dtype=np.float64)
        for edge_id in edges:
            future_bonus = 1.0 + self.importance_weight * importance.get(edge_id, 0.0)
            c_gen[edge_var[edge_id]] = -generated[edge_id] * future_bonus
        served_cols = np.array([i for i in range(n_vars) if c_served[i] != 0.0], dtype=np.int64)
        served_coeffs = np.asarray(c_served[served_cols], dtype=np.float64)
        constraints2 = list(constraints)
        constraints2.append((served_coeffs, served_cols, served_obj + service_slack))
        n_cons2 = len(constraints2)
        matrix2 = lil_matrix((n_cons2, n_vars), dtype=np.float64)
        upper2 = np.full(n_cons2, np.inf, dtype=np.float64)
        for row_idx, (coeffs, cols, ub_row) in enumerate(constraints2):
            matrix2[row_idx, cols] = coeffs
            upper2[row_idx] = ub_row
        status2, x2 = _solve_highs(
            c=c_gen,
            lb=lb,
            ub=ub,
            integrality=integrality,
            matrix_csc=matrix2.tocsc(),
            upper=upper2,
            time_limit_s=self.time_limit_s,
            mip_rel_gap=self.mip_rel_gap,
        )
        if x2 is not None:
            status, x = status2, x2

        activated_edges = [edge_id for edge_id in edges if x[edge_var[edge_id]] >= 0.5]
        flow_by_request: dict[str, float] = {}
        served_ids: list[str] = []
        total_served = 0.0
        for r, paths in enumerate(request_paths):
            total = 0.0
            for i in range(len(paths)):
                total += float(x[flow_var[(r, i)]])
            if total > 1.0e-6:
                flow_by_request[requests[r].request_id] = total
                served_ids.append(requests[r].request_id)
            total_served += total
        return ILPOutcome(
            activated_edges=activated_edges,
            served_request_ids=served_ids,
            served_amount=float(total_served),
            solve_time_s=time.perf_counter() - t0,
            status=status,
            flow_by_request=flow_by_request,
        )
