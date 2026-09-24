from __future__ import annotations

import random
from collections import deque

from qkd_rl.data.scenario_builder import Scenario
from qkd_rl.env.action_resolver import ActionResolver
from qkd_rl.env.graph_builder import GraphBuilder, GraphObservation
from qkd_rl.env.history_buffer import HistoryBuffer
from qkd_rl.env.masks import ActionMaskBuilder
from qkd_rl.env.metrics import MetricsTracker
from qkd_rl.env.qkp import LinkQKPPool
from qkd_rl.env.request import RequestGenerator, RequestHistoryTracker, RequestQueue
from qkd_rl.env.reward import RewardFunction
from qkd_rl.env.routing import RoutingPolicy
from qkd_rl.env.state import EnvState
from qkd_rl.link.rate_provider import EdgeWindow, LazyEdgeWindows, RateProvider


class QKDEnv:
    def __init__(
        self,
        scenario: Scenario,
        rate_provider: RateProvider,
        request_generator: RequestGenerator,
        qkp: LinkQKPPool,
        routing: RoutingPolicy,
        reward_fn: RewardFunction,
        graph_builder: GraphBuilder,
        mask_builder: ActionMaskBuilder,
        action_resolver: ActionResolver,
        metrics: MetricsTracker,
        config: dict,
        history_buffer: HistoryBuffer | None = None,
    ):
        self.scenario = scenario
        self.rate_provider = rate_provider
        self.request_generator = request_generator
        self.qkp = qkp
        self.routing = routing
        self.reward_fn = reward_fn
        self.graph_builder = graph_builder
        self.mask_builder = mask_builder
        self.action_resolver = action_resolver
        self.metrics = metrics
        self.config = config
        self.continuous = bool(config["env"].get("continuous", False))
        self.requests = RequestQueue()
        self.request_history = RequestHistoryTracker()
        self.history_buffer = history_buffer
        self.rng = random.Random(config["seed"]["env_seed"])
        self.t = scenario.start_t
        self.steps = 0
        self.last_activated_edges: list[str] = []
        # Directed arcs actually matched last step (from ResolvedAction), for
        # max_weight_matching re-evaluation of the joint log probability.
        self.last_matched_arcs: list[tuple[str, str]] = []
        self._prev_activated_edges: list[str] = []
        # (t, observation, masks) cache: the observation built at the end of a
        # step (t+1) is exactly what the next step would rebuild at its start.
        self._obs_cache: tuple[int, GraphObservation, dict[str, list[bool]]] | None = None

    def reset(self, seed: int | None = None, start_seed: int | None = None) -> GraphObservation:
        if seed is not None:
            # Per-episode reproducibility: reseed the request stream with the
            # episode seed. Workers reuse one env across episodes, so without
            # this the second episode continues the first one's stream and
            # rerunning the same seed cannot reproduce the same requests.
            self.request_generator.seed(seed)
        if start_seed is not None:
            # The random start is driven by its own seed so request seed and
            # start-day seed stay independent while remaining reproducible.
            self.rng.seed(start_seed)
        elif seed is not None:
            self.rng.seed(seed)
        start_mode = self.config["env"].get("episode_start_mode", "fixed")
        if start_mode == "random_day" and seed is not None:
            episode_steps = int(self.config["env"].get("episode_steps", 400))
            day_steps = int(self.config["env"].get("day_steps", 1440))
            activation_days = int(self.config["env"].get("activation_window_days", 0) or 0)
            activation_end_day = int(self.config["env"].get("activation_window_end_day", -1) or -1)
            if activation_days <= 0 and activation_end_day >= 0:
                activation_start_day = int(self.config["env"].get("activation_window_start_day", 0) or 0)
                activation_days = max(0, activation_end_day - activation_start_day)
            if activation_days > 0:
                # Activation window = allowed range for the episode start.
                # The episode itself may run past the window end; the caller
                # should configure scenario.time_limit.days large enough to
                # include both the activation window and the episode horizon.
                activation_start_day = int(self.config["env"].get("activation_window_start_day", 0) or 0)
                window_start = int(self.scenario.start_t) + activation_start_day * day_steps
                window_end = min(
                    int(self.scenario.end_t),
                    window_start + activation_days * day_steps,
                )
                self.t = self.rng.randint(window_start, max(window_start, window_end - 1))
            else:
                # Legacy full-year behavior: random day boundary, but avoid
                # starting so late that the episode cannot fit before the end.
                max_start = max(self.scenario.start_t, self.scenario.end_t - episode_steps)
                max_day = max(0, (max_start - self.scenario.start_t) // day_steps)
                self.t = self.scenario.start_t + self.rng.randint(0, max_day) * day_steps
        elif start_mode == "random_window" and seed is not None:
            episode_steps = int(self.config["env"].get("episode_steps", 400))
            max_start = max(self.scenario.start_t, self.scenario.end_t - episode_steps)
            self.t = self.rng.randint(self.scenario.start_t, max_start)
        else:
            self.t = self.scenario.start_t
        # ★ 不能用 `get(k, -1) or -1`：Python 里 `0 or -1` 求值为 `-1`
        #   ⟹ **day 0 会被静默忽略**（传 0 = 不钉起日）。实测踩到过：
        #   一个判读脚本要 day0、实得 day330，且只在自检里才发现。
        #   `None` 才是「不钉」，0 是合法的一天（= scenario.start_t）。
        episode_start_day = self.config["env"].get("episode_start_day", None)
        if episode_start_day is not None and int(episode_start_day) >= 0:
            day_steps = int(self.config["env"].get("day_steps", 1440))
            self.t = self.scenario.start_t + int(episode_start_day) * day_steps
        if self.continuous and "activation_window_start_day" in self.config["env"]:
            day_steps = int(self.config["env"].get("day_steps", 1440))
            start_day = int(self.config["env"].get("activation_window_start_day", 0) or 0)
            activation_days = int(self.config["env"].get("activation_window_days", 0) or 0)
            activation_end_day = int(self.config["env"].get("activation_window_end_day", -1) or -1)
            if activation_days <= 0 and activation_end_day >= 0:
                activation_days = max(0, activation_end_day - start_day)
            if activation_days > 0:
                last_allowed = min(
                    activation_days - 1,
                    max(0, int(self.scenario.end_t / day_steps) - start_day - 1),
                )
                start_day += self.rng.randrange(last_allowed + 1)
            self.t = self.scenario.start_t + start_day * day_steps
        self.qkp.reset(self.t)
        self.requests.reset()
        self.request_history.reset()
        if self.history_buffer is not None:
            self.history_buffer.reset()
        self.metrics.reset()
        self.reward_fn.reset()
        self.steps = 0
        self.last_activated_edges = []
        self.last_matched_arcs = []
        self._prev_activated_edges = []
        self._obs_cache = None
        self._prev_waiting_keys = 0.0
        return self._build_observation()

    def step(
        self,
        actions: dict[str, tuple[str, str]],
        action_scores: dict[str, dict[str, float]] | None = None,
        edge_scores: dict[str, float] | None = None,
        expected_matched_edges: list[tuple[str, str]] | None = None,
    ) -> tuple[GraphObservation, float, bool, bool, dict]:
        """Advance one slot given each node's ``(tx_target, rx_source)`` action.

        The resolver turns the proposals into a feasible set of directed arcs
        subject to the dual-port constraints and reports them as deduplicated
        undirected pair edge ids in ``resolved.activated_edges``, so key
        generation / routing / QKP consume the same pair ids as before.
        """
        arrivals = self.request_generator.generate(self.t)
        self.requests.add_arrivals(arrivals)
        self.request_history.record_arrivals(arrivals, self.t)
        self.metrics.add_arrivals(arrivals)

        state_before, masks = self._state_and_masks(self.t)
        resolved = self.action_resolver.resolve(
            actions,
            state_before,
            masks,
            action_scores,
            edge_scores=edge_scores,
        )
        if expected_matched_edges is not None and set(resolved.matched_arcs) != set(expected_matched_edges):
            raise RuntimeError(
                "Resolver executed a different matching than the policy sampled: "
                f"executed={sorted(resolved.matched_arcs)} sampled={sorted(expected_matched_edges)}. "
                "Use action_resolver.mode=mutual_choice for the global matching policy."
            )

        generated = self._generate_keys(resolved.activated_edges, state_before.edge_windows)
        allocation = self.routing.allocate_generated_keys(resolved.activated_edges, generated, self.qkp, self.t)
        # Mark this slot's new keys so the serve phase can attribute each
        # request's consumption between history stock and current activation.
        self.qkp.set_slot_new(allocation.added_by_edge)
        serve_result = self.requests.serve(
            self.qkp, self.routing, self.t,
            deadline_steps=float(self.config.get("requests", {}).get("deadline_steps", 0) or 0),
        )
        self.qkp.clear_slot_new()
        expired_requests = self.requests.expire(self.t)
        self.request_history.record_served(serve_result.served_requests, self.t)
        self.request_history.record_failed(serve_result.failed_requests + expired_requests, self.t)
        expired_keys = self.qkp.expire(self.t)

        # Flow-based waiting penalty: only the *increase* of the waiting stock
        # is penalized (waiting_now - waiting_prev), so a policy that never
        # serves is punished for the new backlog it creates each slot instead
        # of being hit by the ever-growing total stock, which dominated every
        # other reward term (measured ~3.7/step) and drowned the signal.
        waiting_delta = max(0.0, serve_result.waiting_keys - self._prev_waiting_keys)
        self._prev_waiting_keys = serve_result.waiting_keys
        # Reuse the relay importance already computed by GraphBuilder for the
        # model input; this cache reflects the state at observation build time
        # (before the current step's arrivals), and the reward side never
        # recomputes it.
        relay_importance = self.graph_builder.last_relay_importance
        baseline_reward = 0.0
        if self.reward_fn.mode == "baseline_score":
            active_edge_ids = self._active_edge_ids(masks)
            baseline_scores = self._baseline_edge_scores(
                state_before.edge_windows, active_edge_ids
            )
            baseline_reward = sum(
                baseline_scores.get(edge_id, 0.0)
                for edge_id in resolved.activated_edges
            )
        reward_detail = self.reward_fn.compute(
            serve_result=serve_result,
            allocation=allocation,
            expired_requests=expired_requests,
            expired_keys=expired_keys,
            resolved_action=resolved,
            qkp=self.qkp,
            arrived_keys=sum(req.amount for req in arrivals),
            served_keys=serve_result.served_keys,
            waiting_keys=serve_result.waiting_keys,
            waiting_delta=waiting_delta,
            switch_count=len(set(resolved.activated_edges) - set(self.last_activated_edges)),
            keep_active_count=len(set(resolved.activated_edges) & set(self.last_activated_edges)),
            added_by_edge=allocation.added_by_edge,
            relay_importance=relay_importance,
            baseline_reward=baseline_reward,
            serve_events=serve_result.serve_events,
            from_new_by_edge=serve_result.from_new_by_edge,
            storage_pathness=self._compute_storage_pathness(resolved.activated_edges, self.qkp),
        )
        self.metrics.update(resolved, generated, serve_result, reward_detail, self.qkp, expired_requests)

        self._prev_activated_edges = list(self.last_activated_edges)
        self.last_activated_edges = resolved.activated_edges
        self.last_matched_arcs = list(resolved.matched_arcs)
        if self.history_buffer is not None:
            self.history_buffer.push(
                arrivals=arrivals,
                serve_result=serve_result,
                expired_requests=expired_requests,
                qkp=self.qkp,
                edge_windows=state_before.edge_windows,
                activated_edges=resolved.activated_edges,
                pending_requests=self.requests.get_pending(),
                t=self.t,
            )
        self.t += 1
        self.steps += 1
        obs = self._build_observation()
        episode_steps = int(self.config["env"]["episode_steps"])
        terminated = (not self.continuous) and self.steps >= episode_steps
        terminate_on_year_end = bool(self.config["env"].get("terminate_on_year_end", True))
        truncated = self.t >= self.scenario.end_t if terminate_on_year_end else False
        return obs, reward_detail.total, terminated, truncated, self.metrics.last_info(reward_detail)

    def _active_edge_ids(self, masks: dict[str, list[bool]]) -> list[str]:
        """Edge ids legal for both endpoints at the current time step."""
        if self._obs_cache is not None and self._obs_cache[0] == self.t:
            return list(self._obs_cache[1].generation_edge_ids)
        active_edges, _ = self.graph_builder._active_edges(
            masks, self.mask_builder.last_flat_legal
        )
        return [edge.edge_id for edge in active_edges]

    def _compute_storage_pathness(
        self, activated_edges: list[str], qkp: LinkQKPPool
    ) -> dict[str, float]:
        """Mark activated edges that lie on some pending request's usable path.

        A mixed path (activated edges may generate new keys, stocked edges may
        already hold keys) is built over ``activated ∪ positive-stock`` edges.
        The returned map is used by the storage reward to pay only for keys
        stored where they can actually be consumed later.
        """
        reward_cfg = self.config.get("reward", {})
        if not RewardFunction._enabled(reward_cfg, "storage"):
            return {}
        pending = self.requests.get_pending()
        if not pending:
            return {}
        activated_set = set(activated_edges)
        usable = activated_set | set(qkp.positive)
        adj: dict[str, list[tuple[str, str]]] = {}
        for edge in self.routing.edges:
            if edge.edge_id in usable:
                adj.setdefault(edge.src, []).append((edge.dst, edge.edge_id))
                adj.setdefault(edge.dst, []).append((edge.src, edge.edge_id))
        pathness: dict[str, float] = {}
        for req in pending:
            if req.src_gs not in adj or req.dst_gs not in adj:
                continue
            parent: dict[str, tuple[str | None, str | None]] = {req.src_gs: (None, None)}
            queue: deque[str] = deque([req.src_gs])
            found = False
            while queue:
                node_id = queue.popleft()
                if node_id == req.dst_gs:
                    found = True
                    break
                for neighbor, edge_id in adj.get(node_id, ()):
                    if neighbor in parent:
                        continue
                    parent[neighbor] = (node_id, edge_id)
                    queue.append(neighbor)
            if not found:
                continue
            cur = req.dst_gs
            while parent[cur][0] is not None:
                edge_id = parent[cur][1]
                if edge_id in activated_set:
                    pathness[edge_id] = 1.0
                cur = parent[cur][0]
        return pathness

    def _baseline_edge_scores(
        self,
        edge_windows: dict,
        active_edge_ids: list[str],
    ) -> dict[str, float]:
        """Score every legal edge exactly like the dynamic BFS baseline."""
        reward_cfg = self.config.get("reward", {})
        rate_weight = float(reward_cfg.get("baseline_rate_weight", 1.0))
        importance_weight = float(reward_cfg.get("baseline_importance_weight", 2.0))
        rates = {
            edge_id: float(edge_windows[edge_id].rates[0])
            for edge_id in active_edge_ids
        }
        max_rate = max(rates.values(), default=1.0) or 1.0
        importance = self.graph_builder.last_relay_importance
        return {
            edge_id: (
                rate_weight * (rates[edge_id] / max_rate)
                + importance_weight * float(importance.get(edge_id, 0.0))
            )
            for edge_id in active_edge_ids
        }

    def _generate_keys(
        self, activated_edges: list[str], edge_windows: dict[str, EdgeWindow]
    ) -> dict[str, float]:
        # Use the already-built window cache instead of a per-edge scalar read.
        # Switch cost: a link activated this step but not in the previous step
        # (a "switch") generates at a reduced rate for this slot only; links
        # kept active generate at full rate.
        switch_cfg = self.config.get("env", {}).get("switch_cost", {})
        decay = float(switch_cfg.get("rate_decay_factor", 0.5)) if switch_cfg.get("enabled", True) else 1.0
        generated: dict[str, float] = {}
        for edge_id in activated_edges:
            rate = edge_windows[edge_id].rates[0]
            if edge_id not in self.last_activated_edges:
                rate *= decay
            generated[edge_id] = rate * self.scenario.slot_seconds
        return generated

    def _build_state(self) -> EnvState:
        blocks = self.rate_provider.get_window_blocks(self.t)
        return EnvState(
            t=self.t,
            qkp_snapshot=self.qkp.snapshot(),
            pending_requests=self.requests.get_pending(),
            edge_windows=LazyEdgeWindows(self.rate_provider, self.t, blocks),
            last_activated_edges=list(self.last_activated_edges),
            prev_activated_edges=list(self._prev_activated_edges),
            qkp_capacity=dict(self.qkp.capacities),
        )

    def _state_and_masks(self, t: int) -> tuple[EnvState, dict[str, list[bool]]]:
        """Return the state/masks for ``t``, reusing the observation cache."""
        if self._obs_cache is not None and self._obs_cache[0] == t:
            return self._obs_cache[1].state, self._obs_cache[2]
        state = self._build_state()
        masks = self.mask_builder.build(state, self.qkp, self.requests)
        return state, masks

    def _build_observation(self) -> GraphObservation:
        state = self._build_state()
        masks = self.mask_builder.build(state, self.qkp, self.requests)
        obs = self.graph_builder.build(
            state,
            self.requests,
            self.request_history,
            masks,
            flat_masks=self.mask_builder.last_flat_legal,
        )
        self._obs_cache = (self.t, obs, masks)
        return obs
