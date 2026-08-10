from __future__ import annotations

import random

from qkd_rl.data.scenario_builder import Scenario
from qkd_rl.env.action_resolver import ActionResolver
from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.graph_builder import GraphBuilder, GraphObservation
from qkd_rl.env.history_buffer import HistoryBuffer
from qkd_rl.env.masks import ActionMaskBuilder
from qkd_rl.env.metrics import MetricsTracker
from qkd_rl.env.qkp import LinkQKPPool
from qkd_rl.env.request import RequestGenerator, RequestHistoryTracker, RequestQueue
from qkd_rl.env.reward import RewardFunction
from qkd_rl.env.routing import RoutingPolicy
from qkd_rl.env.state import EnvState
from qkd_rl.link.rate_provider import LazyEdgeWindows, RateProvider


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
        self.requests = RequestQueue()
        self.request_history = RequestHistoryTracker()
        self.history_buffer = history_buffer
        self.rng = random.Random(config["seed"]["env_seed"])
        self.t = scenario.start_t
        self.steps = 0
        self.last_activated_edges: list[str] = []
        self._prev_activated_edges: list[str] = []
        # (t, observation, masks) cache: the observation built at the end of a
        # step (t+1) is exactly what the next step would rebuild at its start.
        self._obs_cache: tuple[int, GraphObservation, dict[str, list[bool]]] | None = None

    def reset(self, seed: int | None = None) -> GraphObservation:
        if seed is not None:
            self.rng.seed(seed)
            # Per-episode reproducibility: reseed the request stream with the
            # episode seed too. Workers reuse one env across episodes, so
            # without this the second episode continues the first one's
            # stream and rerunning the same seed cannot reproduce the same
            # requests.
            self.request_generator.seed(seed)
        start_mode = self.config["env"].get("episode_start_mode", "fixed")
        if start_mode == "random_day" and seed is not None:
            # Full-year H5 training: start each episode at a random day boundary
            # so the agent sees every season instead of always re-running day 0.
            episode_steps = int(self.config["env"].get("episode_steps", 400))
            day_steps = int(self.config["env"].get("day_steps", 1440))
            max_start = max(self.scenario.start_t, self.scenario.end_t - episode_steps)
            max_day = max(0, (max_start - self.scenario.start_t) // day_steps)
            self.t = self.scenario.start_t + self.rng.randint(0, max_day) * day_steps
        elif start_mode == "random_window" and seed is not None:
            episode_steps = int(self.config["env"].get("episode_steps", 400))
            max_start = max(self.scenario.start_t, self.scenario.end_t - episode_steps)
            self.t = self.rng.randint(self.scenario.start_t, max_start)
        else:
            self.t = self.scenario.start_t
        self.qkp.reset(self.t)
        self.requests.reset()
        self.request_history.reset()
        if self.history_buffer is not None:
            self.history_buffer.reset()
        self.metrics.reset()
        self.reward_fn.reset()
        self.steps = 0
        self.last_activated_edges = []
        self._prev_activated_edges = []
        self._obs_cache = None
        self._prev_waiting_keys = 0.0
        return self._build_observation()

    def step(
        self,
        actions: dict[str, str],
        action_scores: dict[str, dict[str, float]] | None = None,
        edge_scores: dict[str, float] | None = None,
        expected_matched_edges: list[str] | None = None,
    ) -> tuple[GraphObservation, float, bool, bool, dict]:
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
        if expected_matched_edges is not None and set(resolved.activated_edges) != set(expected_matched_edges):
            raise RuntimeError(
                "Resolver executed a different matching than the policy sampled: "
                f"executed={sorted(resolved.activated_edges)} sampled={sorted(expected_matched_edges)}. "
                "Use action_resolver.mode=mutual_choice for the global matching policy."
            )

        generated = self._generate_keys(resolved.activated_edges, state_before.edge_windows)
        allocation = self.routing.allocate_generated_keys(resolved.activated_edges, generated, self.qkp, self.t)
        serve_result = self.requests.serve(self.qkp, self.routing, self.t)
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
            added_by_edge=allocation.added_by_edge,
            relay_importance=relay_importance,
        )
        self.metrics.update(resolved, generated, serve_result, reward_detail, self.qkp, expired_requests)

        self._prev_activated_edges = list(self.last_activated_edges)
        self.last_activated_edges = resolved.activated_edges
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
        terminated = self.steps >= int(self.config["env"]["episode_steps"])
        terminate_on_year_end = bool(self.config["env"].get("terminate_on_year_end", True))
        truncated = self.t >= self.scenario.end_t if terminate_on_year_end else False
        return obs, reward_detail.total, terminated, truncated, self.metrics.last_info(reward_detail)

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
