from __future__ import annotations

import random

from qkd_rl.data.scenario_builder import Scenario
from qkd_rl.env.action_resolver import ActionResolver
from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.graph_builder import GraphBuilder, GraphObservation
from qkd_rl.env.masks import ActionMaskBuilder
from qkd_rl.env.metrics import MetricsTracker
from qkd_rl.env.qkp import LinkQKPPool
from qkd_rl.env.request import RequestGenerator, RequestHistoryTracker, RequestQueue
from qkd_rl.env.reward import RewardFunction
from qkd_rl.env.routing import RoutingPolicy
from qkd_rl.env.state import EnvState
from qkd_rl.link.rate_provider import RateProvider


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
        self.rng = random.Random(config["seed"]["env_seed"])
        self.t = scenario.start_t
        self.steps = 0
        self.last_activated_edges: list[str] = []

    def reset(self, seed: int | None = None) -> GraphObservation:
        if seed is not None:
            self.rng.seed(seed)
        self.qkp.reset()
        self.requests.reset()
        self.request_history.reset()
        self.metrics.reset()
        self.steps = 0
        self.last_activated_edges = []
        self.t = self.scenario.start_t
        return self._build_observation()

    def step(
        self,
        actions: dict[str, str],
        action_scores: dict[str, dict[str, float]] | None = None,
    ) -> tuple[GraphObservation, float, bool, bool, dict]:
        arrivals = self.request_generator.generate(self.t)
        self.requests.add_arrivals(arrivals)
        self.request_history.record_arrivals(arrivals, self.t)
        self.metrics.add_arrivals(sum(req.amount for req in arrivals))

        state_before = self._build_state()
        masks = self.mask_builder.build(state_before, self.qkp, self.requests)
        resolved = self.action_resolver.resolve(actions, state_before, masks, action_scores)

        generated = self._generate_keys(resolved.activated_edges)
        allocation = self.routing.allocate_generated_keys(resolved.activated_edges, generated, self.qkp, self.t)
        serve_result = self.requests.serve(self.qkp, self.routing, self.t)
        expired_requests = self.requests.expire(self.t)
        self.request_history.record_served(serve_result.served_requests, self.t)
        self.request_history.record_failed(serve_result.failed_requests + expired_requests, self.t)
        expired_keys = self.qkp.expire(self.t)

        reward_detail = self.reward_fn.compute(
            serve_result=serve_result,
            allocation=allocation,
            expired_requests=expired_requests,
            expired_keys=expired_keys,
            resolved_action=resolved,
            qkp=self.qkp,
        )
        self.metrics.update(resolved, generated, serve_result, reward_detail, self.qkp)

        self.last_activated_edges = resolved.activated_edges
        self.t += 1
        self.steps += 1
        obs = self._build_observation()
        terminated = self.steps >= int(self.config["env"]["episode_steps"])
        truncated = self.t >= self.scenario.end_t
        return obs, reward_detail.total, terminated, truncated, self.metrics.last_info(reward_detail)

    def _generate_keys(self, activated_edges: list[str]) -> dict[str, float]:
        return {
            edge_id: self.rate_provider.get_rate(edge_id, self.t) * self.scenario.slot_seconds
            for edge_id in activated_edges
        }

    def _build_state(self) -> EnvState:
        return EnvState(
            t=self.t,
            qkp_snapshot=self.qkp.snapshot(),
            pending_requests=self.requests.get_pending(),
            edge_windows=self.rate_provider.get_all_edge_windows(self.t),
            last_activated_edges=list(self.last_activated_edges),
        )

    def _build_observation(self) -> GraphObservation:
        state = self._build_state()
        masks = self.mask_builder.build(state, self.qkp, self.requests)
        return self.graph_builder.build(state, self.requests, self.request_history, masks)
