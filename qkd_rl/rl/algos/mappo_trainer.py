"""MAPPO trainer: rollout collection, GAE, and PPO parameter updates."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from qkd_rl.rl.algos.checkpoint import load_checkpoint, save_checkpoint
from qkd_rl.rl.algos.policy import MAPPOPolicy
from qkd_rl.rl.algos.rollout_buffer import RolloutBuffer, RolloutStep, state_free_obs
from qkd_rl.env.env import QKDEnv


def _reset_module(module: torch.nn.Module) -> None:
    """Re-initialize a module's parameters in place (identity preserved)."""
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


@dataclass
class UpdateStats:
    update: int
    actor_loss: float
    critic_loss: float
    entropy: float
    kl: float
    mean_reward: float
    mean_return: float
    mean_abs_advantage: float
    mean_success_rate: float
    mean_served_keys: float
    rollout_s: float
    update_s: float
    elapsed_s: float
    mean_ratio: float = 0.0
    actor_grad_norm: float = 0.0
    critic_grad_norm: float = 0.0
    # Critic-fit diagnostics. `advantage` collapsing onto `return` (i.e.
    # value_std << return_std) means the value head is not tracking the return
    # scale and PPO is running without a baseline, which makes every advantage
    # noise. Watching these two is the cheapest way to catch that.
    value_std: float = 0.0
    return_std: float = 0.0
    value_return_corr: float = 0.0


class MAPPOTrainer:
    """Collects episodes, computes GAE, and runs PPO updates on the shared
    actor-critic policy. One episode is one rollout of the full graph; every
    time step is one minibatch item (the GNN encodes the whole graph per item).
    """

    def __init__(
        self,
        env: QKDEnv,
        policy: MAPPOPolicy,
        config: dict,
        output_dir: str | Path,
        device: torch.device | str | None = None,
    ):
        self.env = env
        self.policy = policy
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = config["train"]
        self.device = torch.device(device if device is not None else config["runtime"]["device"])
        self.policy.device = self.device
        self.model = policy.model
        self.model.to(self.device)
        resolver_mode = str(config.get("action_resolver", {}).get("mode", "priority_matching"))
        if resolver_mode not in ("mutual_choice", "priority_matching", "max_weight_matching"):
            raise ValueError(
                f"action_resolver.mode={resolver_mode!r} is incompatible with the global "
                "matching policy: use 'mutual_choice', 'priority_matching', or "
                "'max_weight_matching'."
            )
        self.resolver_mode = resolver_mode
        # Only these resolvers consume the per-edge score dict; `mutual_choice`
        # (the default) executes the sampled matching as-is. Materializing the
        # dict costs one device sync per legal edge, so skip it otherwise.
        self._needs_edge_scores = resolver_mode in ("priority_matching", "max_weight_matching")

        self.gamma = float(train_cfg["gamma"])
        self.gae_lambda = float(train_cfg["gae_lambda"])
        self.ppo_cfg = train_cfg["ppo"]
        self.num_updates = int(train_cfg["num_updates"])
        self.episodes_per_update = int(train_cfg.get("episodes_per_update", 1))
        default_steps = int(env.config["env"].get("episode_steps", 400))
        self.rollout_steps = int(train_cfg.get("rollout_steps", default_steps))

        opt_cfg = train_cfg["optimizer"]
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.encoder.parameters(), "lr": float(opt_cfg["actor_lr"])},
                {"params": self.model.actor.parameters(), "lr": float(opt_cfg["actor_lr"])},
                {"params": self.model.critic.parameters(), "lr": float(opt_cfg["critic_lr"])},
            ]
        )
        # Gradient clipping is applied per role by default: see the note in
        # ``update``. Set ``train.ppo.clip_per_role: false`` to clip the whole
        # model together (the previous behaviour, kept for ablation).
        self.clip_per_role = bool(self.ppo_cfg.get("clip_per_role", True))
        # The encoder is shared, so it belongs to the policy group.
        self._policy_params = list(self.model.encoder.parameters()) + list(self.model.actor.parameters())
        self._value_params = list(self.model.critic.parameters())

        seed = int(config["seed"]["global_seed"])
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.update_count = 0
        self.last_episode_rewards: list[float] = []
        self.last_episode_summaries: list[dict] = []
        self.last_stats: UpdateStats | None = None

        log_cfg = train_cfg.get("logging", {})
        self.checkpoint_interval = int(log_cfg.get("checkpoint_interval", 100))
        self.log_interval = int(log_cfg.get("log_interval", 1))
        self.eval_interval = int(log_cfg.get("eval_interval", 0))
        self.eval_episodes = int(log_cfg.get("eval_episodes", 4))
        self.eval_steps = int(log_cfg.get("eval_steps", 0) or 0)
        self._rollout_pool = None
        self._continuous_obs = None
        self._continuous_obs_list = None
        self._replay_buffers: list[list[RolloutStep]] = []
        self.replay_days = int(train_cfg.get("replay_days", 0) or 0)
        self.continuous_session_updates = 0
        if getattr(env, "continuous", False):
            session_days = int(train_cfg.get("continuous_session_days", 0) or 0)
            if session_days > 0:
                day_steps = int(config["env"].get("day_steps", 1440))
                rollout_steps = int(train_cfg.get("rollout_steps", 1440))
                self.continuous_session_updates = max(
                    1, math.ceil(session_days * day_steps / max(1, rollout_steps))
                )
        self.episode_days_min = int(train_cfg.get("episode_days_min", 1) or 1)
        self.episode_days_max = int(train_cfg.get("episode_days_max", 1) or 1)
        if self.episode_days_max < self.episode_days_min:
            self.episode_days_max = self.episode_days_min
        # Fixed episode length mode: when True, the env's configured
        # episode_steps is honored as-is (no random 1..N-day resampling). The
        # episode start is still randomized inside the training window by
        # episode_start_mode: random_day, and requests stay per-episode random.
        self.episode_steps_fixed = bool(train_cfg.get("episode_steps_fixed", False))
        # Temperature annealing: actor logits are divided by temperature, so >
        # 1.0 = more exploratory, < 1.0 = more greedy. Linearly anneal from
        # `start` to `end` over `updates` updates (both default 1.0 = no anneal).
        ts = train_cfg.get("temperature_schedule", {}) or {}
        self.temp_start = float(ts.get("start", 1.0))
        self.temp_end = float(ts.get("end", 1.0))
        self.temp_updates = max(1, int(ts.get("updates", 1) or 1))
        self.model.actor.temperature = self.temp_start
        if getattr(env, "continuous", False) and self.episodes_per_update != 1:
            raise ValueError("continuous RL training requires episodes_per_update=1.")
        # Lockstep batched-rollout envs (episodes_per_update instances), built
        # lazily so single-episode runs never pay the construction cost.
        self._rollout_envs = None
        self._eval_envs = None
        self.validation_cfg = config.get("validation", {}) or {}
        self._validation_envs = None
        self._validation_seeds: list[int] = []
        self.best_validation_success = -float("inf")
        self._rollout_debug: dict[str, float] = {}
        # Critic target mode: "gae" (default, standard PPO returns) or "mc"
        # (bootstrap-free returns, kept only for ablation; see RolloutBuffer).
        self.value_target = str(train_cfg.get("value_target", "gae"))
        # Optional curriculum: a list of stages that lengthen the rollout and
        # adjust the number of parallel episodes as training progresses. The
        # trainer starts on short episodes (frequent updates, easy horizon)
        # and gradually transitions to the full-day schedule.
        self.curriculum_stages = list(train_cfg.get("curriculum", {}).get("stages", []) or [])
        # Fixed stride for episode seeds: base_seed = env_seed + update * stride.
        # Using the *current* episodes_per_update would shift the multiplier
        # when the curriculum changes it (e.g. 4 -> 2), so stage 3 would reuse
        # stage 2's exact request seeds (verified: update 120 base = 120*2 =
        # 240 = 60*4). The max across stages keeps seeds globally unique while
        # staying stateless (checkpoint resumes update_count -> seeds continue).
        seed_stride = int(train_cfg.get("episodes_per_update", 1))
        for stage in self.curriculum_stages:
            seed_stride = max(seed_stride, int(stage.get("episodes_per_update", seed_stride)))
        self._seed_stride = max(1, seed_stride)
        if bool(train_cfg.get("fixed_episode_seed", False)):
            # Fixed-scenario debugging/tuning: every update replays the exact
            # same request streams (base_seed = env_seed + 0 * stride), so the
            # mean success rate on the rollout is directly comparable across
            # updates and isolates learning in one scenario.
            self._seed_stride = 0

    def _apply_curriculum(self) -> None:
        """Apply the active curriculum stage for the current update count."""
        if not self.curriculum_stages:
            return
        active = None
        for stage in self.curriculum_stages:
            if self.update_count < int(stage.get("until_update", 0)):
                active = stage
                break
        if active is None:
            active = self.curriculum_stages[-1]
        new_steps = int(active.get("rollout_steps", self.rollout_steps))
        new_episodes = int(active.get("episodes_per_update", self.episodes_per_update))
        if new_episodes != self.episodes_per_update:
            # Rebuild the lockstep rollout envs at the new episode count.
            self._rollout_envs = None
        self.rollout_steps = new_steps
        self.episodes_per_update = new_episodes

    # ------------------------------------------------------------------ rollout
    def _sample_episode_steps(self) -> int:
        """Sample an episode length in whole days (used by random_episode)."""
        day_steps = int(self.config["env"].get("day_steps", 1440))
        days = self.rng.randint(self.episode_days_min, self.episode_days_max)
        return days * day_steps

    def collect_rollout(self) -> RolloutBuffer:
        self._reset_rollout_debug()
        n_workers = int(self.config["train"].get("n_rollout_workers", 1))
        if self.env.continuous and n_workers > 1:
            raise ValueError(
                "continuous RL requires n_rollout_workers=1: worker envs call "
                "env.reset() every episode and would break the cross-episode "
                "continuity (use rollout_batch + episodes_per_update=1 instead)."
            )
        if n_workers <= 1:
            if (
                self.episodes_per_update > 1
                and bool(self.config["train"].get("rollout_batch", True))
            ):
                return self._collect_rollout_batched_grouped()
            return self._collect_rollout_serial()
        if self._rollout_pool is None:
            from qkd_rl.rl.algos.rollout_workers import RolloutWorkerPool

            worker_device = self.config["train"].get("rollout_worker_device", str(self.device))
            self._rollout_pool = RolloutWorkerPool(self.config, worker_device, n_workers)
        base_seed = int(self.config["seed"]["env_seed"]) + self.update_count * self._seed_stride
        seeds = [base_seed + ep for ep in range(self.episodes_per_update)]
        if self.env.continuous:
            episode_steps_list = [self.rollout_steps] * len(seeds)
        elif self.episode_steps_fixed:
            episode_steps_list = [
                int(self.config["env"].get("episode_steps", self.rollout_steps))
            ] * len(seeds)
        else:
            episode_steps_list = [self._sample_episode_steps() for _ in seeds]
        weights = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        results = self._rollout_pool.collect(
            weights,
            seeds,
            self.rollout_steps,
            episode_steps_list,
            temperature=float(self.model.actor.temperature),
        )
        buffer = RolloutBuffer(self.gamma, self.gae_lambda, self.device, value_target=self.value_target)
        episode_rewards: list[float] = []
        episode_summaries: list[dict] = []
        for _seed, steps, ep_reward, summary in results:
            buffer.steps.extend(steps)
            episode_rewards.append(ep_reward)
            episode_summaries.append(summary)
        self.last_episode_rewards = episode_rewards
        self.last_episode_summaries = episode_summaries
        return buffer

    def _reset_rollout_debug(self) -> None:
        self._rollout_debug = {
            "steps": 0.0,
            "activated_edges": 0.0,
            "generated_keys": 0.0,
            "served_keys": 0.0,
            "failed_keys": 0.0,
            "waiting_keys": 0.0,
            "qkp_utilization": 0.0,
            "conflict_count": 0.0,
            "arrived_keys": 0.0,
            "reward_total": 0.0,
            "reward_served": 0.0,
            "reward_generated": 0.0,
            "reward_dense": 0.0,
            "reward_storage": 0.0,
            "reward_keep_active": 0.0,
            "reward_failed": 0.0,
            "reward_waiting": 0.0,
            "reward_switch": 0.0,
            "reward_expired": 0.0,
            "reward_conflict": 0.0,
            # attribution split: how much of the served volume earned the full
            # served reward (keys generated this slot) vs the discounted
            # history-stock part vs nothing at all (pure stock service).
            "attributed_served": 0.0,
            "history_utilized": 0.0,
        }

    def _update_rollout_debug(self, info: dict, activated_count: int) -> None:
        debug = self._rollout_debug
        debug["steps"] += 1.0
        debug["activated_edges"] += float(activated_count)
        debug["generated_keys"] += float(info.get("generated_keys", 0.0))
        debug["served_keys"] += float(info.get("served_keys", 0.0))
        debug["failed_keys"] += float(info.get("failed_keys", 0.0))
        debug["waiting_keys"] += float(info.get("waiting_keys", 0.0))
        debug["qkp_utilization"] += float(info.get("qkp_utilization", 0.0))
        debug["conflict_count"] += float(info.get("conflict_count", 0.0))
        debug["arrived_keys"] += float(info.get("arrived_keys", 0.0))
        detail = info.get("reward_detail")
        if detail is not None:
            debug["reward_total"] += float(detail.total)
            debug["reward_served"] += float(detail.served_reward)
            debug["reward_generated"] += float(detail.generated_reward)
            debug["reward_dense"] += float(detail.dense_reward)
            debug["reward_storage"] += float(detail.storage_reward)
            debug["reward_keep_active"] += float(detail.keep_active_reward)
            debug["reward_failed"] += float(detail.failed_penalty)
            debug["reward_waiting"] += float(detail.waiting_penalty)
            debug["reward_switch"] += float(detail.switch_penalty)
            debug["reward_expired"] += float(detail.expired_key_penalty)
            debug["reward_conflict"] += float(detail.conflict_penalty)
            debug["attributed_served"] += float(detail.attributed_served)
            debug["history_utilized"] += float(detail.history_utilized)

    def _rollout_debug_record(self, stats: UpdateStats) -> dict:
        debug = dict(self._rollout_debug)
        n = max(1.0, float(debug.get("steps", 0.0)))
        means = {
            "update": stats.update,
            "steps": int(debug.get("steps", 0.0)),
            "mean_activated_edges": debug.get("activated_edges", 0.0) / n,
            "mean_generated_keys": debug.get("generated_keys", 0.0) / n,
            "mean_served_keys": debug.get("served_keys", 0.0) / n,
            "mean_failed_keys": debug.get("failed_keys", 0.0) / n,
            "mean_waiting_keys": debug.get("waiting_keys", 0.0) / n,
            "mean_arrived_keys": debug.get("arrived_keys", 0.0) / n,
            "mean_qkp_utilization": debug.get("qkp_utilization", 0.0) / n,
            "mean_conflict_count": debug.get("conflict_count", 0.0) / n,
            "mean_reward": debug.get("reward_total", 0.0) / n,
            "mean_reward_served": debug.get("reward_served", 0.0) / n,
            "mean_reward_generated": debug.get("reward_generated", 0.0) / n,
            "mean_reward_dense": debug.get("reward_dense", 0.0) / n,
            "mean_reward_storage": debug.get("reward_storage", 0.0) / n,
            "mean_reward_keep_active": debug.get("reward_keep_active", 0.0) / n,
            "mean_reward_failed": debug.get("reward_failed", 0.0) / n,
            "mean_reward_waiting": debug.get("reward_waiting", 0.0) / n,
            "mean_reward_switch": debug.get("reward_switch", 0.0) / n,
            "mean_reward_expired": debug.get("reward_expired", 0.0) / n,
            "mean_reward_conflict": debug.get("reward_conflict", 0.0) / n,
            "mean_attributed_served": debug.get("attributed_served", 0.0) / n,
            "mean_history_utilized": debug.get("history_utilized", 0.0) / n,
            "actor_loss": stats.actor_loss,
            "critic_loss": stats.critic_loss,
            "entropy": stats.entropy,
            "kl": stats.kl,
            "mean_ratio": stats.mean_ratio,
            "actor_grad_norm": stats.actor_grad_norm,
            "mean_return": stats.mean_return,
            "mean_abs_advantage": stats.mean_abs_advantage,
            "mean_success_rate": stats.mean_success_rate,
        }
        return means

    def _write_rollout_debug(self, stats: UpdateStats) -> None:
        if not self._rollout_debug.get("steps", 0.0):
            return
        record = self._rollout_debug_record(stats)
        path = self.output_dir / "rollout_debug.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _collect_rollout_serial(self) -> RolloutBuffer:
        buffer = RolloutBuffer(self.gamma, self.gae_lambda, self.device, value_target=self.value_target)
        self.model.eval()
        base_seed = int(self.config["seed"]["env_seed"]) + self.update_count * self._seed_stride
        episode_rewards: list[float] = []
        episode_summaries: list[dict] = []
        for ep_idx in range(self.episodes_per_update):
            if self.env.continuous and self._continuous_obs is not None:
                obs = self._continuous_obs
            else:
                if not self.env.continuous and not self.episode_steps_fixed:
                    self.env.config["env"]["episode_steps"] = self._sample_episode_steps()
                obs = self.env.reset(seed=base_seed + ep_idx)
            ep_reward = 0.0
            terminated = False
            truncated = False
            steps = 0
            while steps < self.rollout_steps:
                with torch.no_grad():
                    step = self.policy.act(obs, build_scores=self._needs_edge_scores)
                raw_value = step.value.detach()
                next_obs, reward, terminated, truncated, info = self.env.step(
                    step.actions,
                    step.action_scores,
                    edge_scores=step.edge_scores,
                    expected_matched_edges=(
                        None
                        if self.resolver_mode == "max_weight_matching"
                        else list(step.matched_edges or [])
                    ),
                )
                self._update_rollout_debug(info, len(self.env.last_activated_edges))
                if self.resolver_mode == "max_weight_matching":
                    # max-weight is a deterministic resolver action, so the PPO
                    # target follows the matching the environment executed.
                    matched_edges = list(self.env.last_matched_arcs)
                    mean_lp, mean_entropy = self.policy.log_prob_entropy_for_matching(
                        step.edge_scores, matched_edges
                    )
                else:
                    # The policy samples a global matching in a specific order;
                    # the joint log-prob is the probability of that sampled
                    # sequence, so it must be stored as-is instead of being
                    # recomputed from the resolver's output order.
                    matched_edges = list(step.matched_edges or [])
                    mean_lp = step.mean_log_prob.detach()
                    mean_entropy = step.mean_entropy.detach()
                buffer.add(
                    RolloutStep(
                        obs=state_free_obs(obs),
                        actions=step.actions,
                        log_probs={node: lp.detach() for node, lp in step.log_probs.items()},
                        entropies={node: ent.detach() for node, ent in step.entropies.items()},
                        value=raw_value,
                        reward=float(reward),
                        terminated=terminated,
                        truncated=truncated,
                        mean_log_prob=mean_lp.detach(),
                        mean_entropy=mean_entropy.detach(),
                        matched_edges=matched_edges,
                    )
                )
                ep_reward += float(reward)
                obs = next_obs
                steps += 1
                if terminated or truncated:
                    break
            if terminated:
                last_value = torch.zeros((), dtype=torch.float32, device=self.device)
            else:
                with torch.no_grad():
                    last_value = self.policy.act(obs, build_scores=False).value.detach()
            buffer.finish_episode(last_value)
            episode_rewards.append(ep_reward)
            episode_summaries.append(self.env.metrics.episode_summary())
            if self.env.continuous:
                self._continuous_obs = None if truncated else obs
        self.last_episode_rewards = episode_rewards
        self.last_episode_summaries = episode_summaries
        return buffer

    def _collect_rollout_batched_grouped(self) -> RolloutBuffer:
        """Drive :meth:`_collect_rollout_batched` over groups of envs.

        ``train.rollout_batch_envs`` (default: all episodes) caps how many graphs
        share one block-diagonal forward. Stacking more graphs into one forward
        gets *worse* per graph on this model (measured on the RTX 4060 laptop:
        57 ms/step for 4 graphs vs 409 ms/step for 8, i.e. 14 vs 51 ms per
        graph), so raising ``episodes_per_update`` for a lower-variance gradient
        is only affordable if the episodes are split into groups of ~4.
        """
        n_ep = self.episodes_per_update
        group_size = max(1, int(self.config["train"].get("rollout_batch_envs", n_ep) or n_ep))
        if group_size >= n_ep:
            return self._collect_rollout_batched()
        buffer = RolloutBuffer(self.gamma, self.gae_lambda, self.device, value_target=self.value_target)
        rewards: list[float] = []
        summaries: list[dict] = []
        for start in range(0, n_ep, group_size):
            indices = list(range(start, min(start + group_size, n_ep)))
            self._collect_rollout_batched(env_indices=indices, buffer=buffer)
            rewards.extend(self.last_episode_rewards)
            summaries.extend(self.last_episode_summaries)
        self.last_episode_rewards = rewards
        self.last_episode_summaries = summaries
        return buffer

    def _collect_rollout_batched(self, env_indices=None, buffer: RolloutBuffer | None = None) -> RolloutBuffer:
        """Lockstep rollout: the episodes of one group step in parallel and a
        single block-diagonal policy forward serves every graph of the group per
        time step. Per-graph sampling is identical to serial rollout, so this is
        a pure throughput optimization.

        ``env_indices`` selects which of the ``episodes_per_update`` envs this
        call runs; ``collect_rollout`` drives it over
        ``rollout_batch_envs``-sized groups. One forward over 8 graphs measured
        ~5x the cost of one over 4 (409 ms vs 57 ms per step, i.e. 3.6x worse
        per graph), so stacking every episode into a single block-diagonal graph
        is counterproductive. The index list is explicit rather than positional
        so a caller can rotate which seeds share a group.
        """
        buffer = buffer if buffer is not None else RolloutBuffer(
            self.gamma, self.gae_lambda, self.device, value_target=self.value_target
        )
        self.model.eval()
        n_ep = self.episodes_per_update
        if self._rollout_envs is None:
            from qkd_rl.env.factory import build_env_from_config

            self._rollout_envs = [build_env_from_config(self.config) for _ in range(n_ep)]
        envs = self._rollout_envs
        if env_indices is None:
            env_indices = list(range(n_ep))
        group = [envs[i] for i in env_indices]
        for env in group:
            if not self.episode_steps_fixed:
                env.config["env"]["episode_steps"] = self._sample_episode_steps()
        base_seed = int(self.config["seed"]["env_seed"]) + self.update_count * self._seed_stride
        continuous = group[0].continuous
        if continuous and self._continuous_obs_list is not None:
            obs_list = [self._continuous_obs_list[i] for i in env_indices]
        else:
            obs_list = [envs[i].reset(seed=base_seed + i) for i in env_indices]
        n_group = len(env_indices)
        ep_steps: list[list[RolloutStep]] = [[] for _ in range(n_group)]
        ep_rewards = [0.0] * n_group
        done = [False] * n_group
        terminated = [False] * n_group
        for _ in range(self.rollout_steps):
            if all(done):
                break
            with torch.no_grad():
                step_outs = self.policy.act_batched(
                    obs_list,
                    build_scores=self._needs_edge_scores,
                    # The index-array sampler path avoids rebuilding a ~300-entry
                    # arc dict per graph per step; it is only valid when the
                    # resolver does not need the per-edge score dict.
                    use_edge_arrays=not self._needs_edge_scores,
                )
            for k, env in enumerate(group):
                if done[k]:
                    continue
                step = step_outs[k]
                raw_value = step.value.detach()
                next_obs, reward, term, trunc, info = env.step(
                    step.actions,
                    step.action_scores,
                    edge_scores=step.edge_scores,
                    expected_matched_edges=(
                        None
                        if self.resolver_mode == "max_weight_matching"
                        else list(step.matched_edges or [])
                    ),
                )
                self._update_rollout_debug(info, len(env.last_activated_edges))
                if self.resolver_mode == "max_weight_matching":
                    matched_edges = list(env.last_activated_edges)
                    mean_lp, mean_entropy = self.policy.log_prob_entropy_for_matching(
                        step.edge_scores, matched_edges
                    )
                else:
                    matched_edges = list(step.matched_edges or [])
                    mean_lp = step.mean_log_prob.detach()
                    mean_entropy = step.mean_entropy.detach()
                ep_steps[k].append(
                    RolloutStep(
                        obs=state_free_obs(obs_list[k]),
                        actions=step.actions,
                        log_probs={node: lp.detach() for node, lp in step.log_probs.items()},
                        entropies={node: ent.detach() for node, ent in step.entropies.items()},
                        value=raw_value,
                        reward=float(reward),
                        terminated=term,
                        truncated=trunc,
                        mean_log_prob=mean_lp.detach(),
                        mean_entropy=mean_entropy.detach(),
                        matched_edges=matched_edges,
                    )
                )
                ep_rewards[k] += float(reward)
                obs_list[k] = next_obs
                if term or trunc:
                    done[k] = True
                    terminated[k] = bool(term)
        episode_rewards: list[float] = []
        episode_summaries: list[dict] = []
        for k, env in enumerate(group):
            if not ep_steps[k]:
                episode_rewards.append(ep_rewards[k])
                episode_summaries.append(env.metrics.episode_summary())
                continue
            if done[k] and terminated[k]:
                last_value = torch.zeros((), dtype=torch.float32, device=self.device)
            else:
                with torch.no_grad():
                    last_value = self.policy.act(
                        obs_list[k], build_scores=False
                    ).value.detach()
            for step in ep_steps[k]:
                buffer.add(step)
            buffer.finish_episode(last_value)
            episode_rewards.append(ep_rewards[k])
            episode_summaries.append(env.metrics.episode_summary())
        if continuous:
            self._continuous_obs_list = list(obs_list)
        self.last_episode_rewards = episode_rewards
        self.last_episode_summaries = episode_summaries
        return buffer

    # -------------------------------------------------------------------- update
    def _remember_replay(self, buffer: RolloutBuffer) -> None:
        """Append the latest rollout to the replay ring (disabled by default).

        ``replay_days=0`` (the recommended default) keeps PPO on-policy; the
        replay path is retained only as a legacy/ablation option. Replayed
        steps keep advantages computed by an older critic, so re-enabling it
        makes the update approximately off-policy without importance
        correction -- prefer raising ``episodes_per_update`` instead.
        """
        if self.replay_days <= 0:
            return
        self._replay_buffers.append(list(buffer.steps))
        if len(self._replay_buffers) > self.replay_days:
            self._replay_buffers.pop(0)

    def update(self, buffer: RolloutBuffer) -> UpdateStats:
        ppo = self.ppo_cfg
        epochs = int(ppo["epochs"])
        minibatch_size = int(ppo["minibatch_size"])
        clip_eps = float(ppo["clip_eps"])
        entropy_coef = float(ppo["entropy_coef"])
        value_coef = float(ppo["value_coef"])
        max_grad_norm = float(ppo["max_grad_norm"])
        normalize_adv = bool(ppo.get("normalize_advantages", True))
        target_kl = float(ppo.get("target_kl", 0.0)) or None

        self.model.train()
        replay_steps = [
            step
            for replay in self._replay_buffers
            for step in replay
        ]
        total_actor = 0.0
        total_critic = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_ratio = 0.0
        total_actor_grad = 0.0
        total_critic_grad = 0.0
        total_batches = 0
        stop_for_kl = False
        for _epoch in range(epochs):
            if replay_steps:
                indices = list(range(len(replay_steps)))
                self.rng.shuffle(indices)
                batch_iter = (
                    [replay_steps[idx] for idx in indices[start : start + minibatch_size]]
                    for start in range(0, len(replay_steps), minibatch_size)
                )
            else:
                batch_iter = buffer.sample(minibatch_size, self.rng)
            for batch in batch_iter:
                actor_loss, critic_loss, entropy_mean, kl_mean, ratio_mean = self._loss_for_batch(
                    batch,
                    clip_eps=clip_eps,
                    entropy_coef=entropy_coef,
                    value_coef=value_coef,
                    normalize_adv=normalize_adv,
                )
                loss = actor_loss + critic_loss - entropy_coef * entropy_mean
                self.optimizer.zero_grad()
                loss.backward()
                if self.clip_per_role:
                    # Clip the policy and the value function separately.
                    #
                    # A single global clip lets the critic crowd out the actor:
                    # the critic's gradient norm is an order of magnitude larger
                    # than the actor's on this model (measured 1.07 vs 0.045 per
                    # role), because the actor's loss is a per-decision mean over
                    # ~40 decisions while the critic's is a squared error against
                    # a return of order 1. With one global clip at 0.5 the actor
                    # was left with ~4% of the gradient energy and its effective
                    # step shrank with it; the policy-group norm is now measured
                    # at ~0.42, i.e. below the clip, so the actor's full gradient
                    # survives.
                    grad_norm = torch.nn.utils.clip_grad_norm_(self._policy_params, max_grad_norm)
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(self._value_params, max_grad_norm)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    critic_grad_norm = torch.zeros((), dtype=torch.float32, device=self.device)
                self.optimizer.step()

                total_actor += float(actor_loss.detach().cpu())
                total_critic += float(critic_loss.detach().cpu())
                total_entropy += float(entropy_mean.detach().cpu())
                total_kl += float(kl_mean.detach().cpu())
                total_ratio += float(ratio_mean.detach().cpu())
                total_actor_grad += float(grad_norm.detach().cpu())
                total_critic_grad += float(critic_grad_norm.detach().cpu())
                total_batches += 1

                if target_kl is not None and float(kl_mean.detach().cpu()) > target_kl:
                    stop_for_kl = True
                    break
            if stop_for_kl:
                break
        n = max(1, total_batches)
        mean_reward = sum(self.last_episode_rewards) / max(1, len(self.last_episode_rewards))
        success_rates = [summary.get("success_rate", 0.0) for summary in self.last_episode_summaries]
        served = [summary.get("served_keys", 0.0) for summary in self.last_episode_summaries]
        returns = torch.stack([step.returns.to(self.device) for step in buffer.steps]) if buffer.steps else torch.zeros(())
        mean_return = float(returns.mean().detach().cpu()) if buffer.steps else 0.0
        adv_all = torch.stack([step.advantages.to(self.device) for step in buffer.steps]) if buffer.steps else torch.zeros(())
        mean_abs_advantage = float(adv_all.abs().mean().detach().cpu()) if buffer.steps else 0.0
        # Critic-fit diagnostics: if value_std collapses relative to return_std,
        # GAE has no baseline and every advantage is just the return.
        values_all = (
            torch.stack([step.value.detach().to(self.device).reshape(()) for step in buffer.steps])
            if buffer.steps else torch.zeros(())
        )
        value_std = float(values_all.std().detach().cpu()) if buffer.steps else 0.0
        return_std = float(returns.std().detach().cpu()) if buffer.steps else 0.0
        if buffer.steps and value_std > 0.0 and return_std > 0.0:
            value_return_corr = float(
                torch.corrcoef(torch.stack([values_all, returns]))[0, 1].detach().cpu()
            )
        else:
            value_return_corr = 0.0
        stats = UpdateStats(
            update=self.update_count,
            actor_loss=total_actor / n,
            critic_loss=total_critic / n,
            entropy=total_entropy / n,
            kl=total_kl / n,
            mean_reward=mean_reward,
            mean_return=mean_return,
            mean_abs_advantage=mean_abs_advantage,
            mean_success_rate=sum(success_rates) / max(1, len(success_rates)),
            mean_served_keys=sum(served) / max(1, len(served)),
            rollout_s=0.0,
            update_s=0.0,
            elapsed_s=0.0,
            mean_ratio=total_ratio / n,
            actor_grad_norm=total_actor_grad / n,
            critic_grad_norm=total_critic_grad / n,
            value_std=value_std,
            return_std=return_std,
            value_return_corr=value_return_corr,
        )
        self.last_stats = stats
        return stats

    def _loss_for_batch(
        self,
        batch: list[RolloutStep],
        clip_eps: float,
        entropy_coef: float,
        value_coef: float,
        normalize_adv: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not batch:
            raise ValueError("Empty minibatch in PPO update.")
        advantages = [step.advantages.to(self.device) for step in batch]
        if normalize_adv and len(advantages) > 1:
            stacked = torch.stack(advantages)
            mean = stacked.mean()
            std = stacked.std().clamp_min(1.0e-6)
            advantages = [(adv - mean) / std for adv in advantages]
        returns_batch = torch.stack([step.returns.to(self.device) for step in batch])
        critic_beta = torch.clamp(returns_batch.std(), min=1.0).detach()

        actor_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        critic_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        entropy_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        kl_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        ratio_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        # Batched PPO evaluation: one block-diagonal forward over many minibatch
        # graphs (chunked to bound GPU memory) instead of one forward per step.
        # The math is identical to per-step evaluation; only the CUDA kernels
        # are merged into larger batched ones.
        chunk_size = int(self.config["train"].get("ppo", {}).get("batch_chunk", 256))
        n_steps = 0
        for start in range(0, len(batch), chunk_size):
            chunk = batch[start:start + chunk_size]
            chunk_advantages = advantages[start:start + chunk_size]
            batched_results = self.policy.evaluate_actions_batched(
                [step.obs for step in chunk],
                [step.actions for step in chunk],
                [list(step.matched_edges or []) for step in chunk],
            )
            for step, adv, (log_probs, entropies, value) in zip(chunk, chunk_advantages, batched_results):
                node_ids = step.obs.node_ids
                if node_ids:
                    # PPO is evaluated on the whole matching action, not on the
                    # per-node copies of that same scalar. Every node shares the
                    # same scalar, so any node id yields it.
                    #
                    # Both sides of the ratio are the matching's PER-DECISION
                    # MEAN log probability (see ``MAPPOPolicy._sample_matching``).
                    # The sampler makes one decision per matched arc plus a final
                    # STOP -- up to ~60 of them -- so comparing raw sums would
                    # make a single parameter step look like a 60x larger policy
                    # change than it is: with the sum, clip_eps=0.1 and
                    # target_kl=0.02 were both exceeded after one optimizer step
                    # (kl 0.03-0.17), the KL early stop fired on the second
                    # minibatch of every update, and the actor advanced exactly
                    # one step per update. The entropy term below was already a
                    # per-decision mean, so this also puts all three quantities on
                    # one scale.
                    new_lp = log_probs[node_ids[0]]
                    old_lp = (
                        step.mean_log_prob.to(self.device)
                        if step.mean_log_prob is not None
                        else step.log_probs[node_ids[0]].to(self.device)
                    )
                    ratios = torch.exp(new_lp - old_lp)
                    surr1 = ratios * adv
                    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                    actor_loss = actor_loss - torch.min(surr1, surr2)
                    entropy_sum = entropy_sum + entropies[node_ids[0]]
                    kl_sum = kl_sum + (ratios - 1.0 - (new_lp - old_lp))
                    ratio_sum = ratio_sum + ratios
                returns_target = step.returns.to(self.device)
                # Huber loss keeps the critic robust to high-reward outlier
                # episodes; the beta scales with the current batch so the
                # loss stays comparable across different return magnitudes.
                critic_loss = critic_loss + torch.nn.functional.smooth_l1_loss(
                    value, returns_target, beta=critic_beta
                )
                n_steps += 1
            del batched_results, chunk_advantages
        actor_loss = actor_loss / n_steps
        critic_loss = (critic_loss / n_steps) * value_coef
        return (
            actor_loss,
            critic_loss,
            entropy_sum / n_steps,
            kl_sum / n_steps,
            ratio_sum / n_steps,
        )

    # ------------------------------------------------------------------ evaluate
    def evaluate(self, num_episodes: int | None = None) -> dict:
        """Run deterministic rollouts and report mean reward / success rate.

        With multiple episodes the rollouts run in lockstep and one batched
        forward serves all graphs per step (same deterministic policy, so the
        per-episode trajectories are identical to serial evaluation).
        """
        num_episodes = int(num_episodes or self.eval_episodes)
        self.model.eval()
        base_seed = int(self.config["seed"]["env_seed"]) + 1000000
        rewards: list[float] = []
        success_rates: list[float] = []
        served: list[float] = []
        with torch.no_grad():
            if num_episodes > 1:
                if self._eval_envs is None:
                    from qkd_rl.env.factory import build_env_from_config

                    self._eval_envs = [
                        build_env_from_config(self.config) for _ in range(num_episodes)
                    ]
                envs = self._eval_envs
                if self.eval_steps > 0:
                    for env in envs:
                        env.continuous = False
                        env.config["env"]["episode_steps"] = self.eval_steps
                obs_list = [env.reset(seed=base_seed + i) for i, env in enumerate(envs)]
                ep_rewards = [0.0] * num_episodes
                done = [False] * num_episodes
                while not all(done):
                    outs = self.policy.act_batched(
                        obs_list, deterministic=True, build_scores=self._needs_edge_scores
                    )
                    for i, env in enumerate(envs):
                        if done[i]:
                            continue
                        obs, reward, terminated, truncated, _info = env.step(
                            outs[i].actions,
                            outs[i].action_scores,
                            edge_scores=outs[i].edge_scores,
                            expected_matched_edges=(
                                None
                                if self.resolver_mode == "max_weight_matching"
                                else list(outs[i].matched_edges or [])
                            ),
                        )
                        ep_rewards[i] += float(reward)
                        obs_list[i] = obs
                        done[i] = terminated or truncated
                for i, env in enumerate(envs):
                    summary = env.metrics.episode_summary()
                    rewards.append(ep_rewards[i])
                    success_rates.append(summary.get("success_rate", 0.0))
                    served.append(summary.get("served_keys", 0.0))
            else:
                old_continuous = self.env.continuous
                old_episode_steps = int(self.env.config["env"]["episode_steps"])
                if self.eval_steps > 0:
                    self.env.continuous = False
                    self.env.config["env"]["episode_steps"] = self.eval_steps
                try:
                    obs = self.env.reset(seed=base_seed)
                    ep_reward = 0.0
                    done = False
                    while not done:
                        step = self.policy.act(
                            obs, deterministic=True, build_scores=self._needs_edge_scores
                        )
                        obs, reward, terminated, truncated, _info = self.env.step(
                            step.actions,
                            step.action_scores,
                            edge_scores=step.edge_scores,
                            expected_matched_edges=(
                                None
                                if self.resolver_mode == "max_weight_matching"
                                else list(step.matched_edges or [])
                            ),
                        )
                        ep_reward += float(reward)
                        done = terminated or truncated
                    summary = self.env.metrics.episode_summary()
                finally:
                    self.env.continuous = old_continuous
                    self.env.config["env"]["episode_steps"] = old_episode_steps
                rewards.append(ep_reward)
                success_rates.append(summary.get("success_rate", 0.0))
                served.append(summary.get("served_keys", 0.0))
        self.model.train()
        return {
            "mean_reward": sum(rewards) / max(1, len(rewards)),
            "mean_success_rate": sum(success_rates) / max(1, len(success_rates)),
            "mean_served_keys": sum(served) / max(1, len(served)),
        }

    # ----------------------------------------------------------- validation eval
    @property
    def validation_enabled(self) -> bool:
        window = self.validation_cfg.get("window", {}) or {}
        return window.get("start_day") is not None and window.get("end_day") is not None

    def _build_validation_envs(self, num_episodes: int | None = None, step_limit: int | None = None) -> list[object]:
        if self._validation_envs is not None and num_episodes is None and step_limit is None:
            return self._validation_envs

        import copy

        window = self.validation_cfg.get("window", {}) or {}
        start_day = int(window.get("start_day", 0))
        end_day = int(window.get("end_day", 365))
        seeds = [int(s) for s in (self.validation_cfg.get("request_seeds", []) or [])]
        if not seeds:
            seeds = list(range(7, 7 + int(self.validation_cfg.get("episodes", 3) or 3)))
        # Optional override: number of eval episodes (REUSE the seed list by
        # cycling, exactly like the evaluator: seed = seeds[ep % len(seeds)]).
        if num_episodes is not None and num_episodes > 0:
            cycles = max(1, int(num_episodes))
            seeds = [seeds[i % len(seeds)] for i in range(cycles)]
        episode_steps = int(step_limit) if (step_limit is not None and step_limit > 0) else 0
        if episode_steps <= 0:
            episode_steps = int(self.validation_cfg.get("episode_steps", 0) or 0)
        if episode_steps <= 0:
            # Back-compat: derive from episode_days if episode_steps not set
            episode_days = int(self.validation_cfg.get("episode_days", 1) or 1)
            day_steps = int(self.config["env"].get("day_steps", 1440))
            episode_steps = episode_days * day_steps
        day_steps = int(self.config["env"].get("day_steps", 1440))

        config = copy.deepcopy(self.config)
        # Use the canonical validation start mode (default random_day), NOT a
        # hardcoded value, so trainer validation matches the evaluator.
        config["env"]["episode_start_mode"] = str(
            self.validation_cfg.get("start_mode", "random_day")
        )
        config["env"]["episode_steps"] = episode_steps
        config["env"]["continuous"] = False
        config["env"]["activation_window_start_day"] = start_day
        config["env"]["activation_window_end_day"] = end_day
        config["env"]["activation_window_days"] = max(0, end_day - start_day)
        config["scenario"]["time_limit"]["days"] = end_day + max(
            1, math.ceil(episode_steps / day_steps)
        )
        config["seed"]["env_seed"] = seeds[0]

        from qkd_rl.env.factory import build_env_from_config

        self._validation_seeds = seeds
        self._validation_envs = [build_env_from_config(config) for _ in seeds]
        return self._validation_envs

    def evaluate_validation(self, num_episodes: int | None = None, step_limit: int | None = None) -> dict:
        """Deterministic rollouts on the held-out validation window."""
        envs = self._build_validation_envs(num_episodes=num_episodes, step_limit=step_limit)
        self.model.eval()
        start_seed_base = int(self.validation_cfg.get("start_seed", 0) or 0)
        obs_list = [
            env.reset(seed=seed, start_seed=start_seed_base + seed)
            for env, seed in zip(envs, self._validation_seeds)
        ]
        rewards = [0.0] * len(envs)
        done = [False] * len(envs)
        with torch.no_grad():
            while not all(done):
                outs = self.policy.act_batched(
                    obs_list, deterministic=True, build_scores=self._needs_edge_scores
                )
                for i, env in enumerate(envs):
                    if done[i]:
                        continue
                    obs, reward, terminated, truncated, _info = env.step(
                        outs[i].actions,
                        outs[i].action_scores,
                        edge_scores=outs[i].edge_scores,
                        expected_matched_edges=(
                            None
                            if self.resolver_mode == "max_weight_matching"
                            else list(outs[i].matched_edges or [])
                        ),
                    )
                    rewards[i] += float(reward)
                    obs_list[i] = obs
                    done[i] = terminated or truncated
        summaries = [env.metrics.episode_summary() for env in envs]
        self.model.train()
        return {
            "mean_reward": sum(rewards) / max(1, len(rewards)),
            "mean_success_rate": sum(
                summary.get("success_rate", 0.0) for summary in summaries
            )
            / max(1, len(summaries)),
            "mean_served_keys": sum(
                summary.get("served_keys", 0.0) for summary in summaries
            )
            / max(1, len(summaries)),
            "seeds": list(self._validation_seeds),
        }

    # ------------------------------------------------------------------- control
    def train(self, num_updates: int | None = None) -> dict:
        if num_updates is None:
            num_updates = self.num_updates
        num_updates = int(num_updates)
        target_updates = self.update_count + num_updates
        log_path = self.output_dir / "metrics.jsonl"
        try:
            for _ in range(num_updates):
                self._apply_curriculum()
                iteration_started = time.perf_counter()
                buffer = self.collect_rollout()
                rollout_finished = time.perf_counter()
                # Remember BEFORE updating so this rollout's steps train this
                # update (replay data is otherwise always one rollout stale).
                # No-op when replay_days=0 (the recommended on-policy default).
                self._remember_replay(buffer)
                stats = self.update(buffer)
                update_finished = time.perf_counter()
                stats.rollout_s = rollout_finished - iteration_started
                stats.update_s = update_finished - rollout_finished
                stats.elapsed_s = update_finished - iteration_started
                self.update_count += 1
                stats.update = self.update_count
                # Temperature annealing (exploration -> exploitation)
                if self.temp_start != self.temp_end:
                    frac = min(1.0, self.update_count / self.temp_updates)
                    self.model.actor.temperature = self.temp_start + frac * (self.temp_end - self.temp_start)
                elif self.temp_start != 1.0:
                    self.model.actor.temperature = self.temp_start
                if (
                    self.env.continuous
                    and self.continuous_session_updates > 0
                    and self.update_count % self.continuous_session_updates == 0
                ):
                    self._continuous_obs = None
                    self._continuous_obs_list = None
                self._write_rollout_debug(stats)
                self._log(stats, log_path)
                if self.update_count % self.checkpoint_interval == 0 or self.update_count == target_updates:
                    self.save_checkpoint(self.output_dir / f"checkpoint_update_{self.update_count:06d}.pt", stats)
                if self.eval_interval and self.update_count % self.eval_interval == 0:
                    if self.validation_enabled:
                        # Validated ONLY on the held-out validation config, so
                        # trainer validation is always aligned with the evaluator:
                        # episodes = validation.episodes, seeds cycle from
                        # validation.request_seeds, steps = validation.episode_steps.
                        val_stats = self.evaluate_validation(
                            num_episodes=int(self.validation_cfg.get("episodes", 1) or 1)
                        )
                        self._append_log({"eval_validation": val_stats}, log_path)
                        val_success = float(val_stats["mean_success_rate"])
                        if val_success > self.best_validation_success:
                            self.best_validation_success = val_success
                            self.save_checkpoint(
                                self.output_dir / "checkpoint_best_val.pt", stats
                            )
                            print(
                                f"new best validation success={val_success:.4f} "
                                "-> checkpoint_best_val.pt"
                            )
        finally:
            self.shutdown()
        return {"last_stats": asdict(self.last_stats) if self.last_stats else None}

    def shutdown(self) -> None:
        if self._rollout_pool is not None:
            self._rollout_pool.shutdown()
            self._rollout_pool = None

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

    def save_checkpoint(self, path: str | Path, stats: UpdateStats | None = None) -> Path:
        path = Path(path)
        save_checkpoint(
            path,
            update=self.update_count,
            model=self.model,
            optimizer=self.optimizer,
            config=self.config,
            metrics=asdict(stats) if stats is not None else None,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        data = load_checkpoint(path, self.device)
        self.model.load_state_dict(data.model_state)
        if data.config is not None and data.config.get("reward") != self.config.get("reward"):
            # The critic value head carries the scale of the OLD reward; reset
            # it so the stale value magnitude cannot poison the GAE bootstrap
            # and the advantages while the critic re-adapts. The shared encoder
            # and the actor keep their learned weights.
            self.model.critic.value_head.apply(_reset_module)
            print("reward config differs from checkpoint: re-initialized critic value head")
        if data.optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(data.optimizer_state)
            except ValueError as exc:
                # Pretraining checkpoints may use a single-parameter optimizer
                # while the trainer uses encoder/actor/critic groups. The
                # model weights are what matter for warm-starting; keep a
                # freshly initialized optimizer in that case.
                print(f"optimizer state incompatible ({exc}); starting optimizer fresh")
        self.update_count = data.update
        if data.config is not None:
            # Keep the CURRENT training config (env scenario, train schedule,
            # seeds, runtime) instead of overwriting it with the checkpoint's
            # stale config. Only model/optimizer state and the update counter
            # are resumed; this is what makes "resume and continue with a
            # different/longer schedule" safe.
            old_cfg = data.config
            print(
                f"checkpoint config: update={old_cfg.get('train', {}).get('num_updates')} "
                f"rollout={old_cfg.get('train', {}).get('rollout_steps')} "
                f"episodes={old_cfg.get('train', {}).get('episodes_per_update')} "
                f"seed={old_cfg.get('seed', {}).get('env_seed')} "
                f"device={old_cfg.get('runtime', {}).get('device')} "
                f"scenario={old_cfg.get('scenario', {}).get('mode')}"
            )
            print(f"running config: update={self.num_updates} rollout={self.rollout_steps} episodes={self.episodes_per_update}")

    # --------------------------------------------------------------------- logging
    def _log(self, stats: UpdateStats, log_path: Path) -> None:
        record = asdict(stats)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.log_interval and self.update_count % self.log_interval == 0:
            print(
                f"update={self.update_count:6d} actor_loss={stats.actor_loss:.4f} "
                f"critic_loss={stats.critic_loss:.4f} entropy={stats.entropy:.4f} "
                f"kl={stats.kl:.4f} reward={stats.mean_reward:.3f} "
                f"success_rate={stats.mean_success_rate:.3f} served={stats.mean_served_keys:.1f} "
                f"ratio={stats.mean_ratio:.4f} actor_grad={stats.actor_grad_norm:.4f} "
                f"critic_grad={stats.critic_grad_norm:.4f} "
                f"V_std={stats.value_std:.3f} R_std={stats.return_std:.2f} "
                f"corr(V,R)={stats.value_return_corr:.3f} "
                f"rollout_s={stats.rollout_s:.2f} update_s={stats.update_s:.2f}"
            )

    def _append_log(self, record: dict, log_path: Path) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
