"""MAPPO trainer: rollout collection, GAE, and PPO parameter updates."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from qkd_rl.algos.checkpoint import load_checkpoint, save_checkpoint
from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.algos.rollout_buffer import RolloutBuffer, RolloutStep
from qkd_rl.env.env import QKDEnv


def _reset_module(module: torch.nn.Module) -> None:
    """Re-initialize a module's parameters in place (identity preserved)."""
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


class ValueNormalizer:
    """Running mean/std normalization of GAE returns (official MAPPO style).

    The critic is trained to predict standardized returns (mean 0, std 1) so
    the value target stays well-conditioned regardless of the reward scale.
    During rollout the critic's standardized output is denormalized back into
    the raw scale before it is used as the GAE bootstrap.
    """

    def __init__(self, device: torch.device, tau: float = 0.05):
        self.mean = 0.0
        self.std = 1.0
        self.count = 0
        self.tau = float(tau)
        self.device = torch.device(device)

    def update(self, returns: torch.Tensor) -> None:
        """Exponential-moving-average update of the return statistics.

        The first batch seeds the statistics; afterwards each update moves the
        running mean/std by ``tau`` toward the new batch statistics. A full
        merge (the previous implementation) let one biased rollout permanently
        drag the running mean: because GAE returns are ``V + advantage`` and
        ``V`` is denormalized from the running mean, a biased mean made every
        later return more negative, a positive feedback loop that drifted the
        training return from ~-50 to ~-1100 while the true (bootstrap-free)
        return stayed ~-200. EMA keeps the target distribution stable so the
        critic can actually fit it.
        """
        if returns.numel() == 0:
            return
        m = float(returns.mean())
        s = float(returns.std(correction=0).clamp_min(1.0e-6))
        n = returns.numel()
        if self.count == 0:
            self.mean, self.std, self.count = m, s, n
            return
        self.mean = (1.0 - self.tau) * self.mean + self.tau * m
        self.std = (1.0 - self.tau) * self.std + self.tau * s
        self.count += n

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean + self.std * x

    def state_dict(self) -> dict[str, float]:
        return {
            "mean": float(self.mean),
            "std": float(self.std),
            "count": int(self.count),
            "tau": float(self.tau),
        }

    def load_state_dict(self, state: dict | None) -> None:
        if not state:
            return
        self.mean = float(state.get("mean", self.mean))
        self.std = float(state.get("std", self.std)) or 1.0
        self.count = int(state.get("count", self.count))
        self.tau = float(state.get("tau", self.tau))


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
        self._rollout_pool = None
        # Lockstep batched-rollout envs (episodes_per_update instances), built
        # lazily so single-episode runs never pay the construction cost.
        self._rollout_envs = None
        self._eval_envs = None
        self.value_norm = ValueNormalizer(
            self.device, tau=float(train_cfg.get("ppo", {}).get("value_norm_tau", 0.05))
        )
        # Critic target mode: "gae" (default) or "mc" (bootstrap-free returns,
        # see RolloutBuffer). "mc" breaks the value-normalizer feedback loop
        # that drifted GAE returns to ~5x the true return scale.
        self.value_target = str(train_cfg.get("value_target", "gae"))
        # Optional curriculum: a list of stages that lengthen the rollout and
        # adjust the number of parallel episodes as training progresses. The
        # trainer starts on short episodes (frequent updates, easy horizon)
        # and gradually transitions to the full-day schedule.
        self.curriculum_stages = list(train_cfg.get("curriculum", {}).get("stages", []) or [])

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
    def collect_rollout(self) -> RolloutBuffer:
        n_workers = int(self.config["train"].get("n_rollout_workers", 1))
        if n_workers <= 1:
            if (
                self.episodes_per_update > 1
                and bool(self.config["train"].get("rollout_batch", True))
            ):
                return self._collect_rollout_batched()
            return self._collect_rollout_serial()
        if self._rollout_pool is None:
            from qkd_rl.algos.rollout_workers import RolloutWorkerPool

            worker_device = self.config["train"].get("rollout_worker_device", str(self.device))
            self._rollout_pool = RolloutWorkerPool(self.config, worker_device, n_workers)
        base_seed = int(self.config["seed"]["env_seed"]) + self.update_count * self.episodes_per_update
        seeds = [base_seed + ep for ep in range(self.episodes_per_update)]
        weights = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        results = self._rollout_pool.collect(weights, seeds, self.rollout_steps, self.value_norm.mean, self.value_norm.std)
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

    def _collect_rollout_serial(self) -> RolloutBuffer:
        buffer = RolloutBuffer(self.gamma, self.gae_lambda, self.device, value_target=self.value_target)
        self.model.eval()
        base_seed = int(self.config["seed"]["env_seed"]) + self.update_count * self.episodes_per_update
        episode_rewards: list[float] = []
        episode_summaries: list[dict] = []
        for ep_idx in range(self.episodes_per_update):
            obs = self.env.reset(seed=base_seed + ep_idx)
            ep_reward = 0.0
            terminated = False
            truncated = False
            steps = 0
            while steps < self.rollout_steps:
                with torch.no_grad():
                    step = self.policy.act(obs)
                value_denorm = self.value_norm.denormalize(step.value.detach())
                next_obs, reward, terminated, truncated, _info = self.env.step(
                    step.actions,
                    step.action_scores,
                    edge_scores=step.edge_scores,
                    expected_matched_edges=(
                        None
                        if self.resolver_mode == "max_weight_matching"
                        else list(step.matched_edges or [])
                    ),
                )
                if self.resolver_mode == "max_weight_matching":
                    # max-weight is a deterministic resolver action, so the PPO
                    # target follows the matching the environment executed.
                    matched_edges = list(self.env.last_activated_edges)
                    joint_lp, joint_entropy = self.policy.log_prob_entropy_for_matching(
                        step.edge_scores, matched_edges
                    )
                else:
                    # The policy samples a global matching in a specific order;
                    # the joint log-prob is the probability of that sampled
                    # sequence, so it must be stored as-is instead of being
                    # recomputed from the resolver's output order.
                    matched_edges = list(step.matched_edges or [])
                    joint_lp = step.joint_log_prob.detach()
                    joint_entropy = step.joint_entropy.detach()
                buffer.add(
                    RolloutStep(
                        obs=obs,
                        actions=step.actions,
                        log_probs={node: lp.detach() for node, lp in step.log_probs.items()},
                        entropies={node: ent.detach() for node, ent in step.entropies.items()},
                        value=value_denorm,
                        reward=float(reward),
                        terminated=terminated,
                        truncated=truncated,
                        joint_log_prob=joint_lp.detach(),
                        joint_entropy=joint_entropy.detach(),
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
                    last_value = self.value_norm.denormalize(self.policy.act(obs).value.detach())
            buffer.finish_episode(last_value)
            episode_rewards.append(ep_reward)
            episode_summaries.append(self.env.metrics.episode_summary())
        self.last_episode_rewards = episode_rewards
        self.last_episode_summaries = episode_summaries
        return buffer

    def _collect_rollout_batched(self) -> RolloutBuffer:
        """Lockstep multi-episode rollout: ``episodes_per_update`` envs step in
        parallel and one block-diagonal policy forward serves all graphs per
        time step. Per-graph sampling is identical to serial rollout, so this
        is a pure throughput optimization (the per-step CUDA kernels dominate
        on small graphs and the serial loop launches them once per graph)."""
        buffer = RolloutBuffer(self.gamma, self.gae_lambda, self.device, value_target=self.value_target)
        self.model.eval()
        n_ep = self.episodes_per_update
        if self._rollout_envs is None:
            from qkd_rl.env.factory import build_env_from_config

            self._rollout_envs = [build_env_from_config(self.config) for _ in range(n_ep)]
        envs = self._rollout_envs
        base_seed = int(self.config["seed"]["env_seed"]) + self.update_count * n_ep
        obs_list = [env.reset(seed=base_seed + i) for i, env in enumerate(envs)]
        ep_steps: list[list[RolloutStep]] = [[] for _ in range(n_ep)]
        ep_rewards = [0.0] * n_ep
        done = [False] * n_ep
        terminated = [False] * n_ep
        for _ in range(self.rollout_steps):
            if all(done):
                break
            with torch.no_grad():
                step_outs = self.policy.act_batched(obs_list)
            for i, env in enumerate(envs):
                if done[i]:
                    continue
                step = step_outs[i]
                value_denorm = self.value_norm.denormalize(step.value.detach())
                next_obs, reward, term, trunc, _info = env.step(
                    step.actions,
                    step.action_scores,
                    edge_scores=step.edge_scores,
                    expected_matched_edges=(
                        None
                        if self.resolver_mode == "max_weight_matching"
                        else list(step.matched_edges or [])
                    ),
                )
                if self.resolver_mode == "max_weight_matching":
                    matched_edges = list(env.last_activated_edges)
                    joint_lp, joint_entropy = self.policy.log_prob_entropy_for_matching(
                        step.edge_scores, matched_edges
                    )
                else:
                    matched_edges = list(step.matched_edges or [])
                    joint_lp = step.joint_log_prob.detach()
                    joint_entropy = step.joint_entropy.detach()
                ep_steps[i].append(
                    RolloutStep(
                        obs=obs_list[i],
                        actions=step.actions,
                        log_probs={node: lp.detach() for node, lp in step.log_probs.items()},
                        entropies={node: ent.detach() for node, ent in step.entropies.items()},
                        value=value_denorm,
                        reward=float(reward),
                        terminated=term,
                        truncated=trunc,
                        joint_log_prob=joint_lp.detach(),
                        joint_entropy=joint_entropy.detach(),
                        matched_edges=matched_edges,
                    )
                )
                ep_rewards[i] += float(reward)
                obs_list[i] = next_obs
                if term or trunc:
                    done[i] = True
                    terminated[i] = bool(term)
        episode_rewards: list[float] = []
        episode_summaries: list[dict] = []
        for i, env in enumerate(envs):
            if not ep_steps[i]:
                episode_rewards.append(ep_rewards[i])
                episode_summaries.append(env.metrics.episode_summary())
                continue
            if done[i] and terminated[i]:
                last_value = torch.zeros((), dtype=torch.float32, device=self.device)
            else:
                with torch.no_grad():
                    last_value = self.value_norm.denormalize(self.policy.act(obs_list[i]).value.detach())
            for step in ep_steps[i]:
                buffer.add(step)
            buffer.finish_episode(last_value)
            episode_rewards.append(ep_rewards[i])
            episode_summaries.append(env.metrics.episode_summary())
        self.last_episode_rewards = episode_rewards
        self.last_episode_summaries = episode_summaries
        return buffer

    # -------------------------------------------------------------------- update
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
        # Update the running return statistics and attach standardized critic
        # targets so the value function is well-conditioned at any reward scale.
        if buffer.steps:
            all_returns = torch.stack([step.returns.to(self.device) for step in buffer.steps])
            self.value_norm.update(all_returns)
            for step in buffer.steps:
                step.returns_norm = self.value_norm.normalize(step.returns.to(self.device)).detach()
        total_actor = 0.0
        total_critic = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_ratio = 0.0
        total_actor_grad = 0.0
        total_batches = 0
        stop_for_kl = False
        for _epoch in range(epochs):
            for batch in buffer.sample(minibatch_size, self.rng):
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
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()

                total_actor += float(actor_loss.detach().cpu())
                total_critic += float(critic_loss.detach().cpu())
                total_entropy += float(entropy_mean.detach().cpu())
                total_kl += float(kl_mean.detach().cpu())
                total_ratio += float(ratio_mean.detach().cpu())
                total_actor_grad += float(grad_norm.detach().cpu())
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
        batched_results: list[tuple[dict, dict, torch.Tensor]] = []
        for start in range(0, len(batch), chunk_size):
            chunk = batch[start:start + chunk_size]
            batched_results.extend(
                self.policy.evaluate_actions_batched(
                    [step.obs for step in chunk],
                    [step.actions for step in chunk],
                    [list(step.matched_edges or []) for step in chunk],
                )
            )
        n_steps = 0
        for step, adv, (log_probs, entropies, value) in zip(batch, advantages, batched_results):
            node_ids = step.obs.node_ids
            if node_ids:
                # PPO is now evaluated on the whole matching action, not on
                # the per-node copies of that same scalar.
                new_lp = torch.stack([log_probs[node_id] for node_id in node_ids])[0]
                old_lp = (
                    step.joint_log_prob.to(self.device)
                    if step.joint_log_prob is not None
                    else torch.stack([step.log_probs[node_id].to(self.device) for node_id in node_ids])[0]
                )
                ratios = torch.exp(new_lp - old_lp)
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                actor_loss = actor_loss - torch.min(surr1, surr2)
                entropy_sum = entropy_sum + torch.stack([ent for ent in entropies.values()])[0]
                kl_sum = kl_sum + (ratios - 1.0 - (new_lp - old_lp))
                ratio_sum = ratio_sum + ratios
            returns_target = (
                step.returns_norm
                if getattr(step, "returns_norm", None) is not None
                else step.returns.to(self.device)
            )
            critic_loss = critic_loss + torch.nn.functional.mse_loss(value, returns_target)
            n_steps += 1
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
                obs_list = [env.reset(seed=base_seed + i) for i, env in enumerate(envs)]
                ep_rewards = [0.0] * num_episodes
                done = [False] * num_episodes
                while not all(done):
                    outs = self.policy.act_batched(obs_list, deterministic=True)
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
                obs = self.env.reset(seed=base_seed)
                ep_reward = 0.0
                done = False
                while not done:
                    step = self.policy.act(obs, deterministic=True)
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
                rewards.append(ep_reward)
                success_rates.append(summary.get("success_rate", 0.0))
                served.append(summary.get("served_keys", 0.0))
        self.model.train()
        return {
            "mean_reward": sum(rewards) / max(1, len(rewards)),
            "mean_success_rate": sum(success_rates) / max(1, len(success_rates)),
            "mean_served_keys": sum(served) / max(1, len(served)),
        }

    # ------------------------------------------------------------------- control
    def train(self, num_updates: int | None = None) -> dict:
        num_updates = int(num_updates or self.num_updates)
        target_updates = self.update_count + num_updates
        log_path = self.output_dir / "metrics.jsonl"
        try:
            for _ in range(num_updates):
                self._apply_curriculum()
                iteration_started = time.perf_counter()
                buffer = self.collect_rollout()
                rollout_finished = time.perf_counter()
                stats = self.update(buffer)
                update_finished = time.perf_counter()
                stats.rollout_s = rollout_finished - iteration_started
                stats.update_s = update_finished - rollout_finished
                stats.elapsed_s = update_finished - iteration_started
                self.update_count += 1
                stats.update = self.update_count
                self._log(stats, log_path)
                if self.update_count % self.checkpoint_interval == 0 or self.update_count == target_updates:
                    self.save_checkpoint(self.output_dir / f"checkpoint_update_{self.update_count:06d}.pt", stats)
                if self.eval_interval and self.update_count % self.eval_interval == 0:
                    eval_stats = self.evaluate(self.eval_episodes)
                    self._append_log({"eval": eval_stats}, log_path)
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
            trainer_state={"value_norm": self.value_norm.state_dict()},
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
            self.optimizer.load_state_dict(data.optimizer_state)
        if data.trainer_state is not None:
            self.value_norm.load_state_dict(data.trainer_state.get("value_norm"))
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
                f"rollout_s={stats.rollout_s:.2f} update_s={stats.update_s:.2f}"
            )

    def _append_log(self, record: dict, log_path: Path) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
