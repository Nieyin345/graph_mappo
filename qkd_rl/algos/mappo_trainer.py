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


@dataclass
class UpdateStats:
    update: int
    actor_loss: float
    critic_loss: float
    entropy: float
    kl: float
    mean_reward: float
    mean_return: float
    mean_success_rate: float
    mean_served_keys: float
    elapsed_s: float


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

    # ------------------------------------------------------------------ rollout
    def collect_rollout(self) -> RolloutBuffer:
        buffer = RolloutBuffer(self.gamma, self.gae_lambda, self.device)
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
                next_obs, reward, terminated, truncated, _info = self.env.step(
                    step.actions, step.action_scores
                )
                buffer.add(
                    RolloutStep(
                        obs=obs,
                        actions=step.actions,
                        log_probs={node: lp.detach() for node, lp in step.log_probs.items()},
                        entropies={node: ent.detach() for node, ent in step.entropies.items()},
                        value=step.value.detach(),
                        reward=float(reward),
                        terminated=terminated,
                        truncated=truncated,
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
                    last_value = self.policy.act(obs).value.detach()
            buffer.finish_episode(last_value)
            episode_rewards.append(ep_reward)
            episode_summaries.append(self.env.metrics.episode_summary())
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
        total_actor = 0.0
        total_critic = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_batches = 0
        for _epoch in range(epochs):
            for batch in buffer.sample(minibatch_size, self.rng):
                actor_loss, critic_loss, entropy_mean, kl_mean = self._loss_for_batch(
                    batch,
                    clip_eps=clip_eps,
                    entropy_coef=entropy_coef,
                    value_coef=value_coef,
                    normalize_adv=normalize_adv,
                )
                loss = actor_loss + critic_loss - entropy_coef * entropy_mean
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()

                total_actor += float(actor_loss.detach().cpu())
                total_critic += float(critic_loss.detach().cpu())
                total_entropy += float(entropy_mean.detach().cpu())
                total_kl += float(kl_mean.detach().cpu())
                total_batches += 1

                if target_kl is not None and float(kl_mean.detach().cpu()) > target_kl:
                    break
        n = max(1, total_batches)
        mean_reward = sum(self.last_episode_rewards) / max(1, len(self.last_episode_rewards))
        success_rates = [summary.get("success_rate", 0.0) for summary in self.last_episode_summaries]
        served = [summary.get("served_keys", 0.0) for summary in self.last_episode_summaries]
        returns = torch.stack([step.returns.to(self.device) for step in buffer.steps]) if buffer.steps else torch.zeros(())
        mean_return = float(returns.mean().detach().cpu()) if buffer.steps else 0.0
        stats = UpdateStats(
            update=self.update_count,
            actor_loss=total_actor / n,
            critic_loss=total_critic / n,
            entropy=total_entropy / n,
            kl=total_kl / n,
            mean_reward=mean_reward,
            mean_return=mean_return,
            mean_success_rate=sum(success_rates) / max(1, len(success_rates)),
            mean_served_keys=sum(served) / max(1, len(served)),
            elapsed_s=0.0,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        n_steps = 0
        for step, adv in zip(batch, advantages):
            log_probs, entropies, value = self.policy.evaluate_actions(step.obs, step.actions)
            node_log_probs = [log_probs[node_id] for node_id in step.obs.node_ids]
            node_old_log_probs = [step.log_probs[node_id].to(self.device) for node_id in step.obs.node_ids]
            if node_log_probs:
                ratios = torch.stack([torch.exp(lp - old) for lp, old in zip(node_log_probs, node_old_log_probs)])
                old_lps = torch.stack(node_old_log_probs)
                new_lps = torch.stack(node_log_probs)
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                actor_loss = actor_loss - torch.min(surr1, surr2).mean()
                entropy_sum = entropy_sum + torch.stack([ent for ent in entropies.values()]).mean()
                kl_sum = kl_sum + (ratios - 1.0 - (new_lps - old_lps)).mean()
            critic_loss = critic_loss + torch.nn.functional.mse_loss(value, step.returns.to(self.device))
            n_steps += 1
        actor_loss = actor_loss / n_steps
        critic_loss = (critic_loss / n_steps) * value_coef
        return actor_loss, critic_loss, entropy_sum / n_steps, kl_sum / n_steps

    # ------------------------------------------------------------------ evaluate
    def evaluate(self, num_episodes: int | None = None) -> dict:
        """Run deterministic rollouts and report mean reward / success rate."""
        num_episodes = int(num_episodes or self.eval_episodes)
        self.model.eval()
        base_seed = int(self.config["seed"]["env_seed"]) + 1000000
        rewards: list[float] = []
        success_rates: list[float] = []
        served: list[float] = []
        with torch.no_grad():
            for ep_idx in range(num_episodes):
                obs = self.env.reset(seed=base_seed + ep_idx)
                ep_reward = 0.0
                done = False
                while not done:
                    step = self.policy.act(obs, deterministic=True)
                    obs, reward, terminated, truncated, _info = self.env.step(step.actions, step.action_scores)
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
        for _ in range(num_updates):
            buffer = self.collect_rollout()
            t0 = time.time()
            stats = self.update(buffer)
            stats.elapsed_s = time.time() - t0
            self.update_count += 1
            stats.update = self.update_count
            self._log(stats, log_path)
            if self.update_count % self.checkpoint_interval == 0 or self.update_count == target_updates:
                self.save_checkpoint(self.output_dir / f"checkpoint_update_{self.update_count:06d}.pt", stats)
            if self.eval_interval and self.update_count % self.eval_interval == 0:
                eval_stats = self.evaluate(self.eval_episodes)
                self._append_log({"eval": eval_stats}, log_path)
        return {"last_stats": asdict(self.last_stats) if self.last_stats else None}

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
        if data.optimizer_state is not None:
            self.optimizer.load_state_dict(data.optimizer_state)
        self.update_count = data.update
        if data.config is not None:
            self.config = data.config

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
                f"success_rate={stats.mean_success_rate:.3f} served={stats.mean_served_keys:.1f}"
            )

    def _append_log(self, record: dict, log_path: Path) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")




