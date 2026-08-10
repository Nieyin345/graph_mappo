"""Persistent multiprocessing rollout workers for MAPPO training.

Each worker process owns its own env + policy copy and runs whole episodes,
so the per-step env/model work is spread across CPU cores (and GPU if
``runtime.device`` is cuda). Weights are synced to the workers once per
rollout; finished episodes (returns/advantages computed in the worker) are
returned as ``RolloutStep`` lists.

Payload handoff uses temp files, not the mp queues: a single full-graph
observation is ~1.3 MB, so one update's worth (~300 steps) is hundreds of MB,
which reliably deadlocks Windows mp.Queue pipes. The queues only carry tiny
control messages (weights state dict + result seeds).
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import shutil
import tempfile
from pathlib import Path

import torch

from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.algos.rollout_buffer import RolloutBuffer, RolloutStep
from qkd_rl.env.factory import build_env_from_config
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic


def _run_episode(env, policy, seed: int, rollout_steps: int, gamma: float, gae_lambda: float, value_mean: float = 0.0, value_std: float = 1.0, value_target: str = "gae"):
    """Run one episode and return finished ``RolloutStep``s (CPU tensors).

    Returns ``(steps, episode_reward, episode_summary)``.
    """
    obs = env.reset(seed=seed)
    buffer = RolloutBuffer(gamma, gae_lambda, device="cpu", value_target=value_target)
    ep_reward = 0.0
    steps = 0
    while steps < rollout_steps:
        with torch.no_grad():
            step = policy.act(obs)
        resolver_mode = env.action_resolver.mode
        next_obs, reward, terminated, truncated, _info = env.step(
            step.actions,
            step.action_scores,
            edge_scores=step.edge_scores,
            expected_matched_edges=(
                None
                if resolver_mode == "max_weight_matching"
                else list(step.matched_edges or [])
            ),
        )
        if resolver_mode == "max_weight_matching":
            matched_edges = list(env.last_activated_edges)
            joint_lp, joint_entropy = policy.log_prob_entropy_for_matching(
                step.edge_scores, matched_edges
            )
            joint_lp = joint_lp.detach().cpu()
            joint_entropy = joint_entropy.detach().cpu()
        else:
            matched_edges = list(step.matched_edges or [])
            joint_lp = step.joint_log_prob.detach().cpu()
            joint_entropy = step.joint_entropy.detach().cpu()
        buffer.add(
            RolloutStep(
                obs=obs,
                actions=step.actions,
                log_probs={node: lp.detach().cpu() for node, lp in step.log_probs.items()},
                entropies={node: ent.detach().cpu() for node, ent in step.entropies.items()},
                value=(value_mean + value_std * step.value.detach().cpu()),
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                joint_log_prob=joint_lp,
                joint_entropy=joint_entropy,
                matched_edges=matched_edges,
            )
        )
        ep_reward += float(reward)
        obs = next_obs
        steps += 1
        if terminated or truncated:
            break
    if terminated:
        last_value = torch.zeros((), dtype=torch.float32)
    else:
        with torch.no_grad():
            last_value = (value_mean + value_std * policy.act(obs).value.detach().cpu())
    buffer.finish_episode(last_value)
    return buffer.steps, ep_reward, env.metrics.episode_summary()


def _worker_entry(config: dict, device: str, task_queue, result_queue, job_dir: str, _worker_id: int) -> None:
    """Long-lived worker: build env + model once, then run episodes on demand."""
    torch.set_num_threads(1)
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config).to(device)
    policy = MAPPOPolicy(model, device)
    policy.model.eval()
    gamma = float(config["train"]["gamma"])
    gae_lambda = float(config["train"]["gae_lambda"])
    value_target = str(config["train"].get("value_target", "gae"))
    while True:
        task = task_queue.get()
        if task is None:
            break
        weights, seed, rollout_steps, value_mean, value_std = task
        if weights is not None:
            policy.model.load_state_dict(weights)
        steps, ep_reward, summary = _run_episode(env, policy, seed, rollout_steps, gamma, gae_lambda, value_mean, value_std, value_target)
        path = Path(job_dir) / f"job_{seed}.pkl"
        with path.open("wb") as f:
            pickle.dump((steps, ep_reward, summary), f, protocol=4)
        result_queue.put(seed)


class RolloutWorkerPool:
    """Spawned once and reused across updates; dispatch episodes to workers."""

    def __init__(self, config: dict, device, n_workers: int):
        self.n_workers = int(n_workers)
        self.device = str(device)
        self.config = config
        self.job_dir = Path(tempfile.mkdtemp(prefix="qkd_rollout_"))
        ctx = mp.get_context("spawn")
        self.task_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self._closed = False
        self.processes = [
            ctx.Process(
                target=_worker_entry,
                args=(config, self.device, self.task_queue, self.result_queue, str(self.job_dir), i),
            )
            for i in range(self.n_workers)
        ]
        for process in self.processes:
            process.start()

    def collect(
        self,
        weights: dict,
        seeds: list[int],
        rollout_steps: int,
        value_mean: float = 0.0,
        value_std: float = 1.0,
    ) -> list[tuple[int, list, float, dict]]:
        """Dispatch one episode per seed; returns results sorted by seed."""
        for seed in seeds:
            self.task_queue.put((weights, seed, int(rollout_steps), float(value_mean), float(value_std)))
        results: list[tuple[int, list, float, dict]] = []
        for _ in seeds:
            seed_done = self.result_queue.get()
            path = self.job_dir / f"job_{seed_done}.pkl"
            with path.open("rb") as f:
                steps, ep_reward, summary = pickle.load(f)
            path.unlink(missing_ok=True)
            results.append((seed_done, steps, ep_reward, summary))
        results.sort(key=lambda item: item[0])
        return results

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            for _ in self.processes:
                self.task_queue.put(None)
            for process in self.processes:
                process.join(timeout=60)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
        finally:
            for queue in (self.task_queue, self.result_queue):
                try:
                    queue.close()
                    queue.join_thread()
                except Exception:
                    pass
            shutil.rmtree(self.job_dir, ignore_errors=True)

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
