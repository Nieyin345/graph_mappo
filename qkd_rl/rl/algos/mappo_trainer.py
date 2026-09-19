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
from qkd_rl.rl.algos.rollout_buffer import (
    RolloutBuffer,
    RolloutStep,
    accumulate_rollout_debug,
    new_rollout_debug,
    state_free_obs,
)
from qkd_rl.env.env import QKDEnv


def _reset_module(module: torch.nn.Module) -> None:
    """Re-initialize a module's parameters in place (identity preserved)."""
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def _expand_last_dim(t: torch.Tensor, new_dim: int) -> torch.Tensor:
    """把张量最后一维加宽到 ``new_dim``，**新增的列/行初始化为零**。

    专为「给已有模型的 encoder 输入接上一段新特征」而写：新特征的投影权重
    置零 ⟹ 前向传播逐位等于旧模型（新分支贡献恰好为 0），于是**旧 checkpoint
    仍然等价可用**，新分支再从零开始学。

    只处理最后一维变宽的情形；其余形状差异直接交给调用方判定为"不可升级"。
    """
    if t.dim() == 0 or t.size(-1) >= new_dim:
        raise ValueError("_expand_last_dim: %s 无法加宽到 %d" % (tuple(t.shape), new_dim))
    pad = list(t.shape[:-1]) + [new_dim - t.size(-1)]
    return torch.cat([t, t.new_zeros(pad)], dim=-1)


def _upgrade_state_dict_for_model(
    model: torch.nn.Module, state: dict
) -> tuple[dict, list[str], list[str], list[str], list[str]]:
    """把 checkpoint 的 state_dict 适配到**当前**模型的结构上。

    返回 ``(新 state, 已加宽, 已丢弃, 缺失, 多余)``。

    为什么需要它：本项目里「改结构」与「沿用权重」一直被当成互斥的两件事
    （`docs/当前状态与优化方向.md` §4：改形状 = 丢 BC 暖启动）。但其实只有当
    **形状变了且无法零填**时才必须丢。若某个权重只是**最后一维变宽**
    （典型：encoder 的输入维度因为接了新特征而变大），把它按
    ``[旧权重 | 0]`` 加宽就得到**逐位等价**的模型 ⟹ 暖启动可以保留。

    实测（2026-09-20，`history_encoder` 只开节点通道）：
    101 个键里 **100 个形状完全相同**，**只有 1 个**变宽
    （``encoder.node_proj.0.weight (128,17) → (128,81)``）⟹ 正好落在这个函数能救的范围。

    ⚠ **本函数放松了 `load_state_dict(strict=True)` 的两项检查**（缺失键、
    多余键），所以它**必须**把这两项回传给调用方**显式打印**——静默放宽是
    本项目反复吃过的亏（记忆 `failed-launch-must-be-loud`）。
    """
    cur = model.state_dict()
    out: dict = {}
    widened: list[str] = []
    dropped: list[str] = []
    missing: list[str] = []
    for k, v in cur.items():
        old = state.get(k)
        if old is None:
            # 新增的模块（如 history_encoder.*）：用当前初始化值，等价于从零学。
            out[k] = v
            missing.append(k)
        elif tuple(old.shape) == tuple(v.shape):
            out[k] = old
        elif old.dim() >= 1 and old.shape[:-1] == v.shape[:-1] and old.size(-1) < v.size(-1):
            out[k] = _expand_last_dim(old, v.size(-1))
            widened.append("%s %s→%s" % (k, tuple(old.shape), tuple(v.shape)))
        else:
            out[k] = v
            dropped.append("%s %s→%s" % (k, tuple(old.shape), tuple(v.shape)))
    unexpected = sorted(k for k in state if k not in cur)
    return out, widened, dropped, missing, unexpected


def _torch_names(module: torch.nn.Module, prefix: str = "") -> list[str]:
    """返回 ``module`` 中**有 ``.data`` 的属性**的限定名（torch.optim 的收集口径）。

    为什么不用 ``module.state_dict()``：本项目的 ``node_proj`` 是 ``nn.Sequential``，
    字典里是 ``node_proj.0.weight``，而优化器收集的是 ``node_proj._modules['0'].weight``。
    两者顺序**一致**，但字符串不同；这里要的是**与优化器 state 索引对齐**的那一份。
    """
    out = []
    # 顺序必须与 `torch.nn.Module.named_parameters()` 一致：**先自身、再递归子模块**
    # 的前序遍历（torch 的 `_named_members` 就是这么走的）。写反了会让名字与
    # 优化器的 state 索引错位，打印出来的说明就是错的。
    for name, param in module.__dict__.get("_parameters", {}).items():
        if param is not None:
            out.append(prefix + name)
    for name, buf in module.__dict__.get("_buffers", {}).items():
        if buf is not None:
            out.append(prefix + name)
    for name, child in module.__dict__.get("_modules", {}).items():
        if child is not None:
            out.extend(_torch_names(child, prefix + name + "."))
    return out


def _upgrade_optimizer_state_for_model(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, opt_state: dict
) -> list[str]:
    """把 checkpoint 的**优化器**状态适配到当前结构上（就地修改 ``opt_state``）。

    为什么必须有这个：``Optimizer.load_state_dict`` 只校验 **param_groups** 的
    结构（组数、每组参数个数），**完全不校验每个 state 张量的形状**。于是
    模型被 ``_upgrade_state_dict_for_model`` 加宽之后（如
    ``encoder.node_proj.0.weight (128,17)→(128,81)``），Adam 的
    ``exp_avg``/``exp_avg_sq`` 仍是 ``(128,17)``，**静默通过加载**，
    直到**第一次 ``optimizer.step()``** 才抛
    ``RuntimeError: The size of tensor a (17) must match the size of tensor b (81)``。

    实测（2026-09-20）：这个 bug 在 ``seq_len: 240`` 时**被 OOM 掩盖**——
    第一个 update 之前进程就被杀了，从没走到 ``step()``；把 ``seq_len`` 降到 32
    跑得够快，才第一次走到那里并暴露出来。

    加宽规则与模型侧完全一致：**新增的列置零**。这既是数学上正确的
    （新输入的梯度贡献初始为 0 ⟹ 一阶矩本就该是 0），也顺带让
    ``exp_avg`` 与加宽后的权重对齐。**无动量近似**：优化器状态是按
    ``param_groups`` 的**全局顺序**存的扁平列表，形状变了没有可靠的对应关系，
    所以放不进去的一律丢弃并打印。

    返回人类可读的处理说明（调用方必须打印）。**绝不允许静默跳过。**
    """
    msgs: list[str] = []
    if not opt_state or "state" not in opt_state or "param_groups" not in opt_state:
        return msgs
    target = [tuple(p.shape) for g in optimizer.param_groups for p in g["params"]]
    groups = opt_state["param_groups"]
    try:
        flat = [p for g in groups for p in g["params"]]
    except (TypeError, KeyError):
        return ["!! 优化器状态结构无法解析（param_groups 里没有 params），整体丢弃"]

    # 检查：索引必须连续覆盖 [0, len(target))
    if sorted(flat) != list(range(len(target))):
        return ["!! 优化器状态索引不是 0..%d 的排列 ⟹ 无法可靠映射，整体丢弃"
                % (len(target) - 1)]

    # 名字表只为打印用；顺序与 target 一致（见 _torch_names 的说明）。
    names: list[str] = []
    for g in optimizer.param_groups:
        if not g["params"]:
            continue
        root = g["params"][0]
        # 优化器的三组依次绑在 encoder / actor / critic 上（见 __init__）
        for attr in ("encoder", "actor", "critic"):
            sub = getattr(model, attr, None)
            if sub is not None and next(sub.parameters(), None) is root:
                names.extend(_torch_names(sub, attr + "."))
                break
    if len(names) != len(target):
        # 名字对不齐 ⟹ 打印出来的「哪个参数被加宽」会是错的。而这条日志正是
        # 事后核查的依据，宁可不打印也不能打印错的。
        names = ["<参数 %d>" % i for i in range(len(target))]

    # Adam 的 state 是混合的：`step` 是**标量计数器**（优化器超参数，不属于参数形状），
    # `exp_avg` / `exp_avg_sq` 才与参数同形。把前者当参数张量去比形状，
    # 会把**每一个**参数都判成「无法适配」并整槽丢弃 —— 那就等于静默清空 Adam。
    # 实测（2026-09-20）第一版就是这个错：打印出「形状相同 3 / 丢弃 90」。
    _HPARAMS = ("step",)

    n_ok = n_widen = n_drop = 0
    for idx, st in list(opt_state["state"].items()):
        idx = int(idx)
        if idx >= len(target):
            del opt_state["state"][idx]
            n_drop += 1
            msgs.append("    - 优化器槽位 %d 越界（模型只有 %d 个参数）" % (idx, len(target)))
            continue
        cur_shape = target[idx]
        name = names[idx]
        fixed = {}
        bad = None
        for k, t in st.items():
            if k in _HPARAMS or not torch.is_tensor(t) or t.dim() == 0:
                fixed[k] = t          # 超参数/标量：与参数形状无关，原样保留
            elif tuple(t.shape) == cur_shape:
                fixed[k] = t
                n_ok += 1
            elif (t.dim() >= 1 and t.shape[:-1] == cur_shape[:-1]
                  and t.size(-1) < cur_shape[-1]):
                fixed[k] = _expand_last_dim(t, cur_shape[-1])
                n_widen += 1
                msgs.append("    + 优化器 %s.%s %s→%s（新增列置零 ⟹ 无动量近似）"
                            % (name, k, tuple(t.shape), cur_shape))
            else:
                bad = "    - 优化器 %s.%s %s 无法适配 %s ⟹ 丢弃该槽位" % (
                    name, k, tuple(t.shape), cur_shape)
                break
        if bad is not None:
            n_drop += 1
            msgs.append(bad)
            del opt_state["state"][idx]
        else:
            opt_state["state"][idx] = fixed

    # 新参数（如 history_encoder.*）本来就没有优化器状态：Adam 会补齐，无需处理。
    msgs.insert(0, "优化器状态适配：形状相同 %d / 加宽 %d / 丢弃 %d"
                % (n_ok, n_widen, n_drop))
    return msgs


def _mean_ratio(numerators, denominators) -> float:
    """Sum(numerators) / Sum(denominators)，而不是逐项比值再平均。

    为什么按**总和**算：密钥效率要的是"整局一共用了多少密钥服务了多少需求"，
    这正是两个总量的比。逐项比值再平均会被小分母的局放大（有一局服务量很低时
    它的比值会很大），把整体比拉偏。分母为 0 的局直接跳过。
    """
    num = sum(float(x) for x in numerators)
    den = sum(float(x) for x in denominators)
    return float(num / den) if den > 0 else 0.0


# Minimum minibatches a PPO update must run before the KL early stop is allowed
# to fire. With epochs=1 an early stop aborts the whole update, so letting the
# very first minibatch decide throws away the entire epoch on one noisy draw.
_MIN_BATCHES_BEFORE_KL_STOP = 4


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
    # How many minibatches this update actually ran. With epochs=1 the KL early
    # stop aborts the WHOLE update, so a small number here means the update did
    # almost no work -- and `kl` above cannot reveal that, because it is the
    # mean over the batches that did run.
    n_minibatches: int = 0


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
        # 记下配置里的学习率。`load_checkpoint` 里的 load_state_dict 会把
        # param_groups 整组恢复（含 lr），所以热启动时配置值会被 checkpoint 里
        # 存的那个盖掉 —— 必须在加载之后按这份记录重新写回。见 load_checkpoint。
        self._configured_lrs = [
            float(opt_cfg["actor_lr"]),    # encoder（与 actor 共享策略组）
            float(opt_cfg["actor_lr"]),    # actor
            float(opt_cfg["critic_lr"]),   # critic
        ]
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.encoder.parameters(), "lr": self._configured_lrs[0]},
                {"params": self.model.actor.parameters(), "lr": self._configured_lrs[1]},
                {"params": self.model.critic.parameters(), "lr": self._configured_lrs[2]},
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

    def _reset_env(self, env, seed: int, **kwargs):
        """``env.reset`` with the action-sampling RNG pinned to the same seed.

        An episode is reproducible only when BOTH halves are seeded: the env
        seed fixes the request stream, and the policy's sample seed fixes the
        exploration noise drawn against it. Seeding only the env left the
        Gumbel draws to a global RNG that the workers never seed, so two runs
        of one config took different actions from the very first rollout.

        Always reset through here rather than calling ``env.reset`` directly.
        """
        obs = env.reset(seed=seed, **kwargs)
        self.policy.set_sample_seed(seed)
        return obs

    def _reset_envs(self, envs, base_seed: int, env_indices=None) -> list:
        """Batched ``_reset_env``: one sample generator per row.

        ``env_indices`` are the episode indices the caller selected out of the
        full env pool, so seeds stay ``base_seed + episode_index`` rather than
        ``base_seed + position_in_group`` -- the seed an episode gets must not
        depend on how the batch was split.
        """
        if env_indices is None:
            env_indices = list(range(len(envs)))
        obs_list = [env.reset(seed=base_seed + i) for env, i in zip(envs, env_indices)]
        self.policy.set_sample_seed([base_seed + i for i in env_indices])
        return obs_list

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
        for _seed, steps, ep_reward, summary, ep_debug in results:
            buffer.steps.extend(steps)
            episode_rewards.append(ep_reward)
            episode_summaries.append(summary)
            # Workers accumulate the same per-component telemetry the
            # single-process loop does (shared accumulate_rollout_debug), so
            # rollout_debug.jsonl stays available with n_workers > 1.
            for key, value in ep_debug.items():
                self._rollout_debug[key] += value
        self.last_episode_rewards = episode_rewards
        self.last_episode_summaries = episode_summaries
        return buffer

    def _reset_rollout_debug(self) -> None:
        self._rollout_debug = new_rollout_debug()

    def _update_rollout_debug(self, info: dict, activated_count: int) -> None:
        accumulate_rollout_debug(self._rollout_debug, info, activated_count)

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
                obs = self._reset_env(self.env, base_seed + ep_idx)
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
            # Seed each row's exploration noise from its own episode seed, so a
            # row's sampled matching does not depend on how many rows share the
            # batch or on which ones ran first. The returned obs MUST be kept:
            # dropping it (as a refactor once did) leaves `obs_list` unbound on
            # this path -- a latent UnboundLocalError that fires only when
            # `n_rollout_workers <= 1`, i.e. exactly the single-process runs
            # that are otherwise the most reproducible configuration.
            obs_list = self._reset_envs(group, base_seed, env_indices=env_indices)
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

                # Trip on the RUNNING MEAN KL, not on a single minibatch's.
                #
                # With epochs=1 an early stop aborts the ENTIRE update, so a
                # per-batch test is catastrophic: measured on a full-window run,
                # one noisy batch out of ~45 ended an update after 18.4 s of a
                # 302 s budget -- 4% of the work -- and `kl` could not reveal it
                # because it averages only over the batches that ran (it printed
                # 0.0092 while a single batch had exceeded 0.02). A mean still
                # catches a genuinely drifting policy, and the minimum-batch
                # guard stops the very first draw from deciding alone.
                if (
                    target_kl is not None
                    and total_batches >= _MIN_BATCHES_BEFORE_KL_STOP
                    and total_kl / total_batches > target_kl
                ):
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
            n_minibatches=total_batches,
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

        # Collect per-step scalars instead of accumulating in the loop.
        #
        # The math is identical: the final values are the MEAN over steps, and
        # mean(a) + mean(b) == mean(a + b) for equal-length sequences, so
        # accumulating then dividing is the same as stacking then meaning.
        # What changes is the autograd graph: `acc = acc + x` inside the loop
        # builds a chain of N scalar Add nodes, while one `torch.stack(...).mean()`
        # builds a single Sum node over an N-vector. On CPU the backward pass is
        # dominated by per-node engine overhead rather than arithmetic (measured:
        # run_backward tottime 3.5x the forward linear kernels), so collapsing
        # the chain is worth it.
        actor_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        kl_terms: list[torch.Tensor] = []
        ratio_terms: list[torch.Tensor] = []
        critic_terms: list[torch.Tensor] = []
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
                    actor_terms.append(-torch.min(surr1, surr2))
                    entropy_terms.append(entropies[node_ids[0]])
                    kl_terms.append(ratios - 1.0 - (new_lp - old_lp))
                    ratio_terms.append(ratios)
                returns_target = step.returns.to(self.device)
                # Huber loss keeps the critic robust to high-reward outlier
                # episodes; the beta scales with the current batch so the
                # loss stays comparable across different return magnitudes.
                critic_terms.append(torch.nn.functional.smooth_l1_loss(
                    value, returns_target, beta=critic_beta
                ))
                n_steps += 1
            del batched_results, chunk_advantages
        if n_steps == 0:
            raise ValueError("Minibatch produced no steps in PPO update.")
        actor_loss = torch.stack(actor_terms).mean()
        critic_loss = torch.stack(critic_terms).mean() * value_coef
        entropy_mean = torch.stack(entropy_terms).mean()
        kl_mean = torch.stack(kl_terms).mean()
        ratio_mean = torch.stack(ratio_terms).mean()
        return (actor_loss, critic_loss, entropy_mean, kl_mean, ratio_mean)

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
            # Per-seed values, aligned with `seeds` above (envs and
            # `_validation_seeds` are built from the same list, in order).
            #
            # Kept because the seed-to-seed spread dwarfs the effects being
            # measured: single-seed success on a fixed scenario spans ~0.29-0.88,
            # so a 12-seed MEAN has ~0.012 standard error and only same-seed
            # PAIRING gets below that. Without this, comparing two runs' means
            # silently folds the seed spread back in as noise -- measured at
            # ~0.06 unpaired vs ~0.012 paired, which is the difference between
            # seeing a 0.02 effect and not.
            "per_seed_success": [
                float(summary.get("success_rate", 0.0)) for summary in summaries
            ],
            # 2026-09-19 新增：**密钥效率**也逐种子记下来。
            #
            # 为什么（实测驱动，不是预防性加字段）：训练侧 `rollout_debug.jsonl`
            # 显示 RL 的密钥生成量单调掉 41%，而 `mean_reward_generated = 0`、
            # success_rate 与 reward 都不动 —— 奖励有一个很大的零空间。
            # 后来在**同一验证 regime**上把专家并排量，才判定那是**漂对了**：
            # RL 比专家省 36.5% 的密钥（3 个训练种子 22/42/45%，逐个 15/15 同向）
            # 而服务量不降。见 docs/训练诊断记录.md「奖励看不见的行为漂移」。
            #
            # 但那个判定**当时做不了**，因为本函数只存 `per_seed_success`，
            # 正好缺了发生漂移的那两维。补上后，训练过程中就能直接看到
            # 密钥效率、不必事后补评估。`served/generated` 越大越省。
            "per_seed_generated_keys": [
                float(summary.get("generated_keys", 0.0)) for summary in summaries
            ],
            "per_seed_served_keys": [
                float(summary.get("served_keys", 0.0)) for summary in summaries
            ],
            # 注意 `waiting_keys` 是**存量**：`episode_summary` 给的是整局均值
            # （累加÷步数），可与 `rollout_debug.jsonl` 的 `mean_waiting_keys`
            # 直接对照；不要与 `served_keys`（流量总量）混着比。
            "per_seed_waiting_keys_mean": [
                float(summary.get("waiting_keys_mean", 0.0)) for summary in summaries
            ],
            "mean_key_efficiency": _mean_ratio(
                [summary.get("generated_keys", 0.0) for summary in summaries],
                [summary.get("served_keys", 0.0) for summary in summaries],
            ),
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
                        # 与 success_rate 并排打印密钥效率：实测 RL 省 36.5% 密钥，
                        # 而这在 success_rate 上**完全看不见**。不打印出来，等于
                        # 每轮都在丢掉一个已证实的优势信号（见 evaluate_validation）。
                        print(
                            f"  validation success={val_success:.4f} "
                            f"key_eff={float(val_stats.get('mean_key_efficiency', 0.0)):.3f} "
                            "(生成/服务，越低越省)"
                        )
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
        # ★ 不再直接 `load_state_dict`（默认 strict=True，形状一变就抛异常，
        #   于是"改结构"被迫等于"丢暖启动"）。先让 checkpoint 适配当前结构：
        #   能靠**末维补零**救回的键就救（救回后前向逐位等价），其余才丢。
        state, widened, dropped, missing, unexpected = _upgrade_state_dict_for_model(
            self.model, data.model_state
        )
        if widened:
            print("checkpoint 结构升级：加宽 %d 个键（新增维置零 ⟹ 前向逐位等价）" % len(widened))
            for w in widened:
                print("    + %s" % w)
        if dropped:
            print("checkpoint 结构升级：丢弃 %d 个键（形状无法靠补零救回）" % len(dropped))
            for d in dropped:
                print("    - %s" % d)
        # strict=True 之外的两项，pyTorch 会静默放过；这里必须吵出来。
        if missing:
            print("checkpoint 缺失 %d 个键（用当前初始化值，等价于从零学）" % len(missing))
            for k in missing:
                print("    ? %s" % k)
        if unexpected:
            print("!! checkpoint 多余 %d 个键（模型里不存在，**已忽略**）" % len(unexpected))
            for k in unexpected:
                print("    ! %s" % k)
        self.model.load_state_dict(state)
        if data.config is not None and data.config.get("reward") != self.config.get("reward"):
            # The critic value head carries the scale of the OLD reward; reset
            # it so the stale value magnitude cannot poison the GAE bootstrap
            # and the advantages while the critic re-adapts. The shared encoder
            # and the actor keep their learned weights.
            self.model.critic.value_head.apply(_reset_module)
            print("reward config differs from checkpoint: re-initialized critic value head")
        if data.optimizer_state is not None:
            try:
                # ★★ 必须先把**优化器状态**也升级到当前结构，否则模型加宽之后
                #   Adam 的 exp_avg/exp_avg_sq 仍是旧宽度，而
                #   `Optimizer.load_state_dict` **只校验 param_groups 的结构、
                #   不校验 state 张量的形状** ⟹ 静默通过，直到第一次
                #   `optimizer.step()` 才抛 RuntimeError（实测 (17) vs (81)）。
                #   这不是可选的：对照臂是**带 Adam 矩**热启动的，若这里丢掉
                #   优化器状态，臂与对照就差**两**个变量（历史输入 + Adam 动量），
                #   配对不再干净。
                for msg in _upgrade_optimizer_state_for_model(
                    self.model, self.optimizer, data.optimizer_state
                ):
                    print("checkpoint " + msg if msg.startswith("优化器") else msg)
                self.optimizer.load_state_dict(data.optimizer_state)
                # load_state_dict 恢复的是**整组** param_groups，学习率也在里面。
                # 不写回的话，checkpoint 里存的那个 lr 会盖掉配置值，而这一点
                # 完全静默 —— 实测 BC 权重里三组都是 lr=0.001，于是
                # `train.optimizer.actor_lr: 0.0003` 从未生效过，基于它做的
                # 学习率实验（包括把 lr 改成 0.001 的那一组）全部是空跑：
                # 训练指标与对照逐位相同，只有 rollout_s/update_s 不同。
                # Adam 的一二阶矩（恢复时真正需要的部分）保持不动。
                for group, lr in zip(self.optimizer.param_groups, self._configured_lrs):
                    group["lr"] = lr
            except (ValueError, RuntimeError) as exc:
                # Pretraining checkpoints may use a single-parameter optimizer
                # while the trainer uses encoder/actor/critic groups. The
                # model weights are what matter for warm-starting; keep a
                # freshly initialized optimizer in that case.
                # ⚠ RuntimeError 也要抓：形状不匹配抛的是它，早先只抓 ValueError
                #   会让训练**在第一个 optimizer.step() 处**才炸（远离根因）。
                # ⚠ 这里一旦触发，臂与对照就差**两**个变量（结构 + Adam 动量），
                #   实验结果必须按「丢了优化器状态」来读，不能当干净配对。
                print(f"optimizer state incompatible ({type(exc).__name__}: {exc}); "
                      f"starting optimizer fresh —— ⚠ 本 run 与对照的差异不止一处，"
                      f"判读时要写明")
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
                f"nb={stats.n_minibatches} "
                f"rollout_s={stats.rollout_s:.2f} update_s={stats.update_s:.2f}"
            )

    def _append_log(self, record: dict, log_path: Path) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
