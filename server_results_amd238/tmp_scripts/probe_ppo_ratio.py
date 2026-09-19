"""rollout 存下来的 log-prob，和更新时重算的 log-prob 一致吗？

PPO 的比率是 ``exp(new_logp - old_logp)``。它只有在 old 和 new 出自同一个策略、
**同一套定义**时才等于 1。之前的检验只验了"更新期那两个 log-prob 函数互相等价"
（`_matching_log_prob_entropy_arrays` vs `_matching_log_prob_entropy_fast`），
从没验过 **rollout 存下来的那个**和它们是否同一个定义。

如果两者不一致 —— 比如一个按"每次决策平均"、另一个按"总和"，或者候选集/可行掩码
的范围不同 —— 比率会**系统性**偏离 1，PPO 就在优化一个错误的目标，训练再多轮也不会涨。
观察到的现象正好符合：`mean_ratio` 全程稳定在 0.93–0.99（**始终小于 1**），
同时策略熵一路上升（3.85 → 4.29）而成功率不动。

做法：同一份策略、同一批观测，先走 rollout 路径采样并记录 `mean_log_prob`，
再用更新路径对**同一批动作序列**重算，逐条比较，并给出隐含的比率。

用法：
    python .tmp/probe_ppo_ratio.py [--checkpoint PATH] [--steps 40] [--envs 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "rl"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        default="/opt/qkd/graph_mappo/outputs/supervised_pg_phased/supervised_pg_phased_latest.pt",
    )
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from qkd_rl.env.factory import build_env_from_config
    from qkd_rl.rl.algos.checkpoint import load_checkpoint
    from qkd_rl.rl.algos.policy import MAPPOPolicy
    from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic
    from train_graph_mappo import build_config

    ns = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"],
        mode="random_episode",
        run_name="probe_ratio",
        num_updates=None,
        seed=None,
        checkpoint=None,
        device=args.device,
    )
    config = build_config(ns)
    torch.manual_seed(int(config["seed"]["global_seed"]))
    torch.set_float32_matmul_precision("high")

    resolver = config["env"].get("resolver_mode", "mutual_choice")
    needs_edge_scores = resolver in ("priority_matching", "max_weight_matching")
    print(f"resolver_mode = {resolver}")
    print(f"_needs_edge_scores = {needs_edge_scores}  "
          f"-> rollout 用 {'字典' if needs_edge_scores else '数组'} 路径")

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    data = load_checkpoint(args.checkpoint, device=args.device)
    model.load_state_dict(data.model_state)
    model.eval()
    policy = MAPPOPolicy(model, args.device)

    envs = [build_env_from_config(config) for _ in range(args.envs)]
    obs_list = [e.reset(seed=args.seed + i) for i, e in enumerate(envs)]

    diffs: list[float] = []
    lp_roll_all: list[float] = []
    lp_upd_all: list[float] = []
    ent_roll_all: list[float] = []
    ent_upd_all: list[float] = []

    for t in range(args.steps):
        with torch.no_grad():
            steps = policy.act_batched(
                obs_list,
                build_scores=needs_edge_scores,
                use_edge_arrays=not needs_edge_scores,
            )
            evals = policy.evaluate_actions_batched(
                obs_list,
                [s.actions for s in steps],
                [s.matched_edges for s in steps],
            )
        for gi, (s, ev) in enumerate(zip(steps, evals)):
            lp_roll = float(s.mean_log_prob)
            ent_roll = float(s.mean_entropy)
            lp_dict, ent_dict, _v = ev
            if not lp_dict:
                continue
            lp_upd = float(next(iter(lp_dict.values())))
            ent_upd = float(next(iter(ent_dict.values())))
            diffs.append(lp_upd - lp_roll)
            lp_roll_all.append(lp_roll)
            lp_upd_all.append(lp_upd)
            ent_roll_all.append(ent_roll)
            ent_upd_all.append(ent_upd)
        for k, e in enumerate(envs):
            obs, _r, term, trunc, _i = e.step(
                steps[k].actions,
                steps[k].action_scores,
                edge_scores=steps[k].edge_scores,
                expected_matched_edges=list(steps[k].matched_edges or []),
            )
            if term or trunc:
                obs = e.reset(seed=args.seed + 1000 + k)
            obs_list[k] = obs

    n = len(diffs)
    if n == 0:
        print("没有采到样本（观测为空？）")
        return

    d = torch.tensor(diffs, dtype=torch.float64)
    ratios = torch.exp(d)

    print()
    print(f"样本数 {n}（{args.envs} 个环境 × {args.steps} 步）")
    print()
    print("=== log-prob：更新期重算 − rollout 存储 ===")
    print(f"  均值      {float(d.mean()):+.6e}")
    print(f"  标准差    {float(d.std()):.6e}")
    print(f"  最大绝对值 {float(d.abs().max()):.6e}")
    print(f"  隐含比率 exp(diff)：均值 {float(ratios.mean()):.6f}，"
          f"最小 {float(ratios.min()):.6f}，最大 {float(ratios.max()):.6f}")
    print()
    print("=== 参考量级 ===")
    lr_ = torch.tensor(lp_roll_all, dtype=torch.float64)
    lu_ = torch.tensor(lp_upd_all, dtype=torch.float64)
    print(f"  rollout 的 mean_log_prob：均值 {float(lr_.mean()):.4f}，"
          f"标准差 {float(lr_.std()):.4f}")
    print(f"  更新期的 mean_log_prob  ：均值 {float(lu_.mean()):.4f}，"
          f"标准差 {float(lu_.std()):.4f}")
    print()
    er_ = torch.tensor(ent_roll_all, dtype=torch.float64)
    eu_ = torch.tensor(ent_upd_all, dtype=torch.float64)
    ed = eu_ - er_
    print("=== 熵（更新期 − rollout）===")
    print(f"  均值 {float(ed.mean()):+.6e}，最大绝对值 {float(ed.abs().max()):.6e}")
    print(f"  rollout 熵均值 {float(er_.mean()):.4f}，更新期熵均值 {float(eu_.mean()):.4f}")
    print()
    print("判读：")
    print("  diff 的均值若显著非零（相对 log-prob 标准差不是小量），")
    print("  说明 rollout 与更新的 log-prob 定义不一致，PPO 比率是错的。")
    print("PROBE_RATIO_DONE")


if __name__ == "__main__":
    main()
