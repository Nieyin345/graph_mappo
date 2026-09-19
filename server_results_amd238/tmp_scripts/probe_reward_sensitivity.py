"""奖励到底能不能区分"好动作"和"坏动作"？

这是训练不涨的核心疑点。策略梯度只会往"奖励更高的动作"方向推。如果在一个**固定的
环境状态**下，换一个明显不同的合法匹配，即时奖励几乎不变，那这条梯度就没有可学的
信号 —— 加多少轮、调多少学习率都没用。

做法（不需要 deepcopy，避免复制 H5 句柄）：

  1. 建 K+1 个**完全相同**的环境。
  2. 每个 trial 用同一个 seed 把它们 reset 到同一状态，再用**同一个动作**同步走
     t_branch 步 —— 此时 K+1 个环境处在完全相同的状态。
  3. 在这一步分叉：前 K 个各给一个**从策略自身分布里独立采样**的动作（同一状态、
     同一策略、只是 Gumbel 不同），最后 1 个给策略的**确定性（贪心）**动作。
  4. 记录各自的即时奖励。

于是可以拆出两个方差：
  * **局内（动作引起的）**：同一状态下 K 个动作的奖励标准差 —— 这是策略梯度能利用的
    信号量级。
  * **局间（局面引起的）**：同一个动作在不同 trial/步数下的奖励标准差 —— 这是"今天
    运气好不好"。
两者之比就是信噪比。局内远小于局间时，优势里能用来比较动作的部分被淹没了。

用法：
    python .tmp/probe_reward_sensitivity.py [--trials 8] [--branches 5,20,60,150] [--samples 8]
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
    ap.add_argument("--trials", type=int, default=8, help="每个分叉步数跑几个种子")
    ap.add_argument("--branches", default="5,20,60,150", help="在第几步分叉，逗号分隔")
    ap.add_argument("--samples", type=int, default=8, help="同一状态采样几个动作")
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
        run_name="probe_reward",
        num_updates=None,
        seed=None,
        checkpoint=None,
        device=args.device,
    )
    config = build_config(ns)
    torch.manual_seed(int(config["seed"]["global_seed"]))
    torch.set_float32_matmul_precision("high")

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    data = load_checkpoint(args.checkpoint, device=args.device)
    model.load_state_dict(data.model_state)
    model.eval()
    policy = MAPPOPolicy(model, args.device)

    K = args.samples
    envs = [build_env_from_config(config) for _ in range(K + 1)]
    branches = [int(x) for x in str(args.branches).split(",")]
    if not branches:
        branches = [0]

    # info 里有哪些量可以分解奖励（只打印一次）。
    shown_keys = False

    within_stds: list[float] = []
    between_means: list[float] = []
    rows: list[tuple[int, int, float, float, float]] = []

    for t_branch in branches:
        for trial in range(args.trials):
            seed = args.seed + trial
            obs_list = [e.reset(seed=seed) for e in envs]

            # 用同一个动作把所有环境同步推到同一点（它们状态因此完全一致）。
            for _ in range(t_branch):
                with torch.no_grad():
                    s = policy.act(obs_list[0], build_scores=False)
                for k, e in enumerate(envs):
                    obs_list[k], _r, _te, _tr, _i = e.step(
                        s.actions, s.action_scores,
                        edge_scores=s.edge_scores,
                        expected_matched_edges=list(s.matched_edges or []),
                    )

            # 分叉：同一状态，K 个独立采样的动作 + 1 个贪心动作。
            with torch.no_grad():
                sampled = policy.act_batched(
                    [obs_list[0]] * K, build_scores=False, use_edge_arrays=True
                )
                greedy = policy.act(obs_list[0], deterministic=True, build_scores=False)

            rewards = []
            comps = None
            for k in range(K):
                _o, r, _te, _tr, info = envs[k].step(
                    sampled[k].actions, sampled[k].action_scores,
                    edge_scores=sampled[k].edge_scores,
                    expected_matched_edges=list(sampled[k].matched_edges or []),
                )
                rewards.append(float(r))
                if not shown_keys and info:
                    comps = sorted(info.keys())
            _o, r_greedy, _te, _tr, info_g = envs[K].step(
                greedy.actions, greedy.action_scores,
                edge_scores=greedy.edge_scores,
                expected_matched_edges=list(greedy.matched_edges or []),
            )
            r_greedy = float(r_greedy)
            if not shown_keys and info_g:
                comps = sorted(info_g.keys())
                shown_keys = True

            rr = torch.tensor(rewards, dtype=torch.float64)
            within = float(rr.std()) if len(rewards) > 1 else 0.0
            within_stds.append(within)
            between_means.append(r_greedy)
            rows.append((t_branch, trial, float(rr.mean()), within, r_greedy))

            if comps is not None:
                print(f"  info 里的字段: {comps}")
                comps = None

    print()
    print("=== 每个分叉点：同一状态下换动作引起的奖励差异 ===")
    print(f"  {'步':>5} {'动作均':>12} {'局内std':>12} {'贪心':>12} {'vs 动作均':>12}")
    for t_branch, trial, mean_r, within, g in rows:
        print(f"  {t_branch:>5} {mean_r:>12.6f} {within:>12.6f} {g:>12.6f} {g - mean_r:>+12.6f}")

    w = torch.tensor(within_stds, dtype=torch.float64)
    b = torch.tensor(between_means, dtype=torch.float64)
    print()
    print("=== 汇总 ===")
    print(f"  局内标准差（动作引起，同一状态换动作）  均值 {float(w.mean()):.6f}")
    print(f"  局间标准差（局面引起，同一动作换状态）  均值 {float(b.std()):.6f}")
    if float(w.mean()) > 0:
        print(f"  信噪比 局间/局内 = {float(b.std() / w.mean()):.3f}")
    print(f"  奖励量级：均值 {float(b.mean()):.4f}")
    print()
    print("判读：")
    print("  * 局内 std 远小于奖励量级（比如不到 1%）-> 换动作几乎不影响奖励，")
    print("    策略梯度没有可学的信号。")
    print("  * 贪心动作明显优于随机采样 -> 策略的众数是对的，只是探索把奖励摊平了。")
    print("  * 贪心与采样无差别 -> 策略对'哪个动作更好'没有概念。")
    print("PROBE_REWARD_DONE")


if __name__ == "__main__":
    main()
