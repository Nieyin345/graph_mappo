"""同一份权重、两个模型版本，跑同一套确定性评估。

回答一个被训练掩盖的问题：新模型的**前向**在权重完全相同的情况下，是否就已经
改变了策略的确定性动作？

  - 两边相同  -> 前向等价。之前 update 5 的 0.06 差距只能来自训练过程中的数值放大。
  - 两边差 ~0.06 -> 前向本身就改变了决策，与训练无关，是真 bug。

用法（脚本放在任何地方都可以，--repo 指向要用的那份代码）：

    python eval_same_weights.py --repo /opt/qkd/graph_mappo          --checkpoint <ckpt.pt>
    python eval_same_weights.py --repo /opt/qkd/eval_old_model       --checkpoint <ckpt.pt>

关键点是复用 MAPPOTrainer.evaluate()，而不是自己重写一遍评估循环 —— 这样
seed 基准、episode 数、eval_steps、确定性开关全都和训练里那条曲线一致，数字
才和 metrics.jsonl 里的 0.7572 / 0.8174 直接可比。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="要做评估的那份代码的仓库根目录")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--configs",
        nargs="*",
        default=["rl_algorithm.yaml", "train_diag_fast.yaml"],
        help="和训练命令用同一条配置链",
    )
    args = ap.parse_args()

    root = Path(args.repo).resolve()
    # 这个仓库自己的包和训练脚本优先，避免 import 到另一份代码。
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts" / "rl"))

    from qkd_rl.env.factory import build_env_from_config
    from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer
    from qkd_rl.rl.algos.policy import MAPPOPolicy
    from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic
    from train_graph_mappo import build_config

    ns = argparse.Namespace(
        configs=list(args.configs),
        mode="random_episode",
        run_name="eval_same_weights",
        num_updates=None,
        seed=None,
        checkpoint=None,
        device=args.device,
    )
    config = build_config(ns)

    seed = int(config["seed"]["global_seed"])
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, args.device)

    out_dir = Path("/tmp/eval_same_weights")
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, config, out_dir, device=args.device)
    trainer.load_checkpoint(Path(args.checkpoint))

    # 让数字能直接对上 metrics.jsonl 里的 eval_validation：那条记录来自
    # evaluate_validation()（held-out 验证窗口），不是 evaluate()（固定场景）。
    # 两者协议不同，served_keys 差一个量级，别混着比。
    if trainer.validation_enabled:
        n = int(trainer.validation_cfg.get("episodes", 1) or 1)
        summary = trainer.evaluate_validation(num_episodes=n)
        proto = f"evaluate_validation(episodes={n})"
    else:
        summary = trainer.evaluate()
        proto = "evaluate()"

    print(f"REPO={root}")
    print(f"  protocol          = {proto}")
    print(f"  mean_success_rate = {summary['mean_success_rate']:.6f}")
    print(f"  mean_reward       = {summary['mean_reward']:.6f}")
    print(f"  mean_served_keys  = {summary['mean_served_keys']:.1f}")
    print("EVAL_SAME_WEIGHTS_DONE")


if __name__ == "__main__":
    main()
