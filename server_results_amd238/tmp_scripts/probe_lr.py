"""从 checkpoint 热启动之后，优化器实际用的是哪个学习率？

背景（2026-09-15 发现）：`mappo_trainer.load_checkpoint` 里的
`self.optimizer.load_state_dict(data.optimizer_state)` 会**整组恢复
param_groups，包括 lr**。BC 权重里三组存的都是 lr=0.001，于是
`rl_algorithm.yaml` 的 `train.optimizer.actor_lr: 0.0003` 从未生效过，
而且这一点完全静默 —— 把配置改成 0.001 的那组实验和对照**逐位相同**
（只有 rollout_s/update_s 不同），才暴露出来。

修复：加载之后把配置里的学习率写回 param_groups（Adam 的一二阶矩保持不动）。

本脚本直接打印"配置值 / 加载后实际值"，一眼看出修复是否生效。

用法：
    python .tmp/probe_lr.py
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
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from qkd_rl.env.factory import build_env_from_config
    from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer
    from qkd_rl.rl.algos.policy import MAPPOPolicy
    from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic
    from train_graph_mappo import build_config

    ns = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_diag_fast.yaml"],
        mode="random_episode",
        run_name="probe_lr",
        num_updates=None,
        seed=7,
        checkpoint=None,
        device=args.device,
    )
    config = build_config(ns)
    roles = ["encoder", "actor", "critic"]
    configured = [float(config["train"]["optimizer"]["actor_lr"]),
                  float(config["train"]["optimizer"]["actor_lr"]),
                  float(config["train"]["optimizer"]["critic_lr"])]

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, args.device)
    out = Path("/tmp/probe_lr")
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, config, out, device=args.device)

    print("构造后（应等于配置值）：")
    for role, g in zip(roles, trainer.optimizer.param_groups):
        print(f"  {role:<8} lr={g['lr']}")

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stored = ck.get("optimizer_state")
    if stored is None:
        print("checkpoint 里没有 optimizer_state，热启动不影响学习率")
    else:
        print("checkpoint 里存的学习率：")
        for role, g in zip(roles, stored.get("param_groups", [])):
            print(f"  {role:<8} lr={g.get('lr')}")

    trainer.load_checkpoint(Path(args.checkpoint))
    print("热启动之后（实际生效值）：")
    ok = True
    for role, g, want in zip(roles, trainer.optimizer.param_groups, configured):
        flag = "" if abs(g["lr"] - want) < 1e-12 else "   <-- 与配置不符！"
        ok = ok and not flag
        print(f"  {role:<8} lr={g['lr']}{flag}")
    print()
    print("配置值:", configured)
    print("PASS —— 配置的学习率已生效" if ok else "FAIL —— 配置被 checkpoint 盖掉了")

    # 顺带确认 Adam 的一二阶矩确实被恢复了（修复不应破坏续训）。
    n_state = len(trainer.optimizer.state)
    print(f"optimizer.state 条目数 = {n_state}（0 表示没恢复动量，续训会重来）")
    print("PROBE_LR_DONE")


if __name__ == "__main__":
    main()
