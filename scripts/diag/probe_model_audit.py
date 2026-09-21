"""模型体检（修正版）：用**项目自己的 PPO 损失**回传，不自造 loss。

## 上一版的缺陷（自曝）

我手工造了 `loss = -step.mean_log_prob + (step.value**2).sum()`。
但 `step.mean_log_prob` 来自 `_sample_matching` 的 **numpy 标量**（`.detach().cpu()`），
**不带图** ⟹ 反向传不到 `edge_scorer` ⟹ 它显示"0 梯度"是**我的构造错**，
不是模型的病。

⟹ 本版走**项目真正的更新路径**：`MAPPOTrainer.update()` 的那套
（`evaluate_actions_batched` → `log_probs` → PPO 损失 → backward）。

判据：若 `edge_scorer` 在真路径下有梯度 ⟹ 上一版结果作废。
"""
import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "_te", REPO / "scripts" / "train" / "train_graph_mappo.py")
_te = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_te)

from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic  # noqa: E402
from qkd_rl.rl.algos.policy import MAPPOPolicy  # noqa: E402
from qkd_rl.rl.algos.mappo_trainer import build_param_groups  # noqa: E402

BASE = ["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml"]


def main():
    cfg = _te.build_config(argparse.Namespace(
        configs=BASE, mode="random_episode", run_name="audit2",
        num_updates=30, seed=42, checkpoint=None, device="cpu"))
    env = build_env_from_config(cfg)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
    pol = MAPPOPolicy(model, "cpu")

    obs = env.reset(seed=100, start_seed=100)
    # 先采一次，拿到 actions（真路径需要它）
    step = pol.act(obs, deterministic=False, build_scores=True)
    acts = step.actions
    matched = list(step.matched_edges or [])
    print(f"  采样得到 {len(matched)} 条匹配弧，{len(acts)} 个节点的动作")

    # ★ 走**真路径**：evaluate_actions_batched（PPO 更新用的那个）
    log_probs, entropies, value = pol.evaluate_actions_batched(
        [obs], [acts], [matched])[0]

    grads = {}
    for n, p in model.named_parameters():
        def hook(grad, name=n):
            grads[name] = grads.get(name, 0.0) + float(grad.abs().sum())
        p.register_hook(hook)

    # PPO 形状的损失（与 trainer 同构：actor 项 + critic 项 + 熵项）
    # ★ `evaluate_actions_batched` 返回 per-node **dict**（trainer 也是这么用的：
    #   `new_lp = log_probs[node_ids[0]]`，因为每节点共享同一个联合标量）。
    node0 = obs.node_ids[0]
    lp = log_probs[node0]
    ent = entropies[node0]
    adv = torch.tensor(1.0)
    ratio = torch.exp(lp - lp.detach())
    actor_loss = -torch.min(ratio * adv,
                            torch.clamp(ratio, 0.9, 1.1) * adv).mean()
    critic_loss = ((value - torch.tensor(0.7)) ** 2).mean()
    entropy_loss = -ent * 0.01
    loss = actor_loss + critic_loss + entropy_loss
    model.zero_grad()
    loss.backward()

    total = sum(p.numel() for p in model.parameters())
    got = defaultdict(int)
    tot_by_mod = defaultdict(int)
    for n, p in model.named_parameters():
        parts = n.split(".")
        mod = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
        tot_by_mod[mod] += p.numel()
        if grads.get(n, 0.0) != 0.0:
            got[mod] += p.numel()

    print("\n" + "=" * 92)
    print("★ 走**真 PPO 路径**（evaluate_actions_batched）：哪些模块有梯度")
    print("=" * 92)
    print(f"\n  {'二级模块':<40}{'有梯度':>12}{'占该模块':>10}")
    for k in sorted(tot_by_mod, key=lambda x: -tot_by_mod[x]):
        g = got.get(k, 0)
        mark = "  ★ 全无梯度" if g == 0 else ""
        print(f"  {k:<40}{g:>12,}{g/tot_by_mod[k]*100:>9.1f}%{mark}")

    n_nograd = sum(p.numel() for n, p in model.named_parameters()
                   if grads.get(n, 0.0) == 0.0)
    print(f"\n  ★ 完全没梯度的参数: {n_nograd:,} / {total:,}"
          f"  ({n_nograd/total*100:.1f}%)")

    print(f"\n{'='*92}")
    print("对照：上一版（自造 loss，mean_log_prob 已 detach）")
    print("=" * 92)
    print("  edge_scorer 显示 0 梯度 —— 那是**我的构造错**（mean_log_prob 不带图）")


if __name__ == "__main__":
    main()
