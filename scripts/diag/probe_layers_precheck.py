"""num_layers 预检：改层数对 BC 暖启动的代价有多大？

## 为什么这条机理最强

`demand_edge` 实测显著更差 6.5 点 ⟹ **节点表示需要链路上下文**。
而 `model.encoder.num_layers` **正是**「节点能看到几跳外的链路」的旋钮：
GNN 每多一层，节点的感受野就扩一跳。

53/53 条旧臂全是 `num_layers: 3` ⟹ **从未扫过**。

## 但有个代价问题

`hidden_dim` 不变（128），**维度不变**；但层数变了 ⟹ 权重张量**个数**变
⟹ BC checkpoint 里多/少的层会 missing/dropped。
需要量清楚：**这算"保住暖启动"还是"丢掉暖启动"？**

判据：
  · **丢弃 0 键**（dropped=0）⟹ 暖启动保住了（新层从零学，类似加宽补零）
  · 有丢弃 ⟹ 暖启动部分丢失，效果要打折
"""
import argparse
import importlib.util
import sys
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
from qkd_rl.rl.algos.mappo_trainer import _upgrade_state_dict_for_model  # noqa: E402

BASE = ["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml"]
CKPT = REPO / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"


def build(layers, name):
    cfg = _te.build_config(argparse.Namespace(
        configs=BASE, mode="random_episode", run_name=name,
        num_updates=30, seed=42, checkpoint=None, device="cpu"))
    cfg["model"]["encoder"]["num_layers"] = layers
    return cfg


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ck.get("model_state", ck.get("model", ck))

    print("=" * 92)
    print("num_layers 对 BC 暖启动的代价")
    print("=" * 92)
    print(f"\n  {'层数':>5}{'加宽':>7}{'丢弃':>7}{'缺失':>7}{'多余':>7}   判读")
    print("  " + "-" * 74)
    for L in (2, 3, 4, 5):
        cfg = build(L, f"probe_L{L}")
        env = build_env_from_config(cfg)
        model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
        _, widened, dropped, missing, unexpected = _upgrade_state_dict_for_model(
            model, state)
        if L == 3:
            verd = "（基线：应全 0）"
        elif len(dropped) == 0:
            verd = "✓ 丢弃 0 ⟹ 暖启动保住（新层从零学）"
        else:
            verd = f"★ 丢弃 {len(dropped)} ⟹ 暖启动部分丢失"
        print(f"  {L:>5}{len(widened):>7}{len(dropped):>7}{len(missing):>7}"
              f"{len(unexpected):>7}   {verd}")
        if L == 4 and dropped:
            for d in dropped[:4]:
                print(f"        - {d}")
        if L == 4 and missing:
            for m in missing[:4]:
                print(f"        ? {m}")

    # 维度确认
    cfg = build(4, "probe_dim")
    print(f"\n  维度（L=4）：edge_dim={cfg['features']['dims']['edge_dim_resolved']} "
          f"node_dim={cfg['features']['dims']['node_dim_resolved']}")
    print("  （与 L=3 的 44 / 17 比较 ⟹ 应相同）")
    print(f"\n  参数量对比：")
    for L in (3, 4):
        cfg = build(L, f"probe_p{L}")
        env = build_env_from_config(cfg)
        m = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
        n = sum(p.numel() for p in m.parameters())
        print(f"    L={L}: {n:,}")


if __name__ == "__main__":
    main()
