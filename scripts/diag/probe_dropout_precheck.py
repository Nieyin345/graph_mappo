"""dropout 预检：加 dropout 是否会破坏 BC 暖启动？

## 为什么是这条

`configs/train_v2_gelu.yaml` 的注释明写：
「**dropout 不要混进这一臂**。dropout 的作用面是 update 正则化
（`mappo_trainer.py:1074/1414/1509` 的 `self.model.train()`），**不进探索**
（`rollout_workers.py:131` 的 `policy.model.eval()`）——
与激活的作用面不同，混在一臂里 Δ 无法归因。**dropout 单开一臂**。」

而全库 53 条臂 `model.encoder.dropout` **恒为 0.0** ⟹ 从未跑过。

## 关键：它不改维度

`build_mlp(..., dropout)` 的 dropout 是 `nn.Dropout` 层，**无参数** ⟹
权重形状不变 ⟹ **BC 暖启动应逐位保留**。本探针验证。

## 顺带确认它真的"不进探索"

grep `rollout_workers.py` 里是否 `model.eval()`（eval 模式下 nn.Dropout 是恒等）。
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


def build(dropout, name):
    cfg = _te.build_config(argparse.Namespace(
        configs=BASE, mode="random_episode", run_name=name,
        num_updates=30, seed=42, checkpoint=None, device="cpu"))
    cfg["model"]["encoder"]["dropout"] = dropout
    return cfg


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ck.get("model_state", ck.get("model", ck))

    print("=" * 92)
    print("dropout 对 BC 暖启动的代价 + 参数量")
    print("=" * 92)
    print(f"\n  {'dropout':>8}{'加宽':>7}{'丢弃':>7}{'缺失':>7}{'多余':>7}"
          f"{'参数量':>12}   判读")
    print("  " + "-" * 82)
    for d in (0.0, 0.1, 0.2, 0.3):
        cfg = build(d, f"probe_d{d}")
        env = build_env_from_config(cfg)
        m = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
        _, widened, dropped, missing, unexpected = _upgrade_state_dict_for_model(
            m, state)
        n = sum(p.numel() for p in m.parameters())
        if d == 0.0:
            verd = "（基线）"
        elif len(dropped) == 0 and len(widened) == 0:
            verd = "✓ 逐位保留"
        else:
            verd = f"★ 丢弃{widened}"
        print(f"  {d:>8.1f}{len(widened):>7}{len(dropped):>7}{len(missing):>7}"
              f"{len(unexpected):>7}{n:>12,}   {verd}")

    # 维度
    cfg = build(0.2, "probe_dim")
    print(f"\n  维度（dropout=0.2）：edge_dim="
          f"{cfg['features']['dims']['edge_dim_resolved']} node_dim="
          f"{cfg['features']['dims']['node_dim_resolved']}  （应与 44/17 相同）")

    # ★ 确认它不进探索
    print(f"\n  ★ 确认 dropout 不进探索（rollout 时 eval 模式）：")
    r = __import__("subprocess").run(
        f"cd {REPO} && grep -n 'model.eval()\\|\\.eval()' qkd_rl/rl/algos/rollout_workers.py",
        shell=True, capture_output=True, text=True)
    for ln in r.stdout.splitlines()[:4]:
        print(f"      {ln.strip()[:100]}")
    if not r.stdout.strip():
        print("      ⚠ 没找到 eval() —— 需人工确认")


if __name__ == "__main__":
    main()
