"""storage_reward_weight 预检。

## 为什么这条候选

`docs/训练诊断记录.md:14691`：
> **`storage_reward` 是活的**（每步在跑、有非零梯度，与恒 0 的那些不同），
> 但 `served/storage = 647.8 倍 = 2.81 个数量级` ⟹ **以现在的权重它几乎不可能是有效驱动**。

且 `storage_reward_weight` **全库 73/73 都是 0.5** ⟹ 从未扫过。
机理：它奖励「本槽新增且未被消费、且**在待办请求的可用路径上**」的密钥
（`reward.py:333`，`storage_pathness` 由 `env.py:232` 生产）
⟹ **正对已知瓶颈**（最弱跳局部饥荒，见 `key-pool-is-not-the-bottleneck`）。

## 本探针确认

① 维度不变（奖励是标量，不碰模型）
② BC 暖启动完整（模型没变 ⟹ 应 0/0/0/0）
③ 差异字段恰好是 `reward.storage_reward_weight`
④ **打印当前 storage 占比**（用于定剂量）
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


def build(extra, name):
    return _te.build_config(argparse.Namespace(
        configs=BASE + ([extra] if extra else []), mode="random_episode",
        run_name=name, num_updates=30, seed=42, checkpoint=None, device="cpu"))


def flat(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}."))
    else:
        out[prefix.rstrip(".")] = d
    return out


def main():
    base = build(None, "probe_s_base")
    arm = build("train_storage5.yaml", "probe_s_arm")

    fb, fa = flat(base), flat(arm)
    diffs = {k: (fb.get(k, "<缺>"), fa.get(k, "<缺>"))
             for k in set(fb) | set(fa)
             if k not in {"project.run_name"} and fb.get(k, "<缺>") != fa.get(k, "<缺>")}
    print("=" * 90)
    print("① 差异字段（期望恰好 1）")
    for k, (a, b) in sorted(diffs.items()):
        print(f"    {k}: {a} → {b}")
    print(f"  ⟹ {'✓' if set(diffs) == {'reward.storage_reward_weight'} else '★ 意外'}")

    print("\n② 维度（应全不变）")
    for k in ("features.dims.edge_dim_resolved", "features.dims.node_dim_resolved"):
        print(f"    {k}: {fb.get(k)} → {fa.get(k)}")

    print("\n③ BC 暖启动（模型没变 ⟹ 应全 0）")
    env = build_env_from_config(arm)
    m = GraphMAPPOActorCritic(env.action_resolver.action_space, arm)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    st = ck.get("model_state", ck.get("model", ck))
    _, w, dr, mis, unexp = _upgrade_state_dict_for_model(m, st)
    print(f"    加宽 {len(w)} / 丢弃 {len(dr)} / 缺失 {len(mis)} / 多余 {len(unexp)}")
    print(f"  ⟹ {'✓ 完整' if not (w or dr or mis or unexp) else '★ 有变化'}")

    print("\n④ 当前 storage 占比（用于定剂量）")
    print("    实测 rollout_debug：storage/served = 0.141%（§二十一）")
    print("    ⟹ 抬到 ~10% 需约 72×；本波用 5.0（10×）作第一档")


if __name__ == "__main__":
    main()
