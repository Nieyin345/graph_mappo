"""demand_edge 臂预检：确认**不改维度**（因而能带 BC 起点）。

## 这是本臂成立的前提

若维度变了 ⟹ 必须从零训 ⟹ 落高方差族（可检测 0.17）⟹ 白跑。

## 三条检查

① 维度：`edge_dim_resolved` / `node_dim_resolved` 必须**与基线相同**
② 差异字段：**恰好** `{"model.mode"}`
③ BC 暖启动：加载 `supervised_pg_phased_latest.pt` 时**丢弃 0 个键**
   （用 `_upgrade_state_dict_for_model` 实跑）
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


def build(configs, name):
    return _te.build_config(argparse.Namespace(
        configs=configs, mode="random_episode", run_name=name,
        num_updates=30, seed=42, checkpoint=None, device="cpu"))


def flat(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}."))
    else:
        out[prefix.rstrip(".")] = d
    return out


def main():
    print("=" * 92)
    print("demand_edge 臂预检（关键：**维度必须不变**）")
    print("=" * 92)

    base = build(BASE, "probe_de_base")
    arm = build(BASE + ["train_demandedge.yaml"], "probe_de_arm")

    fb, fa = flat(base), flat(arm)
    diffs = {k: (fb.get(k, "<缺>"), fa.get(k, "<缺>"))
             for k in set(fb) | set(fa)
             if k not in {"project.run_name"} and fb.get(k, "<缺>") != fa.get(k, "<缺>")}

    print(f"\n  ① 差异字段 {len(diffs)} 个（期望恰好 1）")
    for k, (a, b) in sorted(diffs.items()):
        mark = "  ← 预期" if k == "model.mode" else "  ★★ 意外"
        print(f"      {k}: {a} → {b}{mark}")

    print(f"\n  ② 维度：")
    same_dim = True
    for k in ("features.dims.edge_dim_resolved", "features.dims.node_dim_resolved",
              "features.dims.demand_edge_dim_resolved",
              "features.dims.physical_edge_dim_resolved"):
        a, b = fb.get(k), fa.get(k)
        ok = a == b
        same_dim &= ok
        print(f"      {k}: {a} → {b}  {'✓' if ok else '★ 变了'}")
    print(f"      ⟹ {'✓ 维度不变 ⟹ 可带 BC 起点' if same_dim else '★ 维度变了 ⟹ 必须从零训'}")

    print(f"\n  ③ BC 暖启动加载（实跑 `_upgrade_state_dict_for_model`）：")
    env = build_env_from_config(arm)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, arm)
    ck = torch.load(
        REPO / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt",
        map_location="cpu", weights_only=False)
    state = ck.get("model_state", ck.get("model", ck))
    _, widened, dropped, missing, unexpected = _upgrade_state_dict_for_model(model, state)
    print(f"      加宽 {len(widened)} / **丢弃 {len(dropped)}** / "
          f"缺失 {len(missing)} / 多余 {len(unexpected)}")
    for d in dropped[:5]:
        print(f"        - {d}")
    ok_load = (len(dropped) == 0 and len(widened) == 0)
    print(f"      ⟹ {'✓ 逐位加载，BC 暖启动完整保留' if ok_load else '★ 有丢弃/加宽'}")

    print("\n" + "=" * 92)
    ok = (set(diffs) == {"model.mode"} and same_dim and ok_load)
    print("✓ 全部通过：单变量 + 维度不变 + BC 可加载 ⟹ 可起臂" if ok
          else "★ 有检查未过 ⟹ 先查清楚再起")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
