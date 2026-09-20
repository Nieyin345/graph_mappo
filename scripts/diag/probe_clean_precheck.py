"""clean 臂预检：确认关掉三项特征后配置链能构造、edge_dim 真的变小。

## 为什么先跑这个（不起训练）

关特征 ⟹ `edge_dim` 变小 ⟹ `encoder.edge_proj_*` 末维**变窄**。
`_upgrade_state_dict_for_model` **只能救变宽**（末维加零），变窄救不回
⟹ **必须从零训**。本探针确认：
  ① 配置链能过 `ConfigValidator`
  ② `edge_dim` 确实变小（并打印前后值）
  ③ 差异字段**恰好**是那三项
"""
import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "_te", REPO / "scripts" / "train" / "train_graph_mappo.py")
_te = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_te)

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
    print("=" * 90)
    print("clean 臂预检")
    print("=" * 90)

    base = build(BASE, "probe_base")
    arm = build(BASE + ["train_clean.yaml"], "probe_clean")

    fb, fa = flat(base), flat(arm)
    diffs = {k: (fb.get(k, "<缺>"), fa.get(k, "<缺>"))
             for k in set(fb) | set(fa)
             if k not in {"project.run_name"} and fb.get(k, "<缺>") != fa.get(k, "<缺>")}

    print(f"\n  差异字段 {len(diffs)} 个：")
    for k, (a, b) in sorted(diffs.items()):
        print(f"    {k}: {a} → {b}")

    print(f"\n  维度：")
    for k in sorted(fb):
        if "dims" in k or "edge_dim" in k or "node_dim" in k:
            print(f"    {k}: {fb[k]} → {fa.get(k)}")

    # 期望恰好三项
    expected = {
        "features.edge.include_relay_importance",
        "features.edge.include_req_hop",
        "features.edge.include_on_pending_path",
    }
    got = set(diffs)
    print(f"\n  预期差异集合 = {sorted(expected)}")
    print(f"  实际差异集合 = {sorted(got)}")
    if got == expected:
        print("  ✓ 恰好三项")
    else:
        print(f"  ⚠ 多了/少了：{sorted(got ^ expected)}")

    # 边维
    eb = base["model"]["encoder"].get("edge_input_dim") or base["features"]["dims"].get("edge_dim_resolved")
    ea = arm["model"]["encoder"].get("edge_input_dim") or arm["features"]["dims"].get("edge_dim_resolved")
    print(f"\n  edge 维：{eb} → {ea}  （若变小 ⟹ 必须从零训，不能带 BC）")


if __name__ == "__main__":
    main()
