"""方案 A（去图）的可行性预检：`num_layers: 0` 能否直接用？

## 为什么这条值得先试（零成本）

`graph_mappo.py:388` 的 `run_layers` 遍历 `self.layers`；`num_layers: 0`
⟹ `self.layers` **为空** ⟹ **完全没有消息传递** ⟹ 节点表示只剩
「自身特征投影 + （mixed 模式下）关联边」⟹ **图结构消失**。

这正是 H₁ **拆分假设**要的：「GNN 相对纯 MAPPO 无增益 ⟹ GNN 贡献不成立」
—— 需要一条"有消息传递 vs 无消息传递"的对照。

## 但文档说它被否决过，理由是"两个变量（Adam 动量会丢）"

`docs/定稿结论.md` §3.2 把 `num_layers: 0` 列为"⚠ 两个变量"。
**本探针核实这条否决还成不成立**：

① 差异字段是否恰好 1 个（`model.encoder.num_layers`）
② 维度是否不变（node_dim/edge_dim）
③ ★ BC 暖启动：`_upgrade_state_dict_for_model` 的四项
   —— 若 `dropped` 只含 `encoder.layers.*`（那些本就不该在）、共享部分逐位保留，
     那"两个变量"的指控**不成立**（丢掉的是**该丢的**）
④ 参数量变化（应减少，因为层没了）
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


def flat(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}."))
    else:
        out[prefix.rstrip(".")] = d
    return out


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ck.get("model_state", ck.get("model", ck))

    print("=" * 92)
    print("方案 A（去图 = num_layers 0）可行性预检")
    print("=" * 92)

    base = build(3, "pa_base")
    arm = build(0, "pa_arm")
    fb, fa = flat(base), flat(arm)
    diffs = {k: (fb.get(k, "<缺>"), fa.get(k, "<缺>"))
             for k in set(fb) | set(fa)
             if k not in {"project.run_name"} and fb.get(k, "<缺>") != fa.get(k, "<缺>")}
    print(f"\n① 差异字段 {len(diffs)} 个")
    for k, (a, b) in sorted(diffs.items()):
        print(f"    {k}: {a} → {b}")

    print(f"\n② 维度")
    for k in ("features.dims.node_dim_resolved", "features.dims.edge_dim_resolved"):
        print(f"    {k}: {fb.get(k)} → {fa.get(k)}")

    print(f"\n③ BC 暖启动（★ 核实「两个变量」的指控）")
    for L in (3, 0):
        cfg = build(L, f"pa_c{L}")
        env = build_env_from_config(cfg)
        m = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
        _, w, dr, mis, unexp = _upgrade_state_dict_for_model(m, st=state) \
            if False else _upgrade_state_dict_for_model(m, state)
        n = sum(p.numel() for p in m.parameters())
        print(f"    num_layers={L}: 参数量 {n:>10,}  加宽 {len(w)}  丢弃 {len(dr)}"
              f"  缺失 {len(mis)}  多余 {len(unexp)}")
        if dr:
            # 丢弃的是不是**全都是** encoder.layers.*？（那些是"该丢的"）
            all_layers = all(d.startswith("encoder.layers.") for d in dr)
            print(f"      丢弃项全部是 encoder.layers.*: {'✓ 是（该丢的）' if all_layers else '★ 否！'}")
            non_layer = [d for d in dr if not d.startswith("encoder.layers.")]
            if non_layer:
                print(f"      ★ 非 layers 的丢弃项（{len(non_layer)}）: {non_layer[:5]}")
        if unexp:
            all_layers = all(d.startswith("encoder.layers.") for d in unexp)
            print(f"      多余项全部是 encoder.layers.*: {'✓ 是' if all_layers else '★ 否'}")

    # ★ 关键：共享部分（node_proj/edge_proj/actor/critic）是否逐位保留？
    print(f"\n④ ★ 关键判据：**共享部分**是否逐位保留（不是只看总数）")
    cfg3, cfg0 = build(3, "x3"), build(0, "x0")
    e3 = build_env_from_config(cfg3); e0 = build_env_from_config(cfg0)
    m3 = GraphMAPPOActorCritic(e3.action_resolver.action_space, cfg3)
    m0 = GraphMAPPOActorCritic(e0.action_resolver.action_space, cfg0)
    s3, w3, d3, mi3, un3 = _upgrade_state_dict_for_model(m3, state)
    s0, w0, d0, mi0, un0 = _upgrade_state_dict_for_model(m0, state)
    shared_keys = [k for k in s0 if not k.startswith("encoder.layers.")]
    mis_shared = [k for k in mi0 if not k.startswith("encoder.layers.")]
    diff_shared = [k for k in shared_keys
                   if k in s3 and not torch.equal(s0[k], s3[k])]
    print(f"    num_layers=0 时非 layers 的键: {len(shared_keys)}")
    print(f"    其中**缺失**（没从 ckpt 拿到）: {len(mis_shared)}  {mis_shared[:5]}")
    print(f"    其中**与 L=3 的装载结果不同**: {len(diff_shared)}  {diff_shared[:5]}")
    print(f"    ⟹ {'✓ 共享部分逐位保留 ⟹ 「两个变量」的指控不成立' if not mis_shared and not diff_shared else '★ 共享部分有变动 ⟹ 确实是两个变量'}")


if __name__ == "__main__":
    main()
