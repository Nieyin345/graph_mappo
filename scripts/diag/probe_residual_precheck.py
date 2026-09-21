"""residual 预检（待测候选 #1）。

## 机理

与今晚唯一显著结果（`demand_edge`：「节点需要链路上下文」）和
`layers4`（「3 层够用」）**同一条机理线**：**节点表示如何聚合信息**。

`graph_mappo.py:399`：
    node_emb = node_emb + next_node_emb if self.residual else next_node_emb

关掉 residual ⟹ 每层**覆盖**而非**累加** ⟹ 更早的信息被冲掉。
若「需要上下文」成立，关掉它应该**有害**（方向可预测，与 demand_edge 同向）。

## 代价

`residual` 是前向的一个分支，**没有任何参数** ⟹ 维度不变、BC 必然完整。
本探针验证。
"""
import argparse, importlib.util, sys
from pathlib import Path
import torch
REPO = Path("/opt/qkd/graph_mappo"); sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("_te", REPO/"scripts/train/train_graph_mappo.py")
_te = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_te)
from qkd_rl.env.factory import build_env_from_config
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic
from qkd_rl.rl.algos.mappo_trainer import _upgrade_state_dict_for_model

BASE = ["rl_algorithm.yaml","train_full_rl.yaml","train_ent01.yaml"]
CKPT = REPO/"outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"

def build(extra, name):
    return _te.build_config(argparse.Namespace(configs=BASE+([extra] if extra else []),
        mode="random_episode", run_name=name, num_updates=30, seed=42,
        checkpoint=None, device="cpu"))

def flat(d, prefix=""):
    out={}
    if isinstance(d, dict):
        for k,v in d.items(): out.update(flat(v, f"{prefix}{k}."))
    else: out[prefix.rstrip(".")] = d
    return out

base = build(None, "r_base")
arm  = build("train_residfalse.yaml", "r_arm")
fb, fa = flat(base), flat(arm)
diffs = {k:(fb.get(k,"<缺>"), fa.get(k,"<缺>")) for k in set(fb)|set(fa)
         if k not in {"project.run_name"} and fb.get(k,"<缺>")!=fa.get(k,"<缺>")}
print("="*88); print("residual 预检"); print("="*88)
print(f"\n① 差异字段 {len(diffs)} 个（期望恰好 1）")
for k,(a,b) in sorted(diffs.items()):
    print(f"    {k}: {a} → {b}  {'← 预期' if k=='model.encoder.residual' else '★★ 意外'}")
print(f"  ⟹ {'✓' if set(diffs)=={'model.encoder.residual'} else '★ 意外'}")

print("\n② 维度")
for k in ("features.dims.edge_dim_resolved","features.dims.node_dim_resolved"):
    print(f"    {k}: {fb.get(k)} → {fa.get(k)}")

print("\n③ BC 暖启动")
env = build_env_from_config(arm)
m = GraphMAPPOActorCritic(env.action_resolver.action_space, arm)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
st = ck.get("model_state", ck.get("model", ck))
_, w, dr, mis, unexp = _upgrade_state_dict_for_model(m, st)
print(f"    加宽 {len(w)} / 丢弃 {len(dr)} / 缺失 {len(mis)} / 多余 {len(unexp)}")
print(f"  ⟹ {'✓ 完整' if not (w or dr or mis or unexp) else '★ 有变化'}")
