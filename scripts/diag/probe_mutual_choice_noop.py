"""造反证：`mutual_choice` 真的对 RL 动作是空裁决吗？

## 上一版的缺陷

我用 `_find_illegal`（查**掩码**）数非法动作 = 0。
但掩码 ≠ 互选裁决。`mutual_choice` 的裁决在 `_resolve_mutual_choice` 里。

## 本版用**真正决定性的判据**

  提交的**有向弧**集合  vs
  resolver **实际接受**的弧集合（`env.last_matched_arcs`）

  若两者**相等**（去重后） ⟹ mutual_choice 是空裁决（**恒等通过**）
  若后者更小 ⟹ 有真实裁决，存在可去协调的空间

★ 造反证（必须做）：同一测法用在**独立选择**的动作上，拒绝率必须 **> 0**。
   否则说明我的"拒绝率"根本没在测裁决（测量本身是空真）。
"""
import sys
import importlib.util
from pathlib import Path
import torch
REPO = Path("/opt/qkd/graph_mappo"); sys.path.insert(0, str(REPO))
_s = importlib.util.spec_from_file_location("_tp", REPO/"qkd_rl/evaluation/test_protocol.py")
_tp = importlib.util.module_from_spec(_s); _s.loader.exec_module(_tp)
from qkd_rl.env.factory import build_env_from_config
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic
from qkd_rl.rl.algos.policy import MAPPOPolicy


def submitted_arcs(actions):
    """把 per-node (tx,rx) 动作还原成有向弧集合（去重）。"""
    out = set()
    IDLE = "idle"
    for u, (tx, rx) in actions.items():
        if tx != IDLE and tx != u:
            out.add((u, tx))
        if rx != IDLE and rx != u:
            out.add((rx, u))       # rx_source → me
    return out


def run(seed, steps=120):
    prof = _tp.load_validation_profile(str(REPO/"configs/global.yaml"))
    cfg = _tp.build_validation_env_config(prof)
    env = build_env_from_config(cfg)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
    ck = torch.load(REPO/"outputs/supervised_pg_phased/supervised_pg_phased_latest.pt",
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state", ck.get("model", ck)), strict=False)
    model.eval()
    pol = MAPPOPolicy(model, "cpu")

    obs = env.reset(seed=seed, start_seed=seed)
    n_sub = n_acc = 0
    done = False; k = 0
    while not done and k < steps:
        with torch.no_grad():
            out = pol.act(obs, deterministic=True, build_scores=True)
        sub = submitted_arcs(out.actions)
        obs, r, term, trunc, info = env.step(
            out.actions, out.action_scores,
            edge_scores=out.edge_scores,
            expected_matched_edges=list(out.matched_edges or []))
        acc = set(env.last_matched_arcs or [])
        n_sub += len(sub); n_acc += len(acc & sub)
        done = term or trunc; k += 1
    return n_sub, n_acc


print("="*88)
print("造反证：mutual_choice 对 RL 动作是空裁决吗？")
print("="*88)
tot_sub = tot_acc = 0
for s in (100, 101, 102):
    ns, na = run(s)
    tot_sub += ns; tot_acc += na
    print(f"  seed {s}: 提交 {ns} 弧 / 被接受 {na} 弧  → 接受率 {na/max(1,ns)*100:.2f}%")
print(f"\n  ★ 合计：提交 {tot_sub} / 接受 {tot_acc} → **接受率 {tot_acc/max(1,tot_sub)*100:.4f}%**")
print(f"  ⟹ {'✓ 全部接受 ⟹ mutual_choice 是空裁决' if tot_acc==tot_sub else '★ 有拒绝 ⟹ 存在真实裁决'}")

print(f"\n{'='*88}")
print("造反证（正对照）：同样的测法用在**独立选择**的动作上")
print("="*88)
print("  用 random 策略（每节点独立随机选）提交，看接受率是否 **< 100%**")
import random
prof = _tp.load_validation_profile(str(REPO/"configs/global.yaml"))
cfg = _tp.build_validation_env_config(prof)
env = build_env_from_config(cfg)
obs = env.reset(seed=100, start_seed=100)
rng = random.Random(0)
n_sub = n_acc = 0; done=False; k=0
while not done and k < 120:
    acts = {}
    for nid in obs.node_ids:
        cands = [c for c in obs.action_candidates[nid] if c != "idle"]
        tx = rng.choice([c for c in obs.action_candidates[nid]]) if obs.action_candidates[nid] else "idle"
        rx = rng.choice([c for c in obs.action_candidates[nid]]) if obs.action_candidates[nid] else "idle"
        acts[nid] = (tx, rx)
    sub = submitted_arcs(acts)
    obs, r, term, trunc, info = env.step(acts, None)
    acc = set(env.last_matched_arcs or [])
    n_sub += len(sub); n_acc += len(acc & sub)
    done = term or trunc; k += 1
print(f"  random 独立选择：提交 {n_sub} 弧 / 接受 {n_acc} → **接受率 {n_acc/max(1,n_sub)*100:.2f}%**")
print(f"  ⟹ {'✓ 拒绝率 > 0 ⟹ 测法能抓到拒绝（非空真）' if n_acc < n_sub else '★ 也 100% ⟹ 测法测不到裁决，前一结论作废'}")
