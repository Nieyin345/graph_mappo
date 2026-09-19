#!/usr/bin/env python
"""决定性探针：actor 的梯度到底是有效信号还是抵消的噪声？

回答"KL≈0.001、成功率纹丝不动"的根因。三个角度：

1) Adam SNR —— 对每个参数取优化器里的 exp_avg(m) 与 exp_avg_sq(v)，
   算 |m| / sqrt(v)。这**就是** Adam 实际步长的方向一致度：
     ≈1  → 梯度在 EMA 窗口内方向一致，是有效信号
     ≪1  → 梯度正负来回抵消，是噪声
   这个方法不需要任何新实验，直接从 checkpoint 里读。

2) 位移 vs 路程 —— 相邻 checkpoint 的 ||Δw|| 是"净位移"。
   若每步方向随机会抵消，净位移 ≪ 步数×单步。用 5 轮间隔的
   checkpoint 算净位移，与 Adam 推的单步幅度比。

3) 有效学习率 —— 从 checkpoint 里的 config 读出来，确认配置链
   （train_full_rl.yaml 没有 train.ppo 段 → 全部回落到 rl_algorithm.yaml）。
"""
from pathlib import Path

import torch

MAIN = Path("/opt/qkd/graph_mappo/outputs")
RUN = "r6_base"
CKPTS = ["checkpoint_update_000005.pt", "checkpoint_update_000010.pt",
         "checkpoint_update_000015.pt", "checkpoint_update_000020.pt"]


def load(rel):
    return torch.load(MAIN / rel, map_location="cpu", weights_only=False)


def group(name):
    low = name.lower()
    if "critic" in low or "value" in low:
        return "critic"
    if "actor" in low:
        return "actor"
    return "other"


ck = load(f"{RUN}/{CKPTS[-1]}")
print("=" * 72)
print("3) 有效配置（从 checkpoint 里读，绕开配置文件链）")
print("=" * 72)
cfg = ck.get("config", {})
train = cfg.get("train", {})
print(f"  checkpoint 里的 train 段键: {sorted(train) if isinstance(train, dict) else train}")
if isinstance(train, dict):
    ppo = train.get("ppo", {})
    opt = train.get("optimizer", {})
    print(f"  train.ppo      = {ppo}")
    print(f"  train.optimizer= {opt}")
print()

# ---------------------------------------------------------------- 1) Adam SNR
print("=" * 72)
print("1) Adam SNR = |m| / sqrt(v)   —— 梯度方向一致度")
print("=" * 72)
opt_state = ck.get("optimizer_state") or {}
ms = opt_state.get("state", {})
pg = opt_state.get("param_groups", [{}])
print(f"  优化器 param_groups[0].lr = {pg[0].get('lr') if pg else '?'}")
print(f"  状态条目数 = {len(ms)}")

model_state = ck["model_state"]
# 优化器 state 的键是参数索引；需要映射回名字。用顺序匹配：
# torch 优化器按 param_groups 的顺序编号，这里用 model_state 的顺序对齐。
names = [k for k, v in model_state.items() if torch.is_tensor(v) and v.is_floating_point()]

buckets = {"actor": [], "critic": [], "other": []}
per_param = []
for idx, st in ms.items():
    if not isinstance(st, dict) or "exp_avg" not in st or "exp_avg_sq" not in st:
        continue
    m, v = st["exp_avg"].float(), st["exp_avg_sq"].float()
    if m.numel() == 0:
        continue
    num = m.norm().item()
    den = v.sqrt().norm().item()
    snr = num / den if den > 1e-12 else 0.0
    # 参数名：优化器 state 的键通常是 int 索引，映射到 names
    nm = names[idx] if isinstance(idx, int) and idx < len(names) else str(idx)
    g = group(nm)
    buckets[g].append(snr)
    per_param.append((nm, g, snr, m.norm().item(), v.sqrt().norm().item()))

for g in ("actor", "critic", "other"):
    vals = buckets[g]
    if not vals:
        print(f"  {g:<7} 无数据")
        continue
    vals_sorted = sorted(vals)
    med = vals_sorted[len(vals_sorted) // 2]
    print(f"  {g:<7} n={len(vals):<4} SNR 均值={sum(vals)/len(vals):.4f} "
          f"中位={med:.4f} 最小={min(vals):.4f} 最大={max(vals):.4f}")

print()
print("  actor 逐参数 SNR（这是 Adam 实际步长方向的纯度）：")
for nm, g, snr, mn, den in sorted(per_param, key=lambda x: x[2]):
    if g == "actor":
        print(f"    {nm:<46} SNR={snr:.5f}  |m|={mn:.3e}  sqrt(v)_norm={den:.3e}")
print()

# ------------------------------------------------------- 2) 位移 vs 路程
print("=" * 72)
print("2) 净位移 vs Adam 单步幅度")
print("=" * 72)
lr = float(pg[0].get("lr", 0.0)) if pg else 0.0
for i in range(len(CKPTS) - 1):
    a = load(f"{RUN}/{CKPTS[i]}")["model_state"]
    b = load(f"{RUN}/{CKPTS[i+1]}")["model_state"]
    per_g = {"actor": [], "critic": []}
    for k in a:
        if k not in b or not torch.is_tensor(a[k]) or not a[k].is_floating_point():
            continue
        g = group(k)
        if g not in per_g:
            continue
        x, y = a[k].float(), b[k].float()
        if x.shape != y.shape:
            continue
        n = x.norm().item()
        if n > 1e-12:
            per_g[g].append((y - x).norm().item() / n)
    for g in ("actor", "critic"):
        if per_g[g]:
            mean = sum(per_g[g]) / len(per_g[g])
            print(f"  {CKPTS[i][-7:-3]}->{CKPTS[i+1][-7:-3]}  {g:<7} "
                  f"5轮净位移(相对) = {mean:.3e}   折合每轮 {mean/5:.3e}")
print()
print(f"  若梯度完全一致，每轮相对位移 ≈ lr = {lr:.1e}")
print(f"  实测 actor 每轮 ≈ 见上；比值 = 实测/lr 指示方向一致度")
