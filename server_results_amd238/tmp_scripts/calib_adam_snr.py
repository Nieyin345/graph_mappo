#!/usr/bin/env python
"""校验「Adam SNR 的纯噪声基线」这个判据，别想当然。

对纯 i.i.d. 噪声梯度跑同样的 EMA，看 ‖m‖/‖sqrt(v)‖ 落在哪。
再把实测的 actor SNR 放进去比。

若实测 SNR < 纯噪声基线，说明梯度不只是"噪声"，而是**系统性反号**
（这一步的梯度和上一步负相关）——那是比"信号弱"更严重的问题。
"""
import torch

torch.manual_seed(0)
B1, B2 = 0.9, 0.999
N = 256 * 64          # 与 actor.edge_scorer.0.weight 同量级
STEPS = 900           # 20 轮 × 45 minibatch

for label, gen in [
    ("纯 i.i.d. 噪声", lambda: torch.randn(N)),
    ("常数信号 + 噪声 (SNR=1)", lambda: 1.0 + torch.randn(N)),
    ("常数信号 + 噪声 (SNR=0.3)", lambda: 0.3 + torch.randn(N)),
    ("反号振荡 (g_k = (-1)^k)", None),
]:
    m = torch.zeros(N)
    v = torch.zeros(N)
    for k in range(STEPS):
        if gen is None:
            g = torch.randn(N) * ((-1.0) ** k)
        else:
            g = gen()
        m = B1 * m + (1 - B1) * g
        v = B2 * v + (1 - B2) * g * g
    snr = m.norm().item() / v.sqrt().norm().item()
    print(f"  {label:<28} ‖m‖/‖√v‖ = {snr:.4f}")

print()
print("  理论纯噪声基线 √((1-β1)/(1+β1)) = "
      f"{(1-B1)/(1+B1):.4f} 的平方根 = {((1-B1)/(1+B1))**0.5:.4f}")
print()
print("  实测 actor   = 0.1102 (均值) / 0.0836 (中位)")
print("  实测 critic  = 0.0332 (均值)")
print("  实测 encoder = 0.0828 (均值)")

# 逐元素版本（norm of ratios 而非 ratio of norms），看两种算法差多少
m = torch.zeros(N); v = torch.zeros(N)
for k in range(STEPS):
    g = torch.randn(N)
    m = B1 * m + (1 - B1) * g
    v = B2 * v + (1 - B2) * g * g
per_elem = (m.abs() / v.sqrt().clamp_min(1e-12)).mean().item()
print(f"\n  纯噪声，逐元素平均 |m|/√v = {per_elem:.4f}（另一种算法口径）")
