#!/usr/bin/env python
"""微基准：两个速度假设能不能兑现？

假设 1（L3 驻留）：configs/rl_algorithm.yaml:70-79 的注释说 batch_chunk
  512 -> 64 是 1.35x 提速，机制是"512 步 ~103MB 掉出 L3，64 步 ~13MB 常驻"。
  若成立，再往小（32）应当还有收益，且曲线应当在某个尺寸后变平（= 落进 L3）。

假设 2（bf16 快速数学）：train_graph_mappo.py:127-131 只在检测到 CUDA 时设
  set_float32_matmul_precision("high")。当前跑的是 **CPU**，所以没设。
  在 CPU 上 "high" 允许 oneDNN 用 bf16 做 matmul —— 可能接近 2x。
  **这条如果成立，是一行改动换大收益。**

本基准只量 matmul/GEMM 本身，不跑 env，所以对机器上同时跑的 R7 不敏感
（相对差异在同样竞争下仍然可比）。

跑法（节点上）：/opt/qkd/venv/bin/python .tmp/bench_matmul.py
"""
import os
import statistics as st
import time

import torch

print(f"torch {torch.__version__}")
print(f"float32_matmul_precision = {torch.get_float32_matmul_precision()}")
print(f"OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS', '(未设)')}  "
      f"torch threads = {torch.get_num_threads()}")
print(f"mkldnn 可用 = {torch.backends.mkldnn.is_available()}")
print()

# 与真实 update 同量级：PPO 在 ~393 条有向边上做，hidden 维度见 graph_mappo.yaml
HIDDEN = 128
EDGES = 393
STEPS = 1440          # rollout 步数
EPISODES = 8
N_EDGE = EDGES

torch.manual_seed(0)


def bench(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return st.median(ts)


print("=" * 72)
print(f"假设 1：batch_chunk 对单块前向+反向的影响（边张量 {N_EDGE}x{HIDDEN}）")
print("=" * 72)
print(f"  {'chunk':>6}{'每块 MB':>10}{'每块 ms':>10}{'总 ms(1000 块)':>16}")
for chunk in (8, 16, 32, 64, 128, 256, 512):
    x = torch.randn(chunk, N_EDGE, HIDDEN)
    w = torch.randn(HIDDEN, HIDDEN, requires_grad=False)

    def run():
        y = x @ w
        y.sum().backward() if y.requires_grad else None

    # 需要梯度才对，重建一次
    xg = x.clone().requires_grad_(True)
    wg = w.clone().requires_grad_(True)

    def run_g():
        y = xg @ wg
        y.sum().backward()

    ms = bench(run_g) * 1000
    mb = chunk * N_EDGE * HIDDEN * 4 / 1e6
    total_ms = ms * (1000 / chunk)
    print(f"  {chunk:>6}{mb:>10.1f}{ms:>10.2f}{total_ms:>16.1f}")

print()
print("=" * 72)
print("假设 2：float32_matmul_precision 的影响")
print("=" * 72)
a = torch.randn(2048, 1024)
b = torch.randn(1024, 1024)
for prec in ("highest", "high", "medium"):
    try:
        torch.set_float32_matmul_precision(prec)
        ms = bench(lambda: a @ b) * 1000
        gf = 2 * 2048 * 1024 * 1024 / (ms / 1000) / 1e9
        print(f"  {prec:<10} {ms:8.2f} ms   {gf:7.1f} GFLOP/s")
    except Exception as exc:  # noqa: BLE001
        print(f"  {prec:<10} 失败：{exc}")
torch.set_float32_matmul_precision("highest")

print()
print("=" * 72)
print("假设 2b：显式 bf16 autocast（对照，看 CPU 上能快到哪去）")
print("=" * 72)
try:
    ms32 = bench(lambda: a @ b) * 1000
    ab, bb = a.to(torch.bfloat16), b.to(torch.bfloat16)

    def bf16mm():
        return ab @ bb

    ms16 = bench(bf16mm) * 1000
    print(f"  fp32  {ms32:8.2f} ms")
    print(f"  bf16  {ms16:8.2f} ms   加速 {ms32 / ms16:.2f}x")
    print("  注：bf16 会改数值结果，只能当**上界参考**，不能直接用在训练里")
except Exception as exc:  # noqa: BLE001
    print(f"  bf16 测试失败：{exc}")

print()
print("=== BENCH_DONE ===")
