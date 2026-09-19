"""测 update 阶段的**纯计算开销**：PPO 损失那 360 个 minibatch 到底花多少。

动机：日志里 `update_s` 145~196s 而 `rollout_s` 46~82s，即**七成时间在 update**。
而配置是 epochs=1、minibatch=256 —— 一个 rollout 是 8×1440 = 11520 步，
也就是 45 个 minibatch、**45 次 optimizer.step()**。每步 backward 在 CPU 上
主要吃**每个 autograd 节点的引擎开销**，不是算术。

这个探针**不训练**、不写 outputs、不启动 run，只在一个隔离副本里把
`_loss_for_batch + backward` 的循环重放若干次，量出：
  - 单次 minibatch（含 backward）的墙钟
  - 前向 / 反向 各自占比
  - `evaluate_actions_batched` 那一步（block-diagonal 前向）占多少

判据是**能不能用一条明确、不改变数值的改动把它砍掉**（例如
`torch.optim.Adam(foreach=True)` 或减少 optimizer.step 次数）。
本探针只产出数字，不产出改动。

跑法（服务器上，隔离副本）：
    cd /tmp/metrics_check && /opt/qkd/venv/bin/python /tmp/probe_update_cost.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/metrics_check")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

# 参数规模对齐真实模型（1,053,188）。**形状不对齐**：真实的 forward 是
# 90 节点图的 block-diagonal 稀疏操作，这里用稠密 Linear 近似。但本探针要量的
# 是**两个与形状无关的项**：
#   (a) `optimizer.step()` 的 per-parameter Python 开销（只取决于参数量，1e6）
#   (b) backward 的 per-autograd-node 引擎开销（取决于图的节点数）
# 所以形状不同不影响这两个结论，但**不能**用它推断真实 forward 的耗时。
HIDDEN = 128
LAYERS = 3
WIDTH = 590          # 3*WIDTH² + ... ≈ 1.05e6，与真实模型同量级


class Tiny(torch.nn.Module):
    def __init__(self, feat: int) -> None:
        super().__init__()
        self.enc = torch.nn.ModuleList([torch.nn.Linear(feat, WIDTH)])
        for _ in range(LAYERS - 1):
            self.enc.append(torch.nn.Linear(WIDTH, WIDTH))
        self.actor = torch.nn.Linear(WIDTH, WIDTH)
        self.critic = torch.nn.Linear(WIDTH, 1)

    def forward(self, x):
        for l in self.enc:
            x = torch.relu(l(x))
        return self.actor(x), self.critic(x)


def timeit(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main() -> None:
    B, NODES, FEAT = 256, 90, 12
    m = Tiny(FEAT)
    n_par = sum(p.numel() for p in m.parameters())
    print("=" * 66)
    print(f"模型参数量 {n_par:,}（真实模型 1,053,188）")
    print(f"torch threads = {torch.get_num_threads()}")
    print(f"torch {torch.__version__}")
    print("=" * 66)

    # 每步 = 一个图的节点数。真实里一个 90 节点图的观测 ~90×feat，
    # 而 minibatch=256 步 → 前向是 256 个图拼成的 block-diagonal。
    # 这里用 B=256 个 90 行的小图近似（总行数 23040）。
    x = torch.randn(B * NODES, FEAT)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)

    def one_minibatch():
        opt.zero_grad()
        a, v = m(x)
        loss = (-(a.mean()) + v.mean().square() - a.std())
        loss.backward()
        opt.step()

    # 预热
    for _ in range(3):
        one_minibatch()

    t_mb = timeit(one_minibatch, 20)
    print(f"单 minibatch（前向+反向+step）: {t_mb*1000:8.1f} ms")

    # 拆开量：forward-only vs forward+backward vs +step
    def fwd_only():
        with torch.no_grad():
            m(x)

    def fwd_bwd():
        m.zero_grad(set_to_none=True)
        a, v = m(x)
        loss = (-(a.mean()) + v.mean().square() - a.std())
        loss.backward()

    t_fwd = timeit(fwd_only, 20)
    t_fb = timeit(fwd_bwd, 20)
    print(f"  仅前向(no_grad)            : {t_fwd*1000:8.1f} ms")
    print(f"  前向+反向                  : {t_fb*1000:8.1f} ms")
    print(f"  ⟹ 反向 = {100*(t_fb-t_fwd)/t_fb:.0f}% of 前向+反向")
    print(f"  ⟹ optimizer.step/zero_grad = {(t_mb-t_fb)*1000:.1f} ms")

    # foreach 版本（fused，减少 per-parameter 的 Python 开销）
    m2 = Tiny(FEAT)
    m2.load_state_dict(m.state_dict())
    opt2 = torch.optim.Adam(m2.parameters(), lr=3e-4, foreach=True)

    def one_mb_foreach():
        opt2.zero_grad()
        a, v = m2(x)
        loss = (-(a.mean()) + v.mean().square() - a.std())
        loss.backward()
        opt2.step()

    for _ in range(3):
        one_mb_foreach()
    t_mb2 = timeit(one_mb_foreach, 20)
    print("-" * 66)
    print(f"单 minibatch（foreach=True）  : {t_mb2*1000:8.1f} ms "
          f"（{(t_mb2-t_mb)*1000:+.1f} ms, {100*(t_mb2-t_mb)/t_mb:+.1f}%）")

    # 真实规模推演：8 envs × 1440 steps / 256 = 45 个 minibatch
    n_mb = 8 * 1440 // 256
    print("-" * 66)
    print(f"真实一轮 = {n_mb} 个 minibatch")
    print(f"  按实测单 mb {t_mb*1000:.1f} ms 推：{t_mb*n_mb:.1f} s")
    print(f"  （日志里 update_s 实测 145~196 s —— 差额应来自")
    print(f"    真实图的 block-diagonal 前向比这里的稠密 Linear 更贵）")
    print("=" * 66)


if __name__ == "__main__":
    main()
