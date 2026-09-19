"""同一份权重下，新旧两个模型的**反向梯度**是否一致？

为什么还需要这一步：前向已经验过了 —— 三份 checkpoint 在新旧两版模型下给出
六位小数相同的确定性评估，说明**决策函数等价**。但训练仍然系统性地分叉（新代码
四个种子 0.7414/0.7535/0.7501/0.7627，旧代码 0.8174，落在分布之外）。前向等价
而训练分叉，差异只能在**反向**：这次重构动的恰恰是反向图的核心 —— 切片改成
``index_add_``、per-graph 池化改成 ``_segment_sum``、两类边分开做 LayerNorm。

做法：同一个权重、同一批观测、同一个标量损失，两边各反传一次，逐参数比梯度。
  - 各参数相对差都在浮点噪声量级（~1e-6）-> 反向也等价，训练分叉只能是优化过程
    对微小差异的放大，不是 bug。
  - 某个参数差出几个数量级 -> 反向图有实质差异，那就是要找的东西。

用法：
    python grad_probe.py --repo <repo> --checkpoint <ckpt> --out <grad.pt>
    python grad_probe.py --compare <grad_a.pt> <grad_b.pt>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def build(repo: Path, device: str):
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "scripts" / "rl"))

    from qkd_rl.env.factory import build_env_from_config
    from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic
    from train_graph_mappo import build_config

    ns = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_diag_fast.yaml"],
        mode="random_episode",
        run_name="grad_probe",
        num_updates=None,
        seed=None,
        checkpoint=None,
        device=device,
    )
    config = build_config(ns)
    torch.manual_seed(int(config["seed"]["global_seed"]))
    torch.set_float32_matmul_precision("high")
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    return env, model


def probe(args) -> None:
    repo = Path(args.repo).resolve()
    env, model = build(repo, args.device)

    from qkd_rl.rl.algos.checkpoint import load_checkpoint

    data = load_checkpoint(args.checkpoint, device=args.device)
    model.load_state_dict(data.model_state)
    model.eval()
    # eval() 会关掉 dropout，但反传本身不受影响；保持 eval 让两边完全同构。

    n = int(args.batch)
    obs_list = [env.reset(seed=args.seed + i) for i in range(n)]

    out = model.batched_forward(obs_list, args.device, want_edge_maps=True)

    # 标量损失：把 critic 的每图价值、actor 的每条候选边打分都加起来。两者
    # 合起来覆盖 encoder + actor 头 + critic 头，也就是整个模型的参数。
    loss = torch.zeros((), dtype=torch.float32)
    for v in out.values:
        loss = loss + v.sum()
    n_arc = 0
    for m in out.edge_score_maps:
        for s in m.values():
            loss = loss + s.sum()
            n_arc += 1

    model.zero_grad(set_to_none=True)
    loss.backward()

    grads = {}
    for name, p in model.named_parameters():
        grads[name] = (
            p.grad.detach().clone() if p.grad is not None
            else torch.zeros_like(p)
        )

    torch.save(
        {
            "grads": grads,
            "loss": float(loss),
            "n_arcs": n_arc,
            "repo": str(repo),
        },
        args.out,
    )
    print(f"REPO={repo}")
    print(f"  loss       = {float(loss):.8f}")
    print(f"  n_params   = {len(grads)}")
    print(f"  n_arcs     = {n_arc}")
    print(f"  saved      = {args.out}")
    print("GRAD_PROBE_DONE")


def compare(args) -> None:
    a = torch.load(args.compare[0], map_location="cpu")
    b = torch.load(args.compare[1], map_location="cpu")
    print(f"A = {a['repo']}  loss={a['loss']:.8f}")
    print(f"B = {b['repo']}  loss={b['loss']:.8f}")
    ga, gb = a["grads"], b["grads"]
    only_a = sorted(set(ga) - set(gb))
    only_b = sorted(set(gb) - set(ga))
    if only_a or only_b:
        print(f"!! 参数集合不同: only-A={only_a[:5]} only-B={only_b[:5]}")

    def norm(t):
        return float(t.norm())

    rows = []
    for name in sorted(set(ga) & set(gb)):
        x, y = ga[name].double(), gb[name].double()
        d = (x - y).norm()
        ref = max(norm(x), norm(y))
        rows.append((name, norm(x), norm(y), float(d), float(d / ref) if ref > 0 else 0.0))

    # 先看整体：所有参数梯度拼起来的相对差
    flat_a = torch.cat([ga[n].reshape(-1).double() for n in sorted(ga)])
    flat_b = torch.cat([gb[n].reshape(-1).double() for n in sorted(gb)]) if set(ga) == set(gb) else None
    if flat_b is not None:
        rel = float((flat_a - flat_b).norm() / max(flat_a.norm(), 1e-300))
        print(f"\n整体相对差 ||dG|| / ||G|| = {rel:.3e}")

    rows.sort(key=lambda r: -r[4])
    print("\n相对差最大的 10 个参数：")
    print(f"  {'参数':<58}{'|g_old|':>12}{'|g_new|':>12}{'相对差':>12}")
    for name, na, nb, d, r in rows[:10]:
        short = name if len(name) <= 56 else "..." + name[-53:]
        print(f"  {short:<58}{na:>12.4e}{nb:>12.4e}{r:>12.3e}")

    rels = [r for *_ , r in rows]
    print(f"\n参数数 {len(rows)}，相对差中位数 {sorted(rels)[len(rels)//2]:.3e}，最大 {max(rels):.3e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo")
    ap.add_argument("--checkpoint")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--compare", nargs=2, metavar=("A.pt", "B.pt"))
    args = ap.parse_args()

    if args.compare:
        compare(args)
    else:
        probe(args)


if __name__ == "__main__":
    main()
