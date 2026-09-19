"""核对 `probe_param_drift.py` 的读数不是口径假象。

`rel_drift = ‖θ−θ₀‖ / ‖θ₀‖` 在 `‖θ₀‖` 很小时会放大。critic 漂移读到 1.43，
必须排除「critic 的初始范数特别小」这个解释。

这里打**绝对**量：各模块的参数范数（BC 时 / 当前）、参数量、逐键最大变化。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

OUT = Path("/opt/qkd/graph_mappo/outputs")
BC = OUT / "supervised_pg_phased" / "supervised_pg_phased_latest.pt"
UPD_RE = re.compile(r"checkpoint_update_(\d+)\.pt$")
MODULES = ("encoder", "actor", "critic")


def module_of(key: str) -> str:
    head = key.split(".", 1)[0]
    return head if head in MODULES else "其它"


def load(p: Path):
    return torch.load(p, map_location="cpu", weights_only=False)["model_state"]


def group(st, mod):
    return {k: v for k, v in st.items()
            if module_of(k) == mod and torch.is_tensor(v) and v.is_floating_point()}


def main(argv: list[str]) -> int:
    run = argv[1] if len(argv) > 1 else "mode_de_s42"
    d = OUT / run
    pts = sorted((int(UPD_RE.search(p.name).group(1)), p)
                 for p in d.glob("checkpoint_update_*.pt") if UPD_RE.search(p.name))
    if not pts:
        print(f"✗ {run} 无检查点")
        return 1
    u, latest = pts[-1]
    base = load(BC)
    cur = load(latest)

    print(f"BC 起点 vs {run} u{u}")
    print()
    print(f"{'模块':<10}{'参数量':>10}{'范数(BC)':>12}{'范数(今)':>12}"
          f"{'绝对变化':>12}{'相对':>9}")
    print("-" * 68)
    for m in MODULES + ("其它",):
        gb, gc = group(base, m), group(cur, m)
        if not gb:
            continue
        nb = sum(v.numel() for v in gb.values())
        fb = sum(float(v.pow(2).sum()) for v in gb.values()) ** 0.5
        fc = sum(float(v.pow(2).sum()) for v in gc.values()) ** 0.5
        dnorm = sum(float((gc[k] - gb[k]).pow(2).sum()) for k in gb if k in gc) ** 0.5
        print(f"{m:<10}{nb:>10,}{fb:>12.4f}{fc:>12.4f}{dnorm:>12.4f}"
              f"{dnorm / fb if fb else 0:>9.4f}")

    # 逐键看谁变化最大 —— 确认不是某一两个键主导
    print()
    print("逐键变化最大的 10 个（相对）：")
    rows = []
    for k in base:
        if k not in cur or not torch.is_tensor(base[k]) or not base[k].is_floating_point():
            continue
        fb = float(base[k].pow(2).sum()) ** 0.5
        if fb <= 0:
            continue
        dn = float((cur[k] - base[k]).pow(2).sum()) ** 0.5
        rows.append((dn / fb, k, dn, fb))
    rows.sort(reverse=True)
    top = rows[:10]
    for r, k, dn, fb in top:
        print(f"  {r:>10.4f}  {k:<44} Δ={dn:.5f} ‖θ₀‖={fb:.5f}")
    print()
    # ★ 统计必须只看 **top**，不是全部 rows。第一版写成 `for ... in rows`，
    #   于是打印"前 10 里：actor 11 个，critic 6 个"—— 加起来 17 > 10。
    #   这种"数字自相矛盾"的输出最容易被顺手信掉。
    n_actor = sum(1 for _, k, _, _ in top if module_of(k) == "actor")
    n_crit = sum(1 for _, k, _, _ in top if module_of(k) == "critic")
    print(f"  前 {len(top)} 里：actor {n_actor} 个，critic {n_crit} 个，"
          f"encoder {sum(1 for _, k, _, _ in top if module_of(k) == 'encoder')} 个")

    # 按**二级前缀**细分：actor/critic 内部到底哪一块在动
    print()
    print("按二级子模块（谁在动）：")
    print(f"  {'子模块':<44}{'参数量':>9}{'Δ':>11}{'‖θ₀‖':>11}{'相对':>9}")
    sub: dict[str, list] = {}
    for k in base:
        if k not in cur or not torch.is_tensor(base[k]) or not base[k].is_floating_point():
            continue
        parts = k.split(".")
        sk = ".".join(parts[:2]) if len(parts) > 2 else parts[0]
        fb = float(base[k].pow(2).sum()) ** 0.5
        dn = float((cur[k] - base[k]).pow(2).sum()) ** 0.5
        sub.setdefault(sk, []).append((base[k].numel(), dn, fb))
    agg = []
    for sk, items in sub.items():
        n = sum(i[0] for i in items)
        dn = sum(i[1] ** 2 for i in items) ** 0.5
        fb = sum(i[2] ** 2 for i in items) ** 0.5
        agg.append((dn / fb if fb else 0, sk, n, dn, fb))
    agg.sort(reverse=True)
    for r, sk, n, dn, fb in agg:
        print(f"  {sk:<44}{n:>9,}{dn:>11.4f}{fb:>11.4f}{r:>9.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
