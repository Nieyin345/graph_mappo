"""读 `stop_logit` 的轨迹：这个**全局标量**到底有没有被学动过？

### 为什么这个问题决定一整条结构方向

`graph_mappo.py:439` 只有**一个** `self.stop_logit = nn.Parameter(torch.zeros(()))` ——
一个标量，**全节点、全状态共用**。STOP 的分数就是它（`policy.py:139/386/680/762/765/869/871`）。

而专家 `PathScoreGreedy(phased=True)` 已实测是"**钉住 54.6 条、游程 9.92 槽**"的行为
（见 `docs/当前状态与优化方向.md` §1.3）——专家**按状态**决定何时停。
一个全局标量**原理上无法**表达这种状态依赖。

所以先把问题分成两半，**顺序不能反**：

| stop_logit 的行为 | 含义 | 下一步 |
|---|---|---|
| **30 轮几乎不动** | 梯度没把它推走 ⟹ 这个参数在损失面上的可动空间很小 ⟹ **状态化是活的方向**（加的是模型本来缺的表达能力） |
| **30 轮明显移动** | 模型**在用**它 ⟹ 说明单标量也够用，或者它在补偿别的东西 ⟹ 先查它往哪走、为什么，再谈改结构 |

**不先测就改结构，是拿成本换猜测。** 这个探针零训练成本。

### 怎么读

`model_state` 就是 `model.state_dict()`（`checkpoint.py:35`），所以只需
`torch.load(...)["model_state"]["actor.stop_logit"]` —— **不必构造模型**，
也就不会拉起 qkd_rl 的 env（内存友好）。

用法（服务器上）：
  /opt/qkd/venv/bin/python scripts/diag/probe_stop_logit.py [run-name ...]
  不给 run-name 时自动扫 outputs/ 下所有含 checkpoint_update_*.pt 的 run。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

OUT = Path("/opt/qkd/graph_mappo/outputs")
BC = OUT / "supervised_pg_phased" / "supervised_pg_phased_latest.pt"
KEY = "actor.stop_logit"
UPD_RE = re.compile(r"checkpoint_update_(\d+)\.pt$")


def read_stop_logit(path: Path):
    """只要那个标量，不构造模型。"""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:                       # noqa: BLE001
        return None, f"读取失败: {exc}"
    state = payload.get("model_state") if isinstance(payload, dict) else None
    if state is None:
        return None, "无 model_state"
    if KEY not in state:
        cand = [k for k in state if "stop" in k.lower()]
        return None, f"无 {KEY}；含 stop 的键: {cand}"
    return float(state[KEY].reshape(-1)[0]), None


def series_for(run_dir: Path):
    """按 update 序号排序的 [(update, value)]，外加 BC 起点。"""
    pts = []
    for p in run_dir.glob("checkpoint_update_*.pt"):
        m = UPD_RE.search(p.name)
        if m:
            pts.append((int(m.group(1)), p))
    pts.sort()
    return pts


def main(argv: list[str]) -> int:
    runs = [OUT / a for a in argv[1:]] if len(argv) > 1 else sorted(
        d for d in OUT.iterdir() if d.is_dir() and any(d.glob("checkpoint_update_*.pt"))
    )
    if not runs:
        print("没有找到含 checkpoint_update_*.pt 的 run")
        return 1

    bc_val = None
    if BC.exists():
        bc_val, err = read_stop_logit(BC)
        print(f"BC 起点  {BC.parent.name:<28}  {KEY} = "
              f"{'%.6f' % bc_val if bc_val is not None else 'ERR: ' + str(err)}")
        print()

    for d in runs:
        pts = series_for(d)
        if not pts:
            continue
        vals = [(u, *read_stop_logit(p)) for u, p in pts]
        ok = [(u, v) for u, v, e in vals if v is not None]
        print(f"{d.name}  （{len(ok)} 个检查点）")
        if not ok:
            print(f"    ✗ {vals[0][2]}")
            continue
        if bc_val is not None:
            print(f"    Δ vs BC 起点 = {ok[0][1] - bc_val:+.6f}")
        # 轨迹：逐点打，密集时压缩
        show = ok if len(ok) <= 12 else ok[:: max(1, len(ok) // 10)] + [ok[-1]]
        line = "  ".join(f"u{u}:{v:+.4f}" for u, v in show)
        print(f"    {line}")

        # ★ 只有 1 个检查点时**不能**判读。第一版照样打「净变化 +0.000000 /
        #   全程极差 0.000000 / 几乎没动」——三个都是**单点的算术恒等式**，
        #   不是观测。这种"看起来有结论"的输出比没有输出更坏。
        if len(ok) < 3:
            print(f"    ⚠ 只有 {len(ok)} 个点，**不足以判读趋势**（至少 3 个）。")
            print(f"      检查点每 5 轮一个，30 轮跑完会有 6 个。")
            print()
            continue

        first, last = ok[0][1], ok[-1][1]
        span = max(v for _, v in ok) - min(v for _, v in ok)
        print(f"    首 {first:+.6f} → 末 {last:+.6f}   净变化 {last - first:+.6f}"
              f"   全程极差 {span:.6f}")
        # 判读
        net = abs(last - first)
        if net < 0.01 and span < 0.05:
            print("    ⟹ **几乎没动** → 状态化是活的方向（模型本来缺的表达能力）")
        elif net >= 0.1 or span >= 0.3:
            print("    ⟹ **明显在动** → 模型在用它；先查往哪走、为什么，再谈改结构")
        else:
            print("    ⟹ 有移动但幅度中等 → 判读需要更多种子/轮数，别急着下结论")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
