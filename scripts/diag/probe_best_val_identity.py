"""best_val 与 final 到底是不是同一份权重？逐臂查，并找出原因。

### 背景

`checkpoint_best_val.pt` 从没被当产出用过，看着像个"白捡的增益"。
但 2026-09-19 实测 `w329p_s42`：两份 checkpoint 的模型权重
**sha256 完全相同、都写着 update=30** ⟹ 它们就是同一份。

### 为什么会长成这样（本脚本要验证的假说）

trainer 在每个 `eval_interval`（5 轮）上算一次验证集均值，
**取最大值**存 best_val。而本项目的验证曲线**到 u30 还在涨**
（已单独实测：u25→u30 还有约 +1 点），所以**最大值就在最后一轮**
⟹ best_val ≡ final，选择偏差无从产生（没有"挑"这个动作）。

★ 反过来这也说明：**只要验证曲线还在升，best_val 就是个空操作**。
  它真正会有用的场合是「验证侧已经见顶回落」——而本项目还没到那儿。

### 判据

逐臂：打印每个 eval 轮次的验证均值，标出 argmax；
再读 checkpoint 里的 `update` 字段，看 best_val 存的是第几轮。
  · argmax == 最后一轮  ⟹ best_val ≡ final（本假说成立）
  · argmax 在中间      ⟹ 两份不同，值得单独评估
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch

OUT = Path("/opt/qkd/graph_mappo/outputs")
ARMS = ["w329p_s42", "w329p_s43", "w329p_s44"]


def per_update(run: str) -> dict[int, float]:
    """{update: 该轮 15 个验证种子的均值}"""
    p = OUT / run / "metrics.jsonl"
    out: dict[int, float] = {}
    n = 0
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = r.get("eval_validation")
        if isinstance(ev, dict) and n:
            v = ev.get("per_seed_success") or ev.get("mean_success_rate")
            if isinstance(v, list) and v:
                out[n] = statistics.mean(float(x) for x in v)
            elif isinstance(v, (int, float)):
                out[n] = float(v)
        elif isinstance(r.get("update"), int):
            n = max(n, int(r["update"]))
    return out


def ckpt_update(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        d = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return f"读不出（{type(exc).__name__}）"  # type: ignore[return-value]
    return d.get("update") if isinstance(d, dict) else getattr(d, "update", None)


def main() -> int:
    print("=" * 78)
    print("逐臂：验证曲线 + best_val 存在第几轮")
    print("=" * 78)
    for run in ARMS:
        d = per_update(run)
        if not d:
            print(f"  {run}: 无 eval 数据")
            continue
        argmax = max(d, key=lambda k: d[k])
        last = max(d)
        ub = ckpt_update(OUT / run / "checkpoint_best_val.pt")
        uf = ckpt_update(OUT / run / "checkpoint_final.pt")
        print(f"\n  ── {run} ──")
        print(f"     逐轮：" + "  ".join(f"u{k}={v:.4f}" for k, v in sorted(d.items())))
        print(f"     argmax = u{argmax}（{d[argmax]:.4f}）   最后一轮 = u{last}"
              f"（{d[last]:.4f}）")
        print(f"     checkpoint_best_val 里 update={ub}   "
              f"checkpoint_final 里 update={uf}")
        if argmax == last:
            print("     ⟹ **best_val ≡ final**（验证曲线在末轮最高，没有「挑」这个动作）")
        else:
            print(f"     ⟹ best_val 是**中间轮 u{argmax}**，与 final 不同，值得单独评估")

    # ---- 权重是否逐位相同 ----
    print()
    print("=" * 78)
    print("权重逐位比对（sha256）")
    print("=" * 78)
    import hashlib
    for run in ARMS:
        hs = {}
        for tag, f in (("final", "checkpoint_final.pt"),
                       ("best_val", "checkpoint_best_val.pt")):
            p = OUT / run / f
            if not p.exists():
                hs[tag] = "缺文件"
                continue
            d = torch.load(str(p), map_location="cpu", weights_only=False)
            ms = d["model_state"] if isinstance(d, dict) else d.model_state
            h = hashlib.sha256()
            for k in sorted(ms):
                h.update(k.encode())
                h.update(ms[k].detach().cpu().numpy().tobytes())
            hs[tag] = h.hexdigest()[:20]
        same = "★ 完全相同" if hs.get("final") == hs.get("best_val") else "不同"
        print(f"  {run:12s} final={hs.get('final')}  best_val={hs.get('best_val')}"
              f"   → {same}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
