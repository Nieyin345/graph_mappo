"""参数漂移探针：**训练到底在动哪一部分**？

### 为什么问这个

本项目已测：训练侧成功率先饱和（~0.86 vs 专家 0.869），加轮数无效
（续训 20 轮增量 +0.0013），`kl≈0.001` 使 `clip` 从未激活。这些都是
「**在同一目标上更努力**」的证据。那么问题是：

  **是梯度推不动整个网络，还是只推得动某几层？**

这个问题决定结构改动的方向：
  · 若 **actor 几乎不动、critic 在动** → 策略已到局部最优，该改的是
    表达能力/目标，不是优化强度
  · 若 **encoder 不动、actor 在动** → 表征已经够用/是瓶颈，该看 encoder
  · 若 **全都在动但指标不动** → 动的是没用的方向（零空间漂移，
    本项目已在 [[rl-key-efficiency-is-unconstrained-drift]] 见过同类）

### 为什么是"零训练成本"

`checkpoint.py:35` 把 `model.state_dict()` 原样存在 `model_state` 下，
所以 `torch.load` 之后**比参数字典就行，不必构造模型**，
也就不会拉起 env（内存友好，可在跑训练的机器上安全执行）。

### 判读口径（跑之前写死）

对每个 run、每个模块 m ∈ {encoder, actor, critic, 其它}：

    drift_m(u) = ‖θ_m(u) − θ_m(BC)‖₂ / ‖θ_m(BC)‖₂      （相对 L2 漂移）

三个判据：
  1. **谁在动**：末轮 drift 的排序。某模块 drift < 0.02 视为"没动"。
  2. **是否饱和**：drift 曲线在末段是否走平（斜率 < 首段斜率的 1/5）。
  3. **动得值不值**：drift 大而指标不动 ⟹ 漂移没换来性能。

用法（服务器上）：
  /opt/qkd/venv/bin/python /tmp/probe_param_drift.py [run-name ...]
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

OUT = Path("/opt/qkd/graph_mappo/outputs")
BC = OUT / "supervised_pg_phased" / "supervised_pg_phased_latest.pt"
UPD_RE = re.compile(r"checkpoint_update_(\d+)\.pt$")

# 模块归属：按 state_dict 键的**前缀**分组。前缀是 `encoder.` / `actor.` /
# `critic.` 这种；分不进去的一律进 "其它"（宁可漏分组，不可错分组）。
MODULES = ("encoder", "actor", "critic")
FROZEN = 0.02          # 相对漂移 < 此值视为"没动"


def state_of(path: Path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:                       # noqa: BLE001
        return None, f"读取失败: {exc}"
    st = payload.get("model_state") if isinstance(payload, dict) else None
    if st is None:
        return None, "无 model_state"
    return st, None


def module_of(key: str) -> str:
    head = key.split(".", 1)[0]
    return head if head in MODULES else "其它"


def norm_of(st: dict, mod: str | None) -> float:
    """模块参数的 L2 范数（None = 全部）。"""
    tot = 0.0
    for k, v in st.items():
        if mod is not None and module_of(k) != mod:
            continue
        if torch.is_tensor(v) and v.is_floating_point():
            tot += float(v.detach().float().pow(2).sum())
    return tot ** 0.5


def rel_drift(cur: dict, base: dict, mod: str | None) -> float | None:
    """‖θ−θ₀‖ / ‖θ₀‖，只在两者键集合一致时才算。"""
    num = 0.0
    keys = [k for k in base if torch.is_tensor(base[k]) and base[k].is_floating_point()]
    if mod is not None:
        keys = [k for k in keys if module_of(k) == mod]
    if not keys:
        return None
    for k in keys:
        if k not in cur:
            return None
        num += float((cur[k].detach().float() - base[k].detach().float()).pow(2).sum())
    den = norm_of({k: base[k] for k in keys}, None)
    return (num ** 0.5) / den if den > 0 else None


def series(run_dir: Path):
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

    base, err = state_of(BC) if BC.exists() else (None, "BC 检查点不存在")
    if base is None:
        print(f"✗ 读不到 BC 起点: {err}")
        return 1
    print(f"BC 起点 {BC.parent.name}   参数键 {len(base)} 个")
    print()

    for d in runs:
        pts = series(d)
        if not pts:
            continue
        rows = []
        for u, p in pts:
            st, e = state_of(p)
            if st is None:
                continue
            rows.append((u, {m: rel_drift(st, base, m) for m in MODULES + ("其它",)}))
        if not rows:
            continue
        print(f"{d.name}  （{len(rows)} 个检查点）")
        if len(rows) < 3:
            print(f"    ⚠ 只有 {len(rows)} 个点，**不足以判读趋势**（至少 3 个）")
            print(f"      u{rows[-1][0]}: " + "  ".join(
                f"{m}={v:.4f}" for m, v in rows[-1][1].items() if v is not None))
            print()
            continue

        print(f"    {'u':>4}" + "".join(f"{m:>12}" for m in MODULES + ("其它",)))
        show = rows if len(rows) <= 12 else rows[:: max(1, len(rows) // 10)] + [rows[-1]]
        for u, dr in show:
            print(f"    {u:>4}" + "".join(
                f"{dr[m]:>12.4f}" if dr[m] is not None else f"{'-':>12}"
                for m in MODULES + ("其它",)))

        # 判读：末轮漂移排序 + 是否饱和
        last = rows[-1][1]
        moved = {m: v for m, v in last.items() if v is not None and v >= FROZEN}
        frozen = [m for m, v in last.items() if v is not None and v < FROZEN]
        print(f"    末轮：在动 = {sorted(moved, key=lambda k: -moved[k]) or '（无）'}"
              f"   没动(<{FROZEN}) = {frozen or '（无）'}")
        # 饱和：末段斜率 vs 首段斜率
        if len(rows) >= 4:
            half = len(rows) // 2
            for m in MODULES:
                vals = [(u, dr[m]) for u, dr in rows if dr[m] is not None]
                if len(vals) < 4:
                    continue
                h = len(vals) // 2
                s1 = (vals[h][1] - vals[0][1]) / max(1, vals[h][0] - vals[0][0])
                s2 = (vals[-1][1] - vals[h][1]) / max(1, vals[-1][0] - vals[h][0])
                if s1 > 0 and s2 < s1 / 5:
                    print(f"    {m}: 首段斜率 {s1:.2e} → 末段 {s2:.2e}"
                          f"  ⟹ **已饱和**（末段 < 首段 1/5）")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
