#!/usr/bin/env python
"""通用**配对平台判读**：逐种子配对 Δ_s = 实验臂 − 对照臂，单样本 t。

### 为什么抽出来

`mode_de_verdict.py` 那套逻辑（取平台窗口逐种子均值 → 配对差 → 单样本 t →
按 df 的临界值判 → 报「测不出」而不是「没影响」）**每一波实验都要重写一遍**。
重写就会重踩，本项目已经在「判据落在错误位置」上栽过三次。

### 用法（服务器上）

    /opt/qkd/venv/bin/python scripts/diag/paired_verdict.py \
        --label 'ent03 vs ent01_t8（熵 0.03 vs 0.01）' \
        --ctrl-fmt 'ent01_t8_s{}' --exp-fmt 'ent03_s{}' --seeds 42,43,44 \
        --expect-negative

    # 平台窗口默认 u25/u30；单臂只有部分轮次时会自动退到可用轮
    --plateau 25,30

### 判据（**必须在跑之前写死**）

  |t| >= 临界值( df = n-1 ) ⟹ 有可测差异，按符号报方向
  |t| <  临界值            ⟹ **报「以 n 的分辨率测不出」，不报「没影响」**

临界值**现算**，不手抄 —— 本项目有过「手抄数字悄悄过期」的教训
（[[thresholds-and-transcribed-numbers]]）。scipy 不可用时退回一张小表。
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
EXPERT = OUT / "eval" / "expert_seeds100_240.json"

# df=1..8 的双侧 95% 临界值（scipy 缺席时的退路；够本项目用）
T_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
           5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}


def val_points(run: str) -> dict[int, list[float]]:
    """{update: per_seed_success}"""
    p = OUT / run / "metrics.jsonl"
    out: dict[int, list[float]] = {}
    if not p.exists():
        return out
    last = None
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "update" in o:
            last = o["update"]
        ev = o.get("eval_validation")
        if isinstance(ev, dict) and ev.get("per_seed_success") and last is not None:
            out[last] = [float(x) for x in ev["per_seed_success"]]
    return out


def plateau_mean(run: str, window: tuple[int, ...]):
    """窗口内逐验证种子取均值；窗口轮次缺失时退到**最后一个可用轮**。"""
    pts = val_points(run)
    if not pts:
        return None, "无 metrics"
    us = [u for u in window if u in pts]
    fell_back = False
    if not us:
        last = max(pts)
        us = [last]
        fell_back = True
    n = len(pts[us[0]])
    if any(len(pts[u]) != n for u in us):
        return None, "窗口内各轮种子数不一致"
    vals = [statistics.mean(pts[u][i] for u in us) for i in range(n)]
    return vals, (f"退到 u{us[0]}（窗口 {list(window)} 无数据）" if fell_back else None)


def t_crit(df: int) -> tuple[float, str]:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df)), "scipy"
    except ImportError:
        return T_TABLE.get(df, float("nan")), "内置表"


def p_value(t: float, df: int) -> tuple[float, str]:
    try:
        from scipy import stats
        return 2 * float(stats.t.sf(abs(t), df)), "scipy"
    except ImportError:
        return float("nan"), "无 scipy"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--ctrl-fmt", required=True, help="如 ent01_t8_s{}")
    ap.add_argument("--exp-fmt", required=True, help="如 ent03_s{}")
    ap.add_argument("--seeds", required=True, help="逗号分隔，如 42,43,44")
    ap.add_argument("--plateau", default="25,30")
    ap.add_argument("--expect-negative", action="store_true",
                    help="跑之前预期实验臂更差（用于措辞，不影响判据）")
    ap.add_argument("--note", default="", help="记账：需要随结论一起说的前提")
    a = ap.parse_args(argv[1:])

    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    window = tuple(int(x) for x in a.plateau.replace(",", " ").split())

    print("=" * 78)
    print(f"配对判读：{a.label}")
    print(f"  Δ_s = {a.exp_fmt.format('<seed>')} − {a.ctrl_fmt.format('<seed>')}"
          f"   种子 {seeds}   平台窗口 u{'/'.join(map(str, window))}")
    print("=" * 78)

    ctrl: dict[int, list[float]] = {}
    exp: dict[int, list[float]] = {}
    notes = []
    for s in seeds:
        c, nc = plateau_mean(a.ctrl_fmt.format(s), window)
        e, ne = plateau_mean(a.exp_fmt.format(s), window)
        ctrl[s], exp[s] = c, e
        for nm, nn in ((a.ctrl_fmt.format(s), nc), (a.exp_fmt.format(s), ne)):
            if nn:
                notes.append(f"{nm}: {nn}")

    missing = [s for s in seeds if ctrl[s] is None or exp[s] is None]
    if missing:
        print(f"  ✗ 数据不全，缺种子 {missing}")
        for s in missing:
            for fmt, d in ((a.ctrl_fmt, ctrl), (a.exp_fmt, exp)):
                if d[s] is None:
                    print(f"      {fmt.format(s)}")
        for n in notes:
            print(f"      注：{n}")
        return 1
    for n in notes:
        print(f"  ⚠ {n}")

    diffs = []
    print()
    print(f"  {'种子':>6}{'对照':>14}{'实验':>14}{'Δ':>12}")
    for s in seeds:
        mc, me = statistics.mean(ctrl[s]), statistics.mean(exp[s])
        d = me - mc
        diffs.append(d)
        print(f"  {s:>6}{mc:>14.4f}{me:>14.4f}{d:>+12.4f}")

    m = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    se = sd / math.sqrt(len(diffs)) if len(diffs) > 1 else float("inf")
    t = m / se if se > 0 else float("inf")
    df = len(diffs) - 1
    crit, csrc = t_crit(df)
    p, psrc = p_value(t, df)

    print()
    print(f"  Δ = {m:+.4f}   SD(Δ_s) = {sd:.4f}   SE = {se:.4f}")
    print(f"  t = {t:+.3f}  (df={df}, 双侧临界值 {crit:.3f} [{csrc}])")
    print(f"  p = {p:.4f}  ({psrc})")
    if a.note:
        print(f"  记账：{a.note}")

    print()
    print("=" * 78)
    if abs(t) >= crit:
        print(f"  ✗ |t| = {abs(t):.3f} ≥ {crit:.3f} ⟹ **有可测差异**")
        print(f"    Δ = {m:+.4f}（实验臂{'更好' if m > 0 else '更差'}）")
        if a.expect_negative and m < 0:
            print("    方向与跑之前写死的预期一致。")
        elif a.expect_negative and m > 0:
            print("    ⚠ 方向**与跑之前写死的预期相反** —— 先查机理，别当好消息收下。")
    else:
        print(f"  ✓ |t| = {abs(t):.3f} < {crit:.3f} ⟹ **以 n={len(seeds)} 的分辨率测不出差异**")
        print(f"    必须报「测不出」，**不能**报「没影响」：")
        print(f"      n={len(seeds)}（df={df}）的分辨率约 **0.035**，比这小的效应测不到。")
        print(f"    要变成结论需加训练种子（每种子约 30 分钟）。")

    if EXPERT.exists():
        ex = json.loads(EXPERT.read_text(encoding="utf-8"))
        print(f"\n  参考：专家 = {statistics.mean(float(x) for x in ex['success']):.4f}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
