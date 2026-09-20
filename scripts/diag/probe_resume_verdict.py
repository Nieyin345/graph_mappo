"""续跑后的综合判读：从零族 vs BC 族的 **u50/u60** 平台。

## 为什么要看 u50/u60 而不是 u30

原判据窗口 u25/u30 是**预注册的**，但那是在"只跑到 u30"的前提下定的。
现在续跑到 u60，若还用 u25/u30 比，等于**忽略了后半段**。
`verdict-window-is-part-of-the-claim`：窗口是结论的一半 ⟹ 两个窗口都报。

## 本脚本报三件事

1. **u30**（原预注册窗口）—— 与之前一致，便于对照
2. **u50/u60**（续跑后的平台）—— 新窗口
3. **可检测效应**（t_crit × SE）—— 「测不出」的强度

## ★ 曲线形状也要报

scratch 族续跑后**验证侧剧烈震荡**（s42: 0.7246→0.3836），
而 kl/entropy/|adv| 平滑 ⟹ 不是训练坏了，是**策略在不同验证日上表现差异极大**。
这一条本身是发现，不只是一个数。
"""
import json
import math
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
EXPERT = OUT / "eval" / "expert_seeds100_240.json"
T = {4: 2.776, 5: 2.571, 8: 2.306, 14: 2.145}


def valpts(run):
    p = OUT / run / "metrics.jsonl"
    out, last = {}, None
    if not p.exists():
        return out
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
        if "eval_validation" in o and last is not None:
            out[int(last)] = list(o["eval_validation"]["per_seed_success"])
    return out


def at(vp, u):
    return statistics.mean(vp[u]) if u in vp else None


def report(label, exp_runs, ctrl_runs, window):
    """逐臂取窗口内可用点的均值，配对。"""
    pairs = []
    for e, c in zip(exp_runs, ctrl_runs):
        ve, vc = valpts(e), valpts(c)
        if not ve or not vc:
            continue
        pts = [u for u in window if u in ve and u in vc]
        if not pts:
            continue
        a = statistics.mean(at(ve, u) for u in pts)
        b = statistics.mean(at(vc, u) for u in pts)
        pairs.append((e, a, b, a - b, pts))
    if len(pairs) < 2:
        print(f"  {label}: 配对样本不足（{len(pairs)}）")
        return
    ds = [p[3] for p in pairs]
    n = len(ds)
    m = statistics.mean(ds)
    sd = statistics.stdev(ds)
    se = sd / math.sqrt(n)
    t = m / se if se else 0.0
    crit = T.get(n - 1)
    detect = crit * se if crit else float("nan")
    print(f"\n  {label}   窗口 u{window}")
    print(f"    {'臂':<16}{'实验':>9}{'对照':>9}{'Δ':>10}")
    for e, a, b, d, pts in pairs:
        print(f"    {e:<16}{a:>9.4f}{b:>9.4f}{d:>+10.4f}")
    print(f"    Δ = {m:+.4f}  SD = {sd:.4f}  SE = {se:.4f}  "
          f"t = {t:+.3f} (df={n-1}, 临界 {crit})")
    verd = "★显著" if crit and abs(t) >= crit else "测不出"
    print(f"    ⟹ {verd}   **可检测效应 = {detect:.4f}**")
    return m, detect, verd


def main():
    exp = json.loads(EXPERT.read_text(encoding="utf-8"))
    e_vals = [float(x) for x in (exp.get("success") or exp["per_seed_success"])]
    print("=" * 92)
    print(f"综合判读   专家 = {statistics.mean(e_vals):.4f}")
    print("=" * 92)

    scr = [f"scratch_s{s}" for s in (42, 43, 44)]
    bc = [f"ent01_rerun_s{s}" for s in (42, 43, 44)]
    cln = [f"clean_s{s}" for s in (42, 43, 44, 45, 46)]
    sc5 = [f"scratch_s{s}" for s in (42, 43, 44, 45, 46)]

    # 1. 续跑：从零 vs BC（u30 原窗口 与 u50/60 新窗口）
    print("\n【1】从零族 vs BC 族（续跑前后）")
    for w in ([30], [50, 55, 60]):
        report(f"从零 vs BC @ u{w}", scr, bc, w)

    # 2. clean vs scratch（u30）
    print("\n【2】clean（拿掉专家先验特征）vs scratch（全特征）")
    report("clean vs scratch", cln, sc5, [25, 30])

    # 3. 各族自身的曲线（看形状）
    print("\n【3】曲线形状（验证侧）")
    for name, runs in (("scratch（续跑到 u60）", scr),
                       ("ent01_rerun（只到 u30）", bc),
                       ("clean", cln)):
        print(f"\n  {name}")
        for r in runs:
            vp = valpts(r)
            if not vp:
                continue
            us = sorted(vp)
            vals = [round(statistics.mean(vp[u]), 4) for u in us]
            print(f"    {r:<16} {vals}")


if __name__ == "__main__":
    main()
