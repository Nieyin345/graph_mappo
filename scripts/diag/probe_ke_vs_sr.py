"""修正版：**只在同一个轮号上比**（u5/u30 混比是混淆）。

上一版把「运行中臂的最新验证点（u5）」和「已跑完臂的末点（u30）」
放进同一张表 ⟹ 跨臂相关被轮号混淆污染。

本版：显式按轮号分组，只报**都是 u30** 的臂。运行中的臂单独列。
"""
import json
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def evals(run):
    """{update: (sr, ke)}"""
    p = OUT / run / "metrics.jsonl"
    out = {}
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
        if "eval_validation" in o and last is not None:
            ev = o["eval_validation"]
            out[int(last)] = (ev.get("mean_success_rate"),
                              ev.get("mean_key_efficiency"))
    return out


def main():
    arms = sorted(d.name for d in OUT.iterdir() if d.is_dir())
    u30, running = [], []
    for a in arms:
        ev = evals(a)
        if not ev:
            continue
        mx = max(ev)
        sr, ke = ev[mx]
        if sr is None:
            continue
        if mx >= 30:
            u30.append((a, sr, ke))
        else:
            running.append((a, mx, sr, ke))

    print("=" * 96)
    print("★ 只在 u30 上比（已完成臂）")
    print("=" * 96)
    print(f"  {'臂':<24}{'SR':>10}{'ke':>10}")
    print("  " + "-" * 44)
    for a, sr, ke in sorted(u30, key=lambda x: (x[2] if x[2] else 1e9)):
        mark = "  ← 超专家(0.6979)" if sr > 0.6979 else ""
        print(f"  {a:<24}{sr:>10.4f}{(f'{ke:>10.1f}' if ke else '         —')}{mark}")

    print("\n" + "=" * 96)
    print("运行中（未到 u30，**不与上面混比**）")
    print("=" * 96)
    print(f"  {'臂':<24}{'到':>5}{'SR':>10}{'ke':>10}")
    for a, mx, sr, ke in sorted(running, key=lambda x: x[0]):
        print(f"  {a:<24}{mx:>5}{sr:>10.4f}{(f'{ke:>10.1f}' if ke else '         —')}")

    # 只在 u30 上算相关
    sub = [(sr, ke) for _, sr, ke in u30 if ke]
    if len(sub) >= 4:
        import statistics
        srs = [x[0] for x in sub]
        kes = [x[1] for x in sub]
        ms, mk = statistics.mean(srs), statistics.mean(kes)
        cov = sum((s - ms) * (k - mk) for s, k in sub) / len(sub)
        ds = (sum((s - ms) ** 2 for s in srs) / len(srs)) ** 0.5
        dk = (sum((k - mk) ** 2 for k in kes) / len(kes)) ** 0.5
        r = cov / (ds * dk) if ds and dk else float("nan")
        print(f"\n  ★ **u30 上**的跨臂相关 r(SR, ke) = {r:+.3f}  (n={len(sub)})")
        print("    期望为负（越省 ⟹ 越高）")
        print("    ⚠ 仍是线索不是证明：不同臂改的旋钮不同，混淆严重")

    # BC 系 vs scratch 系（都到 u30 的）
    print("\n" + "=" * 96)
    print("scratch（从零）vs BC 系（有暖启动）—— 都只取 u30")
    print("=" * 96)
    sc = [(a, sr, ke) for a, sr, ke in u30 if a.startswith("scratch")]
    bc = [(a, sr, ke) for a, sr, ke in u30
          if a.startswith(("ent01_rerun", "v2_bottleneck", "pm_decode", "v1_onpath"))]
    if sc:
        print("  从零：")
        for a, sr, ke in sorted(sc, key=lambda x: -x[1]):
            print(f"    {a:<22}{sr:>10.4f}{(f'{ke:>10.1f}' if ke else '         —')}")
    if bc:
        import statistics
        print(f"  有 BC：n={len(bc)}  均值 SR={statistics.mean(x[1] for x in bc):.4f}"
              f"  ke 均值={statistics.mean(x[2] for x in bc if x[2]):.1f}"
              f"  （范围 {min(x[1] for x in bc):.4f}~{max(x[1] for x in bc):.4f}）")
        print(f"  专家 = 0.6979")


if __name__ == "__main__":
    main()
