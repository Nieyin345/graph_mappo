#!/usr/bin/env python
"""核对"专家基线 0.698"的来源：outputs/eval/ 下所有 json 的口径与两两配对差。

为什么必须做：日志里到处拿 0.698 当参照（"超过专家"的判据全靠它），
但 outputs/eval/ 里躺着 6 个都叫 expert* 的文件，种子集与数值各不相同。
判据错了，"超过专家"这句话就没有意义。
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parents[2] / "server_results" / "runs" / "outputs" / "eval"
if not EVAL.exists():
    EVAL = Path(sys.argv[1])
FROZEN = EVAL / "frozen_baselines.json"


def load(p: Path):
    d = json.loads(p.read_text(encoding="utf-8"))
    s = d.get("success") or []
    if not s:
        return None
    return {
        "policy": d.get("policy"),
        "steps": d.get("steps"),
        "seeds": d.get("seeds") or [],
        "success": s,
        "mean": sum(s) / len(s),
        "sd": statistics.stdev(s) if len(s) > 1 else 0.0,
        "se": (statistics.stdev(s) / len(s) ** 0.5) if len(s) > 1 else 0.0,
    }


def main():
    files = sorted(EVAL.glob("*.json"))
    store = {}
    print(f"{'文件':<30}{'policy':<26}{'n':>3}{'steps':>7}{'seed0':>6}"
          f"{'seedN':>7}{'mean':>9}{'SE':>8}{'min':>7}{'max':>7}")
    print("-" * 112)
    for f in files:
        if f.name == "frozen_baselines.json":
            continue
        r = load(f)
        if r is None:
            print(f"{f.name:<30}(无 success)")
            continue
        store[f.name] = r
        sd = r["seeds"]
        print(f"{f.name:<30}{str(r['policy'])[:25]:<26}{len(r['success']):>3}"
              f"{str(r['steps']):>7}{str(sd[0] if sd else '?'):>6}"
              f"{str(sd[-1] if sd else '?'):>7}{r['mean']:>9.4f}{r['se']:>8.4f}"
              f"{min(r['success']):>7.3f}{max(r['success']):>7.3f}")

    # 两两配对：只对**种子集合完全一致**的做逐种子差
    print("\n=== 同种子集两两配对差 ===")
    names = [n for n in store if n.startswith(("expert", "psg"))]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra, rb = store[a], store[b]
            if set(ra["seeds"]) != set(rb["seeds"]):
                continue
            ia = {s: k for k, s in enumerate(ra["seeds"])}
            ib = {s: k for k, s in enumerate(rb["seeds"])}
            order = [s for s in ra["seeds"] if s in ib]
            diffs = [ra["success"][ia[s]] - rb["success"][ib[s]] for s in order]
            md = sum(diffs) / len(diffs)
            sed = (statistics.stdev(diffs) / len(diffs) ** 0.5
                   if len(diffs) > 1 else 0.0)
            t = md / sed if sed else float("nan")
            print(f"  {a} − {b}:  {md:+.4f}  SE {sed:.4f}  t {t:+.2f}"
                  f"   (n={len(diffs)}, 同种子集 {len(set(ra['seeds']))})")

    # frozen_baselines 里登记的数字
    if FROZEN.exists():
        fr = json.loads(FROZEN.read_text(encoding="utf-8"))
        print(f"\n=== frozen_baselines.json（frozen_at {fr.get('frozen_at')}, "
              f"git_rev {fr.get('git_rev')}）===")
        for pk, pv in fr["protocols"].items():
            print(f"  [{pk}] steps={pv['episode_steps']}"
                  f" seeds={pv['seeds'][0]}–{pv['seeds'][-1]}"
                  f" (n={len(pv['seeds'])}) days={pv['window_days']}")
            for name, pol in pv["policies"].items():
                print(f"      {name:<28} mean={pol['mean_success']:.4f}"
                      f"  n={pol['episodes']}  src={pol['source']}")

    # 交叉核对：frozen 登记的专家值 vs 源文件实算值
    print("\n=== 交叉核对（frozen 登记 vs 源文件实算）===")
    if FROZEN.exists():
        fr = json.loads(FROZEN.read_text(encoding="utf-8"))
        for pk, pv in fr["protocols"].items():
            pol = pv["policies"].get("path_score_greedy_phased")
            if not pol:
                continue
            src = Path(pol["source"]).name
            r = store.get(src)
            if not r:
                print(f"  [{pk}] 源文件 {src} 不在该目录（可能在别的 root）")
                continue
            ok = (abs(r["mean"] - pol["mean_success"]) < 1e-9
                  and len(r["success"]) == pol["episodes"])
            print(f"  [{pk}] {src}: frozen={pol['mean_success']:.4f}"
                  f" ({pol['episodes']} 集)  实算={r['mean']:.4f}"
                  f" ({len(r['success'])} 集)  {'一致' if ok else '★不一致★'}")


if __name__ == "__main__":
    main()
