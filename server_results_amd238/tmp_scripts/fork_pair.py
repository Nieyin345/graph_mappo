#!/usr/bin/env python
"""同起点分叉臂的配对比较：只改一个超参、同种子、从共同检查点恢复。

**为什么要单独写**：`paired_vs_expert.py` 是"RL vs 专家"（对照是固定的 0.6979），
而这里对照是**另一条 run**。两条 run 从同一检查点分叉、同种子、同 base_seed
（`base_seed = env_seed + update_count * stride`，update_count 都由检查点恢复），
所以 u>=fork 轮的请求流逐位相同，逐验证种子配对是有意义的。

**但必须说清这个配对 t **不能**证明什么**：
配对只用 15 个**验证实例**的差异算 SE，它检验的是"在这 15 个留出实例上，
两条臂的差是否可辨"。它**不包含训练种子的变异**——只有 1 个训练种子。
按 docs/测试规范.md §4⑦，单训练种子的分辨率 ~0.035，所以：
  - |配对差| < 0.035 → **测不出差异**（"无效应"），不能说"证明了两者相同"；
  - |配对差| >= 0.035 → 值得再用 ≥3 个训练种子复验，而不是就此定论。

用法（服务器上）：
  python /tmp/fork_pair.py --base ent01_s42 --arm ent01_s42_u25_base --fork 25
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def points(run: str):
    """返回 [(update, mean, per_seed_list, seeds)]，验证行靠前一条训练行定轮次。"""
    p = OUT / run / "metrics.jsonl"
    if not p.exists():
        return []
    last_u, out = None, []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "update" in r:
                last_u = int(r["update"])
            ev = r.get("eval_validation")
            if isinstance(ev, dict) and ev.get("per_seed_success"):
                out.append((last_u, float(ev["mean_success_rate"]),
                            list(ev["per_seed_success"]), list(ev.get("seeds") or [])))
    return out


def val_seeds(run: str):
    """从 resolved_config.yaml 取验证种子，确认两臂口径一致。"""
    p = OUT / run / "resolved_config.yaml"
    if not p.exists():
        return None
    try:
        import yaml
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    v = cfg.get("validation") or {}
    return {
        "request_seeds": v.get("request_seeds"),
        "steps": cfg.get("env", {}).get("episode_steps"),
        "entropy_coef": (cfg.get("train", {}).get("ppo") or {}).get("entropy_coef"),
        "gamma": (cfg.get("train", {}).get("ppo") or {}).get("gamma")
        or (cfg.get("train", {}).get("gamma")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="原臂（有完整曲线）")
    ap.add_argument("--arm", required=True, help="分叉臂（从 --fork 轮恢复）")
    ap.add_argument("--fork", type=int, required=True, help="分叉的轮次")
    args = ap.parse_args()

    print("=== 配置口径核对（两臂必须只剩被测的那一项不同）===")
    cb, ca = val_seeds(args.base), val_seeds(args.arm)
    for k in ("request_seeds", "steps", "entropy_coef", "gamma"):
        vb = (cb or {}).get(k)
        va = (ca or {}).get(k)
        if k == "request_seeds" and isinstance(vb, list):
            vb, va = f"[{vb[0]}..{vb[-1]}] n={len(vb)}", (f"[{va[0]}..{va[-1]}] n={len(va)}"
                                                          if isinstance(va, list) else va)
        flag = "" if vb == va else "   <-- 不同"
        print(f"  {k:<15} base={vb}   arm={va}{flag}")
    print()

    pb, pa = points(args.base), points(args.arm)
    if not pb or not pa:
        print(f"缺数据: base {len(pb)} 点 / arm {len(pa)} 点")
        return

    db = {u: (m, ps) for u, m, ps, _s in pb}
    da = {u: (m, ps) for u, m, ps, _s in pa}
    common = sorted(set(db) & set(da))

    print(f"=== 共同验证轮次：{common}（分叉于 u{args.fork}）===")
    print(f"{'u':>4}{'base':>10}{'arm':>10}{'Δ':>10}{'配对SE':>9}{'t':>7}"
          f"{'胜/负':>8}   配对t覆盖")
    print("-" * 74)
    for u in common:
        mb, psb = db[u]
        ma, psa = da[u]
        if len(psb) != len(psa):
            print(f"{u:>4}  逐种子数不一致 {len(psb)} vs {len(psa)}，跳过")
            continue
        diffs = [psa[i] - psb[i] for i in range(len(psb))]
        md = sum(diffs) / len(diffs)
        sed = statistics.stdev(diffs) / len(diffs) ** 0.5 if len(diffs) > 1 else float("nan")
        t = md / sed if sed and sed == sed else float("nan")
        win = sum(1 for d in diffs if d > 0)
        cover = "分叉前" if u <= args.fork else "分叉后"
        print(f"{u:>4}{mb:>10.4f}{ma:>10.4f}{md:>+10.4f}{sed:>9.4f}{t:>+7.2f}"
              f"{win:>5}/{len(diffs)-win:<3}   {cover}")
        if u > args.fork:
            print(f"{'':<4}逐种子差: " + " ".join(f"{d:+.3f}" for d in diffs))

    print()
    print("=== 判读（按 docs/测试规范.md §4⑦，**先看分辨率再看符号**）===")
    post = [u for u in common if u > args.fork]
    if not post:
        print("  分叉后没有共同验证点，无法判读。")
        return
    for u in post:
        mb, psb = db[u]
        ma, psa = da[u]
        diffs = [psa[i] - psb[i] for i in range(len(psb))]
        md = sum(diffs) / len(diffs)
        sed = statistics.stdev(diffs) / len(diffs) ** 0.5
        verdict = ("**测不出差异**（|Δ| < 单训练种子分辨率 0.035）"
                   if abs(md) < 0.035 else
                   "|Δ| >= 0.035，值得用 ≥3 个训练种子复验（不是就此定论）")
        print(f"  u{u}: Δ={md:+.4f}  配对SE={sed:.4f}  配对t={md/sed:+.2f} → {verdict}")
    print()
    print("  注意：配对 t 只用 15 个**验证实例**算 SE，**不含训练种子的变异**。")
    print("  它检验的是「在这 15 个留出实例上两条臂是否可辨」，")
    print("  **不能**推出「该超参在总体上无效应」——那需要 ≥3 个训练种子。")


if __name__ == "__main__":
    main()
