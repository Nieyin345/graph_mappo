#!/usr/bin/env python
"""按 EVAL_INTERVAL 对齐，输出各 run 的验证曲线（含每种子明细）。

metrics.jsonl 里带验证的行形如：
  {"update": 5, "eval_validation": {"mean_success_rate":..., "per_seed_success":[...]}}
非验证轮没有 eval_validation 键。

用法：
  python .tmp/val_table.py r8_base_s42 ent01_s42 ...
  python .tmp/val_table.py --prefix ent01 --prefix r8_base
  python .tmp/val_table.py --all --field eval_interval
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def load_rows(d: Path):
    m = d / "metrics.jsonl"
    if not m.exists():
        return None
    rows = []
    with m.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows or None


def run_summary(d: Path):
    rows = load_rows(d)
    if not rows:
        return None
    vals = []          # [(update, mean, [per_seed...])]
    for r in rows:
        ev = r.get("eval_validation")
        if isinstance(ev, dict) and ev.get("mean_success_rate") is not None:
            vals.append((r.get("update"), float(ev["mean_success_rate"]),
                         ev.get("per_seed_success") or []))
    cfgp = d / "resolved_config.yaml"
    interval = ent = gamma = None
    if cfgp.exists():
        import yaml
        cfg = yaml.safe_load(cfgp.read_text(encoding="utf-8")) or {}
        tr = cfg.get("train", {}) or {}
        interval = tr.get("eval_interval")
        ent = (tr.get("ppo") or {}).get("entropy_coef")
        gamma = tr.get("gamma")
    return {
        "updates": len(rows),
        "vals": vals,
        "ent": ent,
        "gamma": gamma,
        "interval": interval,
        "mtime": os.path.getmtime(d / "metrics.jsonl"),
    }


def resolve(names, prefixes, all_runs):
    out = list(names)
    for p in prefixes or []:
        out += sorted(os.path.basename(x) for x in glob.glob(str(OUT / f"{p}*"))
                      if os.path.isdir(x))
    if all_runs:
        out += sorted(os.path.basename(x) for x in glob.glob(str(OUT / "*"))
                      if os.path.isdir(x))
    seen, uniq = set(), []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--prefix", action="append", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    names = resolve(args.runs, args.prefix, args.all)
    if not names:
        print("没有匹配的 run")
        return

    print(f"{'run':<24}{'u':>4}{'eval_itv':>9}{'ent':>8}{'gamma':>8}   验证点")
    print("-" * 100)
    for n in names:
        s = run_summary(OUT / n)
        if s is None:
            continue
        vs = "  ".join(f"u{u}:{m:.4f}" for u, m, _ in s["vals"])
        if not s["vals"]:
            vs = f"(无验证点，u={s['updates']})"
        e = f"{s['ent']:.3f}" if isinstance(s["ent"], (int, float)) else "-"
        g = f"{s['gamma']:.3f}" if isinstance(s["gamma"], (int, float)) else "-"
        it = str(s["interval"]) if s["interval"] is not None else "-"
        print(f"{n:<24}{s['updates']:>4}{it:>9}{e:>8}{g:>8}   {vs}")

    # 逐种子明细
    print("\n--- 逐种子（最后验证点）---")
    for n in names:
        s = run_summary(OUT / n)
        if not s or not s["vals"]:
            continue
        u, m, per = s["vals"][-1]
        if per:
            print(f"{n:<24} u{u}: " + " ".join(f"{v:.3f}" for v in per))


if __name__ == "__main__":
    main()
