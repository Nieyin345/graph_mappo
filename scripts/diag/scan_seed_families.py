#!/usr/bin/env python
"""扫**全部** outputs，找所有"同配置多训练种子"的家族，量真实的逐种子 SD。

为什么必须扫全：我用 `ent01_s42/43/44` 的 3 个点在 u30 量到 SD=0.0118，据此说
「3 个种子 t=3.65，够用」。但 **3 个点估 SD，SD 本身的相对不确定度约 50%**
（1/sqrt(2(n-1))）。而 `docs/测试规范.md` ⑦ 写的逐种子跨度是 **0.104**，比我这
读数大 5 倍。两者必有一个不能用来支撑那个结论。

所以不猜、也不挑有利的那个：**把 outputs 里所有 ≥3 个种子的家族都量出来**，
看 SD 的分布。若普遍在 0.01 量级，我的结论成立；若普遍在 0.03-0.04，
那就是**功效不足**，得趁链条还能改赶紧加种子。

用法（服务器上）：python /tmp/scan_seed_families.py
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
PAT = re.compile(r"^(?P<fam>.+?)_s(?P<seed>\d{2})(?P<suf>[a-z0-9_]*)$")


def series(run: str) -> dict[int, float]:
    p = OUT / run / "metrics.jsonl"
    if not p.exists():
        return {}
    out: dict[int, float] = {}
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
            out[last] = float(ev["mean_success_rate"])
    return out


fams: dict[str, dict[int, str]] = defaultdict(dict)
for d in sorted(OUT.iterdir()):
    if not d.is_dir():
        continue
    m = PAT.match(d.name)
    if not m:
        continue
    # 后缀非空的一律排除（续跑 u30to50、重跑 r2 等不是独立种子）
    if m.group("suf"):
        continue
    fams[m.group("fam")][int(m.group("seed"))] = d.name

print("=== 所有 ≥3 个训练种子的家族，逐轮跨种子 SD ===")
rows = []
for fam, seeds in sorted(fams.items()):
    if len(seeds) < 3:
        continue
    ss = {r: series(r) for r in seeds.values()}
    ss = {r: v for r, v in ss.items() if v}
    if len(ss) < 3:
        continue
    common = set.intersection(*(set(v) for v in ss.values()))
    if not common:
        continue
    last_u = max(common)
    vals = [ss[r][last_u] for r in ss]
    sd = statistics.stdev(vals)
    rows.append((fam, len(ss), last_u, statistics.mean(vals), sd,
                 max(vals) - min(vals)))
    print(f"  {fam:<22} n={len(ss)}  末轮 u{last_u:<3} "
          f"均值 {statistics.mean(vals):.4f}  SD {sd:.4f}  跨度 {max(vals)-min(vals):.4f}")

print()
if not rows:
    print("!! 没找到任何 ≥3 种子的家族 —— 上面的结论无从校准")
    raise SystemExit(1)

sds = [r[4] for r in rows]
print("=== SD 的分布 ===")
print(f"  家族数 {len(sds)}   SD 中位 {statistics.median(sds):.4f}   "
      f"最小 {min(sds):.4f}   最大 {max(sds):.4f}")

import math
sd_med = statistics.median(sds)
sd_max = max(sds)
for label, sd in (("中位", sd_med), ("最大", sd_max)):
    for rho in (0.0, 0.5):
        sd_diff = math.sqrt(2 * (1 - rho)) * sd
        for n in (3,):
            se = sd_diff / math.sqrt(n)
            t = 0.035 / se
            print(f"  SD={label}({sd:.4f})  ρ={rho}  n={n}: SE={se:.4f}  "
                  f"Δ=0.035 → t={t:.2f}  {'够' if t > 2 else '**不够**'}")
print()

# 需要多少种子
print("=== 若按最保守的 SD 算，要几个种子才能分辨 0.035（t>2）===")
sd_diff_max = math.sqrt(2) * sd_max
n_need = (2 * sd_diff_max / 0.035) ** 2
print(f"  SD_max={sd_max:.4f}, ρ=0 → SD(diff)={sd_diff_max:.4f}")
print(f"  n ≥ (2×{sd_diff_max:.4f}/0.035)² = {n_need:.1f}")
print()
print("=== 判读 ===")
print("  若中位 SD 在 0.01 量级 → 3 个种子够用，我的原结论成立。")
print("  若中位 SD 在 0.03 量级 → 3 个种子**不够**，需要更多种子或接受仅能")
print("  检出更大的效应。**在链条还在跑的时候就要决定**，不能等 7.8h 后。")
