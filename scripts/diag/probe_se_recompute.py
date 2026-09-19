# -*- coding: utf-8 -*-
"""重算配对/不配对 SE —— 因为文档里那两个数与彼此的比值**不自洽**。

### 起因

`probe_number_audit.py` 的阈值审计抓到一处：日志 5421-5424 行并列写着

    不配对 SE = SD/√15 = 0.0362
    配对 SE（逐种子相减） = 0.0063
    配对缩小 = 8.1×

但 `0.0362 / 0.0063 = 5.75`，**不是 8.1**。三者不可能同时成立。
这正是 [[thresholds-and-transcribed-numbers]] 警告的那类错误，
而且我把它当作 8.07× 一路带进了会话摘要。

### 做法：不信文档，从原始数据重算

数据在 `outputs/<run>/metrics.jsonl` 的 `eval_validation` 行里，含
`per_seed_success`（15 个逐种子成功率）。按 15 个请求种子对齐后：

  · 不配对 SE：把 8 个 run 的 15 个值各自**不配对**地用（SD/√15）
  · 配对 SE：**同一个请求种子**上，两个 run 逐种子**相减**，再对差值求 SE
    配对的收缩量 = SD(diff) / SD(individual)，因为配对消掉了"这个种子本身
    难不难"这一项共同方差。

用法（服务器上）：python3 /tmp/probe_se_recompute.py
"""
import json
import math
import statistics as st
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
# 用**同一基线、同类配置**的臂，配对才有意义（同配置不同训练种子）
RUNS = ["ent01_s42", "ent01_s43", "ent01_s44", "ent01_s45", "ent01_s46"]


def val_rows(run):
    p = OUT / run / "metrics.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = r.get("eval_validation")
        if ev and ev.get("per_seed_success"):
            out.append((r.get("update"), list(ev["per_seed_success"]), ev.get("seeds")))
    return out


data = {}
for run in RUNS:
    rows = val_rows(run)
    if rows:
        data[run] = rows
        print(f"{run}: {len(rows)} 个 eval 点, 每点 {len(rows[0][1])} 个种子")

if not data:
    raise SystemExit("没有 eval_validation 数据")

# ---- 不配对：所有 run 所有点的逐种子值，按各自 run 内 SD/√n
print("\n" + "=" * 78)
print("① 不配对 SE（每个 run 内 SD/√n，再看各 run 是否一致）")
print("=" * 78)
unpaired = []
for run, rows in data.items():
    for u, vals, _ in rows:
        n = len(vals)
        sd = st.stdev(vals) if n > 1 else 0.0
        unpaired.append(sd / math.sqrt(n))
print(f"  点数 {len(unpaired)}  均值 {st.mean(unpaired):.4f}  "
      f"范围 {min(unpaired):.4f}~{max(unpaired):.4f}")
print(f"  SD(per-seed) 均值 = {st.mean([st.stdev(v) for _, v, _ in [r for rows in data.values() for r in rows]]):.4f}")

# ---- 配对：同一个 (update, 种子序号) 上，两个 run 逐种子相减
print("\n" + "=" * 78)
print("② 配对 SE（逐种子相减后再求 SE）")
print("=" * 78)
pair_ses = []
pair_ratios = []
for run_a, run_b in (("ent01_s42", "ent01_s43"), ("ent01_s42", "ent01_s44"),
                     ("ent01_s43", "ent01_s44"), ("ent01_s43", "ent01_s45"),
                     ("ent01_s44", "ent01_s46")):
    if run_a not in data or run_b not in data:
        continue
    by_u_a = {u: v for u, v, _ in data[run_a]}
    by_u_b = {u: v for u, v, _ in data[run_b]}
    common = sorted(set(by_u_a) & set(by_u_b))
    for u in common:
        va, vb = by_u_a[u], by_u_b[u]
        if len(va) != len(vb):
            continue
        diffs = [x - y for x, y in zip(va, vb)]
        n = len(diffs)
        sd_diff = st.stdev(diffs)
        se_diff = sd_diff / math.sqrt(n)
        sd_a = st.stdev(va)
        pair_ses.append(se_diff)
        pair_ratios.append(sd_diff / sd_a)

print(f"  配对点 {len(pair_ses)} 个")
if pair_ses:
    print(f"  配对 SE  均值 {st.mean(pair_ses):.4f}  范围 {min(pair_ses):.4f}~{max(pair_ses):.4f}")
    print(f"  SD(diff)/SD(indiv) 均值 = {st.mean(pair_ratios):.4f}")
    print(f"  ⟹ 配对把 SE 缩小 **{1/st.mean(pair_ratios):.2f}x**（= 1/该比值）")

print("\n" + "=" * 78)
print("③ 与日志里的三个数对照")
print("=" * 78)
u_mean = st.mean(unpaired)
p_mean = st.mean(pair_ses) if pair_ses else float("nan")
print(f"  {'量':<28}{'日志':>12}{'现算':>12}{'':>6}")
print(f"  {'不配对 SE':<28}{0.0362:>12.4f}{u_mean:>12.4f}  {'ok' if abs(u_mean-0.0362)<0.004 else '!!'}")
print(f"  {'配对 SE':<28}{0.0063:>12.4f}{p_mean:>12.4f}  {'ok' if abs(p_mean-0.0063)<0.004 else '!!'}")
ratio = u_mean / p_mean if p_mean else float("nan")
print(f"  {'缩小倍数 = 不配对/配对':<28}{8.1:>12.2f}{ratio:>12.2f}  {'ok' if abs(ratio-8.1)<1.5 else '!!'}")
print(f"  {'缩小倍数 = 1/(SD比)':<28}{8.1:>12.2f}{1/st.mean(pair_ratios) if pair_ratios else float('nan'):>12.2f}")

print("\n" + "=" * 78)
print("判读：")
print(f"  0.0362 / 0.0063 = {0.0362/0.0063:.2f} —— 日志里的 8.1x 与这两个数")
print("  不能同时成立。上面现算给出哪个是对的。")
print("=" * 78)
