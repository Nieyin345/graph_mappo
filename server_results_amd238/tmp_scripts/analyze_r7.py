#!/usr/bin/env python
"""第七浪全面分析：修复的效果、g999 是否还在、H1 签名、以及计时。

四个问题：
  Q1 终止语义修复本身有没有用？   r7_base vs r6_base（同配置，只差代码）
  Q2 g999 的收益在修复后还在吗？  r7_fix_g999 vs r7_base，并与 r6_g999 对照
  Q3 H1（优势塌陷）被治好了吗？   前后 1/3 对比
  Q4 build_scores 修复省了多少？  rollout_s 对比（r6 vs r7，需同条件）

跑法：/opt/qkd/venv/bin/python .tmp/analyze_r7.py
"""
import json
import math
import pathlib
import statistics as st

MAIN = pathlib.Path("/opt/qkd/graph_mappo")


def load(run):
    p = MAIN / "outputs" / run / "metrics.jsonl"
    tr, ev = [], []
    if not p.exists():
        return tr, ev
    for line in p.open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict):
            continue
        if "update" in d:
            tr.append(d)
        elif "eval_validation" in d:
            ev.append(d["eval_validation"])
    return tr, ev


def paired(a, b):
    n = min(len(a), len(b))
    if n < 2:
        return float("nan"), float("nan"), 0
    diffs = [a[i] - b[i] for i in range(n)]
    md = sum(diffs) / n
    var = sum((d - md) ** 2 for d in diffs) / (n - 1)
    return md, (var / n) ** 0.5, n


def tt(md, se):
    return md / se if se else float("nan")


def sig(t):
    return "显著" if isinstance(t, float) and abs(t) >= 2 else "不显著"


R7 = ["r7_base", "r7_fix_g999", "r7_fix_ent", "r7_fix_g999_long"]
R6 = ["r6_base", "r6_g999", "r6_lowv025", "r6_stor10", "r6_base_lr1e4"]
data = {r: load(r) for r in R7 + R6}

print("=" * 96)
print("eval 序列（验证集 success_rate，每 5 轮一个读数）")
print("=" * 96)
for r in R7 + R6:
    tr, ev = data[r]
    if not ev:
        print(f"{r:<22} (无验证记录)")
        continue
    seq = " ".join(f"{e.get('mean_success_rate', float('nan')):.4f}" for e in ev)
    d = ev[-1].get("mean_success_rate", float("nan")) - ev[0].get("mean_success_rate", float("nan"))
    print(f"{r:<22} n={len(ev):<2} {seq}   Δ={d:+.4f}")

print()
print("=" * 96)
print("Q1  终止语义修复本身有没有用？ r7_base vs r6_base（同配置，只差代码版本）")
print("=" * 96)
a, b = data["r7_base"][1], data["r6_base"][1]
if a and b:
    k = min(len(a), len(b))
    print(f"  共同 eval 次数 {k}")
    for i in range(k):
        pa, pb = a[i]["per_seed_success"], b[i]["per_seed_success"]
        md, se, n = paired(pa, pb)
        t = tt(md, se)
        print(f"    @update{5 * (i + 1):<4} {md:+.4f} ± {se:.4f}  t={t:+.2f}  {sig(t)}")
    tail_a = st.mean(e["mean_success_rate"] for e in a[-2:])
    tail_b = st.mean(e["mean_success_rate"] for e in b[-2:])
    print(f"  末两次均值：修复后 {tail_a:.4f}  修复前 {tail_b:.4f}  差 {tail_a - tail_b:+.4f}")
    print(f"  首次数值：  修复后 {a[0]['mean_success_rate']:.4f}  修复前 {b[0]['mean_success_rate']:.4f}")

print()
print("=" * 96)
print("Q2  g999 的收益在修复后还在吗？")
print("=" * 96)
ctrl = data["r7_base"][1]
for r in ["r7_fix_g999", "r7_fix_ent"]:
    ev = data[r][1]
    if not ev or not ctrl:
        continue
    k = min(len(ev), len(ctrl))
    md, se, n = paired(ev[k - 1]["per_seed_success"], ctrl[k - 1]["per_seed_success"])
    t = tt(md, se)
    print(f"  {r:<18} vs r7_base @update{5 * k:<4} {md:+.4f} ± {se:.4f}  t={t:+.2f}  {sig(t)}")
print("  ---- 对照：第六浪（修复前）----")
c6 = data["r6_base"][1]
for r, lab in [("r6_g999", "g999"), ("r6_lowv025", "lowv025"), ("r6_stor10", "stor10")]:
    ev = data[r][1]
    if not ev or not c6:
        continue
    k = min(len(ev), len(c6))
    md, se, n = paired(ev[k - 1]["per_seed_success"], c6[k - 1]["per_seed_success"])
    t = tt(md, se)
    print(f"  r6_{lab:<16} vs r6_base @update{5 * k:<4} {md:+.4f} ± {se:.4f}  t={t:+.2f}  {sig(t)}")

print()
print("=" * 96)
print("Q2b g999 长臂（60 轮）还在涨吗？")
print("=" * 96)
tr, ev = data["r7_fix_g999_long"]
if ev:
    seq = " ".join(f"{e.get('mean_success_rate', float('nan')):.4f}" for e in ev)
    print(f"  序列（每 5 轮）：{seq}")
    half = len(ev) // 2
    if half >= 2:
        h1 = st.mean(e["mean_success_rate"] for e in ev[:half])
        h2 = st.mean(e["mean_success_rate"] for e in ev[half:])
        print(f"  前半 {h1:.4f} -> 后半 {h2:.4f}  ({h2 - h1:+.4f})")
        k = min(len(ev), len(ctrl)) if ctrl else 0
        if k:
            md, se, n = paired(ev[k - 1]["per_seed_success"], ctrl[k - 1]["per_seed_success"])
            t = tt(md, se)
            print(f"  vs r7_base @update{5 * k}  {md:+.4f} ± {se:.4f}  t={t:+.2f}  {sig(t)}")

print()
print("=" * 96)
print("Q3  H1（优势塌陷）：前 1/3 vs 后 1/3")
print("=" * 96)
print(f"  {'run':<22}{'轮':>4}{'|A|前':>9}{'|A|后':>9}{'收缩':>7}{'corr后':>8}{'kl后':>9}{'ent后':>8}{'grad后':>9}")
for r in R7 + ["r6_base", "r6_g999"]:
    tr = data[r][0]
    if len(tr) < 6:
        continue
    k = max(1, len(tr) // 3)

    def m(key, lo, hi):
        xs = [x.get(key) for x in tr[lo:hi]]
        xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
        return st.mean(xs) if xs else float("nan")

    a1 = m("mean_abs_advantage", 0, k)
    a2 = m("mean_abs_advantage", len(tr) - k, len(tr))
    shr = a2 / a1 if a1 else float("nan")
    print(f"  {r:<22}{len(tr):>4}{a1:>9.3f}{a2:>9.3f}{shr:>7.2f}"
          f"{m('value_return_corr', len(tr) - k, len(tr)):>8.3f}"
          f"{m('kl', len(tr) - k, len(tr)):>9.4f}"
          f"{m('entropy', len(tr) - k, len(tr)):>8.3f}"
          f"{m('actor_grad_norm', len(tr) - k, len(tr)):>9.3f}")

print()
print("=" * 96)
print("Q4  计时：build_scores 修复省了多少？（丢前 3 轮，取稳态，同线程=2）")
print("=" * 96)
print(f"  {'run':<22}{'rollout_s':>11}{'update_s':>11}{'一轮':>9}{'轮数':>6}")
for r in R7 + R6:
    tr = data[r][0]
    steady = tr[3:] if len(tr) > 3 else tr
    if not steady:
        continue
    ro = st.mean(x["rollout_s"] for x in steady if isinstance(x.get("rollout_s"), (int, float)))
    up = st.mean(x["update_s"] for x in steady if isinstance(x.get("update_s"), (int, float)))
    print(f"  {r:<22}{ro:>11.1f}{up:>11.1f}{ro + up:>9.1f}{len(tr):>6}")

print()
print("=== ANALYZE_R7_DONE ===")
