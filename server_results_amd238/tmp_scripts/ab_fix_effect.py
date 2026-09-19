#!/usr/bin/env python
"""对照 r7_base 与 r6_base 的逐轮指标 —— 量出"终止语义修复"本身的效果。

依据：同机、同配置、同线程、同种子的训练是**逐位确定性**的
（docs/测试规范.md §6 与 .tmp/probe_thread_repro 的记录）。
所以 r6_base（修复前代码）与 r7_base（修复后代码）之间的差异，
就是这次修复单独造成的差异 —— 不需要任何统计推断。

跑法：/opt/qkd/venv/bin/python .tmp/ab_fix_effect.py
"""
import json
import pathlib

MAIN = pathlib.Path("/opt/qkd/graph_mappo")

KEYS = [
    "mean_return", "mean_abs_advantage", "critic_loss", "actor_loss",
    "kl", "entropy", "mean_ratio", "value_std", "return_std",
    "value_return_corr", "mean_reward", "mean_success_rate",
]


def load(run):
    p = MAIN / "outputs" / run / "metrics.jsonl"
    rows = {}
    if not p.exists():
        return rows
    for line in p.open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and "update" in d:
            rows[d["update"]] = d
    return rows


old = load("r6_base")      # 修复前
new = load("r7_base")      # 修复后
common = sorted(set(old) & set(new))
print(f"r6_base 有 {len(old)} 轮，r7_base 有 {len(new)} 轮，共同 {len(common)} 轮\n")

if not common:
    print("还没有可比的轮次，稍后再跑。")
    raise SystemExit(0)

print("=" * 92)
print("逐轮差异（r7_base - r6_base）。零 = 修复对该指标无影响")
print("=" * 92)
hdr = f"{'update':>7}" + "".join(f"{k[:13]:>15}" for k in KEYS[:5])
print(hdr)
print("-" * 92)
for u in common:
    a, b = old[u], new[u]
    cells = []
    for k in KEYS[:5]:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            cells.append(f"{vb - va:>15.6g}")
        else:
            cells.append(f"{'--':>15}")
    print(f"{u:>7}" + "".join(cells))

print()
print("=" * 92)
print("均值对照（共同轮次），看修复有没有改变量级")
print("=" * 92)
print(f"  {'指标':<22}{'r6_base(旧)':>16}{'r7_base(新)':>16}{'差异':>14}{'相对':>10}")
for k in KEYS:
    va = [old[u].get(k) for u in common if isinstance(old[u].get(k), (int, float))]
    vb = [new[u].get(k) for u in common if isinstance(new[u].get(k), (int, float))]
    if not va or not vb:
        continue
    ma, mb = sum(va) / len(va), sum(vb) / len(vb)
    rel = (mb - ma) / abs(ma) * 100 if ma else float("nan")
    flag = "  <<<" if abs(rel) > 5 else ""
    print(f"  {k:<22}{ma:>16.6g}{mb:>16.6g}{mb - ma:>14.6g}{rel:>9.1f}%{flag}")

print()
print("=" * 92)
print("逐位相同性检查（同机同线程同种子应当逐位一致，除修复影响外）")
print("=" * 92)
for k in KEYS:
    same = all(
        old[u].get(k) == new[u].get(k) for u in common
    )
    print(f"  {k:<22}{'逐位相同' if same else '有差异'}")
print("=== AB_FIX_DONE ===")
