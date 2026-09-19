#!/usr/bin/env python
"""分析 cProfile 产物，按模块拆清楚"时间花在哪"。

动机：docs/deployment/server.md §5.2 那份 profile 是 chunk=512 时代测的，
结论（"环境侧 39% 摊在十来个函数、对 1978 条链路逐条走 Python"）已经被
2026-09-18 的 env 向量化改动推翻。那段注释自己就写着"优化前必须重新 profile"。

跑法（节点上）：/opt/qkd/venv/bin/python .tmp/analyze_profile.py /tmp/prof_full.out
"""
import pstats
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/prof_full.out"

st = pstats.Stats(PATH)
total_tt = sum(v[2] for v in st.stats.values())
total_ct = st.total_tt
print(f"总 self 时间 {total_tt:.1f}s（cProfile 自身的 total_tt = {total_ct:.1f}s）\n")

# ---------------------------------------------------------------- 按模块聚合
BUCKETS = [
    ("env（环境/仿真）", r"qkd_rl[/\\]env"),
    ("link（速率数据）", r"qkd_rl[/\\]link"),
    ("models（GNN/actor/critic）", r"qkd_rl[/\\]rl[/\\]models"),
    ("algos（PPO/rollout/GAE）", r"qkd_rl[/\\]rl[/\\]algos"),
    ("core/data（配置/数据）", r"qkd_rl[/\\](core|data)"),
    ("torch", r"torch[/\\]"),
    ("numpy/scipy", r"(numpy|scipy)[/\\]"),
    ("multiprocessing", r"multiprocessing[/\\]"),
]

print("=" * 78)
print("按模块聚合")
print("=" * 78)
seen = set()
rows = []
for name, pat in BUCKETS:
    buf = []
    for (fn, _ln, fname), _v in st.stats.items():
        if fn == "~":
            continue
        key = (fn, fname)
        if key in seen:
            continue
        if __import__("re").search(pat, fn + "/" + fname):
            seen.add(key)
            buf.append(_v)
    tt = sum(b[2] for b in buf)
    nc = sum(b[1] for b in buf)
    rows.append((name, tt, nc))
for name, tt, nc in sorted(rows, key=lambda r: -r[1]):
    pct = tt / total_tt * 100 if total_tt else 0.0
    bar = "#" * int(pct / 2)
    print(f"  {name:<28}{tt:9.1f}s {pct:5.1f}%  n={nc:<9} {bar}")

rest = total_tt - sum(r[1] for r in rows)
print(f"  {'其余（解释器/IO/其它）':<28}{rest:9.1f}s {rest / total_tt * 100:5.1f}%")

# ---------------------------------------------------------------- 分模块 top
for label, pat in (("ENV", r"qkd_rl[/\\]env"), ("RL/ALGOS+MODELS", r"qkd_rl[/\\]rl")):
    print()
    print("=" * 78)
    print(f"{label} 内部 top 20（按 tottime）")
    print("=" * 78)
    st.sort_stats("tottime").print_stats(pat, 20)


# ---------------------------------------------------------------- cumtime top
print()
print("=" * 78)
print("全局 top 15（按 cumulative，看调用树根部）")
print("=" * 78)
st.sort_stats("cumulative").print_stats(15)
