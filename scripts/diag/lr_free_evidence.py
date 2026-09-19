# -*- coding: utf-8 -*-
"""免费的 lr 证据：r6_base (0.0003) vs r6_base_lr1e4 (0.0001) —— 只差一个字段。

为什么现在特别值得看：实测 KL 长期停在 0.0005~0.003，而 clip_eps=0.1
对应 ratio=1±0.1，即 KL 要到 0.01 量级 clip 才开始起作用。也就是说
**PPO 跑在远离工作点的位置**（clip 从未激活），那么步长（lr）就是
唯一在起作用的东西 —— 它的方向因此格外重要。

局限（必须写在结论里）：这是**单种子**比较，按 docs/测试规范.md §4⑦
分辨率只有 ~0.035。所以它能给的是**方向**，不是判决。要判决必须跑种子。
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
EXPERT = OUT / "eval" / "expert_seeds100_240.json"


def load_val(name):
    p = OUT / name / "metrics.jsonl"
    if not p.exists():
        return None, None
    out, seed = {}, None
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
            out[last] = [float(x) for x in ev["per_seed_success"]]
    # 从 resolved_config 拿种子
    rc = OUT / name / "resolved_config.yaml"
    if rc.exists():
        for ln in rc.read_text(encoding="utf-8", errors="replace").splitlines():
            if "global_seed" in ln:
                seed = ln.split(":")[-1].strip()
                break
    return out, seed


d = json.loads(EXPERT.read_text(encoding="utf-8"))
ex = dict(zip(d["seeds"], (float(x) for x in d["success"])))
base = [ex[s] for s in sorted(ex)]

A, sa = load_val("r6_base")
B, sb = load_val("r6_base_lr1e4")
print(f"r6_base        actor_lr=0.0003  seed={sa}  验证点 {len(A) if A else 0} 个")
print(f"r6_base_lr1e4  actor_lr=0.0001  seed={sb}  验证点 {len(B) if B else 0} 个")
if not A or not B:
    print("数据缺失，退出")
    raise SystemExit

common = sorted(set(A) & set(B))
print(f"共同验证轮: {common}")
if sa != sb:
    print(f"  ⚠ 两个 run 的种子不同（{sa} vs {sb}）—— **不是配对比较**，只能看粗均值")
    print(f"    这会把种子间的 0.10 跨度混进来，方向都可能被掩盖。\n")

print()
print("=" * 84)
print("1. 逐轮对照（均值 = 15 个验证种子的成功率）")
print("=" * 84)
print(f"  {'轮':>4}{'lr=3e-4':>12}{'lr=1e-4':>12}{'差(A-B)':>12}{'对专家A':>12}{'对专家B':>12}")
diffs = []
for u in common:
    a, b = st.mean(A[u]), st.mean(B[u])
    ea = st.mean([A[u][i] - base[i] for i in range(15)])
    eb = st.mean([B[u][i] - base[i] for i in range(15)])
    diffs.append(a - b)
    print(f"  {u:>4}{a:>12.4f}{b:>12.4f}{a-b:>+12.4f}{ea:>+12.4f}{eb:>+12.4f}")

print()
print("=" * 84)
print("2. 平台读数（末段 5 个点，避开单点噪声）")
print("=" * 84)
tail_u = common[-5:]
a_t = st.mean([st.mean(A[u]) for u in tail_u])
b_t = st.mean([st.mean(B[u]) for u in tail_u])
print(f"  末 5 点均值：lr=3e-4 → {a_t:.4f}   lr=1e-4 → {b_t:.4f}   "
      f"差 {a_t-b_t:+.4f}")
print(f"  逐轮差的均值 {st.mean(diffs):+.4f}  SD {st.stdev(diffs):.4f}")
print()
print(f"  分辨率提醒：单种子 ~0.035。|{a_t-b_t:+.4f}| "
      f"{'低于' if abs(a_t-b_t) < 0.035 else '高于'}分辨率 —— "
      f"{'**不能据此下结论**' if abs(a_t-b_t) < 0.035 else '值得跟种子复筛'}。")
