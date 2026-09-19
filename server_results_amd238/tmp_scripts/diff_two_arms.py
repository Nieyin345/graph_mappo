#!/usr/bin/env python
"""两个同配置臂（g999 / g999_long）逐轮对照 —— 排查它们为何数值不同。

理论上二者同配置同种子，只有 --num-updates 不同，应当逐位一致。
实测第 20 轮 eval 却是 0.6321 vs 0.6691。找出分歧从哪一轮开始。
"""
import json
import pathlib

MAIN = pathlib.Path("/opt/qkd/graph_mappo")


def load(run):
    d = {}
    p = MAIN / "outputs" / run / "metrics.jsonl"
    if not p.exists():
        return d
    for line in p.open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(j, dict) and "update" in j:
            d[j["update"]] = j
    return d


a = load("r7_fix_g999")        # 20 轮
b = load("r7_fix_g999_long")   # 60 轮
common = sorted(set(a) & set(b))
print(f"g999 有 {len(a)} 轮，g999_long 有 {len(b)} 轮，共同 {len(common)} 轮\n")

keys = ["mean_reward", "mean_served_keys", "mean_success_rate", "mean_abs_advantage",
        "actor_loss", "critic_loss", "rollout_s", "update_s"]
print(f"{'u':>3} " + "".join(f"{k[:14]:>17}" for k in keys))
print("-" * (4 + 17 * len(keys)))
first_diff = None
for u in common:
    cells = []
    same = True
    for k in keys:
        va, vb = a[u].get(k), b[u].get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if va != vb:
                same = False
            cells.append(f"{vb - va:>17.6g}")
        else:
            cells.append(f"{'--':>17}")
    if not same and first_diff is None:
        first_diff = u
    tag = "  <-- 首次分歧" if u == first_diff else ""
    print(f"{u:>3} " + "".join(cells) + tag)

print()
print(f"首次分歧在第 {first_diff} 轮" if first_diff else "全部逐位相同")
print()
print("=== 前 3 轮的完整关键项 ===")
for u in common[:3]:
    print(f"--- update {u} ---")
    for k in ("mean_reward", "mean_served_keys", "mean_success_rate", "rollout_s"):
        print(f"    {k:<20} a={a[u].get(k)!r:<24} b={b[u].get(k)!r}")
print("=== DONE ===")
