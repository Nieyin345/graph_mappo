# -*- coding: utf-8 -*-
"""奖励是不是**就是**指标？—— 若是，"目标错位"这个假说在本项目里结构上不成立。

### 为什么问这个

`docs/训练诊断记录.md` 里有两个并列的观察：
  · reward 与 success_rate 都**平**（reward +0.7%，success_rate +0.8%）
  · 但生成量掉 31%（奖励零空间）
一个自然的追问是"策略是不是在优化一个已与指标脱钩的代理"。
但若 reward **本身就是** served 的仿射函数，这个假说连提出都不该提 ——
它们不是"相关"，是**同一个量**。

### 生效的奖励（resolved_config.yaml 实测，不是 yaml 源码）

    mode: shaped, success_delta_enabled: false, reward_scale: 0.002
    served_enabled: true,  served_weight: 50.0, served_reference: 100000.0
    raw_generation_enabled: true,  generated_weight: 0.0      ← 0
    dense_enabled: false                                       ← 关
    waiting_enabled: false, switch_enabled: false,
    conflict_enabled: false, attribution_enabled: false
    storage_enabled: true (0.5/1e6), keep_active_enabled: true (0.001)
    expired_key_enabled: true (0.01/1e6), failed_enabled: true (5.0)

奖励里**没有任何一项给"生成密钥"定价**。supply 侧唯一的通道是 served
（要服务必须先用密钥）与 storage/keep_active（实测各占 0.1%）。

### 判据

`reward ≈ a · served_keys + b`。用**逐轮**数据回归（跨 run、跨轮次），
看 R²。R²≈1 ⟹ reward 与 served 是同一个量的两种写法。

同时给出 success_rate 与 served 的关系（success_rate = served/arrived，
故它是否也是仿射取决于 arrived 稳不稳）。

用法（服务器上）：python3 /tmp/probe_reward_identity.py
"""
import json
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")

RUNS = [
    "ent01_s42", "ent01_s43", "ent01_s44",
    "ent01_s45", "ent01_s46",
    "safe_ep2e1_s42", "safe_ep2e1_s43", "safe_ep2e1_s44",
    "safe_mini512e1_s42", "safe_mini512e1_s43", "safe_mini512e1_s44",
    "safe_vcoef1_s42", "safe_vcoef1_s43", "safe_vcoef1_s44",
]


def ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx if sxx else 0.0
    b = my - a * mx
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return a, b, r2


rows = []
for run in RUNS:
    p = OUT / run / "rollout_debug.jsonl"
    if not p.exists():
        continue
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({
            "run": run,
            "u": r["update"],
            "reward": r["mean_reward"],
            "served": r["mean_reward_served"],
            "failed": r["mean_reward_failed"],
            "served_keys": r["mean_served_keys"],
            "arrived_keys": r["mean_arrived_keys"],
            "gen": r["mean_generated_keys"],
            "sr": r["mean_success_rate"],
        })

print("=" * 74)
print(f"样本：{len(rows)} 个（run, update）点，来自 {len({r['run'] for r in rows})} 个 run")
print("=" * 74)

# 1) reward vs reward_served：分解项能否解释总值
a, b, r2 = ols([r["served"] for r in rows], [r["reward"] for r in rows])
print(f"\n① mean_reward  ~  mean_reward_served")
print(f"   slope={a:.6f}  intercept={b:.6f}  R²={r2:.6f}")

# 2) reward_served vs served_keys：served 项是否就是 served_keys 的线性函数
a, b, r2 = ols([r["served_keys"] for r in rows], [r["served"] for r in rows])
print(f"\n② mean_reward_served  ~  mean_served_keys")
print(f"   slope={a:.3e}  intercept={b:.6f}  R²={r2:.6f}")
print(f"   （配置预测 slope = served_weight/served_reference = "
      f"50/100000 = {50/100000:.3e}，乘以 reward_scale 0.002 = {50/100000*0.002:.3e}）")

# 3) success_rate vs served_keys：指标是不是也是同一个量
a, b, r2 = ols([r["served_keys"] for r in rows], [r["sr"] for r in rows])
print(f"\n③ mean_success_rate  ~  mean_served_keys")
print(f"   slope={a:.3e}  intercept={b:.6f}  R²={r2:.6f}")

# 4) 决定性的一条：reward vs success_rate 直接回归
a, b, r2 = ols([r["sr"] for r in rows], [r["reward"] for r in rows])
print(f"\n④ ★ mean_reward  ~  mean_success_rate  （两者是不是同一个量）")
print(f"   slope={a:.6f}  intercept={b:.6f}  R²={r2:.6f}")
print(f"   相关系数 r = {r2 ** 0.5:.6f}")

# 5) arrived 稳不稳（决定 success_rate 是否也是仿射）
arr = [r["arrived_keys"] for r in rows]
mv = sum(arr) / len(arr)
sd = (sum((x - mv) ** 2 for x in arr) / len(arr)) ** 0.5
print(f"\n⑤ mean_arrived_keys：均值 {mv:,.1f}  SD {sd:,.1f}  变异系数 {sd/mv*100:.2f}%")
print("   （arrived 是外生需求，近似常数 ⟹ success_rate 也是 served 的仿射）")

# 6) 零空间：reward 对生成量的敏感度
a, b, r2 = ols([r["gen"] for r in rows], [r["reward"] for r in rows])
print(f"\n⑥ ★ mean_reward  ~  mean_generated_keys  （奖励对生成量敏感吗）")
print(f"   slope={a:.3e}  R²={r2:.6f}")
print(f"   生成量跨 {min(r['gen'] for r in rows)/1e6:.2f}M ~ "
      f"{max(r['gen'] for r in rows)/1e6:.2f}M（{max(r['gen'] for r in rows)/min(r['gen'] for r in rows):.2f}x）")
print(f"   而 reward 跨 {min(r['reward'] for r in rows):.4f} ~ {max(r['reward'] for r in rows):.4f}")

print("\n" + "=" * 74)
if r2 > 0.9:
    print("判读：reward 与 success_rate 高度共线 ⟹ **奖励就是指标**，")
    print("      '优化了代理而指标不动'在本配置下结构上不可能。")
print("=" * 74)
