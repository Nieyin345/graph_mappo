#!/usr/bin/env python
"""对比 r6_base 首轮/中轮/末轮的 rollout 统计，看诊断量是否在动。"""
import json
import sys
from pathlib import Path

run = sys.argv[1] if len(sys.argv) > 1 else "r6_base"
p = Path("/opt/qkd/graph_mappo/outputs") / run / "rollout_debug.jsonl"
rows = [json.loads(l) for l in p.open(encoding="utf-8")]
idxs = [0, len(rows) // 2, len(rows) - 1]
hdr = ["u%d" % rows[i]["update"] for i in idxs]

KEYS = [
    "mean_activated_edges", "mean_arrived_keys", "mean_served_keys",
    "mean_failed_keys", "mean_waiting_keys", "mean_generated_keys",
    "mean_qkp_utilization", "mean_success_rate",
    "mean_reward", "mean_reward_served", "mean_reward_failed",
    "mean_reward_storage", "mean_reward_keep_active", "mean_reward_expired",
    "mean_abs_advantage", "mean_return", "kl", "entropy",
    "actor_grad_norm", "critic_loss", "actor_loss",
]


def fmt(v):
    if v is None:
        return "NA"
    if abs(v) >= 10000:
        return "{:,.0f}".format(v)
    if abs(v) >= 1:
        return "{:.4f}".format(v)
    return "{:.3e}".format(v)


print("运行 {}  共 {} 轮".format(run, len(rows)))
print("{:<24}{:>14}{:>14}{:>14}".format("指标", *hdr))
for k in KEYS:
    print("{:<24}{:>14}{:>14}{:>14}".format(
        k, *[fmt(rows[i].get(k)) for i in idxs]))

# 成功率与失败率的分解
print()
print("=== 服务分解（末轮）===")
r = rows[-1]
arr, srv, fail, wait = (r["mean_arrived_keys"], r["mean_served_keys"],
                        r["mean_failed_keys"], r["mean_waiting_keys"])
print("  arrived  = {:,.0f}".format(arr))
print("  served   = {:,.0f}  ({:.1%} of arrived)".format(srv, srv / arr))
print("  failed   = {:,.0f}  ({:.1%} of arrived)".format(fail, fail / arr))
print("  waiting  = {:,.0f}  (积压, {:.1f}x arrived)".format(wait, wait / arr))
print("  served+failed = {:,.0f}  ({:.1%})".format(
    srv + fail, (srv + fail) / arr))
print("  generated= {:,.0f}  (是 arrived 的 {:.0f} 倍)".format(
    r["mean_generated_keys"], r["mean_generated_keys"] / arr))
