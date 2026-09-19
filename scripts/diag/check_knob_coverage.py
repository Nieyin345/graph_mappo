#!/usr/bin/env python
"""actor_lr 的实测覆盖度 —— 诊断记录里我写了「53 个 0.0003、1 个 0.0001」，
这是一个可核的事实，核一遍。

顺带核 entropy_coef / gamma / clip_eps / target_kl 的覆盖度，
看还有哪些旋钮**从来没被扫过**。

用法（服务器上）：/opt/qkd/venv/bin/python /tmp/check_knob_coverage.py
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")

KNOBS = {
    "actor_lr": ("train", "optimizer", "actor_lr"),
    "critic_lr": ("train", "optimizer", "critic_lr"),
    "entropy_coef": ("train", "ppo", "entropy_coef"),
    "clip_eps": ("train", "ppo", "clip_eps"),
    "target_kl": ("train", "ppo", "target_kl"),
    "gamma": ("train", "gamma"),
    "gae_lambda": ("train", "gae_lambda"),
    "epochs": ("train", "ppo", "epochs"),
    "minibatch_size": ("train", "ppo", "minibatch_size"),
    "value_coef": ("train", "ppo", "value_coef"),
    "max_grad_norm": ("train", "ppo", "max_grad_norm"),
}


def dig(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def load_yaml(p: Path):
    try:
        import yaml
    except ImportError:
        return None
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


runs = sorted(d for d in OUT.iterdir() if d.is_dir())
have = [d for d in runs if (d / "resolved_config.yaml").exists()]

print("=" * 78)
print(f"旋钮覆盖度：{len(have)} / {len(runs)} 个 run 有 resolved_config.yaml")
print("=" * 78)

tables: dict[str, Counter] = {k: Counter() for k in KNOBS}
seen = 0
for d in have:
    cfg = load_yaml(d / "resolved_config.yaml")
    if not isinstance(cfg, dict):
        continue
    seen += 1
    for name, path in KNOBS.items():
        v = dig(cfg, path)
        if v is not None:
            tables[name][repr(v)] += 1

print(f"  （成功解析 {seen} 个）")
print()
for name in KNOBS:
    c = tables[name]
    if not c:
        print(f"  {name:<16} **无读数**")
        continue
    tot = sum(c.values())
    items = ", ".join(f"{v}×{n}" for v, n in c.most_common())
    flag = ""
    if len(c) == 1:
        flag = "   ⟸ ★ **从未被扫过**（全部同一个值）"
    print(f"  {name:<16} n={tot:<4} {items}{flag}")

print()
print("=" * 78)
print("判读")
print("=" * 78)
never = [k for k in KNOBS if len(tables[k]) == 1]
once = [k for k in KNOBS if len(tables[k]) == 2]
print(f"  从未被扫过（只有一个取值）：{never or '（无）'}")
print(f"  只扫过一次（两个取值）  ：{once or '（无）'}")
print()
print("  ⚠ 注意：`resolved_config.yaml` 是**生效值**，历史上有 load_checkpoint")
print("     bug 让 actor_lr 实际跑在 0.001 上 —— 所以「配置覆盖度」不等于")
print("    「真实覆盖度」。修 bug 前后的 run 不可混为一谈。")
print("=" * 78)
