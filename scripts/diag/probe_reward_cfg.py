# -*- coding: utf-8 -*-
"""把**生效的**奖励配置打出来，而不是看 yaml 源码。

为什么要这一道：`env_full.yaml` 里 14 个 `*_enabled` 开关 + 十几个权重，
光看文件很容易把"写在文件里"当成"生效中"。实测 `rollout_debug` 显示
除 served/failed 外**所有分项恒为 0.0000** —— 需要确认这是配置使然
（开关关着/权重为 0），而不是某处静默失效。

用法（服务器上）：python3 /tmp/probe_reward_cfg.py
"""
import re
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
RUNS = ["ent01_s42", "ent01_s43", "ent01_s44", "ent01_s45", "ent01_s46"]

# rollout_debug 里恒为 0 的分项 ↔ 它们在配置里对应的开关/权重
PAIRS = [
    ("generated", "raw_generation_enabled", "generated_weight"),
    ("dense", "dense_enabled", "dense_generation_importance_weight"),
    ("waiting", "waiting_enabled", "waiting_weight"),
    ("storage", "storage_enabled", "storage_weight"),
    ("switch", "switch_enabled", "switch_weight"),
    ("keep_active", "keep_active_enabled", "keep_active_weight"),
    ("expired", "expired_key_enabled", "expired_key_weight"),
    ("conflict", "conflict_enabled", "conflict_weight"),
    ("failed", "failed_enabled", "failed_weight"),
    ("served", "served_enabled", "served_weight"),
]

for run in RUNS:
    p = OUT / run / "resolved_config.yaml"
    if not p.exists():
        print(f"{run}: 缺 resolved_config.yaml")
        continue
    txt = p.read_text(encoding="utf-8", errors="replace")
    # reward 段是 env_full.yaml 里那个 mode: shaped 块
    m = re.search(r"^  reward:\n(.*?)(?=^  [a-z_]+:\n)", txt, re.S | re.M)
    block = m.group(1) if m else txt

    def get(key):
        mm = re.search(rf"^\s*{key}:\s*(\S+)\s*$", block, re.M)
        return mm.group(1) if mm else "—"

    print("=" * 70)
    print(f"{run}   mode={get('mode')}  "
          f"success_delta_enabled={get('success_delta_enabled')}")
    print("-" * 70)
    print(f"  {'分项':<14}{'enabled':<10}{'weight':<12}{'rollout_debug 实测'}")
    for name, flag, wkey in PAIRS:
        en = get(flag)
        w = get(wkey)
        # rollout_debug 里的分项名
        live = {"served": "0.0666（唯一非零项）",
                "failed": "0.0011（唯一非零惩罚）"}.get(name, "0.0000")
        print(f"  {name:<14}{en:<10}{w:<12}{live}")
