#!/usr/bin/env python
"""确认各 run 实际走的 rollout 路径，从而判断 obs_list 那个 bug 影响谁。"""
import yaml
from pathlib import Path

MAIN = Path("/opt/qkd/graph_mappo/outputs")
for d in sorted(MAIN.iterdir()):
    if not d.is_dir():
        continue
    p = d / "resolved_config.yaml"
    if not p.exists():
        continue
    c = yaml.safe_load(p.read_text(encoding="utf-8"))
    t = c.get("train", {})
    env = c.get("env", {})
    nw = t.get("n_rollout_workers")
    rb = t.get("rollout_batch")
    epu = t.get("episodes_per_update")
    cont = env.get("continuous")
    # 会走到 _collect_rollout_batched 的条件
    batched = (nw is not None and int(nw) <= 1 and int(epu or 1) > 1 and bool(rb))
    print(f"{d.name:<34} workers={nw:<3} rollout_batch={str(rb):<5} "
          f"episodes_per_update={epu:<3} continuous={str(cont):<5} "
          f"-> {'**batched(受影响)**' if batched else 'serial/worker(不受影响)'}")
