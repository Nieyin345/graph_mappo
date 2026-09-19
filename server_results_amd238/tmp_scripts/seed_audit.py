#!/usr/bin/env python
"""列出各 run 的训练种子与 entropy_coef，用于确认配对对照是否同种子。

用法：python /tmp/seed_audit.py r7_base r7_fix_ent r8_base_s42 r8_ent01_s42
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

R = Path("/opt/qkd/graph_mappo/outputs")


def main():
    names = sys.argv[1:] or ["r7_base", "r7_fix_ent"]
    print(f"{'run':<18}{'global_seed':>12}{'env_seed':>10}{'entropy':>10}"
          f"{'episodes':>10}{'workers':>9}")
    print("-" * 70)
    for n in names:
        p = R / n / "resolved_config.yaml"
        if not p.exists():
            print(f"{n:<18}  (无 resolved_config.yaml)")
            continue
        with p.open(encoding="utf-8") as fh:
            c = yaml.safe_load(fh) or {}
        s = c.get("seed") or {}
        tr = c.get("train") or {}
        ppo = tr.get("ppo") or {}
        print(f"{n:<18}{str(s.get('global_seed')):>12}"
              f"{str(s.get('env_seed')):>10}"
              f"{str(ppo.get('entropy_coef')):>10}"
              f"{str(tr.get('episodes_per_update')):>10}"
              f"{str(tr.get('n_rollout_workers')):>9}")


if __name__ == "__main__":
    main()
