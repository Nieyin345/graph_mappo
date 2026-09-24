"""核对每条 v2 臂**运行时**的 resolved_config.yaml。

★ 为什么不能只看源码：`whitelist-from-docs-not-from-keys` —— 我按源码/文档
判"接上了"，但真正生效的是 `build_config()` 的产物。这里读产物。

对每条臂核对四件：
  1. `features.edge.include_on_pending_path` == True（开关真开了）
  2. `features.dims.edge_dim_resolved` == 45（维度真是 45，说明这一列进了观测）
  3. `train.ppo.entropy_coef` == 0.01（对照组同位；与 ent01_rerun 同口径）
  4. 训练窗口仍是训练 regime（不是验证 regime）
"""
import sys
from pathlib import Path

import yaml

OUT = Path("/opt/qkd/graph_mappo/outputs")
ARMS = [f"v2_bottleneck_s{s}" for s in (42, 43, 44, 45, 46)]

print(f"{'arm':<20} {'on_path':<8} {'edge_dim':<9} {'ent':<7} {'ep_steps':<9} {'win':<12} 判定")
print("-" * 88)
bad = []
for arm in ARMS:
    f = OUT / arm / "resolved_config.yaml"
    if not f.exists():
        print(f"{arm:<20} (还没有 resolved_config.yaml)")
        bad.append(arm)
        continue
    c = yaml.safe_load(f.read_text())
    e = c["features"]["edge"]
    onp = e.get("include_on_pending_path")
    dim = c["features"]["dims"].get("edge_dim_resolved")
    ent = c["train"]["ppo"].get("entropy_coef")
    eps = c["env"].get("episode_steps")
    win = (f"{c['env'].get('activation_window_start_day')}"
           f"-{c['env'].get('activation_window_end_day')}")
    ok = (onp is True) and (dim == 45) and (abs(float(ent) - 0.01) < 1e-12)
    print(f"{arm:<20} {str(onp):<8} {str(dim):<9} {str(ent):<7} {str(eps):<9} {win:<12} "
          f"{'✓' if ok else '★ 不对'}")
    if not ok:
        bad.append(arm)

print()
if bad:
    print(f"DECISION=CONFIG_MISMATCH  {bad}")
    sys.exit(2)
print("DECISION=CONFIG_OK  5 条臂的开关/维度/熵系数都是预期的")
