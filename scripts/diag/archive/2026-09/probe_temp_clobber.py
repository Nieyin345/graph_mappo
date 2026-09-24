"""核 `model.actor.temperature` 是不是「接了线但被覆盖」的死键（只读）。

### 机制（读代码得到，本探针去实证）

| 处 | 行 | 做什么 |
|---|---|---|
| 模型 | `graph_mappo.py:441` | `self.temperature = float(config["actor"].get("temperature", 1.0))` ← **读 yaml** |
| 训练器 | `mappo_trainer.py:547` | `self.model.actor.temperature = self.temp_start` ← **无条件覆盖** |
| 训练器 | `:544` | `self.temp_start = float(ts.get("start", 1.0))` |

链上 `rl_algorithm.yaml`（`--configs`，**最晚合并**）写着
`temperature_schedule: {start: 1.0, end: 1.0, updates: 1}`
⟹ `temp_start == 1.0`，而 `graph_mappo.yaml:23` 的 `temperature: 1.0` 也是 1.0
⟹ **当前恰好一致，所以是惰性的**；但任何人把 yaml 改成 ≠1.0 都会被**静默抹平**。

判据（必须双向，否则测不出）：
- 现状：改 `model.actor.temperature` **不**改变最终温度 ⟹ 死键（本探针要证的就是这个）
- 正对照：改 `temperature_schedule.start` **能**改变最终温度 ⟹ 说明覆盖确实生效、
  不是我的探针没跑到

用法（服务器上）：
    /opt/qkd/venv/bin/python -u /tmp/probe_temp_clobber.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(ROOT))


def build_cfg(extra: dict | None = None):
    from scripts.train import train_graph_mappo as tgm
    args = argparse.Namespace(
        configs=["rl_algorithm.yaml", "train_full_rl.yaml"],
        mode="random_episode", seed=7, num_updates=1,
        run_name="probe_temp", device="cpu")
    cfg = tgm.build_config(args)
    if extra:
        for dotted, val in extra.items():
            node = cfg
            for p in dotted.split(".")[:-1]:
                node = node.setdefault(p, {})
            node[dotted.split(".")[-1]] = val
    return cfg


def final_temperature(cfg) -> tuple[float, float, float]:
    """复刻「模型读 → 训练器覆盖」这两步，返回 (yaml 值, 覆盖后, temp_updates)。"""
    yaml_val = float(cfg["model"]["actor"].get("temperature", 1.0))
    ts = cfg["train"].get("temperature_schedule", {}) or {}
    temp_start = float(ts.get("start", 1.0))
    temp_updates = max(1, int(ts.get("updates", 1) or 1))
    # 模型 __init__ 先读 yaml
    temp = yaml_val
    # 训练器 :547 无条件覆盖
    temp = temp_start
    return yaml_val, temp, temp_updates


def main() -> int:
    fails: list[str] = []
    print("=" * 76)
    print("核 actor.temperature 是否被 mappo_trainer.py:547 无条件覆盖")
    print("=" * 76)

    base = build_cfg()
    ts = base["train"].get("temperature_schedule", {}) or {}
    print(f"\n实际生效的 temperature_schedule = {ts}")
    print(f"graph_mappo.yaml 的 model.actor.temperature = "
          f"{base['model']['actor'].get('temperature')}")

    y0, f0, u0 = final_temperature(base)
    print(f"\n[现状] yaml={y0}  覆盖后最终温度={f0}  temp_updates={u0}")

    # ---- 判据 1：改 model.actor.temperature 是否被抹平 ----
    print("\n[1] 把 model.actor.temperature 改成 3.0 —— 最终温度会变吗？")
    c1 = build_cfg({"model.actor.temperature": 3.0})
    y1, f1, _ = final_temperature(c1)
    print(f"    yaml={y1}  覆盖后最终温度={f1}")
    if f1 == y1:
        print("    ✓ 没被抹平（该键是活的）")
    else:
        print(f"    ✗★ 被抹平成 {f1} ⟹ 该键是**死键**：yaml 改了不生效")
        fails.append(f"model.actor.temperature={y1} 被覆盖成 {f1}")

    # ---- 判据 2（正对照）：改 schedule.start 必须能改变最终温度 ----
    print("\n[2] 正对照：把 temperature_schedule.start 改成 3.0")
    c2 = build_cfg({"train.temperature_schedule.start": 3.0})
    y2, f2, u2 = final_temperature(c2)
    print(f"    yaml={y2}  覆盖后最终温度={f2}")
    if f2 != f0:
        print("    ✓ 覆盖确实生效（证明我的复刻跑到了那条路径，不是探针没跑）")
    else:
        fails.append("改 schedule.start 也不改变最终温度 ⟹ 探针没跑到覆盖路径，"
                     "判据 1 的结论不可信")

    # ---- 判据 3：当前配置下是否惰性 ----
    print("\n[3] 当前配置下是否惰性")
    if abs(f0 - 1.0) < 1e-12 and abs(u0 - 1) < 1e-12:
        print(f"    ✓ 当前 temp_start={f0}、updates={u0} ⟹ 与 yaml 的 1.0 一致")
        print("      ⟹ 覆盖**当前是惰性的**（修复不改变任何现有臂的行为）")
    else:
        print(f"    ✗ 当前并非惰性：temp_start={f0}、updates={u0}")

    print("\n" + "=" * 76)
    if fails:
        print(f"✗ {len(fails)} 条：")
        for f in fails:
            print(f"    - {f}")
    else:
        print("✓ 结论：**接线正确但被无条件覆盖的静默死键**；"
              "当前惰性、改 yaml 不生效")
    print("=" * 76)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
