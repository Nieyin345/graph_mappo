"""删掉 `actor.idle_scorer` 到底是不是"零风险"？—— 直接查 checkpoint 的键。

### 为什么查这个

`docs/训练诊断记录.md` 的动作清单里，第一行写着：

    删掉 `idle_scorer`  |  代码，**零风险**  |  死重；但要先确认没有别的调用方

前半句（是死重）已有两条独立证据（`probe_idle_dead.py` 逐位不动；
`padded_logits` 无人消费）。但**"零风险"是没验证过的断言**，
而它恰恰是最容易错的那半：

  · `mappo_trainer.py:1226` 的 `self.model.load_state_dict(data.model_state)`
    **没有 `strict=False`** ⟹ 默认 strict=True
  · 若 checkpoint 里存着 `actor.idle_scorer.*` 的键，删模块后它们就成了
    "unexpected keys" ⟹ **加载直接抛异常**
  · 而所有训练都从 BC 起点 `supervised_pg_phased_latest.pt` 起 ⟹
    删掉就等于**打断所有现有起点的加载**

本脚本只做一件事：把 checkpoint 里的键数出来，让"零风险"这句话有个依据。

用法（服务器上）：
  /opt/qkd/venv/bin/python /tmp/probe_idle_del_risk.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path("/opt/qkd/graph_mappo")
CKPTS = [
    "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt",
    "outputs/ent01_t8_s42/checkpoint_best_val.pt",
    "outputs/ent01_t8_s42/checkpoint_latest.pt",
]


def main() -> int:
    print("=" * 78)
    print("删掉 actor.idle_scorer 的风险：查 checkpoint 的键")
    print("=" * 78)
    any_hit = False
    for rel in CKPTS:
        p = ROOT / rel
        if not p.exists():
            print(f"  {rel}\n    ✗ 不存在，跳过")
            continue
        try:
            d = torch.load(str(p), map_location="cpu", weights_only=False)
        except Exception as e:                        # noqa: BLE001
            print(f"  {rel}\n    ✗ 读不了: {type(e).__name__}: {e}")
            continue
        sd = d.get("model_state", d)
        if not hasattr(sd, "keys"):
            print(f"  {rel}\n    ✗ 不是 state_dict（类型 {type(sd).__name__}）")
            continue
        keys = list(sd.keys())
        idle = [k for k in keys if "idle_scorer" in k]
        print(f"  {rel}")
        print(f"    总键数 {len(keys)}   idle_scorer 相关 **{len(idle)}** 个")
        for k in idle[:8]:
            try:
                print(f"      {k:<44} {tuple(sd[k].shape)}")
            except Exception:                          # noqa: BLE001
                print(f"      {k}")
        if len(idle) > 8:
            print(f"      … 还有 {len(idle) - 8} 个")
        any_hit = any_hit or bool(idle)
        print()

    print("=" * 78)
    print("判读")
    print("=" * 78)
    if any_hit:
        print("  ⟹ checkpoint 里**确实存着** `actor.idle_scorer.*` 的键。")
        print("     而 `mappo_trainer.py:1226` 的 load_state_dict **没有 strict=False**")
        print("     ⟹ 直接删模块会让加载抛 'Unexpected key(s)'，")
        print("        **打断所有从 BC 起点起的训练**。")
        print()
        print("  ⟹ 所以「零风险」**不成立**。正确处置是二选一：")
        print("     (a) 删模块 + 同时加一段**键过滤**的迁移代码（一次性、需测）")
        print("     (b) **不删** —— 它只是死重，不产生梯度，留着的代价是每次前向")
        print("         多算 16,641 参数的一层 MLP（省不到可测量的量）")
        print()
        print("  ★ 在「没有可测量的收益 + 有真实的破坏面」下，选 (b)。")
    else:
        print("  ⟹ checkpoint 里**没有** idle_scorer 的键 ⟹ 删模块不影响加载，")
        print("     「零风险」成立。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
