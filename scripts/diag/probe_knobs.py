"""找**不改维度**的旋钮 —— 只有它们能用 BC 族（分辨率 0.008）测。

## 为什么这是关键约束

今晚确立的死结：改特征 ⟹ 改 `edge_dim`/`node_dim` ⟹ 变窄补零救不回 ⟹
必须从零训 ⟹ 高方差（SD 0.2）⟹ 可检测效应 0.17 ⟹ **测不出**。

而 **BC 族的可检测效应是 0.0081**（低 20 倍）。
⟹ 能测的只有**不改维度的旋钮** —— 改完还能用 BC 暖启动 ⟹ 进低方差族。

## 本脚本做什么

读全库 `resolved_config.yaml`，统计每个配置键的**取值个数**。
`取值个数 == 1` 且**不在维度链上**的键 = 从未测过、且可测的候选。

⚠ 「取值个数 == 1」只说明**本库没变过**，不等于"物理上不能变"
（记忆 `ablation-name-is-not-the-parameter`）。
"""
import json
import re
from collections import defaultdict
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")

# 维度链：改这些会改 edge_dim/node_dim ⟹ 必须从零训 ⟹ 落在高方差族
DIM_KEYS = re.compile(
    r"features\.|dims\.|node_dim|edge_dim|history")
# 纯训练/动作空间/kpi 旋钮：不改维度
IGNORE_PREFIX = re.compile(
    r"^(project|runtime|seed|scenario\.time|validation|train\.|reward\.|"
    r"action_resolver|requests\.deadline|qkp\.|rate_provider\.h5)")


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        out[prefix.rstrip(".")] = d
    return out


def main():
    seen = defaultdict(set)
    n_runs = 0
    for cfgp in OUT.glob("*/resolved_config.yaml"):
        try:
            import yaml
            c = yaml.safe_load(cfgp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(c, dict):
            continue
        n_runs += 1
        for k, v in flatten(c).items():
            try:
                seen[k].add(json.dumps(v, sort_keys=True))
            except TypeError:
                pass

    print(f"扫描 {n_runs} 条臂的 resolved_config，共 {len(seen)} 个键")
    print("=" * 92)
    print("取值个数 == 1 的键（= 从未被本库变过）")
    print("=" * 92)

    single = [(k, list(v)[0]) for k, v in seen.items() if len(v) == 1]
    single.sort()

    # 分两类
    on_dim = [(k, v) for k, v in single if DIM_KEYS.search(k)]
    off_dim = [(k, v) for k, v in single if not DIM_KEYS.search(k)]

    print(f"\n  【A】不在维度链上（**可测**：改完仍能带 BC 起点）—— {len(off_dim)} 个")
    print(f"  {'键':<46}{'当前值':>28}")
    print("  " + "-" * 76)
    for k, v in off_dim:
        vs = v if len(v) < 26 else v[:25] + "…"
        print(f"  {k:<46}{vs:>28}")

    print(f"\n  【B】在维度链上（改了就落高方差族，**难测**）—— {len(on_dim)} 个")
    for k, v in on_dim[:12]:
        print(f"    {k}")
    if len(on_dim) > 12:
        print(f"    … 共 {len(on_dim)} 个")


if __name__ == "__main__":
    main()
