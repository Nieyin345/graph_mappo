#!/usr/bin/env python
"""比较 296-329 天与训练窗口 0-295 的数据特征，确认这 34 天可用。

若这 34 天的密钥率分布与训练窗口一致 → 直接扩大训练窗口是零成本改进。
若明显不同（例如率极低）→ 说明它被排除是有原因的。
"""
import sys
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

p = ROOT / "dataset/global/link_data.h5"
DAY = 1440

with h5py.File(p, "r") as f:
    k = f["k_max"]
    print(f"k_max shape={k.shape}\n")

    def stats(a, b, label):
        # 抽稀读取，避免全量载入（391MB 文件）
        idx = np.arange(a * DAY, b * DAY, 37)   # 每 37 步取一行
        sub = k[idx, :]
        nz = sub[sub > 0]
        return {
            "label": label,
            "rows": sub.shape[0],
            "mean": float(sub.mean()),
            "nonzero_frac": float((sub > 0).mean()),
            "nz_mean": float(nz.mean()) if nz.size else 0.0,
            "nz_p90": float(np.percentile(nz, 90)) if nz.size else 0.0,
            "col_active": float((sub > 0).any(axis=0).mean()),
        }

    cells = [
        (0, 60, "训练 0-59"),
        (60, 150, "训练 60-149"),
        (150, 240, "训练 150-239"),
        (240, 296, "训练 240-295"),
        (296, 330, "**闲置 296-329**"),
        (330, 365, "**验证 330-365**"),
    ]
    print(f"{'区间':<20}{'均值':>12}{'非零占比':>10}{'非零均值':>12}{'非零P90':>12}{'活跃列':>9}")
    for a, b, lab in cells:
        s = stats(a, b, lab)
        print(f"{s['label']:<20}{s['mean']:>12.1f}{s['nonzero_frac']:>10.3f}"
              f"{s['nz_mean']:>12.1f}{s['nz_p90']:>12.1f}{s['col_active']:>9.3f}")
