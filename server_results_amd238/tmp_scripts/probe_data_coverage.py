#!/usr/bin/env python
"""查 H5 数据集的时间覆盖范围，以及请求种子在各天的分布。

动机：训练窗口只用 day 0-295，验证用 330-365，中间 296-329 从未被使用。
若数据覆盖到 365 天，训练窗口有空隙可补；若只到 330，那验证窗口就是硬边界。
"""
import sys
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

p = ROOT / "dataset/global/link_data.h5"
print(f"文件 {p}  大小 {p.stat().st_size/1e6:.1f} MB\n")

with h5py.File(p, "r") as f:
    print("=== 顶层结构 ===")
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name:<40} shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"  {name}/")
    f.visititems(show)

    # 找时间维度
    print("\n=== 时间轴探测 ===")
    for key in ("time", "timestamps", "t", "times", "seconds"):
        if key in f:
            arr = f[key][:]
            print(f"  f['{key}'] shape={arr.shape}")
            if arr.size:
                print(f"    min={arr.min()} max={arr.max()}")
                if arr.size > 1:
                    dt = np.diff(np.asarray(arr).ravel()[:1000])
                    print(f"    步长中位数={np.median(dt)}")
                    span = float(arr.max()) - float(arr.min())
                    print(f"    跨度={span}  折合天数(1440步/天)={span/1440:.1f}")
            break
    else:
        # 从某个 dataset 的 shape 推断
        for name in f:
            obj = f[name]
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                print(f"  用 f['{name}'] shape={obj.shape} 推断时间维度")
                for ax, sz in enumerate(obj.shape):
                    print(f"    axis{ax}: {sz}  (若为时间轴则 {sz/1440:.1f} 天)")
                break
