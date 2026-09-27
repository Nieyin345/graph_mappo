"""接触计划前瞻特征（2026-09-27）。

卫星/HAP 链路何时可见由轨道决定，是确定、可预知的（接触计划）。失败解剖（实验日志 §11.12）
显示最大的可控损失是"没提前囤货"，需要 30–120 步前瞻，而原特征只有未来 6 步。
本模块把每条边未来 H 步的可见性汇总成 buckets+2 维：

  [0:buckets]   每个等长时间桶内可见的比例
  [buckets]     当前接触还会持续多久：从 t+1 起连续可见的步数 / H
  [buckets+1]   这段接触结束后要等多久才再次可见 / H
                （H 内不再出现 = 1.0；整段 H 都可见 = 0.0）
"""
from __future__ import annotations

import numpy as np


def contact_lookahead_features(avail: np.ndarray, buckets: int) -> np.ndarray:
    """avail: (H, n) 布尔，行 = 时隙 t+1..t+H → (n, buckets+2) float32。"""
    a = np.asarray(avail, dtype=bool)
    if a.ndim != 2:
        raise ValueError(f"avail 必须是 (H, n)，得到 {a.shape}")
    H, n = a.shape
    if buckets <= 0 or H % buckets:
        raise ValueError(f"steps={H} 必须能被 buckets={buckets} 整除")
    frac = a.reshape(buckets, H // buckets, n).mean(axis=1).T
    if n == 0:
        return np.zeros((0, buckets + 2), dtype=np.float32)
    off = ~a
    first_off = np.where(off.any(axis=0), off.argmax(axis=0), H)
    after = a & (np.arange(H)[:, None] >= first_off[None, :])
    has_next = after.any(axis=0)
    next_on = np.where(has_next, after.argmax(axis=0), H)
    gap = np.where(first_off >= H, 0, np.where(has_next, next_on - first_off, H))
    out = np.concatenate([frac, (first_off / H)[:, None], (gap / H)[:, None]], axis=1)
    return out.astype(np.float32)
