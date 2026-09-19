#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""t 分布双侧临界值：**唯一来源**，不许在别处再写一份。

### 为什么单独抽出来

2026-09-20 审计 `scripts/diag/*.py` 发现三处各写各的：

| 脚本 | 写法 | 对不对 |
|---|---|---|
| `gae90_n5_chain.py:349` | `{1:12.706, 2:4.303, 3:3.182, 4:2.776, 5:2.571}.get(df, 2.0)` | df 1–5 对 |
| `check_w329_verdict.py:138` | `{1:12.706, 2:4.303, 3:3.182, 4:2.776}.get(df, 2.0)` | df 1–4 对 |
| **`ent001_verdict.py:119`** | `CRIT3 if df==2 else (2.776 if df==4 else 2.145)` | **df=1 错成 2.145（应 12.706）** |

`ent001` 那处在 n=2 训练种子时**差点宣布假显著**：本次 t=−0.69 离两条线都远，
所以结论没受影响，但这是运气，不是设计。

### 两个必须守住的点

1. **未知 df 一律报错，不许回退**。
   `.get(df, 2.0)` 式的静默宽松回退最危险：df=5 应 2.571 却给 2.0
   ⟹ **更容易判显著** ⟹ 直接制造假发现。
   宁可炸，不要默默地松。

2. **"临界值"必须与 df 绑定写在调用处**。
   本项目栽过的坑（记忆 `thresholds-and-transcribed-numbers`）就是
   拿 df=14 的 2.145 去判 df=1 的 t。

### 用法

    from stats_crit import t_crit, describe
    crit = t_crit(df)                  # df 不在表内 ⟹ 抛 KeyError
    print(describe(df, t))
"""
from __future__ import annotations

# 双侧 α=0.05 的 t 分位数（0.975 分位）。df = n − 1。
#
#   来源：标准 t 表。row 是 df。
#   df=1  12.706   ← ent001 那处错成 2.145 的就是这一格
#   df=2   4.303   ← 本项目 n=3 训练种子的常用值（"不是 2"）
#   df=4   2.776   ← n=5
#   df=14  2.145   ← n=15 请求种子
T_CRIT_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    20: 2.086,
    30: 2.042,
}

# 常用别名，让调用处读起来就是它想说的意思
CRIT_N3 = T_CRIT_975[2]    # 3 个训练种子
CRIT_N5 = T_CRIT_975[4]    # 5 个训练种子
CRIT_N15 = T_CRIT_975[14]  # 15 个请求种子（配对后）


def t_crit(df: int) -> float:
    """df 对应的双侧临界值。**未知 df 抛 KeyError，不回退。**

    需要表外的 df 时，请显式加进 `T_CRIT_975` 并写下来源——
    不要在这里加 default。
    """
    if df in T_CRIT_975:
        return T_CRIT_975[df]
    raise KeyError(
        "df=%r 不在临界值表里。**不要回退到一个宽松的默认值**——"
        "那会制造假显著。请把该 df 的真实分位数加进 T_CRIT_975（注明来源）。"
        "表内现有 df: %s" % (df, sorted(T_CRIT_975))
    )


def describe(df: int, t: float) -> str:
    """一行判读，把 df、临界值、t 绑在一起说，避免拿错值去比。"""
    c = t_crit(df)
    verdict = "过线（显著）" if abs(t) >= c else "**未过线（测不出）**"
    return "df=%d 临界=%.3f  t=%+.3f  ⟹ %s" % (df, c, t, verdict)


if __name__ == "__main__":
    # 自检：把本文件曾出过的错钉住
    assert t_crit(1) == 12.706, "df=1 必须是 12.706（ent001 曾错成 2.145）"
    assert t_crit(2) == 4.303, "df=2 必须是 4.303（本项目反复强调'不是 2'）"
    assert t_crit(4) == 2.776, "df=4 必须是 2.776（n=5）"
    assert t_crit(14) == 2.145, "df=14 必须是 2.145（n=15 请求种子）"
    try:
        t_crit(99)
    except KeyError:
        pass
    else:
        raise AssertionError("未知 df 必须抛错，不许静默回退")
    print("stats_crit 自检通过：")
    for df in (1, 2, 3, 4, 5, 14, 30):
        print("  %s" % describe(df, 0.0))
    print("  未知 df → KeyError ✓（不会静默回退成宽松值）")
