"""平台期到底在哪一轮？—— 若是 u15 而非 u30，所有实验直接快一倍。

### 为什么这是当前最值钱的一问

速度这条线上，代码层已无油水（backward 57% + 前向 23% = 98%），
线程数虽实测 **1.49x**，但它**会改变训练结果**（差 0.018，与 0.035 的分辨率同量级）
⟹ 采用它必须「整批重跑 + 重建基线」，在一天窗口里做不完。

而**轮数**是另一回事：若验证曲线在 u15 就已经到平台，那么把
`num_updates` 从 30 降到 15 就是**纯 2x**。它不改任何机制、不碰任何张量布局，
因此**不引入新的基线重建问题** —— 唯一要确认的是"平台真的到了"。

已有读数只说 u30≈u40（−0.0023）；**没有人查过 u15**。

### 判据（跑之前写死）

对每个 run 取验证序列 `per_seed_success`（15 个种子，逐种子可比），
把晚期窗口两两配对相减（**配对**，不是比均值）：

  · u25 与 u30 无差异 **且** u20 与 u30 无差异 **且** u15 与 u30 无差异
      ⟹ 平台早于 u15 ⟹ **可以砍到 15 轮**
  · 只有 u25/u20 无差异，u15 有差异
      ⟹ 平台在 u15~u20 之间 ⟹ 砍到 20 轮
  · 到 u25 都还有差异
      ⟹ 平台在 u25 之后 ⟹ **不能砍**，继续 30 轮

★ 这里**必须配对**（逐种子相减）：不配对时 SE≈0.036，配对后≈0.006，
  差 6 倍 —— 用不配对的口径会把"早就平了"读成"还在涨"（本项目已踩过）。

用法（服务器上）：
  /opt/qkd/venv/bin/python /tmp/probe_plateau_round.py
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
T_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
           6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

RUNS = ["ent01_t8_s42", "ent01_t8_s43", "ent01_t8_s44",
        "ent03_s42", "ent03_s43", "ent03_s44"]


def load(run: str) -> dict[int, dict]:
    """{update: eval_validation}。

    ★ 实测的行结构（别再猜）：`metrics.jsonl` 里**验证行是独立的行**，
      只有 `eval_validation` 一个键、**没有 `update` 字段**，每 5 个 update
      之后出现一行：

          [ 4] update=5   ← 普通训练行
          [ 5] EVAL        ← 这一行就是 u5 的验证
          ...
          [34] update=30
          [35] EVAL        ← u30 的验证

      所以轮次**不能从行里的 `update` 取**（验证行没有这个字段），
      要靠"**到这行为止已经过了几个 update 行**"来推 —— 这样也自动
      适配 `eval_interval` 改变。
    """
    p = OUT / run / "metrics.jsonl"
    out: dict[int, dict] = {}
    if not p.exists():
        return out
    n_updates = 0
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = r.get("eval_validation")
        if isinstance(ev, dict):
            if ev.get("per_seed_success") and n_updates:
                out[n_updates] = ev
        elif isinstance(r.get("update"), int):
            n_updates = max(n_updates, int(r["update"]))
    return out


def paired(a: list[float], b: list[float]):
    n = min(len(a), len(b))
    if n < 2:
        return None
    d = [a[i] - b[i] for i in range(n)]
    m = statistics.mean(d)
    sd = statistics.stdev(d)
    se = sd / math.sqrt(n)
    if se == 0.0:
        t = 0.0 if abs(m) < 1e-12 else float("inf")
    else:
        t = m / se
    return m, sd, se, t, n


def main() -> int:
    print("=" * 92)
    print("平台期在哪一轮：逐种子配对比较晚期窗口（零训练成本，只读 metrics.jsonl）")
    print("=" * 92)

    for run in RUNS:
        ser = load(run)
        if not ser:
            print(f"  {run}: 无 eval_validation 序列，跳过")
            continue
        us = sorted(ser)
        print(f"\n  ── {run} ──  有验证点的轮次 {us}")
        if 30 not in ser and max(us) < 25:
            print("     轮次不够（需要跑到 u30 附近），跳过")
            continue
        base_u = 30 if 30 in ser else max(us)
        base = ser[base_u]["per_seed_success"]
        print(f"     基准 = u{base_u}（n={len(base)} 种子）")
        print(f"     {'对比':<12}{'Δ 均值':>12}{'SD':>10}{'SE':>10}{'t':>9}"
              f"{'临界':>8}{'可测?':>8}")
        print("     " + "-" * 68)
        for u in (10, 15, 20, 25):
            if u not in ser:
                continue
            r = paired(ser[u]["per_seed_success"], base)
            if not r:
                continue
            m, sd, se, t, n = r
            crit = T_TABLE.get(n - 1, 2.145)
            ok = "是" if abs(t) >= crit else "否"
            print(f"     u{u} vs u{base_u:<6}{m:>+12.5f}{sd:>10.5f}{se:>10.5f}"
                  f"{t:>9.3f}{crit:>8.3f}{ok:>8}")

    print()
    print("=" * 92)
    print("判读规则（写死在文件头）")
    print("=" * 92)
    print("  · 若 u15 vs u30 在所有 run 上都「否」（不可测）")
    print("      ⟹ 平台早于 u15 ⟹ **num_updates 可砍到 15，纯 2x**")
    print("  · 若只有 u20/u25 是「否」、u15 是「是」")
    print("      ⟹ 平台在 15~20 之间 ⟹ 砍到 20 轮（1.5x）")
    print("  · 若 u25 就是「是」⟹ 不能砍")
    print()
    print("  ⚠ n=3 个种子时配对分辨率约 SE×临界值；「否」只说明**没测出**，")
    print("    不等于完全相同 —— 报的时候按「测不出」写。")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
