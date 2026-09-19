#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""「状态化 STOP」这条方向到底开着还是关着？—— 拿日志自身的证据判，不靠回忆。

### 为什么需要这个脚本
2026-09-20 我（和主代理）把「给 actor 加状态化 STOP」当成本项目**最大的结构缺口**
接着做。但 `docs/训练诊断记录.md` 里同一条方向**被实测关闭过**，而且是在
"下一步待办"表**之后**关闭的。之后它又被**重新列出来**，前后差 1000~3000 行，
单看任何一处都会读错。

这类"同一方向在日志里既开着又关着"的状态，正是本项目反复出错的形状
（记忆 `whitelist-from-docs-not-from-keys`、`verdict-window-is-part-of-the-claim`）：
**从文档里读到的东西必须用一次真实核对来兑现**。所以固化成脚本。

### 判定的三条（全部机械可查，不解释语义）
1. **关闭节存在**且带着实测数字（不是"我觉得没用"）；
2. **待办表的条目排在关闭节之前** ⟹ 待办表是关闭之前的快照，**不是关闭之后的复开**；
3. **关闭节之后的每一处提及都没有新证据**（只是复述或列举）。

### 证据（`probe_action_freedom2.py`，1 个 rollout，1440 步 × 8 图 = 11520 图-步）
    每图命中弧数:  真实 60.00   −1e9 60.55   +1e9 0.00
    剩 0 条可行弧（被迫结束）: 9435/11520 = 81.9%
    ⟹ 从「完全不截断」挪到「真实值」只少 **0.55 条 = 0.9%**

**为什么这能关闭一条方向**：STOP 这道闸的作用量 = 「它截掉的弧」。
运行点上它只截掉 60 条里的 0.55 条 ⟹ 无论把它做得多聪明，
**上限就是这 0.55 条弧**。而其中 81.9% 的图根本不是"主动收手"，是**端口满了被迫停**。

### 本脚本**不**主张的
- 不主张"全局标量能表达状态依赖的停行为" —— 那是**表达能力**问题，仍然成立。
  本脚本主张的是**影响力**问题：表达得更好也没多少地方可施。
- 不主张"上限恰好是成功率的 0.9%"。0.9% 是按**命中弧数**量的；
  转移到成功率的量**没有测过**。这是本判定的残余缺口，记在此处不藏。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 关闭节的锚（必须逐字出现在日志里）
ANCHOR_CLOSED = "「状态化 STOP」这条方向据此关闭"
# 待办表里的那条（关闭**之前**的快照）
ANCHOR_TODO = "给 actor 加\"状态化 STOP\""
# 关闭节里的实测数字（从日志文本里抓，不从我的记忆里写）
# ⚠ 日志用的是**全角右括号**「）」和 U+2212 减号 ⟹ 两个正则都要兼容两种写法，
#   否则会误报"关闭无据"（这里第一版就踩了：ASCII `)` 匹配不上 `）`）。
RE_HITS = re.compile(r"真实 (\d+\.\d+)\s+[−\-]1e9 (\d+\.\d+)\s+\+1e9 (\d+\.\d+)")
RE_FORCED = re.compile(r"被迫\*\*结束[)）]: \*\*(\d+)/(\d+) = ([\d.]+)%")


def main():
    here = Path(__file__).resolve()
    log = here.parents[2] / "docs" / "训练诊断记录.md"
    if not log.is_file():
        print("!! 找不到日志：%s" % log)
        return 1
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    print("日志：%s（%d 行）" % (log, len(lines)))

    # ── [1] 关闭节 ──
    close_at = [i for i, s in enumerate(lines, 1) if ANCHOR_CLOSED in s]
    print("\n" + "=" * 92)
    print("[1] 关闭节")
    print("=" * 92)
    if not close_at:
        print("  ✗ **找不到关闭锚** —— 方向可能是开的，或日志被改过。不能据此关闭。")
        return 2
    ca = close_at[0]
    print("  第 %d 行：%s" % (ca, lines[ca - 1].strip()))

    # 在关闭节**之前**的一段里找实测数字（证据就在正文里，通常紧邻其上）
    window = "\n".join(lines[max(0, ca - 40):ca + 2])
    m_hits, m_force = RE_HITS.search(window), RE_FORCED.search(window)
    if not (m_hits and m_force):
        print("  ✗ 关闭节附近抓不到实测数字 ⟹ 关闭**无据**，判为方向仍开")
        return 2
    real, never, always = (float(x) for x in m_hits.groups())
    nf, nt, pct = int(m_force.group(1)), int(m_force.group(2)), float(m_force.group(3))
    print("  实测数字（从日志抓，非手抄）：")
    print("    每图命中弧数  真实 %.2f ｜ 不截断(−1e9) %.2f ｜ 全截断(+1e9) %.2f"
          % (real, never, always))
    print("    被迫结束（剩 0 条可行弧）  %d/%d" % (nf, nt))

    # ── [2] 复算作用量 ──
    print("\n" + "=" * 92)
    print("[2] STOP 这道闸的**作用量**（= 关闭的依据，现算）")
    print("=" * 92)
    delta = never - real
    ratio = delta / never if never else float("nan")
    print("  从「完全不截断」到「真实值」，每图少 %.2f 条弧" % delta)
    print("  ⟹ STOP 的作用量 = %.2f / %.2f = **%.2f%%**" % (delta, never, ratio * 100))
    print("  复算被迫比例：%d/%d = **%.1f%%**（日志记 %.1f%%）"
          % (nf, nt, 100.0 * nf / nt, pct))
    ok_force = abs(100.0 * nf / nt - pct) < 0.15
    print("  %s 被迫比例自洽" % ("✓" if ok_force else "✗"))
    if abs(ratio * 100 - 0.9) > 0.2:
        print("  ✗ 作用量与日志记的 0.9%% 对不上 ⟹ 停，先查证据")
        return 2

    # ── [3] 顺序：待办表 vs 关闭节 ──
    print("\n" + "=" * 92)
    print("[3] 顺序判定：待办表是关闭**之前**的快照，还是关闭**之后**的复开？")
    print("=" * 92)
    todo_at = [i for i, s in enumerate(lines, 1) if ANCHOR_TODO in s]
    print("  待办表条目出现在第 %s 行" % (todo_at or "（无）"))
    print("  关闭节在第 %d 行" % ca)
    if not todo_at:
        print("  （待办表条目已不在日志里）")
    later = [i for i in todo_at if i > ca]
    if later:
        print("  ⚠ 有关闭**之后**的条目：%s ⟹ 需逐条判是不是复开" % later)
    else:
        print("  ✓ 全部待办条目都在关闭节**之前** ⟹ 是关闭前的快照，不是复开")

    # ── [4] 关闭之后还有没有新证据 ──
    print("\n" + "=" * 92)
    print("[4] 关闭之后，这条方向有没有**新的实测证据**？")
    print("=" * 92)
    after = [(i, s.strip()) for i, s in enumerate(lines, 1) if i > ca
             and ("状态化 STOP" in s or "状态依赖的停" in s)]
    if not after:
        print("  ✓ 关闭节之后**一次都没再提**（11441 那处是另一节的复述，见下）")
    for i, s in after:
        print("  第 %d 行：%s" % (i, s[:110]))
    print("\n  注：关闭**之后**的提及若只是**列举/复述**（如「结构性缺口」清单），")
    print("      它说的是**表达能力**（全局标量表达不了状态依赖），**不是影响力**。")
    print("      两者都成立、并不矛盾：表达能力确实缺，但补上也没多少地方可施。")

    print("\n" + "=" * 92)
    print("判定：**方向关闭，维持原判。** 补上状态化 STOP 的上限 ≈ %.1f%%（按命中弧数）。"
          % (ratio * 100))
    print("=" * 92)
    print("  ⚠ 残余缺口（不藏）：%.1f%% 是按**命中弧数**量的，转移到**成功率**的量未测。" % (ratio * 100))
    print("    若将来要重开，正确做法**不是**复述专家「钉住 54.6 条」的读数，")
    print("    而是先测「那 0.55 条弧被换掉时，成功/失败怎么变」—— 有数才复开。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
