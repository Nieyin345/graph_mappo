# -*- coding: utf-8 -*-
"""`mem_pss.py` 的两视角回归测试：它们必须**代数上真的不同**。

配对先例：`test_launch_gate.py`（对 `launch_g2.py` 的门做已知答案断言）。
本文件对 `mem_pss.py` 的外推判据做同类断言。**纯算术，不连服务器。**

## 锁住的缺陷（B-8，2026-09-21 核实）

旧实现：

    other = MemTotal - MemFree - cur_total
    after = MemTotal - other - extrap_need

代入 `other`：

    after = MemTotal - (MemTotal - MemFree - cur_total) - extrap_need
          = MemFree + cur_total - extrap_need
          = MemFree - (extrap_need - cur_total)
          = MemFree - incremental          ← ∵ incremental := extrap_need - cur_total

⟹ **`after ≡ free_now - incremental`** —— 旧的「视角 B」算的是**同一个残余量**，
  不是注释里说的"绝对视角"。而两个视角的**判据**分别是
  `残余 ≥ 0`（A）与 `残余 ≥ 17`（B）⟹ **B 蕴含 A** ⟹ `A ∧ B ≡ B` ⟹
  **A 是冗余的**，"缺一不可"不成立。

★ 本文件第一版把这条写成了"两视角必然同号"，**是错的** ——
  它们在残余 ∈ [0, 17) 上会异号（A 过、B 不过）。**是测试逼我把缺陷说准的**：
  "同一个量" 不等于 "同一个谓词"。断言写错方向，比不写还坏。

## 判据

  1. 旧实现的 `after` 恒等于 `MemFree − incremental`（**正对照**，必须成立）
  2. 旧实现的**真缺陷**：`A` 从不比 `B` 更严 ⟹ A 提供不了 B 之外的信息
  3. 新实现的两个视角用**不同的量**（MemFree vs MemAvailable），
     互相**都不蕴含对方**（两个方向各给一个场景）
"""
from __future__ import annotations

import random
import sys

sys.stdout.reconfigure(encoding="utf-8")

_ok = [True]


def check(label, got, want):
    good = (got == want)
    _ok[0] = _ok[0] and good
    print("  %s %-56s got=%r want=%r" % ("✓" if good else "✗", label, got, want))


MIN_MARGIN = 17.0


def old_views(mem_total, mem_free, cur_total, extrap_need):
    """旧实现（B-8）：after 恒等于 free_now - incremental。"""
    incremental = extrap_need - cur_total
    other = mem_total - mem_free - cur_total
    after = mem_total - other - extrap_need
    return incremental, other, after


def new_views(mem_total, mem_free, mem_avail, cur_total, extrap_need):
    """新实现：硬口径 MemFree、乐观口径 MemAvailable。"""
    incremental = extrap_need - cur_total
    after_opt = mem_avail - incremental
    return incremental, after_opt


print("=" * 78)
print("1. 旧实现的两个视角**代数同一**（这就是 B-8）")
print("=" * 78)
rng = random.Random(20260921)
worst = 0.0
for _ in range(2000):
    tot = rng.uniform(100, 300)
    free = rng.uniform(0, tot)
    cur = rng.uniform(0, tot - free)
    need = cur + rng.uniform(-50, 50)          # 含负增量
    inc, _other, after = old_views(tot, free, cur, need)
    worst = max(worst, abs(after - (free - inc)))
check("2000 组随机输入下 |after − (MemFree − incremental)| 恒为 0",
      worst < 1e-9, True)
print("     ⟹ 旧『视角 B』算的是**同一个残余量**（不是注释说的『绝对视角』）。")
print("     ⚠ 但**别**说成『两个 ✓/✗ 永远同号』：A=`残余≥0`、B=`残余≥17` 是**两个**")
print("       谓词，在残余 ∈ [0,17) 上会异号。真缺陷见第 2 节。")

print()
print("=" * 78)
print("2. 旧实现的**真缺陷**：A 从不比 B 更严 ⟹ A 提供不了 B 之外的信息")
print("=" * 78)
# ★ 第一版这里断言「异号次数 == 0」，**是错的**：A 是 `残余 ≥ 0`、B 是 `残余 ≥ 17`，
#   在残余 ∈ [0,17) 上 A 过 B 不过 ⟹ 会异号。**是测试逼我把缺陷说准的**。
# ★ 第二版把蕴含方向写反了（写成 A ⟹ B）。**正确方向是 B ⟹ A**：
#   `残余 ≥ 17` 必然推出 `残余 ≥ 0`。所以 `A ∧ B ≡ B` ⟹ **A 完全冗余**。
#   "同一个量" ≠ "同一个谓词"，蕴含方向更是最容易写反的一处 —— 写反了断言就
#   恰好测到反面，还照样"通过"或"失败"得很有道理。
viol_B_implies_A = 0
disagree = 0
for _ in range(2000):
    tot = rng.uniform(100, 300)
    free = rng.uniform(0, tot)
    cur = rng.uniform(0, tot - free)
    need = cur + rng.uniform(-50, 50)
    inc, _o, after = old_views(tot, free, cur, need)
    a = free >= inc
    b = after >= MIN_MARGIN
    if b and not a:
        viol_B_implies_A += 1            # 若出现，说明 B 不蕴含 A ⟹ 两者才真独立
    if a != b:
        disagree += 1
check("旧实现里 B **蕴含** A：`B∧¬A` 一次都没出现", viol_B_implies_A, 0)
check("旧实现**确实会**异号（A 过 B 不过）——所以第一版断言是错的",
      disagree > 0, True)
print("     ⟹ 旧实现的缺陷是 **A ∧ B ≡ B**（A 冗余），不是『同号』。")

print()
print("=" * 78)
print("3. 新实现的两个视角**互不蕴含**（各自能单独否决）")
print("=" * 78)
# 方向一：硬口径更严 —— MemFree 不够，但总可用量（含缓存）够
#   incremental = need - cur = 75 - 60 = 15；MemFree=10 < 15 ⟹ A ✗
#   after_opt = avail - inc = 120 - 15 = 105 ≥ 17 ⟹ B ✓
inc, after_opt = new_views(251.0, 10.0, 120.0, 60.0, 75.0)
a, b = (10.0 >= inc), (after_opt >= MIN_MARGIN)
print("     场景一 MemFree=10 MemAvailable=120 增量=%+.1f ⟹ A=%s B=%s (after_opt=%.1f)"
      % (inc, a, b, after_opt))
check("场景一：**A 单独否决**（硬口径严）", (a, b), (False, True))

# 方向二：乐观口径更严 —— MemFree 够，但缓存极少 ⟹ 绝对余量 < 17
#   incremental = 5；MemFree=30 ≥ 5 ⟹ A ✓；after_opt = 20 - 5 = 15 < 17 ⟹ B ✗
inc2, ao2 = new_views(251.0, 30.0, 20.0, 60.0, 65.0)
a2, b2 = (30.0 >= inc2), (ao2 >= MIN_MARGIN)
print("     场景二 MemFree=30 MemAvailable=20 增量=%+.1f ⟹ A=%s B=%s (after_opt=%.1f)"
      % (inc2, a2, b2, ao2))
check("场景二：**B 单独否决**（绝对余量严）", (a2, b2), (True, False))

print("     ⟹ 两个方向都能单独否决 ⟹ 这一对**是**真的双视角。")
print("        （对比旧实现：只有 B 能单独否决，A 永远跟着 B）")

print()
print("=" * 78)
print("4. 源码断言：旧那两行不得复活")
print("=" * 78)
from pathlib import Path
SRC = Path(__file__).resolve().parent / "mem_pss.py"
live = "\n".join(l.split("#")[0] for l in SRC.read_text(encoding="utf-8").splitlines())
check("活代码里不得再出现 `MemTotal - other - extrap_need`",
      "MemTotal\", 0) - other - extrap_need" in live, False)
check("活代码里不得再把 `other` 算进 after", "- other -" in live, False)
check("新视角用了 MemAvailable", "MemAvailable" in live, True)
check("陈旧除数 /24 已删（25 不是常数）", "/ 24" in live, False)
check("改为按配置查表的提示在位", "别拿 25 当常数" in live, True)

print()
print("=" * 78)
print("5. 空集第三态：**顺序**必须对（先判空，再打两视角）")
print("=" * 78)
# ★ 缺陷（2026-09-21 自查）：空集分支原先是写在**两视角打印之后**的。
#   空集 ⟹ incremental 恒 0 ⟹ 两视角**必然都 ✓** ⟹ 先输出
#   「还要长 +0.0 GiB ... ✓ ✓」再输出「无从外推」，前后自相矛盾。
#   **恒 ✓ 的判据不是判据**（同族：`gate-must-print-its-inputs` ——
#   恒真的门不报错）。修法不是改措辞，是**把判空提前**。
i_empty = live.find("if not groups:")
i_viewA = live.find("视角 A")
check("判空分支存在", i_empty > 0, True)
check("★ 判空分支在『视角 A』**之前**（否则先打出恒 ✓ 的对比）",
      i_empty < i_viewA, True)
empty_seg = live[i_empty:i_viewA] if 0 < i_empty < i_viewA else ""
check("判空分支里不得出现『可以安全跑完』", "可以安全跑完" in empty_seg, False)
check("判空分支里给出了按配置查表的表", "hist32" in empty_seg, True)
check("上限除数用的是 25（且已夹下界，不会出负数）",
      "max(0, int((free_now - MIN_MARGIN) / 25))" in empty_seg, True)
# 正对照：非空分支**仍然要**打那句「可以安全跑完」——否则上面那条
# 「不得出现」会因为整段被删而假通过（空切片恒真）。
check("正对照：非空分支仍在说『可以安全跑完』", "可以安全跑完" in live, True)

print()
print("=" * 78)
print("6. ★★ 孤儿告警**恒为空**（子代理漏报；本文件核查时发现）")
print("=" * 78)
# 机制：孤儿按定义 PPid==1；而 ancestor_run() 一走到 ppid<=1 就 return ""
# ⟹ 「是孤儿」与「能被归属」**互斥**。旧代码把孤儿统计写在
# `if not g: continue` **之后** ⟹ 永远够不到 ⟹ orphans 恒为空。
#
# 用**模拟两种次序**造反证（不是只 grep 源码）：
PSS = {"trainer": 14.0, "worker_ok": 1.1, "worker_orphan": 1.1, "worker_orphan2": 1.1}
# ★ 关键前提：**活着的 worker 的 PPid 是 trainer 的 pid，不是 1**；只有孤儿
#   才 PPid==1。第一版我把 worker_ok 也写成 PPid==1 ⟹ 它被误判成孤儿
#   ⟹ 「旧次序恒为空」测不出来。**夹具错了，断言就测到反面。**
PPID = {"trainer": 1, "worker_ok": 4242,          # 4242 = trainer 的 pid
        "worker_orphan": 1, "worker_orphan2": 1}
# worker_ok 的祖先链到 trainer；两个 orphan 直接挂 init（PPid==1）。
# trainer 自己就命中 `--run-name` ⟹ 它的 ancestor 就是它自己那条臂
# （真实代码里 `ancestor_run()` 的 `if cur in owner: return owner[cur]`）。
ANCESTOR = {"trainer": "arm_x", "worker_ok": "arm_x",
            "worker_orphan": "", "worker_orphan2": ""}
IS_MULTIPROCESS = {"worker_orphan": True, "worker_orphan2": True, "worker_ok": True}
IS_PY = {k: True for k in PSS}


def simulate(orphan_inside_guard):
    """orphan_inside_guard=True 复刻旧次序（孤儿在 `if not g: continue` 之后）。"""
    groups, orphans, total = {}, {}, 0.0
    for pid in PSS:
        if not IS_PY[pid]:
            continue
        v = PSS[pid]
        if v <= 0:
            continue
        if not orphan_inside_guard:
            if PPID[pid] == 1 and IS_MULTIPROCESS.get(pid):
                orphans[pid] = v
                total += v
                continue
        g = ANCESTOR[pid]
        if not g:
            continue                      # ← 旧实现把孤儿丢在这里
        total += v
        groups[pid] = v
        if orphan_inside_guard:
            if PPID[pid] == 1 and IS_MULTIPROCESS.get(pid):
                orphans[pid] = v          # ← 永远够不到
    return groups, orphans, total


g_old, o_old, t_old = simulate(orphan_inside_guard=True)
g_new, o_new, t_new = simulate(orphan_inside_guard=False)
check("旧次序：**孤儿恒为空**（告警是死代码）", len(o_old), 0)
check("新次序：孤儿被抓到（2 个 worker）", len(o_new), 2)
check("新次序：孤儿进入合计（14.0 + 3×1.1 = 17.3）", round(t_new, 1), 17.3)
check("旧次序漏掉孤儿的内存（合计只有 15.1）", round(t_old, 1), 15.1)
check("★ 两个次序的合计**必须不同**——否则这个测试测不出东西",
      round(t_old, 1) != round(t_new, 1), True)

# 源码层：孤儿的判据必须在 ancestor_run 之前（否则不可达）
i_orph = live.find("ppid_of(pid) == 1")
i_anc = live.find("g = ancestor_run(pid)")
check("源码：孤儿判据**先于**归属判定（否则恒不可达）",
      0 < i_orph < i_anc, True)
check("源码：孤儿用独立桶名，不硬塞给某个 run",
      '(未归属 · 孤儿 worker)' in live, True)
# ⚠ `live` 是**去注释**的视图（`l.split("#")[0]`）⟹ 断言里不能含 `#` 注释文本，
#   否则在构造上不可能为真。要断言注释得读**原始**源码。
_raw = SRC.read_text(encoding="utf-8")
check("源码：孤儿进 total_pss（确实在吃内存）",
      "total_pss += v" in live and "# 孤儿" in _raw, True)

print()
print("=" * 78)
print("全部通过 ✓" if _ok[0] else "**有失败 ✗**")
sys.exit(0 if _ok[0] else 1)
