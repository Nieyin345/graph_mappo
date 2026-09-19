#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地验 gae90_n5_chain.py：只做语法检查 + 纯函数逻辑测试。

本机禁止 import torch/qkd_rl，但该脚本只依赖标准库（json/math/statistics/
glob/os/re/subprocess/sys/time），因此可以在这里跑。

★ 为什么要先验：这个脚本要在**几小时后**才可能出结果，
   如果里面有语法错或判读逻辑错，失败会伪装成「还在跑」（记忆 failed-launch-must-be-loud）。
"""
import ast
import importlib.util
import math
import os
import statistics as st
import sys

SRC = r"D:\destop\work_space\learning_space\论文\QKD-SAGIN生产端调度\qkd_rl\.tmp\gae90_n5_chain.py"

# ---- 1. 语法 ----
with open(SRC, encoding="utf-8") as f:
    text = f.read()
try:
    tree = ast.parse(text)
    print("[1] 语法 OK，%d 个顶层节点" % len(tree.body))
except SyntaxError as e:
    print("[1] !! 语法错误 %s" % e)
    sys.exit(1)

# ---- 2. 静态检查：那行 NUL 修复必须存在 ----
if '.replace("\\0", " ")' not in text:
    print("[2] !! 缺 cmdline 的 NUL 规范化 ⟹ live_runs() 会恒为空 ⟹ 门恒通过")
    sys.exit(1)
print("[2] NUL 规范化存在 OK")

# ---- 3. 门必须打印输入（不是只打印 verdict）----
for tok in ("可用", "在跑", "待涨", "余量"):
    if tok not in text:
        print("[3] !! 门没有打印 %s ⟹ 恒真时不报错" % tok)
        sys.exit(1)
print("[3] 门打印了 可用/在跑/待涨/余量 OK")

# ---- 4. 三态标记 ----
for tok in ("STATE=done", "STATE=timeout", "STATE=failed"):
    if tok not in text:
        print("[4] !! 缺 %s" % tok)
        sys.exit(1)
print("[4] done/timeout/failed 三态齐全 OK")

# ---- 5. 判读逻辑：用假数据跑一遍 verdict 的核心算术 ----
# 模拟：对照全 0.69，臂在 s42/43/44/45/46 上分别 +0.0128/+0.0302/+0.0071/+0.0250/+0.0080
# 期望 Δ≈+0.01662, SD≈0.00975, t≈3.81 ⟹ df=4 临界 2.776 ⟹ 过线
ds = [0.0128, 0.0302, 0.0071, 0.0250, 0.0080]
m = st.mean(ds)
sd = st.stdev(ds)
se = sd / math.sqrt(len(ds))
t = m / se
crit = 2.776
print("[5] 判读算术：Δ=%+.4f SD=%.4f SE=%.4f t=%+.2f 临界=%.3f ⟹ %s"
      % (m, sd, se, t, crit, "过线" if abs(t) >= crit else "未过线"))

# n=3 同三个种子（用前三个）应当**不过线**——这正是本波要解决的问题
ds3 = ds[:3]
m3, sd3 = st.mean(ds3), st.stdev(ds3)
t3 = m3 / (sd3 / math.sqrt(3))
print("[6] 同样数据只取 n=3：Δ=%+.4f SD=%.4f t=%+.2f 临界=%.3f ⟹ %s"
      % (m3, sd3, t3, 4.303, "过线" if abs(t3) >= 4.303 else "未过线"))
if abs(t3) >= 4.303:
    print("    !! 与预期不符：n=3 不该过线")
    sys.exit(1)

# ---- 7. 临界值表方向检查（大 df 应更松）----
crits = [12.706, 4.303, 3.182, 2.776, 2.571]
if crits != sorted(crits, reverse=True):
    print("[7] !! 临界值表方向错（df 越大应越松）")
    sys.exit(1)
print("[7] 临界值表 df1→df5 单调递减 OK（12.706→2.571）")

# ---- 8. ★ need_for 必须随剩余条数递减，且第一条要算全部 4 条 ----
# 这是我在自己代码里抓到的真 bug：写成常量 4*25+17=117 时，
# 起完第 1 条后第 2 条的门仍要求 117G，而第 1 条正在爬坡占着内存
# ⟹ 余量永远差 25G ⟹ 剩下 3 条永远起不来。
ns = {}
exec(compile("\n".join(
    l for l in text.splitlines() if l.startswith("STEADY") or l.startswith("FLOOR")
    or l.startswith("def need_for") or l.startswith("    return")
), "<x>", "exec"), ns)
need_for = ns["need_for"]
vals = [need_for(k) for k in (4, 3, 2, 1)]
print("[8] need_for(4,3,2,1) = %s" % vals)
if vals != [92.0, 67.0, 42.0, 17.0]:
    print("[8] !! need_for 不对（期望 92/67/42/17）")
    sys.exit(1)
if vals != sorted(vals, reverse=True):
    print("[8] !! need_for 不随剩余条数递减 ⟹ 起了一条之后永远放行不了下一条")
    sys.exit(1)
# 第一条必须为「还没起的 4 条」预留：4-1=3 份爬坡 + 1 份本臂稳态 + floor
if need_for(4) != 3 * 25.0 + 17.0:
    print("[8] !! need_for(4) 应为 3×25+17（本臂 25 在 NEED 里，另 3 条在 grow 里）")
    sys.exit(1)
# 常量写法（bug）与正确写法在 n_left=4 时相同 —— 所以这个 bug **看不出来**，
# 只有在起完第一条之后才现形。这正是它危险的地方。
print("[8] need_for 递减、且第一条为全部 4 条预留 OK（常量写法的 bug 已修）")

print("\n全部通过。")
