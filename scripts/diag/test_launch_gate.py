# -*- coding: utf-8 -*-
"""对 launch_g2.py 的门做**已知答案断言**（不连服务器，纯算术）。

这是记忆里反复出现的形状：门的公式被改坏时**不报错**，只会在 OOM 时现形。
所以这里把每个场景的期望值写死，跑出别的值就失败。
"""
import importlib.util, sys, io, os
# ★ Windows 控制台是 GBK：不 reconfigure 的话第一句就 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 把模块读进来但不执行 main（它有 __main__ 守卫）
spec = importlib.util.spec_from_file_location(
    "launch_g2",
    r"D:\destop\work_space\learning_space\论文\QKD-SAGIN生产端调度\qkd_rl\scripts\diag\launch_g2.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

CASES = []
def case(desc, runs, kind, memtotal, memfree, avail, exp_inc, exp_abs, exp_pend, exp_other):
    CASES.append((desc, runs, kind, memtotal, memfree, avail,
                  exp_inc, exp_abs, exp_pend, exp_other))

# 场景 1：实测现场（2026-09-20 22:0x）。5 条在跑，3 base u~10 + 2 hist u4。
#   base: 22.2+21.2+21.2 = 64.6   hist: 59.1+59.9 = 119.0   合计 183.6
#   待涨 = base 3×(25−~21.5)=10.4  +  hist 2×(63−59.5)=7.0  ⟹ 17.4
#   新臂 base 稳态 25
#   MemAvailable 79.7 ⟹ 增量 = 79.7 − 17.4 − 25 = 37.3
#   MemTotal 251.2, MemFree 73.0, 已用 PSS 183.6 ⟹ 其它 = 251.2−73.0−183.6 = −5.4
#   Σ全部稳态 = 25×3 + 63×2 + 25 = 226 ⟹ 绝对 = 251.2 − (−5.4) − 226 = 30.6
#   ⚠ 我第一次手算写成 183.6+25（把 hist 的稳态当成实测的 59.5 而不是 63）
case("现场·起第 6 条 base",
     [("a","base",22.2),("b","base",21.2),("c","base",21.2),
      ("h1","hist",59.1),("h2","hist",59.9)],
     "base", 251.2, 73.0, 79.7, 37.3, 30.6, 17.4, -5.4)

# 场景 2：**稳态总和已超总量** —— 绝对视角必须拒绝，而增量视角会放行。
#   10 条 base 稳态 = 250G，但都还在 u3（每条 12G ⟹ 在跑 PSS 120G）
#   待涨 = 10 × 13 = 130 ⟹ 增量 = 200 − 130 − 25 = 45 ≥ 17 ⟹ **A 放行**（危险！）
#   其它 = 251.2 − 80 − 120 = 51.2 ⟹ 绝对 = 251.2 − 51.2 − 275 = −75 ⟹ **B 拒绝** ✓
case("增量放行但绝对必须拒绝（两视角的意义）",
     [("r%d"%i,"base",12.0) for i in range(10)],
     "base", 251.2, 80.0, 200.0, 45.0, -75.0, 130.0, 51.2)

# 场景 3：hist 臂的待涨量必须被算进去（u24 仍未到 63G ⟹ 不能截断）
#   1 条 hist 在 u24、57G，无其它。待涨 = 63 − 57 = 6
#   可用 200 ⟹ 增量 = 200 − 6 − 63 = 131；其它 = 251.2 − 150 − 57 = 44.2
#   绝对 = 251.2 − 44.2 − (63 + 63) = 81   ← 本臂也是 hist，稳态 63 不是 25
case("hist 在 u24 仍有待涨（截断会漏掉这 6G）",
     [("h","hist",57.0)], "hist", 251.2, 150.0, 200.0, 131.0, 81.0, 6.0, 44.2)

# 场景 4：空机起第一条
#   待涨 0；增量 = 245 − 0 − 25 = 220；其它 = 251.2 − 245 − 0 = 6.2
#   绝对 = 251.2 − 6.2 − 25 = 220
case("空机",
     [], "base", 251.2, 245.0, 245.0, 220.0, 220.0, 0.0, 6.2)

bad = 0
for (desc, runs, kind, mt, mf, av, e_inc, e_abs, e_pend, e_other) in CASES:
    m.memtotal_gb = lambda mt=mt: mt
    m.memfree_gb   = lambda mf=mf: mf
    m.avail_gb     = lambda av=av: av
    inc, absolute, pend, other = m.two_views(runs, kind)
    checks = [("增量", inc, e_inc), ("绝对", absolute, e_abs),
              ("待涨", pend, e_pend), ("其它", other, e_other)]
    ok = all(abs(g - e) < 0.05 for _n, g, e in checks)
    print(("  ✓ " if ok else "  ✗ ") + desc)
    for n, g, e in checks:
        if abs(g - e) >= 0.05:
            print("      %s 算出 %.1f，期望 %.1f" % (n, g, e))
            bad += 1
    # 额外断言：场景 2 必须 A 过 B 不过
    if desc.startswith("增量放行"):
        if not (inc >= 17 and absolute < 17):
            print("      ✗ 场景 2 的两视角方向不对：A=%s B=%s" % (inc >= 17, absolute >= 17))
            bad += 1
        else:
            print("      ✓ 场景 2 确认：A 放行、B 拒绝 —— 两个视角都必要")

print()
print("结论：%s" % ("全部通过 ✓" if bad == 0 else "有 %d 处不符 ✗" % bad))
sys.exit(1 if bad else 0)
