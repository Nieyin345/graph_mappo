# -*- coding: utf-8 -*-
"""对 launch_g2.py 的门做**已知答案断言**（不连服务器，纯算术）。

这是记忆里反复出现的形状：门的公式被改坏时**不报错**，只会在 OOM 时现形。
所以这里把每个场景的期望值写死，跑出别的值就失败。
"""
import importlib.util, json, sys, io, os
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
    # ★★ 桩按**名字**打，并且先断言这个名字真的存在。
    #
    #   教训：`a5d9fad`（内存单位修复）把 `avail_gb` 改名成 `avail_gib`，
    #   **但没改这里的桩** ⟹ 三个桩全部打空 ⟹ `avail_gib()` 落回真实的
    #   `/proc/meminfo`（Windows 上读不到 ⟹ 0.0）⟹ 每个场景都算出一堆垃圾数
    #   ⟹ 本测试**从那一刻起就是假失败**，而不是"门坏了"。
    #   失败信息("增量 算出 -25.0，期望 220.0")也完全指不到根因。
    #
    #   `hasattr` 守卫把那句难懂的话换成"桩名不存在" ⟹ 改名会**响亮报错**。
    #   （同族：记忆 `test-harness-must-use-real-launch-path` —— 测试夹必须
    #     走真实路径；这里是反面：桩必须真打上，否则测的是空气。）
    for _fname, _val in (("memtotal_gib", mt), ("memfree_gib", mf),
                         ("avail_gib", av)):
        if not hasattr(m, _fname):
            raise SystemExit(
                "!! 桩名 `%s` 不在 launch_g2 里 ⟹ **桩是死的**，测的是空气。\n"
                "   launch_g2 函数改名了？请同步本文件（别再让测试假失败）。"
                % _fname)
        setattr(m, _fname, lambda _v=_val: _v)

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

# ============================================================================
# 场景 5：★ **driver 预留**（phase A 与 wave263 并行时才有）
#
#   两个启动器各自独立读内存、独立决策，对**同一个内存池**做判断。
#   g2 的绝对视角原先**完全不知道 driver 即将起的那一条** ⟹ 会把机器填到
#   自己的上限，driver 再叠一条 ⟹ 超发（竞态窗口只有一个轮询周期，超发上限
#   约一条臂压在 FLOOR 上）。
#
#   现场构造：6 条已在稳态（3 base + 2 hist + 1 base = 225G），driver 还欠
#   一条 hist32_s44（63G），候选是 gae90_s43（base 25G），MemAvailable 70G。
#
#     不预留：增量 = 70 − 0 − 25       = 45 ≥ 17  ⟹ **放行**（危险）
#             绝对 = 251.2 + 43.8 − 250 = 45 ≥ 17  ⟹ 也放行
#     有预留：增量 = 70 − 0 − 25 − 63  = −18      ⟹ **拒绝** ✓
#             绝对 = 251.2 + 43.8 − 313 = −18     ⟹ 也拒绝 ✓
#
#   ⟹ 同一时刻、同一份内存读数，**预留与否给出相反的判决**。这就是修复的意义。
# ============================================================================
print()
print("场景 5：driver 预留（phase A）")
import os as _os, tempfile as _tf

_runs5 = [("ent01_rerun_s42", "base", 25.0), ("ent01_rerun_s43", "base", 25.0),
          ("ent01_rerun_s44", "base", 25.0), ("hist32_s42", "hist", 63.0),
          ("hist32_s43", "hist", 63.0), ("gae90_s42", "base", 25.0)]
_MT5, _MF5, _AV5 = 251.2, 70.0, 70.0
_OTHER5 = _MT5 - _MF5 - sum(p for _n, _k, p in _runs5)     # = -43.8

for _fname, _val in (("memtotal_gib", _MT5), ("memfree_gib", _MF5),
                     ("avail_gib", _AV5)):
    if not hasattr(m, _fname):
        raise SystemExit("!! 桩名 `%s` 不在 launch_g2 里 ⟹ 桩是死的" % _fname)
    setattr(m, _fname, lambda _v=_val: _v)

# 写一份**真实的** driver 日志（用真实的措辞 "已起 <name>（"），
# 让 driver_reserve 走它自己的解析路径 —— 不 stub 掉它，那才是测真的。
_drv = _os.path.join(_tf.gettempdir(), "test_driver_reserve.log")
with open(_drv, "w", encoding="utf-8") as _f:
    for _nm in ("ent01_rerun_s42", "ent01_rerun_s43", "ent01_rerun_s44",
                "hist32_s42", "hist32_s43"):
        _f.write("[t]   已起 %s（base，稳态 25 GB）\n" % _nm)
m.DRIVER_LOG = _drv

_rsv = m.driver_reserve(_runs5)
print("  driver 预留 = %s" % _rsv)
if _rsv == [("hist32_s44", "hist")]:
    print("  ✓ 待办正确推得：只剩 hist32_s44（已起的不重复计、在跑的不计）")
else:
    print("  ✗ 待办算错：期望 [('hist32_s44','hist')]，得到 %s" % _rsv)
    bad += 1

_inc_no, _abs_no, _, _ = m.two_views(_runs5, "base")            # 无预留
_inc_rs, _abs_rs, _, _ = m.two_views(_runs5, "base", "A")       # 有预留
print("  无预留：增量 %+.1f 绝对 %+.1f  ⟹ %s"
      % (_inc_no, _abs_no, "放行（危险）" if min(_inc_no, _abs_no) >= 17 else "拒绝"))
print("  有预留：增量 %+.1f 绝对 %+.1f  ⟹ %s"
      % (_inc_rs, _abs_rs, "放行" if min(_inc_rs, _abs_rs) >= 17 else "拒绝 ✓"))
if abs(_inc_rs - (-18.0)) > 0.05 or abs(_abs_rs - (-18.0)) > 0.05:
    print("  ✗ 有预留的期望是 −18.0/−18.0，得到 %+.1f/%+.1f" % (_inc_rs, _abs_rs))
    bad += 1
elif not (min(_inc_no, _abs_no) >= 17 and min(_inc_rs, _abs_rs) < 17):
    print("  ✗ 预留必须**反转判决**（无预留放行、有预留拒绝），方向不对")
    bad += 1
else:
    print("  ✓ 预留反转了判决：同一份内存读数，无预留放行、有预留拒绝")

# 未预留时（phase 非 A）必须与旧行为完全一致 —— 保证 phase B/C 不受影响
if abs(_inc_no - 45.0) > 0.05:
    print("  ✗ phase 非 A 时增量为 %+.1f，期望 +45.0（旧的、无预留的行为）" % _inc_no)
    bad += 1
else:
    print("  ✓ phase B/C（无预留）行为不变：增量 +45.0")

# ★ 未知臂必须**响亮报错**，不许当成 0（当成 0 = 静默偏松 = 等于没修）
with open(_drv, "a", encoding="utf-8") as _f:
    _f.write("[t]   已起 some_unknown_arm（base，稳态 25 GB）\n")
try:
    m.driver_reserve(_runs5)
    print("  ✗ 静态表外的臂**没有报错** —— 预留量会静默失准")
    bad += 1
except RuntimeError as _e:
    print("  ✓ 静态表外的臂响亮报错（不当成 0）")
_os.remove(_drv)

# ============================================================================
# 场景 6：★ `updates_done()` 必须**按轮号**读，不能**数行数**
#
#   2026-09-21 实测：`metrics.jsonl` 每轮不止一行 —— `eval_validation` 行也
#   占一行，而它**没有 `update` 键**。旧写法 `sum(1 for _ in f)` 把 eval 行
#   也算成"跑过一轮" ⟹ **系统性多算** `轮数 // eval_interval` 轮。
#
#   现场（`ent01_rerun_s43` u30 / `eval_interval=5`）：
#     旧写法报 u=34、真实 u=30；`gae90_s42` 报 22 / 真实 20；
#     `hist32_s43` 报 18 / 真实 15 —— 全都偏**快**。
#   这会掩盖"冻结"类故障（行数停住，但停在哪一轮是错的）。
#
#   判据用**真实的文件格式**：30 条训练行 + 6 条 eval 行 = 36 行。
#   期望 30（不是 36）。
# ============================================================================
print()
print("场景 6：updates_done 按轮号读（不数行数）")

_fake = _os.path.join(_tf.gettempdir(), "test_updates_done_root")
_out = _os.path.join(_fake, "outputs", "armx")
_os.makedirs(_out, exist_ok=True)
with open(_os.path.join(_out, "metrics.jsonl"), "w", encoding="utf-8") as _f:
    for _i in range(1, 31):                      # 30 条训练行
        _f.write(json.dumps({"update": _i, "mean_success_rate": 0.5}) + "\n")
        if _i % 5 == 0:                          # eval_interval=5 ⟹ 6 条 eval 行
            _f.write(json.dumps({"eval_validation": {"mean_success_rate": 0.6}}) + "\n")

_real_root = m.ROOT
m.ROOT = _fake
try:
    _got = m.updates_done("armx")
    _lines = 36
    if _got == 30:
        print("  ✓ 36 行（30 训练 + 6 eval）⟹ 读到 u=30（不是 %d）" % _lines)
    else:
        print("  ✗ 期望 u=30，得到 %s —— eval 行被算成轮数了？" % _got)
        bad += 1

    # 不存在的臂：确实一轮没跑 ⟹ 0 是真的，不能是 −1
    if m.updates_done("no_such_arm") == 0:
        print("  ✓ 目录不存在 ⟹ 0（「确实没跑」，与「读不出」区分开）")
    else:
        print("  ✗ 目录不存在的返回值不是 0")
        bad += 1

    # ★ 读不出必须返回 −1，**不许退化成 0**（0 与「还没跑」同形 = 静默偏松）。
    #   触发手段：把 `metrics.jsonl` **本身**做成目录 ⟹ `os.path.exists()` 为真
    #   （所以不会走 `not exists` 那条早返回），而 `open()` 必抛 OSError。
    #   ⚠ 第一版我写的是「把 outputs/<name> 做成目录」—— 那样 `metrics.jsonl`
    #   并不存在，先命中了 `exists=False` 分支返回 0，**测的根本不是这条路径**。
    #   这类"桩打歪了"的失败与真失败同样费时间，所以在这里记一笔。
    os.makedirs(os.path.join(_fake, "outputs", "adir", "metrics.jsonl"), exist_ok=True)
    if m.updates_done("adir") == -1:
        print("  ✓ 打不开（metrics.jsonl 是目录）⟹ −1，且显示成 `u=??` 而不是数字")
    else:
        print("  ✗ 打不开时返回的不是 −1 —— 无法与「一轮没跑」区分")
        bad += 1

    if "-1" not in m._u_str(-1) and "??" in m._u_str(-1):
        print("  ✓ `_u_str(-1)` = %r ⟹ 读不出在日志里看得出来" % m._u_str(-1))
    else:
        print("  ✗ `_u_str(-1)` = %r —— 读不出会被显示成数字" % m._u_str(-1))
        bad += 1
finally:
    m.ROOT = _real_root

print()
print("结论：%s" % ("全部通过 ✓" if bad == 0 else "有 %d 处不符 ✗" % bad))
sys.exit(1 if bad else 0)
