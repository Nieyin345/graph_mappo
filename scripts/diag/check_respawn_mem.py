#!/usr/bin/env python
"""wake_parse.maybe_respawn 的内存判据：**两个视角**，都得看。

这次改动的由来（2026-09-18 17:42）：`maybe_respawn` 在余量 62.8G 时启动了第 4 个
run（ent01_g999_s43_r2），机器上已有 3 个 vcoef1 在跑。我停掉它才回到 58.7G。

我第一版「修正」是：把新 run 的 `RUN_BASE + GROWTH×update + 垫` 换成
「在跑的补足增长 + 新 run 到顶 + 垫」。**但变异测试证明这个修正照样放行**
（62.8 >= 55.98）—— 也就是说我原本以为「改完就会拒绝」是**错的**。
一个没被测试过的判据改动和没改一样，所以这个文件留着当证据。

看清楚之后，真正的结构是：**同一个余量有两个视角，它们必须都成立**。

  视角 A（增量）：MemAvailable 还够不够填未来的增长？
      need_delta = 在跑的各补足增长 + 新 run 到顶 + 安全垫
      17:42: 16.38 + 29.6 + 10 = 55.98 ≤ 62.8 → **过**

  视角 B（峰值）：全部跑到头时，机器还剩多少绝对余量？
      peak = 现在的占用 + 未来的增长
      17:42: 62.6 + 16.38 + 29.6 = 108.6 GB，机器 125.4 → 余 16.8 GB

视角 B 才是本项目记录里那条线：「125 GB 机器上并发 3 稳、**4 勉强（108 GB，
余 17 GB）**、5 一定 OOM」。**算出来的 108.6 GB 与记录里的 108 GB 对上了** ——
这既验证了公式，也说明 4 并发落在**记录的「勉强」线上**。

所以旧判据的错不是「放行」，而是**它把 need 少算了 21.8 GB**，于是
「勉强」和「宽裕」在它眼里没有区别。修法是让判据看得见这条线：

    视角 B 的余量 < MIN_HEADROOM_GB(17)  → 拒绝

用法（服务器上）：python /tmp/check_respawn_mem.py
"""
from __future__ import annotations

RUN_BASE_GB = 23.3      # 实测 PSS
RUN_GROWTH_GB = 0.21    # 多点拟合均值（+0.085~+0.298）
SAFETY_GB = 10.0
# 记录里的「4 并发 = 108 GB = 勉强」对应余 17 GB。把这条件为硬线：
# **到不了 17 GB 余量就拒绝**，因为机器上还有探针、pytest、ssh 会话要吃内存。
MIN_HEADROOM_GB = 17.0
TARGET_UPDATES = 30
FULL = RUN_BASE_GB + RUN_GROWTH_GB * TARGET_UPDATES   # 单 run 跑到头


def meminfo():
    mi = {}
    with open("/proc/meminfo", encoding="ascii") as f:
        for line in f:
            k, _, v = line.partition(":")
            mi[k.strip()] = int(v.split()[0]) / 1048576.0
    return mi


def old_need(update: int) -> float:
    """旧判据：只看新 run 自己（漏掉在跑的还会涨）。"""
    return RUN_BASE_GB + RUN_GROWTH_GB * update + SAFETY_GB


def growth_of_running(n_running: int, u_now: int) -> float:
    return n_running * RUN_GROWTH_GB * max(0, TARGET_UPDATES - u_now)


def delta_need(n_running: int, u_now: int) -> float:
    """视角 A：未来的增长还需要多少余量。"""
    return growth_of_running(n_running, u_now) + FULL + SAFETY_GB


def peak_total(total: float, avail: float, n_running: int, u_now: int) -> float:
    """视角 B：全部跑到头时的**物理占用**。

    = 现在的占用 + 在跑的补足增长 + 新 run 从 0 到顶
    （不含安全垫 —— 这里量的是物理占用，垫是判据上的余量。）
    """
    return (total - avail) + growth_of_running(n_running, u_now) + FULL


mi = meminfo()
total, avail = mi["MemTotal"], mi["MemAvailable"]

print("=== 现在这台机器 ===")
print(f"  MemTotal {total:.1f} GB   MemAvailable {avail:.1f} GB   "
      f"已用 {total - avail:.1f} GB")
print()

# ---------- 1. 复现 17:42 的情形 ----------
print("=== 1. 复现 2026-09-18 17:42（3 个 vcoef1 在 u4，余量 62.8G）===")
A_avail, N, U = 62.8, 3, 4
g = growth_of_running(N, U)
o, d = old_need(U), delta_need(N, U)
print(f"  旧判据 need = {RUN_BASE_GB} + {RUN_GROWTH_GB}×{U} + {SAFETY_GB} = {o:.2f} GB")
print(f"     {A_avail} >= {o:.2f} → 放行（这就是当初发生的事）")
print()
print(f"  视角 A（增量）= 在跑 {N} 个补增长 {g:.2f} + 新 run 到顶 {FULL:.2f}"
      f" + 垫 {SAFETY_GB} = {d:.2f} GB")
print(f"     {A_avail} >= {d:.2f} → 放行")
print("     （我第一版修正就停在这里，变异测试证明它照样放行 ——")
print("      我以为改完会拒绝，这是错的，留档）")
print()
p = peak_total(total, A_avail, N, U)
head = total - p
print(f"  视角 B（峰值）= 已用 {total - A_avail:.2f} + 在跑补增长 {g:.2f}"
      f" + 新 run 到顶 {FULL:.2f} = {p:.1f} GB")
print(f"     机器 {total:.1f} GB → 余 {head:.1f} GB")
print("     记录里的线：4 并发 = 108 GB = 勉强（余 17 GB）")
print(f"     算出的 {p:.1f} GB 与记录 108 GB 对上 → 公式可信")
print(f"     {head:.1f} < {MIN_HEADROOM_GB:.0f} → 拒绝")
print()
print(f"  旧判据少算的量 = {d - o:.1f} GB"
      "（这就是「勉强」和「宽裕」在它眼里没区别的原因）")
print()

# ---------- 2. 正确判据的完整形式 ----------
print("=== 2. 判据（两个视角都要过）===")
print("  视角 A: MemAvailable >= delta_need(在跑数, 当前轮)")
print(f"  视角 B: MemTotal - peak_total(...) >= {MIN_HEADROOM_GB:.0f} GB")
print()
print("  加视角 B 的作用：把「勉强」这条线变成硬线。")
print(f"  17:42 的情形：A 过（62.8 >= {d:.2f}）、B 不过（{head:.1f} < "
      f"{MIN_HEADROOM_GB:.0f}）→ 拒绝。")
print()

# ---------- 3. 边界：不能变成「永远说不」 ----------
print("=== 3. 边界：机器空闲、只起 1 个 → 必须放行 ===")
p1 = peak_total(total, avail, 0, 0)
h1 = total - p1
print(f"  peak = 已用 {total - avail:.1f} + 新 run 到顶 {FULL:.1f} = {p1:.1f} GB")
print(f"  余 {h1:.1f} GB  {'>=' if h1 >= MIN_HEADROOM_GB else '<'} "
      f"{MIN_HEADROOM_GB:.0f} → "
      f"{'放行 ✓' if h1 >= MIN_HEADROOM_GB else '拒绝 ✗（注意：机器现在就紧）'}")
print()

# ---------- 4. 满编 3 个的情形（实验常态）----------
print("=== 4. 满编 3 个跑到头 ===")
p3 = 3 * FULL
print(f"  3 × {FULL:.1f} = {p3:.1f} GB，余 {total - p3:.1f} GB")
print(f"  → 3 并发是稳的（记录说 3 稳），余量 {total - p3:.1f} GB 与那句一致")
print()
print("=== 结论 ===")
print(f"  旧判据把 need 少算了 {d - o:.1f} GB，于是分不清「勉强」与「宽裕」。")
print("  修正后两个视角都要过；17:42 那一次会被拒绝 —— 但**拒绝的理由是")
print(f"  视角 B 的绝对余量 {head:.1f} GB**，不是我第一版以为的视角 A。")
