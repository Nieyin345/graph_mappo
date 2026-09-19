# -*- coding: utf-8 -*-
"""四则运算基准：把本会话用到的所有判据阈值**现算一遍**，不对任何手抄数字。

### 为什么必须有这个

本项目有一条已记的教训（[[thresholds-and-transcribed-numbers]]）：一个会话里
出现过**四个静默反向的错误**，全都是"手抄的常数"——df=2 该用 4.303 却用了 2、
连分式初值、教科书常数、配对差符号。数字要么**现算**，要么**对 oracle**。

本会话我又手写过好几个阈值（t 临界值、SE、分辨率、比例），这个脚本把它们
全部交给 scipy/统计库重算，任何一个对不上就打出来。**跑一次，几秒钟。**

用法（服务器上，venv 里有 scipy）：
    /opt/qkd/venv/bin/python /tmp/probe_number_audit.py
"""
from __future__ import annotations

import math

FAILS = []


def check(label, got, want, tol=5e-3):
    ok = abs(got - want) <= tol * max(1.0, abs(want))
    print(f"  {'ok ' if ok else '!! '} {label:<44} 现算={got:<14.6g} 手抄={want:<14.6g}")
    if not ok:
        FAILS.append(label)


print("=" * 84)
print("① t 分布临界值（双侧 95%）—— 本会话多次用于小样本配对判据")
print("=" * 84)
try:
    from scipy import stats
    for df, claimed in ((1, 12.706), (2, 4.303), (4, 2.776), (8, 2.306)):
        check(f"t(0.975, df={df})", float(stats.t.ppf(0.975, df)), claimed)
except ImportError:
    print("  !! 没有 scipy —— 用查表值兜底（但那就退化成手抄了）")
    FAILS.append("scipy 缺失")

print()
print("=" * 84)
print("② 测量分辨率：SE / 配对 SE / 单种子分辨率")
print("=" * 84)
# 验证 protocol：15 个请求种子，逐种子成功率 SD 0.1404（本项目实测）
sd = 0.1404
n = 15
unpaired_se = sd / math.sqrt(n)
check("未配对 SE = SD/√15", unpaired_se, 0.0362, tol=0.01)

# 配对 SE：由 SD_pair 与 n 反推（日志·测试规范记 SD_pair ≈ 0.0244）
sd_pair = 0.0244
paired_se = sd_pair / math.sqrt(n)
check("配对 SE = SD_pair/√15", paired_se, 0.0063, tol=0.02)

# ⚠ 日志 5421-5424 并列写 0.0362 / 0.0063 / "配对缩小 8.1×"，但
#   0.0362/0.0063 = 5.75 —— **三者不能同时成立**。
#   这里**不**把它当成 FAILS 的一项（那是文档的不自洽，不是本脚本的判据失败），
#   只单独报出来，免得它盖住真正的阈值错误。
print(f"  !!  [文档不自洽] 0.0362/0.0063 = {0.0362/0.0063:.2f}，"
      f"而日志写 8.1x")
print(f"      由 SD_pair=0.0244 反推 ⟹ 配对 SE={paired_se:.4f}，"
      f"缩小 {unpaired_se/paired_se:.2f}x")
print(f"      原始数据重算（probe_se_recompute）⟹ 配对 SE 0.0055，缩小 6.7~7.1x")

# 单种子分辨率（本项目记为 ~0.035）
check("未配对 SE vs 单种子分辨率", unpaired_se, 0.035, tol=0.05)

print()
print("=" * 84)
print("③ 奖励恒等式：served 项的斜率必须与配置**逐位**吻合")
print("=" * 84)
# resolved_config 实测：served_weight 50, served_reference 100000, reward_scale 0.002
w, ref, scale = 50.0, 100000.0, 0.002
pred = w / ref * scale
check("served_weight/served_reference×scale", pred, 1.000e-06, tol=1e-6)
print(f"      （探针回归实测 slope = 1.0000e-06 ⟹ R²=1.00000，是恒等式不是拟合）")

print()
print("=" * 84)
print("④ dense 幅度：份额型公式的两步推算")
print("=" * 84)
# 实测 dense = 6e-6（seed 100）；weight 0.02, penalty 0.01
dense_measured = 6e-6
weight, penalty = 0.02, 0.01
inner = dense_measured / weight          # = mean(1.01·imp − 0.01)，与 weight 无关
print(f"      mean(1.01·imp − 0.01) = dense/weight = {inner:.6g}")
imp = (inner + penalty) / (1.0 + penalty)
print(f"      ⟹ mean(importance) = (inner+penalty)/(1+penalty) = {imp:.6g}")
check("mean(importance) 约 1%（探针 P 行为侧同量级）", imp, 0.0102, tol=0.05)
# penalty 把有效增益压小的倍数
check("无惩罚/有惩罚 = (1.01·imp)/inner", (1.01 * imp) / inner, 34.0, tol=0.05)
# 要占到 served 的 6%（served≈0.0412）需多大 weight。
# ⚠ 我初稿写成 0.06*served/(inner/weight) —— 那个除法把 weight 又乘了回去，
#   代数上等于 0.06*served/dense_measured*weight²，算出 0.16（错了一个量级）。
#   正确：dense = weight × inner，而 inner 与 weight 无关（份额型），
#   所以 weight_needed = target / inner。**这里被审计脚本自己抓到了。**
served = 0.041174
target = 0.06 * served
need = target / inner
print(f"      目标 dense = 6% × served = {target:.6g}")
check("达到 served 6% 所需 weight", need, 8.2, tol=0.10)
print(f"      （注意：实测 dense 只报到 6 位小数 6e-6，inner 有 ±8% 的不确定度，"
      f"故 weight 落在 7.6~8.9）")

print()
print("=" * 84)
print("⑤ 密钥效率：省下的比例")
print("=" * 84)
expert = 214.48
for name, val, claimed in (("ent01_s42", 166.73, 22.3), ("ent01_s43", 124.69, 41.9),
                           ("ent01_s44", 117.17, 45.4)):
    check(f"{name} 省下 %", (1 - val / expert) * 100, claimed, tol=0.01)
mean_saving = sum((1 - v / expert) * 100 for v in (166.73, 124.69, 117.17)) / 3
check("三种子均值 省下 %", mean_saving, 36.5, tol=0.02)

print()
print("=" * 84)
if FAILS:
    print(f"FAIL —— {len(FAILS)} 项对不上：")
    for f in FAILS:
        print("  ✗ " + f)
    raise SystemExit(1)
print("PASS —— 全部阈值现算与手抄一致")
print("=" * 84)
