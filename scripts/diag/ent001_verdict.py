# -*- coding: utf-8 -*-
"""ent001 判读：entropy_coef = 0.001 (A0) vs 本节点干净对照 ent01_rerun。

### 背景
`entropy_coef` 曾是「最强单参数线索」（0.01 优于 0.001），但 2026-09-20 复核发现
n=5「超过专家 p=0.0074」是**节点混淆**（五个种子整批跑在旧节点 amd238，
而本节点硬件偏置 +0.0197 ⟹ 去偏置后 −0.002）。三把刀见 `configs/train_ent001.yaml`。

### 判据（跑之前写死，见 train_ent001.yaml 文件头）
对照 = `ent01_rerun_s{42,43,44}`（本节点、8 线程、同一 BC 起点）
臂   = `ent001_s{42,43,44}`（BASE 与之逐项相同，**只差 entropy_coef**）

  · |Δ| < 0.035 ⟹ **「测不出」⟹ 正式关闭这条线索**
  · Δ ≥ +0.035 ⟹ 线索**复活**（0.001 明显更差）
  · Δ ≤ −0.035 ⟹ 0.001 反而更好，同样关闭「0.01 有效」的说法

⚠ n=3（df=2）临界值 **4.303**，不是 2。
"""
import json
import math
import os
import statistics as st

R = "/opt/qkd/graph_mappo/outputs"
CTRL = "ent01_rerun"
ARM = "ent001"
SEEDS = (42, 43, 44)
# ★ 临界值一律从唯一来源取（见 stats_crit.py）。
#   2026-09-20 修复：原先是
#       crit = CRIT3 if df == 2 else (2.776 if df == 4 else 2.145)
#   这条链**只覆盖 df=2 和 df=4**，其余一律落到 2.145（那是 **df=14** 的值）。
#   本次 ent001_s44 平台窗口不全 ⟹ df 降到 1 ⟹ 用 2.145 判 df=1
#   （真值 **12.706**）会让 |t| 在 2.145~12.706 之间时**假显著**。
#   本例 t=−0.69 离两条线都远，结论未受影响——**是运气不是设计**。
from stats_crit import t_crit  # noqa: E402

EXPERT = 0.6979220689
T_LO, T_HI = -0.035, 0.035


def read(a, s):
    p = "%s/%s_s%d/metrics.jsonl" % (R, a, s)
    out = []
    if not os.path.exists(p):
        return out
    last = None
    for ln in open(p, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        o = json.loads(ln)
        if "update" in o:
            last = o["update"]
        ev = o.get("eval_validation")
        if isinstance(ev, dict) and ev.get("per_seed_success") and last is not None:
            out.append((last, [float(x) for x in ev["per_seed_success"]],
                        o.get("rollout_s"), o.get("update_s")))
    return out


def plat_ps(rows):
    """平台窗口 (u25,u30) 的逐请求种子均值。窗口必须**两侧都命中**。"""
    d = {u: p for u, p, _, _ in rows}
    if 25 not in d or 30 not in d or len(d[25]) != len(d[30]):
        return None
    return [st.mean([d[25][i], d[30][i]]) for i in range(len(d[25]))]


def u1_of(run):
    p = "%s/%s/metrics.jsonl" % (R, run)
    try:
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                return json.loads(ln).get("mean_success_rate")
    except OSError:
        pass
    return None


print("=" * 74)
print("ent001 判读：entropy_coef 0.001 vs 0.01   对照 = %s（本节点、8 线程、同 BC）" % CTRL)
print("=" * 74)

# ---- 0. u1 守卫 ----
print("\n### 0. ★ u1 守卫（同 BC + 同种子 + 同节点 ⟹ 必须逐位相同）")
print("    注：ent001 是**纯配置改动**（只改 entropy_coef），不碰任何权重形状，")
print("    所以这条守卫**必须**通过；不通过说明配置链没按预期生效。")
guard_ok = True
for s in SEEDS:
    a, b = u1_of("%s_s%d" % (ARM, s)), u1_of("%s_s%d" % (CTRL, s))
    if a is None or b is None:
        print("   s%d  ent001=%s  ctrl=%s  （数据缺）" % (s, a, b)); guard_ok = False; continue
    d = a - b
    ok = abs(d) < 1e-9
    guard_ok = guard_ok and ok
    print("   s%d  ent001=%.12f  ctrl=%.12f  Δ=%.2e  %s"
          % (s, a, b, d, "✓ 逐位相同" if ok else "✗ **不相同**"))
print("   ⟹ %s" % ("配置链生效、起点相同，配对干净" if guard_ok
                    else "**配置链没按预期生效，先查 --configs 顺序**"))

# ---- 1. 主判据 ----
print("\n### 1. 主判据：平台 u25/u30，逐请求种子配对")
avail = [s for s in SEEDS if read(CTRL, s) and read(ARM, s)]
print("   可用训练种子：%s" % avail)
if len(avail) < 2:
    print("   !! 数据不足（需要两侧都有 u25 与 u30），无法判读")
    raise SystemExit(1)

per_seed = []
nc = None
for s in avail:
    Aps, Cps = plat_ps(read(ARM, s)), plat_ps(read(CTRL, s))
    if Aps is None or Cps is None or len(Aps) != len(Cps):
        print("   s%d：平台窗口不全（对照或臂缺 u25/u30）" % s); continue
    nc = len(Aps)
    per_seed.append(st.mean([Aps[i] - Cps[i] for i in range(nc)]))
if not per_seed:
    print("   !! 没有任何种子两侧都命中平台窗口")
    raise SystemExit(1)

m = st.mean(per_seed)
sd = st.stdev(per_seed) if len(per_seed) > 1 else float("nan")
se = sd / math.sqrt(len(per_seed)) if len(per_seed) > 1 else float("nan")
t = m / se if se and se == se and se != 0 else float("nan")
df = len(per_seed) - 1
crit = t_crit(df)     # 未知 df 会抛错，不会静默回退成宽松值
print("   n=%d 训练种子 × %d 请求种子   df=%d  临界值=%.3f" % (len(per_seed), nc, df, crit))
print("   Δ = %+.4f   SD(训练种子) = %.4f   SE = %.4f   t = %+.2f  %s"
      % (m, sd, se, t, "★ 过线" if abs(t) > crit else "未过线"))
print("   逐训练种子 Δ: %s" % " ".join("%+.4f" % x for x in per_seed))
print()
if m >= T_HI:
    verd = "★ 0.001 **明显更差**（Δ ≥ +0.035）⟹ 线索**复活**"
elif m <= T_LO:
    verd = "★ 0.001 反而更好（Δ ≤ −0.035）⟹ 同样关闭「0.01 有效」"
else:
    verd = "**测不出**（|Δ| < 0.035）⟹ 按预注册**正式关闭 entropy_coef 这条线索**"
print("   ⟹ 判据：%s" % verd)

# ---- 2. 逐轮配对 Δ ----
print("\n### 2. 逐轮配对 Δ（逐请求种子配对后取均值）")
evs = [1, 5, 10, 15, 20, 25, 30]
print("   %-10s" % "臂" + "".join("%9s" % ("u%d" % u) for u in evs))
row = "   %-10s" % ARM
for u in evs:
    dd = []
    for s in avail:
        A = {x[0]: x[1] for x in read(ARM, s)}
        C = {x[0]: x[1] for x in read(CTRL, s)}
        if u in A and u in C and len(A[u]) == len(C[u]):
            dd.append(st.mean([A[u][i] - C[u][i] for i in range(len(A[u]))]))
    row += "%+9.4f" % st.mean(dd) if dd else "%9s" % "--"
print(row)

# ---- 3. 逐轮跨种子 SD ----
print("\n### 3. 逐轮跨种子 SD（紧 vs 散是形状信息，与均值同等重要）")
print("   %-10s" % "臂" + "".join("%9s" % ("u%d" % u) for u in evs))
for a in (CTRL, ARM):
    row = "   %-10s" % a
    for u in evs:
        vs = [st.mean(p) for s in avail for (uu, p, _, _) in read(a, s) if uu == u]
        row += "%9.4f" % st.stdev(vs) if len(vs) > 1 else "%9s" % "--"
    print(row)

# ---- 4. 绝对水平 vs 专家 ----
print("\n### 4. 绝对水平 vs 专家锚 %.4f" % EXPERT)
print("   ⚠ 臂 vs 专家**有节点偏置（+0.0197）不抵消**；臂 vs 臂才抵消。仅供定位。")
for a in (CTRL, ARM):
    vals = []
    for s in avail:
        A = read(a, s)
        if len(A) >= 2:
            vals.append(st.mean([st.mean(x[1]) for x in A[-2:]]))
    if vals:
        print("   %-10s 平台=%.4f  Δ(臂−专家)=%+.4f   逐种子 %s"
              % (a, st.mean(vals), st.mean(vals) - EXPERT,
                 " ".join("%.4f" % v for v in vals)))

# ---- 5. 吞吐（确认两臂可比）----
print("\n### 5. 吞吐（对照与臂应基本相同——只差一个系数，不该有速度差）")
print("   %-10s %10s %10s %10s" % ("臂", "rollout_s", "update_s", "合计"))
for a in (CTRL, ARM):
    rs, us = [], []
    for s in avail:
        rows = [x for x in read(a, s) if x[2] and x[3]]
        if rows:
            rs.append(st.mean([x[2] for x in rows]))
            us.append(st.mean([x[3] for x in rows]))
    if rs:
        print("   %-10s %10.1f %10.1f %10.1f" % (a, st.mean(rs), st.mean(us),
                                                 st.mean(rs) + st.mean(us)))
print()
print("=" * 74)
