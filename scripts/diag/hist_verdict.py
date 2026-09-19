# -*- coding: utf-8 -*-
"""hist 判读：时序编码器（只开节点通道）vs 本节点干净对照 ent01_rerun。

### 判据（2026-09-20 定版，见 configs/train_hist.yaml 文件头）
加载器（commit 7e1ee69）落地后，臂与对照**都从同一个 BC 策略出发**（u1 逐位相同），
所以这是**干净的单变量配对**：

  · Δ ≥ +0.035  ⟹ 时序信息**有用**，值得加深
  · |Δ| < 0.035 ⟹ n=3 分辨率下**测不出**（不是"无用"）
  · Δ ≤ −0.035  ⟹ 时序编码器**有害**

⚠ n=3（df=2）临界值 **4.303**，不是 2。报 t 但以 Δ 的阈值为主判据。

### 必须一起看的两项
1. **u1 守卫**：hist 与 ent01_rerun 的 u1 必须**逐位相同**（同 BC、同种子、同节点）。
   不同 ⟹ 加载器没做到承诺的等价 ⟹ 结果不可信。
2. **吞吐代价**：LSTM 每步对 90 节点跑一次 ⟹ `rollout_s`/`update_s` 可能显著变长。
   本实验是"**值不值得继续投入**"的判断，变慢必须算进去。
"""
import json
import math
import os
import statistics as st

R = "/opt/qkd/graph_mappo/outputs"
CTRL = "ent01_rerun"
ARM = "hist"
SEEDS = (42, 43, 44)
CRIT3 = 4.303
EXPERT = 0.6979220689
T_LO, T_HI = -0.035, 0.035


def read(a, s):
    """返回 [(update, per_seed_success, rollout_s, update_s)]"""
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
    """平台窗口 (u25,u30) 的逐请求种子均值。窗口必须**两侧都命中**，否则返回 None。"""
    d = {u: p for u, p, _, _ in rows}
    if 25 not in d or 30 not in d or len(d[25]) != len(d[30]):
        return None
    return [st.mean([d[25][i], d[30][i]]) for i in range(len(d[25]))]


def paired(Arows, Crows):
    """逐请求种子配对，跨训练种子求 t。返回 (Δ, SD, t, 逐种子Δ, nc) 或 None。"""
    Aps, Cps = plat_ps(Arows), plat_ps(Crows)
    if Aps is None or Cps is None or len(Aps) != len(Cps):
        return None
    nc = len(Aps)
    ds = [st.mean([Aps[i] - Cps[i] for i in range(nc)])]
    return ds, nc


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
print("hist 判读：时序编码器（只开节点通道）  对照 = %s（本节点、8 线程、同 BC）" % CTRL)
print("=" * 74)

# ---- 0. u1 守卫 ----
print("\n### 0. ★ u1 守卫（同 BC + 同种子 + 同节点 ⟹ 必须逐位相同）")
guard_ok = True
for s in SEEDS:
    a, b = u1_of("%s_s%d" % (ARM, s)), u1_of("%s_s%d" % (CTRL, s))
    if a is None or b is None:
        print("   s%d  hist=%s  ctrl=%s  （数据缺）" % (s, a, b)); guard_ok = False; continue
    d = a - b
    ok = abs(d) < 1e-9
    guard_ok = guard_ok and ok
    print("   s%d  hist=%.12f  ctrl=%.12f  Δ=%.2e  %s"
          % (s, a, b, d, "✓ 逐位相同" if ok else "✗ **不相同**"))
print("   ⟹ %s" % ("暖启动确实保住了，配对干净" if guard_ok
                    else "**加载器没做到承诺的等价，下面结果不可信**"))

# ---- 1. 主判据 ----
print("\n### 1. 主判据：平台 u25/u30，逐请求种子配对")
avail = [s for s in SEEDS if read(CTRL, s) and read(ARM, s)]
print("   可用训练种子：%s" % avail)
if len(avail) < 2:
    print("   !! 数据不足，无法判读")
    raise SystemExit(1)

per_seed = []
nc = None
for s in avail:
    r = paired(read(ARM, s), read(CTRL, s))
    if r is None:
        print("   s%d：平台窗口不全（对照或臂缺 u25/u30）" % s); continue
    (ds, n), = [r]
    per_seed.append(ds[0]); nc = n
if not per_seed:
    print("   !! 没有任何种子两侧都命中平台窗口")
    raise SystemExit(1)

m = st.mean(per_seed)
sd = st.stdev(per_seed) if len(per_seed) > 1 else float("nan")
se = sd / math.sqrt(len(per_seed)) if len(per_seed) > 1 else float("nan")
t = m / se if se and se == se and se != 0 else float("nan")
df = len(per_seed) - 1
crit = CRIT3 if df == 2 else (2.776 if df == 4 else 2.145)
print("   n=%d 训练种子 × %d 请求种子   df=%d  临界值=%.3f" % (len(per_seed), nc, df, crit))
print("   Δ = %+.4f   SD(训练种子) = %.4f   SE = %.4f   t = %+.2f  %s"
      % (m, sd, se, t, "★ 过线" if abs(t) > crit else "未过线"))
print("   逐训练种子 Δ: %s" % " ".join("%+.4f" % x for x in per_seed))
print()
if m >= T_HI:
    verd = "★ 时序信息**有用**（Δ ≥ +0.035）⟹ 值得加深（多通道 / 接边）"
elif m <= T_LO:
    verd = "★ 时序编码器**有害**（Δ ≤ −0.035）"
else:
    verd = "**测不出**（|Δ| < 0.035）—— n=3 分辨率下的正确读法，不是「无用」"
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

# ---- 5. ★ 吞吐代价 ----
print("\n### 5. ★ 吞吐代价（LSTM 变慢必须算进「值不值」）")
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
rc, uc = [], []
ra, ua = [], []
for s in avail:
    c = [x for x in read(CTRL, s) if x[2] and x[3]]
    a_ = [x for x in read(ARM, s) if x[2] and x[3]]
    if c and a_:
        rc.append(st.mean([x[2] for x in c])); uc.append(st.mean([x[3] for x in c]))
        ra.append(st.mean([x[2] for x in a_])); ua.append(st.mean([x[3] for x in a_]))
if rc and ra:
    print("   ⟹ 臂/对照 倍数：rollout %.2fx  update %.2fx  整轮 %.2fx"
          % (st.mean(ra) / st.mean(rc), st.mean(ua) / st.mean(uc),
             (st.mean(ra) + st.mean(ua)) / (st.mean(rc) + st.mean(uc))))
    print("   ⚠ 变慢 ⟹ 并发数要下调 ⟹ **总吞吐**下降，这一点必须与 Δ 一起权衡")
print()
print("=" * 74)
