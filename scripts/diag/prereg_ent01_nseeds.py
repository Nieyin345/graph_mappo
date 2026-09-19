#!/usr/bin/env python
"""**预注册**：要几个训练种子才能判定 `ent01` 是否真的超过专家？

### 起因

订正后的结论是：ent01 相对专家 Δ≈+0.0199（u25/u30 平台），
但在 n=3（df=2，临界值 4.303）下 t=3.46、**p=0.0743，未达显著**。
（原写 0.106，复现不出来，2026-09-19 订正；见 docs/训练诊断记录.md 同节。）

⚠ **本脚本曾被一个「静默返回 0」的 bug 骗过**：手写不完全贝塔的 Lentz
连分式，分母初值必须是 `d = 0.0`。我写成 `d = 1.0`，导致 i=0 就满足
收敛判据 `|1 - d*c| < 1e-14`（0.5*2=1），函数**立刻返回 0**。后果是
t_crit 二分收敛到 0、p 收敛到 0，「未达显著」被读成「显著」——一个
恰好把结论**翻向反面**的静默错误。现在 `_betainc` 有 `_selfcheck()`
与 scipy 交叉验证，并断言 df=2 的临界值确实是 4.303。

这不是"没有效应"，是**功效不够**。所以问题是：**再加几个种子才算够？**
回答它必须用**已有的 n=3 数据**去算，而不是跑完再看（跑完再看就是 p-hacking）。

### 这个比较的 SD 与 power_check.py 那个不是同一个

- `power_check.py` 算的是 **arm_s − ent01_s**（两者都随种子变）
  → SD(diff) 里 ρ 很重要。
- 这里算的是 **ent01_s − 专家**，**专家是一个固定的数**
  → SD(Δ_s) = SD(ent01_s)，与 ρ 无关。

所以直接用 ent01 三种子在平台区的散布：
  u25/u30 的逐种子平台均值 = +0.0084 / +0.0256 / +0.0256 → SD = 0.0099

### 预注册内容（**跑之前**写死；下面的 n 由本脚本第 2 节算出并打印）

- 加种子 **45、46**（与 42/43/44 不重复），同配置、同 BC、同 30 轮。
- 判据：**合并 n=5**，取 u25/u30 平台均值，单样本 t 检验，
  df=4，**临界值 2.776**，双侧 **p<0.05** 判"显著超过专家"。
- **无论方向如何都照实报**：若合并后不显著，结论就是"未测出"，
  不许再补种子直到显著（那正是 p-hacking）。

用法（服务器上）：python scripts/diag/prereg_ent01_nseeds.py
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
EXPERT = OUT / "eval" / "expert_seeds100_240.json"

# 已有三种子（用于估 SD，**不用于定结论**）
OLD = [42, 43, 44]
PLATEAU = (25, 30)


def t_crit(df: int, p: float = 0.975) -> float:
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _t_sf(mid, df) > 1.0 - p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _t_sf(t: float, df: int) -> float:
    if t == 0:
        return 0.5
    x = df / (df + t * t)
    half = 0.5 * _betainc(df / 2.0, 0.5, x)
    return half if t > 0 else 1.0 - half


def _betainc(a: float, b: float, x: float) -> float:
    """手写不完全贝塔（Lentz 连分式）。

    ⚠ **d 的初值必须是 0.0**。写成 1.0 时 i=0 就有 d*c = 0.5*2 = 1，
    收敛判据 |1-d*c| < 1e-14 立刻成立，函数静默返回 0 —— 于是 t_crit
    收敛到 0、p 收敛到 0，把「未达显著」读成「显著」。已在另一个脚本里
    真实踩到过。下面的 _selfcheck() 就是防它的。
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    tiny = 1e-300
    f, c, d = 1.0, 1.0, 0.0          # ← d 不是 1.0
    for i in range(0, 400):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = tiny if abs(d) < tiny else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = tiny if abs(c) < tiny else c
        f *= d * c
        if abs(1.0 - d * c) < 1e-14:
            break
    return front * (f - 1.0)


def _selfcheck() -> None:
    """把手写实现钉在 scipy 上，并断言 df=2 的临界值就是 4.303。

    ⚠ 这里**不抄教科书常数**——第一版抄了 4 个，其中两个是错的：
    `12.7062` 是 df=1（不是 df=2）、`2.36462` 是 df=10（不是 df=8）。
    手抄的常数和手抄的 p 值一样会悄悄过期/串行。改成拿 scipy 当 oracle，
    没有 scipy 才退回教科书值。

    为什么值得这么较真：上面那个「静默返回 0」的 bug 会让本脚本**反向**
    下结论（未达显著 → 显著），而输出看起来完全正常。断言是唯一的护栏。
    """
    try:
        from scipy import stats as _st
        for df in (1, 2, 4, 8, 20):
            ref = float(_st.t.sf(_st.t.ppf(0.975, df), df))
            assert abs(_t_sf(float(_st.t.ppf(0.975, df)), df) - ref) < 1e-6, df
        crit = {2: 4.30265, 4: 2.77645}
        for df, ref in crit.items():
            got = t_crit(df)
            assert abs(got - ref) < 0.01, f"t_crit({df}) = {got}, 期望 {ref}"
        print("  （与 scipy 逐位对齐）")
    except ImportError:                      # 没 scipy 时退回教科书值
        for df, t, ref in ((2, 4.30265, 0.025), (4, 2.77645, 0.025),
                           (8, 2.30600, 0.025)):
            got = _t_sf(t, df)
            assert abs(got - ref) < 2e-4, f"t_sf({t},{df}) = {got}, 期望 ≈{ref}"
        assert abs(t_crit(2) - 4.303) < 0.01, f"t_crit(2) = {t_crit(2)}"
        assert abs(t_crit(4) - 2.776) < 0.01, f"t_crit(4) = {t_crit(4)}"



def power(c: float, ncp: float, df: int) -> float:
    try:
        from scipy import stats as _st
        return float(_st.nct.sf(c, df, ncp))
    except Exception:                                          # noqa: BLE001
        return 0.5 * math.erfc((c - ncp) / math.sqrt(2))


d = json.loads(EXPERT.read_text(encoding="utf-8"))
ex = dict(zip(d["seeds"], (float(x) for x in d["success"])))
seeds = sorted(ex)
base = [ex[s] for s in seeds]


def points(run: str) -> dict[int, list[float]]:
    p = OUT / run / "metrics.jsonl"
    out: dict[int, list[float]] = {}
    last = None
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "update" in o:
            last = o["update"]
        ev = o.get("eval_validation")
        if isinstance(ev, dict) and ev.get("per_seed_success") and last is not None:
            out[last] = [float(x) for x in ev["per_seed_success"]]
    return out


per_seed_old = []
for s in OLD:
    pts = points(f"ent01_s{s}")
    us = [statistics.mean([pts[u][i] - base[i] for i in range(len(seeds))])
          for u in PLATEAU if u in pts and len(pts[u]) == len(seeds)]
    if us:
        per_seed_old.append(statistics.mean(us))

m_old = statistics.mean(per_seed_old)
sd = statistics.stdev(per_seed_old)

_selfcheck()          # ← 先把手写 t 分布钉死，再拿它下结论
print("  [自检] 手写 t 分布已与 scipy 对齐；t_crit(2)=4.303 / t_crit(4)=2.776")
print()

print("=" * 74)
print("1. 用已有 n=3 数据估这个比较的尺度")
print("=" * 74)
for s, v in zip(OLD, per_seed_old):
    print(f"  ent01_s{s}: u25/u30 平台配对差 = {v:+.4f}")
print(f"  Δ = {m_old:+.4f}   SD(Δ_s) = {sd:.4f}   （专家是固定值 ⟹ 与 ρ 无关）")
_t_obs = m_old / (sd / math.sqrt(len(per_seed_old)))
_p_obs = 2 * _t_sf(abs(_t_obs), len(per_seed_old) - 1)
print(f"  n=3 现状：t = {_t_obs:+.3f} (df=2)，双侧 p = {_p_obs:.4f}"
      f"  → {'显著' if _p_obs < 0.05 else '**未达显著**'}")
print(f"  （这个 p 是**现算**的，不是抄的：抄来的数字会悄悄过期，"
      f"本项目已因此订正过两次）")
print()

print("=" * 74)
print("2. 功效 → 需要几个种子")
print("=" * 74)
print(f"  {'n':>3} {'SE':>8} {'ncp':>7} {'t_crit':>8} {'功效':>8}")
need = None
for n in range(3, 11):
    se = sd / math.sqrt(n)
    df, tc = n - 1, t_crit(n - 1)
    pw = power(tc, m_old / se, df)
    print(f"  {n:>3} {se:>8.4f} {m_old/se:>7.2f} {tc:>8.3f} {pw:>8.3f}"
          f"{'  ← 首次 ≥0.8' if pw >= 0.8 and need is None else ''}")
    if pw >= 0.8 and need is None:
        need = n
print()
print(f"  ⟹ 达到 0.8 功效需要 **n = {need}** 个训练种子 → 再加 **{need-3}** 个。")
print()
print("  ⚠ 这是用 n=3 估的 SD（约 50% 不确定度）。若真实 SD 更大，需要更多种子；")
print("    所以**预先写明**：若合并后仍未达显著，结论就是『未测出』，")
print("    不再追加种子——追加到显著为止是 p-hacking。")
print()

print("=" * 74)
print("3. 预注册判据（照此执行，不事后改）")
print("=" * 74)
df_f = need - 1
_new = [45, 46, 47, 48][:need - 3]
print(f"  新种子：**{', '.join(map(str, _new))}**（不与 {OLD} 重复）")
print(f"  配置：configs/train_ent01.yaml，同 BC checkpoint，30 轮")
print(f"  判据：合并 **n={need}**，u25/u30 平台均值，单样本 t 检验")
print(f"        df={df_f}，临界值 **{t_crit(df_f):.3f}**，双侧 **p < 0.05**")
print(f"  报法：无论方向，照实报。显著 → 说「超过」；不显著 → 说「未测出」。")
print(f"  预算：{need-3} 个种子 = 1 波，约 2h")

# =====================================================================
# 4. 合并读数 —— **含预先写好的选择性偏差处置**（2026-09-19 补）
#
# 为什么必须两读并报：预注册窗口 u25/u30 是用 42/43/44 这三条曲线**挑出来的**
# （见 docs/训练诊断记录.md「预注册的窗口本身就是选出来的」）。选点与评估用同一批
# 数据 ⟹ 42/43/44 那部分读数带选择效应，实测偏 **+0.0135**，占效应量的 68%。
#
# 45/46 是**新数据，没参与过选窗口**，所以它们是干净的那一半。
# 合并 n=5 = 3/5 有偏 + 2/5 干净 ⟹ **合并不消除偏差，只稀释**（约稀释到 60%）。
#
# ★ 这一节在 s45/s46 **跑完之前**就写好了。若合并后显著，**不许**只报合并读数
#   ——必须同时报「只用 45/46」的读数，哪怕 n=2 分辨率很差（df=1，临界值 12.706）。
#   先写下来才叫预注册；跑完再想就是事后找理由。
# =====================================================================
print()
print("=" * 74)
print("4. 合并读数（**含预先写好的选择性偏差处置**）")
print("=" * 74)

ALL = [42, 43, 44, 45, 46]
BIASED = [42, 43, 44]      # 参与过选窗口 → 有偏
CLEAN = [45, 46]           # 新数据 → 干净


def plateau_delta(seed: int):
    """该种子在 u25/u30 平台的配对差（对专家）。数据不够返回 None。"""
    pts = points(f"ent01_s{seed}")
    vals = [statistics.mean([pts[u][i] - base[i] for i in range(len(seeds))])
            for u in PLATEAU if u in pts and len(pts[u]) == len(seeds)]
    if not vals:
        return None
    return statistics.mean(vals)


def report(label: str, vals: list[float]) -> None:
    if len(vals) < 2:
        print(f"  {label:<26} 数据不足（n={len(vals)}），跳过")
        return
    m = statistics.mean(vals)
    s = statistics.stdev(vals)
    df = len(vals) - 1
    tc = t_crit(df)
    t = m / (s / math.sqrt(len(vals)))
    p = 2 * _t_sf(abs(t), df)
    sig = "**显著**" if abs(t) > tc else "未达显著"
    print(f"  {label:<26} n={len(vals)}  Δ={m:+.4f}  SD={s:.4f}  "
          f"t={t:+.3f} (df={df}, 临界 {tc:.3f})  p={p:.4f}  → {sig}")


d_all = {s: plateau_delta(s) for s in ALL}
have = [s for s in ALL if d_all[s] is not None]
print(f"  已有平台读数的种子: {have}")
for s in ALL:
    v = d_all[s]
    tag = "（有偏：参与过选窗口）" if s in BIASED else "（干净：新数据）"
    print(f"    ent01_s{s}: {'—' if v is None else f'{v:+.4f}'}  {tag}")
print()

if len(have) >= 3:
    report("① 预注册：合并 n=5", [d_all[s] for s in ALL if d_all[s] is not None])

    # 只用干净种子 —— 这一步是**必须**的，不是可选的
    clean_v = [d_all[s] for s in CLEAN if d_all[s] is not None]
    biased_v = [d_all[s] for s in BIASED if d_all[s] is not None]
    report("② 只用干净种子 45/46", clean_v)
    if biased_v:
        report("③ 只用有偏种子 42/43/44", biased_v)

    print()
    print("  ★ 判读（**跑之前就定好的**）：")
    print("    · 若 ① 显著而 ② 方向不一致或远弱 → 效应量里很大一块是**窗口选择**，")
    print("      不能写成「ent01 超过专家」。")
    print("    · 若 ① 与 ② 同向且量级相当 → 选择效应不是主因，结论才站得住。")
    print("    · ② 的 n=2（df=1，临界值 12.706）**几乎没有功效**——它用来**证伪**，")
    print("      不是用来**证实**。方向相反就是硬信号；方向相同只是不矛盾。")
    print(f"    · 若 ① 不显著 → 结论就是「未测出」，**不追加种子到显著为止**（p-hacking）。")
else:
    print(f"  还没凑够（目前 {len(have)} 个种子有 u25/u30 读数）—— 等 s45/s46 跑完再跑本脚本。")
