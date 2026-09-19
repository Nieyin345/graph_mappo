"""复核 w329p：判读时用的到底是哪个 update 的 eval？

背景（2026-09-19 发现的错）：`/tmp/wait_wave2.log` 的达标判据数的是
`metrics.jsonl` 的**行数**，而 eval 行是独立行 ⟹ 30 轮训练实际产生 36 行。
于是 s44 在只有 **27 轮**时就报了"达标"，`/tmp/w329p.go` 提前落盘，
已写进 `docs/训练诊断记录.md` 的判读用的是一个**没跑完的**值。

本脚本不猜：把每臂**每个 update** 的验证成功率逐行打出来，
再在**两侧都有**的 update 上重做配对判读，看结论对窗口是否敏感。

用法（服务器上）：
    /opt/qkd/venv/bin/python scripts/diag/probe_w329p_recheck.py
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
SEEDS = [42, 43, 44]
ARMS = {"对照 ent01_t8": "ent01_t8_s{}", "实验 w329p": "w329p_s{}"}

# n=3, df=2 双侧临界值（本项目已在别处栽过一次 df=2 用 2.0 的错）
T_CRIT_DF2 = 4.303


def load_eval_by_update(path: Path) -> dict[int, dict]:
    """返回 {update轮次: eval_validation 字典}。

    ★ 行结构是**实测**的，不是假设的：训练行有 `update`，eval 行是
      **独立行**、只有 `eval_validation` 键、**没有** `update` 字段。
      所以要用"最近一次见到的 update"给 eval 行归属轮次。
    """
    out: dict[int, dict] = {}
    n = 0
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = r.get("eval_validation")
        if isinstance(ev, dict):
            if n:
                out[n] = ev          # 同一轮若有多条 eval，后者覆盖前者
        elif isinstance(r.get("update"), int):
            n = max(n, int(r["update"]))
    return out


def summarize(key: str, ev: dict) -> tuple[float, int]:
    """取 eval 行里那个 key 的均值，返回 (均值, 种子数)。"""
    v = ev.get(key)
    if isinstance(v, list) and v:
        xs = [float(x) for x in v if isinstance(x, (int, float))]
        if xs:
            return sum(xs) / len(xs), len(xs)
    if isinstance(v, (int, float)):
        return float(v), 1
    return float("nan"), 0


def paired(deltas: list[float]) -> str:
    n = len(deltas)
    if n < 2:
        return f"n={n}，不足以判读"
    m = sum(deltas) / n
    var = sum((d - m) ** 2 for d in deltas) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    t = m / se if se > 0 else float("inf")
    df = n - 1
    crit = T_CRIT_DF2 if df == 2 else float("nan")
    verdict = "可测" if abs(t) > crit else "测不出"
    return (f"Δ={m:+.4f}  SD={sd:.4f}  SE={se:.4f}  t={t:+.3f}  "
            f"(df={df}, 临界 {crit})  ⟹ {verdict}")


def main() -> int:
    data: dict[str, dict[int, dict]] = {}
    for arm, fmt in ARMS.items():
        data[arm] = {}
        for s in SEEDS:
            run = fmt.format(s)
            data[arm][s] = load_eval_by_update(OUT / run / "metrics.jsonl")
            ups = sorted(data[arm][s])
            print(f"  {run:16s} 有 eval 的轮次: {ups}")
        print()

    # ---- 1. 每臂逐轮验证成功率 ----
    print("=" * 78)
    print("① 逐轮验证成功率（key = success_rate 类；把全部数值键都打出来对一遍）")
    print("=" * 78)
    for arm in ARMS:
        for s in SEEDS:
            d = data[arm][s]
            if not d:
                print(f"  {arm} s{s}: （无 eval 行）")
                continue
            ks = sorted(d[max(d)])
            print(f"  {arm} s{s}: 可用键 {ks}")
        print()

    # 选主指标：优先 success_rate，否则退回第一个数值键
    primary = None
    for arm in ARMS:
        for s in SEEDS:
            for ev in data[arm][s].values():
                for k, v in ev.items():
                    if "success" in k.lower():
                        primary = k
                        break
                if primary:
                    break
            if primary:
                break
    if not primary:
        print("  ✗ 找不到含 success 的键，终止")
        return 1
    print(f"  主指标取：{primary}\n")

    print("=" * 78)
    print("② 每臂逐轮（主指标均值）")
    print("=" * 78)
    allups = sorted({u for arm in ARMS for s in SEEDS for u in data[arm][s]})
    hdr = f"  {'update':>6s}" + "".join(f"{arm:>22s}" for arm in ARMS)
    print(hdr)
    for u in allups:
        row = f"  {u:>6d}"
        for arm in ARMS:
            vals = []
            for s in SEEDS:
                ev = data[arm][s].get(u)
                if ev:
                    vals.append(summarize(primary, ev)[0])
            row += f"{('%.4f (n=%d)' % (sum(vals) / len(vals), len(vals))) if vals else '—':>22s}"
        print(row)

    # ---- 3. 配对判读：只在该 update 三臂都有的地方做 ----
    print()
    print("=" * 78)
    print("③ 配对判读（Δ = 实验 − 对照，逐种子）")
    print("=" * 78)
    ctrl, exp = "对照 ent01_t8", "实验 w329p"
    for u in allups:
        deltas, rows = [], []
        for s in SEEDS:
            ec, ee = data[ctrl][s].get(u), data[exp][s].get(u)
            if ec and ee:
                a = summarize(primary, ec)[0]
                b = summarize(primary, ee)[0]
                deltas.append(b - a)
                rows.append(f"      s{s}: 对照 {a:.4f}  实验 {b:.4f}  Δ={b - a:+.4f}")
        if not deltas:
            continue
        print(f"  ── update {u} ──")
        for r in rows:
            print(r)
        print(f"    {paired(deltas)}")
        if len(deltas) < 3:
            print(f"    ⚠ 只有 {len(deltas)} 个种子配对（该轮有臂还没跑到）")
        print()

    # ---- 4. 末轮对比：文档里写的值到底来自哪一轮 ----
    print("=" * 78)
    print("④ 各臂可用的最后一轮（判断「跑满 30 轮」是否成立）")
    print("=" * 78)
    for arm in ARMS:
        for s in SEEDS:
            d = data[arm][s]
            last = max(d) if d else None
            v = f"{summarize(primary, d[last])[0]:.4f}" if last else "—"
            flag = "" if last == 30 else "   ⚠ 未到 30"
            print(f"  {arm} s{s}: 最后的 eval 在 update {last}  值 {v}{flag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
