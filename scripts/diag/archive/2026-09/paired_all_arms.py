"""配对判读（验证种子 100–114 是共享的）。两种模式：

## 模式 A：`arms` —— 逐臂 vs 一个对照臂

对每个臂取指定轮号的 eval，与对照臂做**逐验证种子配对差**，
报 Δ、SD、SE、t、df、临界值、同向种子数。

⚠ 这个模式的对照是**另一个训练种子** ⟹ 训练种子噪声混在 Δ 里，
   它回答的是"这条臂**这一份权重**和对照差多少"，不是"这个**改动**有没有用"。
   要回答后者用模式 B。

## 模式 B：`train-pairs` —— 改动有没有用（**正确的检验**）

`--treat v1_onpath --ctrl ent01_rerun` ⟹ 找 `v1_onpath_s<K>` 与
`ent01_rerun_s<K>` 在 **K 相同**时配成一对（**同种子 ⟹ 训练种子噪声抵消**），
每对用 15 个验证种子的均值，再对 K 做配对 t（n=5 ⟹ df=4 ⟹ 临界 2.776）。

★ 独立重复的单位是**训练种子**，不是验证种子。同一个训练种子上的
  15 个验证种子是**同一个策略**的 15 次测量，不是 15 次独立重复。

## 两条硬规矩

★ `--at-u` 强制两侧在**同一轮号**取点。默认取最后一次 eval ⟹ 续跑臂
  （hist32_v3_s43 到 u=53、s44 到 u=76）会拿晚轮去比早轮，是
  [[window-aggregation-must-align-both-sides]] 那个坑。
★ df 必须现算，不许 `.get(df, 2.0)` 兜底（那会制造假显著）。
  未知 df **抛错**，不许回退成默认值。
"""
import argparse
import json
import sys
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")

# 主臂集（u=30 的完整一批 + 续跑臂）
ARMS = [
    "ent01_rerun_s42", "ent01_rerun_s43", "ent01_rerun_s44",
    "ent01_rerun_s45", "ent01_rerun_s46",
    "v1_onpath_s42", "v1_onpath_s43", "v1_onpath_s44",
    "v1_onpath_s45", "v1_onpath_s46",
    "v2_gelu_s42", "v2_gelu_s43", "v2_gelu_s44",
    "hist32_v3_s42", "hist32_v3_s43", "hist32_v3_s44",
]
BASE = "ent01_rerun_s42"

# t 分布临界值（双侧 95%）。**必须现查，不许兜底默认值。**
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
       13: 2.160, 14: 2.145, 15: 2.131}


def evals(path):
    """返回 [(更新轮号, per_seed_success), ...]。轮号只能取**前面最近**的 update 行。

    `eval_validation` 行**没有 update 键** ⟹ 不能从行自身读轮号，
    也不能按行数推（续跑段是续着编且追加的）。
    """
    if not path.exists():
        return []
    out = []
    last_u = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "update" in d:
            last_u = d["update"]
        ev = d.get("eval_validation")
        if isinstance(ev, dict):
            ps = ev.get("per_seed_success")
            if not isinstance(ps, list) or not ps:
                raise RuntimeError(f"{path.parent.name}: eval 记录缺 per_seed_success")
            out.append((last_u, [float(x) for x in ps]))
    return out


def pick(recs, at_u, name):
    """按 --at-u 取点；at_u=None 取最后一次。取不到就抛错，不许静默退让。"""
    if not recs:
        return None, None
    if at_u is None:
        return recs[-1]
    hit = [r for r in recs if r[0] == at_u]
    if not hit:
        return None, None
    if len(hit) > 1:
        raise RuntimeError(f"{name}: u={at_u} 有 {len(hit)} 条 eval ⟹ 不许猜")
    return hit[0]


def paired(diffs, label, crit_lookup=True):
    n = len(diffs)
    m = sum(diffs) / n
    if n < 2:
        return m, float("nan"), float("nan"), float("nan"), n - 1, float("nan"), 0
    var = sum((x - m) ** 2 for x in diffs) / (n - 1)
    sd = var ** 0.5
    se = sd / n ** 0.5
    t = m / se if se > 0 else float("inf")
    df = n - 1
    if crit_lookup and df not in T95:
        raise RuntimeError(f"{label}: df={df} 不在临界值表里 ⟹ 不许兜底")
    crit = T95.get(df, float("nan"))
    npos = sum(1 for x in diffs if x > 0)
    return m, sd, se, t, df, crit, npos


def mode_arms(args):
    data = {}
    for a in dict.fromkeys([BASE] + args.arms):
        recs = evals(OUT / a / "metrics.jsonl")
        u, ps = pick(recs, args.at_u, a)
        if ps is None:
            why = (f"无 u={args.at_u} 的 eval" if recs else "无 eval_validation")
            print(f"  （跳过 {a}：{why}）")
            continue
        data[a] = (u, ps)

    if BASE not in data:
        print(f"✗ 对照臂 {BASE} 在 u={args.at_u or '最后'} 没有 eval ⟹ 无法配对 ⟹ 退出")
        return 2

    bu, bp = data[BASE]
    n = len(bp)
    print("=" * 104)
    print(f"【模式 A】逐臂 vs 对照   对照 = {BASE}（u={bu}，均值 {sum(bp)/n:.4f}，n={n}）")
    print(f"  取点：{'最后一次 eval' if args.at_u is None else f'u={args.at_u}'}")
    print("  ⚠ 对照是**另一个训练种子** ⟹ Δ 里含训练种子噪声。"
          "问「改动有没有用」请用模式 B。")
    print("=" * 104)
    print(f"{'臂':<22}{'u':>4}{'均值':>9}{'Δ vs 对照':>11}{'SD':>8}{'SE':>8}"
          f"{'t':>8}{'df':>4}{'临界':>7}{'同向':>7}{'结论':>10}")
    print("-" * 104)
    for a, (u, ps) in sorted(data.items(), key=lambda kv: -sum(kv[1][1]) / len(kv[1][1])):
        if a == BASE:
            print(f"{a:<22}{u:>4}{sum(ps)/len(ps):>9.4f}"
                  f"{'—':>11}{'—':>8}{'—':>8}{'—':>8}{'—':>4}{'—':>7}{'—':>7}{'对照':>10}")
            continue
        if len(ps) != n:
            print(f"{a:<22}  人口不一致（{len(ps)} vs {n}）⟹ 跳过")
            continue
        d = [x - y for x, y in zip(ps, bp)]
        m, sd, se, t, df, crit, npos = paired(d, a)
        verdict = "显著" if abs(t) >= crit else "测不出"
        print(f"{a:<22}{u:>4}{sum(ps)/len(ps):>9.4f}"
              f"{m:>+11.4f}{sd:>8.4f}{se:>8.4f}{t:>+8.2f}{df:>4}{crit:>7.3f}"
              f"{npos:>4}/{n}{verdict:>10}")
    print("-" * 104)
    return 0


def mode_train_pairs(args):
    """同训练种子配对 —— 回答"这个改动有没有用"。"""
    treat, ctrl = args.treat, args.ctrl
    seeds = []
    for s in range(0, 1000):
        if ((OUT / f"{treat}_s{s}" / "metrics.jsonl").exists()
                and (OUT / f"{ctrl}_s{s}" / "metrics.jsonl").exists()):
            seeds.append(s)

    print("=" * 104)
    print(f"【模式 B】同训练种子配对   {treat}  vs  {ctrl}")
    print(f"  取点：{'最后一次 eval' if args.at_u is None else f'u={args.at_u}'}"
          f"   独立重复单位 = 训练种子（不是验证种子）")
    print("=" * 104)
    if not seeds:
        print("✗ 没有同种子的可配对臂 ⟹ 退出")
        return 2

    rows = []
    for s in seeds:
        tu, tp = pick(evals(OUT / f"{treat}_s{s}" / "metrics.jsonl"), args.at_u, f"{treat}_s{s}")
        cu, cp = pick(evals(OUT / f"{ctrl}_s{s}" / "metrics.jsonl"), args.at_u, f"{ctrl}_s{s}")
        if tp is None or cp is None:
            print(f"  （跳过 s{s}：u 取不到 treat={tu} ctrl={cu}）")
            continue
        if len(tp) != len(cp):
            print(f"  ★ 跳过 s{s}：人口不一致（{len(tp)} vs {len(cp)}）")
            continue
        if tu != cu:
            print(f"  ★ 跳过 s{s}：轮号不对齐（{tu} vs {cu}）⟹ 窗口必须两侧一致")
            continue
        rows.append((s, tu, sum(tp) / len(tp), sum(cp) / len(cp)))

    if len(rows) < 2:
        print(f"✗ 只有 {len(rows)} 对 ⟹ 无法做配对 t ⟹ 退出")
        return 2

    print(f"{'种子':>5}{'u':>5}{'处理均值':>11}{'对照均值':>11}{'Δ(同种子)':>12}")
    print("-" * 104)
    for s, u, tm, cm in rows:
        print(f"{s:>5}{u:>5}{tm:>11.4f}{cm:>11.4f}{tm - cm:>+12.4f}")
    print("-" * 104)

    diffs = [tm - cm for _, _, tm, cm in rows]
    m, sd, se, t, df, crit, npos = paired(diffs, f"{treat} vs {ctrl}")
    print(f"  Δ 均值 {m:+.4f}   SD {sd:.4f}   SE {se:.4f}   t = {t:+.3f}   "
          f"df = {df}   临界 {crit:.3f}   同向 {npos}/{len(rows)}")
    # 汇总均值也要报（审稿人会看）
    print(f"  {treat} 汇总均值 {sum(r[2] for r in rows)/len(rows):.4f}   "
          f"{ctrl} 汇总均值 {sum(r[3] for r in rows)/len(rows):.4f}")
    verdict = "显著" if abs(t) >= crit else "测不出"
    print(f"  结论：{verdict}（|t| {'≥' if abs(t) >= crit else '<'} {crit:.3f}）")
    if npos == len(rows) and verdict == "测不出":
        print(f"  ⚠ {npos}/{len(rows)} 同向但未过线 ⟹ 方向一致、**判别力不足**，"
              f"只能说「未测出」，不许说「有效」")
    print()
    print(f"DECISION=train_pairs   n={len(rows)}  delta={m:+.6f}  t={t:+.3f}  verdict={'PASS' if abs(t)>=crit and m>0 else 'NO'}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["arms", "train-pairs"], default="arms")
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--at-u", type=int, default=None,
                    help="在指定轮号取 eval（两侧对齐）；默认取最后一次")
    ap.add_argument("--treat", default="v1_onpath")
    ap.add_argument("--ctrl", default="ent01_rerun")
    args = ap.parse_args()
    return mode_arms(args) if args.mode == "arms" else mode_train_pairs(args)


if __name__ == "__main__":
    raise SystemExit(main())
