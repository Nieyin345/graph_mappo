"""pm_decode 五臂的完整对照表：训练侧 + 验证侧 + 与专家。

判读要点：
  · pm_decode 是唯一变量 = action_resolver.mode（mutual_choice → priority_matching）
  · 它**没有** BC 暖启动损失（结构没变 ⟹ critic 不重置）
  · 需要看：s46 是"这条种子本身差"还是"训练发散"
"""
import json
import statistics
import math
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
EXPERT = OUT / "eval" / "expert_seeds100_240.json"
T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 14: 2.145}


def valpts(run):
    """{update: (seeds, per_seed)} —— eval_validation 是独立行，轮号取前一个 update"""
    p = OUT / run / "metrics.jsonl"
    out = {}
    last = None
    if not p.exists():
        return out
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
        if "eval_validation" in o:
            ev = o["eval_validation"]
            out[int(last)] = (ev["seeds"], ev["per_seed_success"])
    return out


def trained(run):
    """{update: (kl, entropy, train_sr)}"""
    p = OUT / run / "metrics.jsonl"
    out = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "update" in o:
            out[o["update"]] = (o.get("kl"), o.get("entropy"), o.get("success_rate"))
    return out


def main():
    exp = json.loads(EXPERT.read_text(encoding="utf-8"))
    e_seeds = [int(s) for s in exp["seeds"]]
    e_vals = [float(x) for x in (exp.get("success") or exp["per_seed_success"])]

    print("=" * 96)
    print("pm_decode（解码器消融）—— 完整对照")
    print("=" * 96)
    print(f"\n专家（种子 100-114）= {statistics.mean(e_vals):.6f}\n")

    arms = [f"pm_decode_s{s}" for s in (42, 43, 44, 45, 46)]
    ctrls = [f"v2_bottleneck_s{s}" for s in (42, 43, 44, 45, 46)]

    print(f"{'臂':<18}{'u5':>9}{'u15':>9}{'u25':>9}{'u30':>9}   {'u30 vs 对照':>13}")
    print("-" * 96)
    acc = []
    for a, c in zip(arms, ctrls):
        va, vc = valpts(a), valpts(c)
        if not va:
            print(f"{a:<18}  (无验证点)")
            continue
        us = sorted(va)
        row = []
        for u in (5, 15, 25, 30):
            row.append(statistics.mean(va[u][1]) if u in va else float("nan"))
        d = None
        if 30 in va and 30 in vc:
            if va[30][0] == vc[30][0]:
                d = statistics.mean(va[30][1]) - statistics.mean(vc[30][1])
            else:
                d = float("nan")
        print(f"{a:<18}" + "".join(f"{x:>9.4f}" for x in row) +
              f"   {d:>+13.4f}" if d is not None else "")
        if d is not None and not math.isnan(d):
            acc.append(d)

    if len(acc) >= 2:
        m = statistics.mean(acc)
        sd = statistics.stdev(acc)
        se = sd / math.sqrt(len(acc))
        crit = T.get(len(acc) - 1)
        print(f"\n  Δ 均值 = {m:+.6f}   SD = {sd:.4f}   SE = {se:.4f}")
        print(f"  t = {m/se:+.3f}  (df={len(acc)-1}, 临界 {crit})")

    # 训练侧：s46 是发散还是本来就差？
    print("\n" + "=" * 96)
    print("训练侧体征（区分「发散」与「这条种子本来就差」）")
    print("=" * 96)
    print(f"\n{'臂':<18}{'u=kl':>10}{'u=ent':>9}{'train_sr u1':>13}{'u30':>10}")
    print("-" * 96)
    for a in arms:
        tr = trained(a)
        if not tr:
            continue
        us = sorted(tr)
        first, last = us[0], us[-1]
        kl1, ent1, sr1 = tr[first]
        kl2, ent2, sr2 = tr[last]
        print(f"{a:<18}{kl1:>10.5f}{ent1:>9.3f}{sr1:>13.4f}{sr2:>10.4f}")


if __name__ == "__main__":
    main()
