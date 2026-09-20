"""把「BC 起点 / 专家 / RL u30」三者放进同一张**配对**表。

动机：BC 起点此前从未被测过。刚测出 = 0.653732。
本脚本用同一个判读函数把三条并列，逐种子配对，回答：

  ① BC 起点 vs 专家      —— RL 的**起点**离专家多远
  ② RL u30 vs BC 起点    —— **RL 训练本身**贡献了多少
  ③ RL u30 vs 专家        —— 最终是否超过专家（这是 H₁ 那一栏）

★ 关键是 ②：如果 ② 显著为正，说明 RL 训练**确实有用**（此前无人量过）；
  如果 ② ≈ 0，才轮到"训练无效"这个结论。
"""
import json
import math
import statistics
from pathlib import Path

REPO = Path("/opt/qkd/graph_mappo")
EXPERT = REPO / "outputs" / "eval" / "expert_seeds100_240.json"
BC = Path("/tmp/probe_bc_eval_bc_start.json")

T_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
           6: 2.447, 7: 2.365, 8: 2.306, 14: 2.145}


def val_points(run: str):
    """{update: per_seed_success} —— eval_validation 是独立行，无 update 键，
    轮号取前面最近的 update 行（记忆 eval-update-number-not-from-position）。"""
    p = REPO / "outputs" / run / "metrics.jsonl"
    out = {}
    if not p.exists():
        return out
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
        if "eval_validation" in o and isinstance(o["eval_validation"], dict):
            ps = o["eval_validation"].get("per_seed_success")
            seeds = o["eval_validation"].get("seeds")
            if ps and seeds:
                out[int(last)] = (list(seeds), [float(x) for x in ps])
    return out


def paired(a_seeds, a_vals, b_seeds, b_vals, label_a, label_b):
    """逐种子配对。必须同一批种子、同一顺序。"""
    if a_seeds != b_seeds:
        print(f"  !! 种子表不一致，拒绝配对（{label_a} vs {label_b}）")
        return None
    d = [x - y for x, y in zip(a_vals, b_vals)]
    n = len(d)
    mean = statistics.mean(d)
    sd = statistics.stdev(d) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    t = mean / se if se > 0 else float("inf")
    df = n - 1
    crit = T_TABLE.get(df)
    if crit is None:
        raise SystemExit(f"  !! df={df} 无临界值表项 —— 不许兜底（silent-lenient-fallback）")
    same = sum(1 for x in d if x > 0)
    return {"n": n, "mean": mean, "sd": sd, "se": se, "t": t, "df": df,
            "crit": crit, "same": same, "d": d}


def show(label, r, label_a, label_b):
    if r is None:
        return
    sig = "★显著" if abs(r["t"]) >= r["crit"] else "测不出"
    print(f"\n  {label}")
    print(f"    逐种子 Δ = {label_a} − {label_b}")
    print(f"    Δ = {r['mean']:+.6f}   SD = {r['sd']:.4f}   SE = {r['se']:.4f}")
    print(f"    t = {r['t']:+.3f}  (df={r['df']}, 临界 {r['crit']})   同向 {r['same']}/{r['n']}")
    print(f"    判读：{sig}")


def main():
    exp = json.loads(EXPERT.read_text(encoding="utf-8"))
    e_seeds = [int(s) for s in exp["seeds"]]
    # 专家 JSON 的键是 `success`；RL 侧是 `per_seed_success` —— 两个都认，
    # 都认不到就抛错（不许静默兜底成空列表，那会让配对变空真）。
    e_raw = exp.get("per_seed_success") or exp.get("success")
    if not e_raw:
        raise SystemExit(f"{EXPERT} 里找不到 success/per_seed_success ⟹ 不许兜底")
    e_vals = [float(x) for x in e_raw]

    bc = json.loads(BC.read_text(encoding="utf-8"))
    b_seeds = [int(s) for s in bc["seeds"]]
    b_vals = [float(x) for x in bc["per_seed_success"]]

    print("=" * 96)
    print("BC 起点 / 专家 / RL u30 —— 同种子配对")
    print("=" * 96)
    print(f"\n  专家      (n={len(e_vals)}) mean = {statistics.mean(e_vals):.6f}")
    print(f"  BC 起点   (n={len(b_vals)}) mean = {statistics.mean(b_vals):.6f}")
    print(f"  种子表一致：{e_seeds == b_seeds}")

    show("① BC 起点 vs 专家", paired(b_seeds, b_vals, e_seeds, e_vals, "BC", "专家"),
         "BC", "专家")

    # ② 每条 ent01_rerun 臂的 u30 vs BC 起点
    for run in ["ent01_rerun_s42", "ent01_rerun_s43", "ent01_rerun_s44",
                "ent01_rerun_s45", "ent01_rerun_s46"]:
        vp = val_points(run)
        if not vp:
            continue
        u = max(vp.keys())
        seeds, vals = vp[u]
        if seeds != b_seeds:
            print(f"  !! {run} 种子表与 BC 不一致，跳过")
            continue
        r2 = paired(seeds, vals, b_seeds, b_vals, run, "BC")
        r3 = paired(seeds, vals, e_seeds, e_vals, run, "专家")
        print(f"\n--- {run}  (u={u}, mean={statistics.mean(vals):.6f}) ---")
        show("② RL 末轮 vs BC 起点", r2, run, "BC")
        show("③ RL 末轮 vs 专家", r3, run, "专家")


if __name__ == "__main__":
    main()
