"""汇总一轮筛选的结果：训练侧读 + 验证侧的**配对**比较。

为什么以验证侧为准：训练侧的 `mean_success_rate` 是**采样**成功率，单轮噪声大；
`eval_validation` 是固定场景、12 个种子、确定性策略，才是低方差读数。

为什么还要配对：12 个种子的均值标准误约 0.012，但均值对均值仍然带着种子间的
巨大跨度（单种子 0.29–0.88）。同种子对同种子的**配对差**才压得下去（见
docs/测试规范.md ⑦）。所以每个变体都和 `base` 逐种子相减。

用法（节点上）：
    cd /opt/qkd/graph_mappo && /opt/qkd/venv/bin/python .tmp/summarize_screen.py base lr1e4 ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MAIN = Path("/opt/qkd/graph_mappo")
BC_BASELINE = MAIN / "outputs" / "eval" / "bc_diag_perseed.json"
EXPERT_DIAG = MAIN / "outputs" / "eval" / "expert_diag.json"


def load(name: str) -> dict | None:
    p = MAIN / "outputs" / f"r2_{name}" / "metrics.jsonl"
    if not p.exists():
        return None
    train: list[dict] = []
    evals: list[dict] = []
    for line in p.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "update" in d:
            train.append(d)
        elif "eval_validation" in d:
            evals.append(d["eval_validation"])
    return {"train": train, "evals": evals}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def paired_vs_base(base: list[float], other: list[float]) -> tuple[float, float, float, int]:
    """返回 (配对差, 标准误, t, 配对数)。两边的种子顺序必须一致。"""
    n = min(len(base), len(other))
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    diffs = [base[i] - other[i] for i in range(n)]
    md = mean(diffs)
    var = sum((d - md) ** 2 for d in diffs) / (n - 1)
    se = (var / n) ** 0.5
    t = md / se if se > 0 else float("nan")
    return md, se, t, n


def main() -> None:
    names = sys.argv[1:]
    results = {}
    for name in names:
        data = load(name)
        if data is None:
            print(f"  (缺 {name})")
            continue
        results[name] = data

    if not results:
        print("没有任何结果")
        return

    base_seeds = None
    if "base" in results and results["base"]["evals"]:
        base_seeds = results["base"]["evals"][-1].get("per_seed_success")

    # BC 起点（.tmp/probe_bc_baseline.py 写的）。这是"训练到底有没有用"的参照 ——
    # 对 base 的配对差只说明变体之间谁好，对 BC 的配对差才说明训练有没有把
    # 起点推上去。
    bc_seeds = None
    if BC_BASELINE.exists():
        bc = json.loads(BC_BASELINE.read_text(encoding="utf-8"))
        bc_seeds = bc.get("per_seed_success")
        print(f"BC 起点（同协议）：{bc['mean_success_rate']:.4f}  "
              f"{len(bc_seeds or [])} 个种子")
    else:
        print(f"（没有 {BC_BASELINE} —— 先跑 .tmp/probe_bc_baseline.py）")
    # 专家（启发式）在同一场景上。这是"天花板"参照：训练超过它才算真的学到了
    # 比启发式更好的东西，而不是只把 BC 权重微微推了一点。
    expert_seeds = None
    if EXPERT_DIAG.exists():
        ex = json.loads(EXPERT_DIAG.read_text(encoding="utf-8"))
        expert_seeds = ex.get("success")
        print(f"专家（同协议）：{mean(expert_seeds or [0.0]):.4f}  "
              f"{len(expert_seeds or [])} 个种子")
    else:
        print(f"（没有 {EXPERT_DIAG} —— 先跑 scripts/eval/eval_expert.py "
              f"--config configs/var2_diag.yaml --seeds 7-18 --steps 240）")
    print()

    # 按最后一次验证成功率排序
    order = sorted(
        results,
        key=lambda n: (
            results[n]["evals"][-1]["mean_success_rate"] if results[n]["evals"] else -1.0
        ),
        reverse=True,
    )

    print(f"{'变体':<14}{'训练前/后半':>17}{'熵首->末':>16}{'平均kl':>10}"
          f"{'验证序列':>34}{'末次':>8}{'对BC配对差':>13}{'t':>7}{'对专家':>10}{'对base':>10}")
    print("-" * 140)
    for name in order:
        d = results[name]
        tr = d["train"]
        ev = d["evals"]
        if tr:
            h = len(tr) // 2
            a = mean([r.get("mean_success_rate", 0.0) for r in tr[:h]])
            b = mean([r.get("mean_success_rate", 0.0) for r in tr[h:]])
            e0 = tr[0].get("entropy", 0.0)
            e1 = tr[-1].get("entropy", 0.0)
            kl = mean([r.get("kl", 0.0) for r in tr])
            train_s = f"{a:.4f}->{b:.4f}"
            ent_s = f"{e0:.2f}->{e1:.2f}"
            kl_s = f"{kl:.5f}"
        else:
            train_s = ent_s = kl_s = "—"
        series = " ".join(f"{e['mean_success_rate']:.3f}" for e in ev) or "—"
        last = f"{ev[-1]['mean_success_rate']:.4f}" if ev else "—"
        if bc_seeds and ev:
            # 取 `paired_vs_base(变体, BC)` 而不是反过来：**两张表都必须"正数=更好"**。
            # 第一版写成了 `(BC, 变体)`，于是这一列的正负和"对专家"那列相反 ——
            # 同一行里一个负数其实是进步、一个正数其实是退步，读的人必然看反。
            md, se, t, _n = paired_vs_base(ev[-1].get("per_seed_success", []), bc_seeds)
            bc_s = f"{md:+.4f}±{se:.4f}"
            bct_s = f"{t:+.2f}"
        else:
            bc_s = bct_s = "—"
        if name != "base" and base_seeds and ev:
            md, _se, t, _n = paired_vs_base(ev[-1].get("per_seed_success", []), base_seeds)
            base_s = f"{md:+.4f}({t:+.1f})"
        else:
            base_s = "—"
        if expert_seeds and ev:
            # 三列统一成**正数 = 该变体更好**。这件事必须一致：同一行里
            # 一个正号是进步、另一个正号是退步的话，读的人必然看反 —— 第一版
            # 就犯了这个错（对 BC 那列的方向反了）。
            md2, _se2, t2, _n2 = paired_vs_base(
                ev[-1].get("per_seed_success", []), expert_seeds
            )
            expert_s = f"{md2:+.4f}({t2:+.1f})"
        else:
            expert_s = "—"
        print(f"{name:<14}{train_s:>17}{ent_s:>16}{kl_s:>10}{series:>34}"
              f"{last:>8}{bc_s:>13}{bct_s:>7}{expert_s:>20}{base_s:>10}")

    print()
    print("口径提醒：")
    print("  * 训练侧是**采样**成功率，只作参考；判据看验证侧。")
    print("  * 对 BC 的配对差 > 0 且 |t|>=2，才算训练真的把起点推上去了。")
    print("  * 对 base 的配对差是变体之间的比较；两者都与 base 同向时不矛盾。")
    print("  * 配对分辨率约 0.012；不配对约 0.06。差 0.02 在配对下看得见，不配对看不见。")
    print("SCREEN_SUMMARY_DONE")


if __name__ == "__main__":
    main()
