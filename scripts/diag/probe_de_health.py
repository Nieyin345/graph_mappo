"""demandedge 判读的**健康检查**：这个显著负结果是真信号还是某条臂坏了？

## 为什么必须查

`Δ=−0.0518 t=−3.207` 是今晚唯一的显著结果，且**方向为负**。
按 `never-run-code-path-hides-bugs`：`demand_edge` 这条分支**从来没跑过**
⟹ 打开前要假设里面有 bug。

三条判据：
① 训练侧体征正常吗（kl/entropy/train_sr）
② 验证侧是**均匀下移**还是**个别臂崩**
③ `model.mode` 确实生效了吗（运行时值）
"""
import json
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def vitals(run):
    rows = []
    for line in (OUT / run / "metrics.jsonl").read_text(
            encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "update" in o:
            rows.append(o)
    return rows


def vals(run):
    out, last = [], None
    for line in (OUT / run / "metrics.jsonl").read_text(
            encoding="utf-8", errors="replace").splitlines():
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
            out.append((last, o["eval_validation"]["mean_success_rate"]))
    return out


def main():
    print("=" * 94)
    print("① 训练侧体征（u5 / u15 / u30）")
    print("=" * 94)
    for grp, runs in (("demandedge", [f"demandedge_s{s}" for s in range(42, 47)]),
                      ("ent01_rerun", [f"ent01_rerun_s{s}" for s in range(42, 47)])):
        print(f"\n  【{grp}】")
        print(f"    {'臂':<18}{'kl':>10}{'entropy':>9}{'clip':>9}{'train_sr':>10}")
        for r in runs:
            rs = vitals(r)
            if not rs:
                continue
            last = rs[-1]
            def f(x, w=9, p=5):
                return f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{'—':>{w}}"
            print(f"    {r:<18}{f(last.get('kl'),10)}{f(last.get('entropy'))}"
                  f"{f(last.get('clip_frac'))}{f(last.get('mean_success_rate'),10)}")

    print("\n" + "=" * 94)
    print("② 验证侧：是均匀下移还是个别臂崩？")
    print("=" * 94)
    for grp, runs in (("demandedge", [f"demandedge_s{s}" for s in range(42, 47)]),
                      ("ent01_rerun", [f"ent01_rerun_s{s}" for s in range(42, 47)])):
        print(f"\n  【{grp}】")
        for r in runs:
            v = vals(r)
            if not v:
                continue
            print(f"    {r:<18}{[round(x[1], 3) for x in v]}")
        means = [statistics.mean([x[1] for x in vals(r)][-2:]) for r in runs
                 if vals(r)]
        print(f"    末两点均值：{statistics.mean(means):.4f}  "
              f"范围 [{min(means):.4f}, {max(means):.4f}]")


if __name__ == "__main__":
    main()
