"""从零训 vs BC 暖启动：逐臂轨迹与末轮趋势。

关键问题：scratch_s42 单条已经到 0.7246（超专家 2.7 点），
但 s43/s44 停在 0.51 —— 是"分化"还是"慢"？看**末段趋势**能分辨：
  · 仍在上冲 ⟹ 30 轮不够，慢
  · 已平/下掉 ⟹ 分化，起点确实重要
"""
import json
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
EXPERT = 0.697922


def traj(run):
    v = []
    for line in (OUT / run / "metrics.jsonl").read_text(
            encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "eval_validation" in o:
            v.append(o["eval_validation"]["mean_success_rate"])
    return v


def main():
    print("=" * 92)
    print("从零训（无 BC）vs BC 暖启动 —— 逐臂轨迹")
    print("=" * 92)
    print(f"\n  专家 = {EXPERT:.4f}\n")

    for kind, pref in (("从零（无 BC）", "scratch_s"),
                       ("BC 暖启动", "ent01_rerun_s")):
        print(f"  --- {kind} ---")
        for s in (42, 43, 44):
            v = traj(f"{pref}{s}")
            if not v:
                print(f"    {pref}{s}: 无数据")
                continue
            vals = "  ".join(f"{x:.4f}" for x in v)
            tail = v[-1] - v[-2] if len(v) >= 2 else 0.0
            print(f"    {pref}{s}: [{vals}]")
            print(f"    {'':<{len(pref)+3}}末段 Δ = {tail:+.4f}   "
                  f"vs 专家 {v[-1]-EXPERT:+.4f}")
        print()

    # 合并
    print("=" * 92)
    sc = {s: traj(f"scratch_s{s}")[-1] for s in (42, 43, 44)}
    bc = {s: traj(f"ent01_rerun_s{s}")[-1] for s in (42, 43, 44)}
    print(f"  从零  均值 = {statistics.mean(sc.values()):.4f}   "
          f"（各臂 {[round(x,4) for x in sc.values()]}）")
    print(f"  有BC  均值 = {statistics.mean(bc.values()):.4f}   "
          f"（各臂 {[round(x,4) for x in bc.values()]}）")
    d = [sc[s] - bc[s] for s in (42, 43, 44)]
    print(f"  配对差 Δ = {statistics.mean(d):+.4f}   "
          f"（各 {[round(x,4) for x in d]}）")
    print(f"\n  ⟹ 有 BC 的**赢了**？需看逐臂：s42 从零反而高 "
          f"{d[0]:+.4f}，s43/s44 低 {d[1]:+.4f}/{d[2]:+.4f}")


if __name__ == "__main__":
    main()
