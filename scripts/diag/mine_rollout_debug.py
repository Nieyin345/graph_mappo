"""从 rollout_debug.jsonl 抽序列 —— 每轮一行，30 轮，**免费的高价值诊断数据**。

为什么值得单独看：这个文件每轮记了 `actor_grad_norm` 与 `mean_abs_advantage`，
而「actor 被优势塌缩冻住」这个诊断**从来没被逐轮验证过**，一直是两点对比
（开头 1.43 / 结尾 0.047 那种）。逐轮序列能直接看出：

  1. `actor_grad_norm` 有没有超过 `max_grad_norm`（0.5）
     ⟹ 这直接判定梯度裁剪是不是活的（不必等探针）
  2. `mean_abs_advantage` 是**单调塌陷**还是**早就平了**
     ⟹ 决定「等它涨回来」是不是一个伪希望

用法（服务器上）：
  /opt/qkd/venv/bin/python /tmp/mine_rollout_debug.py --run ent03_s42
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")

COLS = [
    ("actor_grad_norm", "actor|g|"),
    ("mean_abs_advantage", "|A|"),
    ("kl", "kl"),
    ("entropy", "熵"),
    ("critic_loss", "criticL"),
    ("actor_loss", "actorL"),
    ("mean_success_rate", "sr"),
    ("mean_waiting_keys", "waiting"),
    ("mean_served_keys", "served"),
    ("mean_generated_keys", "generated"),
    ("mean_qkp_utilization", "qkp_util"),
]


def load(run: str) -> list[dict]:
    p = OUT / run / "rollout_debug.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def fmt(v: object, w: int = 9) -> str:
    if isinstance(v, (int, float)):
        av = abs(float(v))
        if av >= 1e6:
            return f"{float(v) / 1e6:>{w - 1}.2f}M"
        if av >= 1e3:
            return f"{float(v) / 1e3:>{w - 1}.1f}k"
        if av >= 1:
            return f"{float(v):>{w}.3f}"
        return f"{float(v):>{w}.5f}"
    return f"{'—':>{w}}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ent03_s42")
    ap.add_argument("--compare", default="",
                    help="逗号分隔的另一个 run，逐轮并排差值")
    ap.add_argument("--max-norm", type=float, default=0.5)
    a = ap.parse_args()

    rows = load(a.run)
    if not rows:
        print(f"✗ {a.run} 没有 rollout_debug.jsonl（对照臂 ent01_t8_* 就没有）")
        return 1

    hdr = "".join(f"{lbl:>{9}}" for _, lbl in COLS)
    print("=" * 100)
    print(f"rollout_debug 逐轮序列：{a.run}（{len(rows)} 轮）")
    print("=" * 100)
    print(f"{'u':>3}{hdr}")
    print("-" * 100)
    for r in rows:
        u = r.get("update", "?")
        line = "".join(fmt(r.get(k)) for k, _ in COLS)
        print(f"{u:>3}{line}")

    print()
    print("=" * 100)
    print("判读")
    print("=" * 100)

    # 1) 梯度裁剪是否活跃
    gn = [float(r["actor_grad_norm"]) for r in rows
          if isinstance(r.get("actor_grad_norm"), (int, float))]
    if gn:
        over = sum(1 for g in gn if g > a.max_norm)
        print(f"  ① actor 梯度范数 vs max_grad_norm={a.max_norm}")
        print(f"     中位 {statistics.median(gn):.4f}  最大 {max(gn):.4f}  "
              f"最小 {min(gn):.4f}")
        print(f"     超过阈值的轮数: **{over}/{len(gn)} = "
              f"{100 * over / len(gn):.1f}%**")
        if over == 0:
            print(f"     ⟹ 裁剪**从未触发** ⟹ `max_grad_norm` 不是杠杆，"
                  f"调它没用。")
        elif over / len(gn) > 0.3:
            print(f"     ⟹ 裁剪经常触发 ⟹ 它是活跃限制器，往上调**有机制支撑**。")
        else:
            print(f"     ⟹ 偶尔触发，多数轮次没到阈值 ⟹ 杠杆很弱，"
                  f"调它的期望收益小。")

    # 2) 优势是不是真在塌缩，还是早就平了
    aa = [float(r["mean_abs_advantage"]) for r in rows
          if isinstance(r.get("mean_abs_advantage"), (int, float))]
    if len(aa) >= 4:
        h = len(aa) // 2
        f_h, s_h = statistics.mean(aa[:h]), statistics.mean(aa[h:])
        print()
        print(f"  ② |A| 的形态（前半 {h} 轮 vs 后半 {len(aa) - h} 轮）")
        print(f"     前半均值 {f_h:.4f}  后半均值 {s_h:.4f}  "
              f"变化 {100 * (s_h - f_h) / f_h:+.1f}%")
        if abs(s_h - f_h) / f_h < 0.05:
            print("     ⟹ **早就平了**，不是「还在塌」 ⟹ 「等优势涨回来」是伪希望，"
                  "要换机制。")
        else:
            print("     ⟹ 前后半仍有可测差异，塌缩过程未结束。")

    # 3) 与另一个 run 逐轮比
    if a.compare:
        other = load(a.compare)
        if other and len(other) == len(rows):
            print()
            print(f"  ③ 与 {a.compare} 的逐轮差（实验 − 对照）")
            for key, lbl in COLS[:4]:
                d = [float(r[key]) - float(o[key]) for r, o in zip(rows, other)
                     if isinstance(r.get(key), (int, float))
                     and isinstance(o.get(key), (int, float))]
                if d:
                    print(f"     {lbl:<10} 平均差 {statistics.mean(d):+.5f}  "
                          f"SD {statistics.pstdev(d):.5f}")
        else:
            print(f"\n  ③ {a.compare} 不可比（行数不同或缺文件）")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
