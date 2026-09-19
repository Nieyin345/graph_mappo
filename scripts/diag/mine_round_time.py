"""一轮时间花在哪：从 metrics.jsonl 读 rollout_s / update_s 的真实分解。

### 为什么值得单独读

「update 占一轮 75%」这个数字来自日志 3740 行的剖面，**不是**这波 run 的实测。
而线程数是个敏感旋钮（改它要整批重跑，见记忆 `thread-count-changes-training`），
所以**先确认真实占比**，再决定值不值得为它重建基线。

★ 若能拿到的实际占比与 75% 差得多，整轮的提速外推就完全不同：
   update 占 75% 时，update 快 2 倍 ⟹ 整轮 1.6x
   update 占 40% 时，同样 2 倍 ⟹ 整轮只有 1.25x

用法（服务器上）：
  /opt/qkd/venv/bin/python /tmp/mine_round_time.py --run ent03_s42
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def load(run: str) -> list[dict]:
    p = OUT / run / "metrics.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ent03_s42")
    a = ap.parse_args()

    rows = load(a.run)
    if not rows:
        print(f"✗ {a.run} 无 metrics")
        return 1

    # 找出计时字段（名字可能不同，先列出来）
    keys = set()
    for r in rows:
        keys |= set(r.keys())
    tkeys = sorted(k for k in keys if k.endswith("_s") or "time" in k.lower()
                   or "sec" in k.lower())
    print("=" * 88)
    print(f"{a.run}：{len(rows)} 轮")
    print("=" * 88)
    print("计时相关字段:", tkeys or "（没有）")
    print()

    def series(k: str) -> list[float]:
        return [float(r[k]) for r in rows
                if isinstance(r.get(k), (int, float))]

    roll = series("rollout_s")
    upd = series("update_s")
    if not roll or not upd:
        print("✗ 缺 rollout_s / update_s，逐行键如下（第一行）：")
        print(" ", sorted(rows[0].keys()))
        return 1

    n = min(len(roll), len(upd))
    # 丢掉第 1 轮（含惰性初始化，不是稳态）
    roll_s, upd_s = roll[1:n], upd[1:n]
    r_m, u_m = statistics.mean(roll_s), statistics.mean(upd_s)
    tot = [r + u for r, u in zip(roll_s, upd_s)]
    print(f"  {'轮':>4}{'rollout_s':>12}{'update_s':>12}{'一轮':>10}{'update 占比':>14}")
    print("  " + "-" * 52)
    for i, (r, u) in enumerate(zip(roll[1:n], upd[1:n]), start=2):
        print(f"  {i:>4}{r:>12.1f}{u:>12.1f}{r + u:>10.1f}"
              f"{100 * u / (r + u):>13.1f}%")
    print("  " + "-" * 52)
    print(f"  稳态均值（丢第 1 轮）: rollout {r_m:.1f}s  update {u_m:.1f}s  "
          f"合计 {r_m + u_m:.1f}s")
    share = u_m / (r_m + u_m)
    print(f"  ⟹ **update 占一轮的 {100 * share:.1f}%**")
    print()

    # 首尾对比：增长的是谁
    if n >= 6:
        h = len(roll_s) // 2
        print(f"  前半（2~{1 + h} 轮）均值: rollout {statistics.mean(roll_s[:h]):.1f}s  "
              f"update {statistics.mean(upd_s[:h]):.1f}s")
        print(f"  后半均值:             rollout "
              f"{statistics.mean(roll_s[h:]):.1f}s  "
              f"update {statistics.mean(upd_s[h:]):.1f}s")
        gr = statistics.mean(roll_s[h:]) / statistics.mean(roll_s[:h])
        gu = statistics.mean(upd_s[h:]) / statistics.mean(upd_s[:h])
        print(f"  后半/前半: rollout **{gr:.2f}x**  update **{gu:.2f}x**")
        print()
        print("  ★ 若两者都在涨，过载症状就是「每轮耗时上涨」而不是「进程变慢」"
              "（CLAUDE.md 已记）。")

    print()
    print("=" * 88)
    print("线程数提速的整轮外推（用上面的真实占比，不用日志里的 75%）")
    print("=" * 88)
    for speedup in (1.5, 2.0):
        whole = 1.0 / ((1 - share) + share / speedup)
        print(f"  若 update 快 {speedup:.1f}x ⟹ 整轮提速 **{whole:.2f}x**"
              f"（rollout 段不变）")
    print()
    print("  ⚠ rollout 段不吃线程（它是 8 个 worker 进程并行），所以只有 update 段受益。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
