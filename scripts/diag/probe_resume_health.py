"""核实 scratch 续跑（u30→u60）的下滑是**真的**还是**污染**。

## 三个可能的解释，各自预测不同

**H1 双写**：同一目录被两个进程写 ⟹ update 号**重复**。
**H2 配置漂移**：续跑读的配置与前半段不同 ⟹ 训练侧体征会突变。
**H3 真实下滑**：优化本身在 u30 后变差（过训练/发散）。

## 判据

- H1 ⟹ `update` 号序列有重复
- H2 ⟹ 对比 `resolved_config.yaml` 的 mtime，或看 kl/entropy 在 u30 处**跳变**
- H3 ⟹ kl/entropy 平滑演变，只有验证侧下滑
"""
import json
from collections import Counter
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def main():
    for run in ("scratch_s42", "scratch_s43", "scratch_s44"):
        p = OUT / run / "metrics.jsonl"
        us, rows = [], []
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "update" in o:
                us.append(o["update"])
                rows.append(o)

        cnt = Counter(us)
        dup = {u: c for u, c in cnt.items() if c > 1}
        print("=" * 88)
        print(f"{run}   update 号 {min(us)}..{max(us)}  共 {len(us)} 条")
        print(f"  重复的 update 号：{dup if dup else '（无 ⟹ 无双写）'}")
        # 连续性
        gaps = [b - a for a, b in zip(sorted(us), sorted(us)[1:]) if b - a != 1]
        print(f"  非连续跳跃：{gaps if gaps else '（连续）'}")

        # 训练侧体征：u30 前后
        print(f"\n  {'u':>4}{'kl':>10}{'entropy':>9}{'clip':>9}{'|adv|':>9}"
              f"{'train_sr':>10}")
        for u in (25, 30, 31, 35, 40, 45, 50, 55, 60):
            r = next((x for x in rows if x["update"] == u), None)
            if not r:
                continue
            def f(x, w=9, p=5):
                return f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{'—':>{w}}"
            print(f"  {u:>4}{f(r.get('kl'),10)}{f(r.get('entropy'))}{f(r.get('clip_frac'))}"
                  f"{f(r.get('mean_abs_advantage'))}{f(r.get('mean_success_rate'),10)}")

        # 配置 mtime
        c = OUT / run / "resolved_config.yaml"
        if c.exists():
            import datetime
            print(f"  resolved_config mtime: "
                  f"{datetime.datetime.fromtimestamp(c.stat().st_mtime).strftime('%m-%d %H:%M')}")
        print()


if __name__ == "__main__":
    main()
