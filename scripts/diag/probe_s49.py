"""查 `clean_s49` 崩到 0.002 是**训练发散**还是**策略退化**。

判据：
- 训练发散 ⟹ kl 爆炸 / clip 满 / entropy 崩
- 策略退化（学到坏解）⟹ 训练侧正常但验证侧低
- 数据问题 ⟹ 服务量/到达量异常
"""
import json
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def main():
    for run in ("clean_s49", "clean_s46"):
        p = OUT / run / "metrics.jsonl"
        print("=" * 90)
        print(f"{run}")
        print("=" * 90)
        print(f"  {'u':>4}{'kl':>10}{'entropy':>9}{'clip':>9}{'|adv|':>9}"
              f"{'train_sr':>10}")
        vals = []
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "update" not in o:
                continue
            u = o["update"]
            if u in (1, 5, 10, 15, 20, 25, 28, 29, 30):
                def f(x, w=9, pp=5):
                    return f"{x:>{w}.{pp}f}" if isinstance(x, (int, float)) else f"{'—':>{w}}"
                print(f"  {u:>4}{f(o.get('kl'),10)}{f(o.get('entropy'))}"
                      f"{f(o.get('clip_frac'))}{f(o.get('mean_abs_advantage'))}"
                      f"{f(o.get('mean_success_rate'),10)}")

        print("\n  验证侧 + 服务/到达：")
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
            if "eval_validation" in o:
                ev = o["eval_validation"]
                print(f"    u{last}: sr={ev['mean_success_rate']:.4f}  "
                      f"served={ev.get('mean_served_keys', 0):,.0f}  "
                      f"ke={ev.get('mean_key_efficiency', 0):.1f}")
        print()


if __name__ == "__main__":
    main()
