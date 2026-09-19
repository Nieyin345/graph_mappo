"""专家 vs 策略，在各窗口上按种子配对比较。

为什么要配对：逐种子成功率跨度极大（专家在留出窗口 0.29–0.88），不配对的标准误
约 0.06，测不出 0.02 的差距；配对后约 0.012。这个项目里配对不是可选步骤。

回答的问题：策略相对**专家**的差距，在训练窗口和留出窗口上是否一样大？
  * 训练窗口追平、只在留出窗口掉 -> 泛化问题。
  * 各窗口都低差不多的量        -> 能力问题。

输入：
  outputs/eval/expert_win_<窗口>.json      —— eval_expert.py 写的（--out 指定）
  outputs/eval/win_<tag>_perseed.json     —— screen_windows_rl.sh 写的

用法：
    python .tmp/compare_windows.py bc
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WINDOWS = ["early_train", "mid_train", "late_train", "heldout"]


def load_expert(name: str) -> dict[int, float]:
    p = Path(f"outputs/eval/expert_win_{name}.json")
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return dict(zip(d["seeds"], d["success"]))


def load_policy(tag: str) -> dict[str, dict[int, float]]:
    p = Path(f"outputs/eval/win_{tag}_perseed.json")
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {w: dict(zip(v["seeds"], v["success"])) for w, v in d.items()}


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "bc"
    pol = load_policy(tag)
    if not pol:
        print(f"没有 outputs/eval/win_{tag}_perseed.json")
        return

    print(f"{'窗口':<14}{'专家':>9}{tag:>9}{'配对差':>10}{'标准误':>9}{'t':>8}{'种子数':>7}")
    print("-" * 68)
    gaps = {}
    for w in WINDOWS:
        exp = load_expert(w)
        got = pol.get(w, {})
        common = sorted(set(exp) & set(got))
        if not common:
            print(f"{w:<14}{'(缺数据)':>9}")
            continue
        e = [exp[s] for s in common]
        g = [got[s] for s in common]
        diffs = [a - b for a, b in zip(e, g)]
        n = len(diffs)
        mean_e = sum(e) / n
        mean_g = sum(g) / n
        md = sum(diffs) / n
        if n > 1:
            var = sum((d - md) ** 2 for d in diffs) / (n - 1)
            se = (var / n) ** 0.5
        else:
            se = float("nan")
        t = md / se if se and se > 0 else float("nan")
        gaps[w] = md
        flag = "  *" if abs(t) >= 2 else ""
        print(f"{w:<14}{mean_e:>9.4f}{mean_g:>9.4f}{md:>+10.4f}{se:>9.4f}{t:>8.2f}{n:>7}{flag}")

    if "heldout" in gaps and "mid_train" in gaps:
        extra = gaps["heldout"] - (gaps["mid_train"] + gaps["late_train"]) / 2
        print()
        print("差中之差（留出窗口的差距 − 训练窗口的平均差距）：")
        print(f"  {extra:+.4f}")
        print("  接近 0  -> 各窗口差距一致，是能力问题")
        print("  明显更负 -> 留出窗口掉得更多，是泛化问题")
    print()
    print("（* = |t| >= 2）  参照分辨率：配对约 0.012，单点不配对约 0.06")
    print("COMPARE_WINDOWS_DONE")


if __name__ == "__main__":
    main()
