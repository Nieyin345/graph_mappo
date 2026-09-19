"""把训练记录里的关键信号打出来，用来判断 actor 到底有没有在学。

关注这几列，它们回答的是不同的问题：
  succ    训练期 rollout 的成功率（随机策略、8 局）
  H       策略熵。**持续上升 = 策略在变随机，不是在变果断**，是策略梯度没学到东西的典型征兆。
  kl      每轮策略移动了多少（nats）。太小 = 每轮几乎没动，跑再多轮也没用。
  ratio   新旧策略概率比。健康时贴着 1.0；偏离说明在移动。
  V_std   价值函数输出的标准差；R_std 是回报的标准差。
  corr    **价值与回报的相关系数 —— 这列最关键**。critic 能不能预测回报，直接决定
          GAE 的优势里有多少是信号、多少是噪声。接近 0 意味着优势基本是噪声，
          PPO 的梯度也就基本是噪声。
  a_grad / c_grad   actor / critic 的梯度范数。

用法：
    python .tmp/summarize_training_signal.py <run 目录> [<run 目录> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "update" in d:
            rows.append(d)
    return rows


def main() -> None:
    for arg in sys.argv[1:]:
        run = Path(arg)
        f = run / "metrics.jsonl"
        if not f.exists():
            print(f"{run}: 没有 metrics.jsonl")
            continue
        rows = load(f)
        if not rows:
            print(f"{run}: 没有训练记录")
            continue

        print("=" * 108)
        print(f"{run}   （{len(rows)} 轮）")
        print("=" * 108)
        head = (f"  {'轮':>3} {'succ':>7} {'H':>7} {'kl':>9} {'ratio':>7} "
                f"{'V_std':>7} {'R_std':>7} {'corr':>7} {'a_grad':>8} {'c_grad':>8}")
        print(head)
        for d in rows:
            print(f"  {d['update']:>3} "
                  f"{d.get('mean_success_rate', float('nan')):>7.4f} "
                  f"{d.get('entropy', float('nan')):>7.3f} "
                  f"{d.get('kl', float('nan')):>9.2e} "
                  f"{d.get('mean_ratio', float('nan')):>7.4f} "
                  f"{d.get('value_std', float('nan')):>7.4f} "
                  f"{d.get('return_std', float('nan')):>7.4f} "
                  f"{d.get('value_return_corr', float('nan')):>7.4f} "
                  f"{d.get('actor_grad_norm', float('nan')):>8.4f} "
                  f"{d.get('critic_grad_norm', float('nan')):>8.4f}")

        # 趋势：前半 / 后半 的均值，直接回答"有没有在变好"。
        half = len(rows) // 2
        def avg(key: str, lo: int, hi: int) -> float:
            vals = [d[key] for d in rows[lo:hi] if key in d]
            return sum(vals) / len(vals) if vals else float("nan")

        print()
        print(f"  前半 {half} 轮  ->  后半 {len(rows) - half} 轮：")
        for key, label in (
            ("mean_success_rate", "训练成功率"),
            ("entropy", "策略熵"),
            ("value_return_corr", "价值-回报相关"),
            ("kl", "每轮 KL"),
        ):
            a, b = avg(key, 0, half), avg(key, half, len(rows))
            arrow = "↑" if b > a else "↓"
            print(f"    {label:<14} {a:>8.4f}  ->  {b:>8.4f}   {arrow}")
        print()

        # 评估记录单独列出来（协议和训练期 rollout 不同，别混着看）。
        evals = [json.loads(l)["eval_validation"]
                 for l in f.read_text(encoding="utf-8").splitlines()
                 if '"eval_validation"' in l]
        if evals:
            print("  验证评估（held-out，12 种子确定性）：")
            for i, e in enumerate(evals, 1):
                print(f"    第 {i} 次: success={e.get('mean_success_rate', float('nan')):.4f}")
        print()


if __name__ == "__main__":
    main()
