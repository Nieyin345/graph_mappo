"""线程数是否改变训练结果 —— 判读。

四组运行：2 线程 ×2 次、4 线程 ×2 次，同一配置、同一 checkpoint、同时起。
两种差分开看：
  * 同线程两次之间  = 该条件下的纯随机重复性
  * 跨线程之间      = 线程数带来的额外差异
前者远小于后者 => 线程数真的改变结果，跨并行度的比较不成立；
两者同量级     => 只是随机漂移。

用法（节点上）：
    cd /opt/qkd/graph_mappo && /opt/qkd/venv/bin/python .tmp/analyze_thread_repro.py
"""

from __future__ import annotations

import json
from pathlib import Path

MAIN = Path("/opt/qkd/graph_mappo")


def series(name: str) -> tuple[dict | None, list[list[float]]]:
    p = MAIN / "outputs" / name / "metrics.jsonl"
    if not p.exists():
        return None, []
    last_round = None
    pers = []
    for line in p.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "update" in d:
            last_round = d
        elif "eval_validation" in d:
            per = d["eval_validation"].get("per_seed_success")
            if per:
                pers.append([float(v) for v in per])
    return last_round, pers


def paired(a: list[float], b: list[float]) -> tuple[float, float, float, int]:
    n = min(len(a), len(b))
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    diffs = [a[i] - b[i] for i in range(n)]
    md = sum(diffs) / n
    var = sum((d - md) ** 2 for d in diffs) / (n - 1)
    se = (var / n) ** 0.5
    return md, se, (md / se if se > 0 else float("nan")), n


def main() -> None:
    runs = {}
    print("=== 各次运行的末轮状态 ===")
    print(f"{'运行':<14}{'末轮验证':>10}{'熵':>9}{'平均kl':>10}{'验证次数':>9}")
    for name in ("repro_t2_1", "repro_t2_2", "repro_t4_1", "repro_t4_2"):
        last, pers = series(name)
        runs[name] = (last, pers)
        if last is None or not pers:
            print(f"{name:<14}  (缺数据)")
            continue
        final = pers[-1]
        print(f"{name:<14}{sum(final) / len(final):>10.4f}"
              f"{last['entropy']:>9.4f}{last['kl']:>10.6f}{len(pers):>9}")

    print()
    print("=== 同线程、两次独立运行之间（纯随机）===")
    same = {}
    for th in (2, 4):
        a = runs.get(f"repro_t{th}_1")
        b = runs.get(f"repro_t{th}_2")
        if not a or not b or not a[1] or not b[1]:
            continue
        md, se, t, n = paired(a[1][-1], b[1][-1])
        same[th] = (md, se)
        print(f"  {th} 线程: 末次验证差 {md:+.4f} ± {se:.4f}  t={t:+.2f}"
              f"  （{n} 种子配对）")

    print()
    print("=== 跨线程：同一配置，只改 OMP_NUM_THREADS ===")
    cross = []
    for rep in (1, 2):
        a = runs.get(f"repro_t2_{rep}")
        b = runs.get(f"repro_t4_{rep}")
        if not a or not b or not a[1] or not b[1]:
            continue
        md, se, t, n = paired(a[1][-1], b[1][-1])
        cross.append((md, se))
        print(f"  第 {rep} 次: 2线程 − 4线程 = {md:+.4f} ± {se:.4f}  t={t:+.2f}")

    print()
    print("=== 判读 ===")
    if same and cross:
        same_abs = sum(abs(m) for m, _ in same.values()) / len(same)
        cross_abs = sum(abs(m) for m, _ in cross) / len(cross)
        print(f"  同线程两次的平均绝对差 : {same_abs:.4f}")
        print(f"  跨线程的平均绝对差     : {cross_abs:.4f}")
        print(f"  倍数                   : {cross_abs / same_abs:.1f}×")
        if cross_abs < 3 * same_abs:
            print("  => 跨线程差异没有明显超出重复性噪声，线程数**不**实质改变结果。")
            print("     但这也意味着：那份'两次同配置差 0.06'不能只归给线程数，")
            print("     还要找别的差异（见文档）。")
        else:
            print("  => 跨线程差异明显超出重复性噪声，**线程数真的改变训练结果**。")
            print("     跨并行度的比较全部不成立，必须固定线程数，或改成不看绝对值的判据。")
    print("THREAD_REPRO_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
