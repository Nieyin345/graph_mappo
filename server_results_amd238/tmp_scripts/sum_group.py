"""按组汇总一轮筛选：组内每个变体都和该组的 base **逐种子配对**。

为什么另起一个：`.tmp/summarize_screen.py` 把产物目录写死成 `outputs/r2_*`
（第一/二轮的名字），第三浪落在 `outputs/r3_*`、真实配置落在 `outputs/real_*`。
与其改那个正在被别的结论引用的脚本，不如加一个按前缀取组的。

判读依据仍然是验证侧 `eval_validation`（固定场景、确定性），不是训练侧的
采样 `mean_success_rate`。配对分辨率约 0.012（见 docs/测试规范.md ⑦）。

第三浪还要额外对两个参照物配对：
  * **BC 起点**（`outputs/eval/bc_diag_perseed.json`）—— 训练到底有没有把 BC
    权重往上推。只有它 > 0 且 |t| >= 2，"训练有用"才成立。
  * **专家**（`outputs/eval/expert_diag.json`，PG-Phased 启发式）—— 天花板。

这两个参照物是诊断场景协议的产物，对真实配置（`real_*`）不适用，脚本会自动跳过。

用法（节点上）：
    /opt/qkd/venv/bin/python .tmp/sum_group.py r3
    /opt/qkd/venv/bin/python .tmp/sum_group.py real
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MAIN = Path("/opt/qkd/graph_mappo")
BC_BASELINE = MAIN / "outputs" / "eval" / "bc_diag_perseed.json"
EXPERT_DIAG = MAIN / "outputs" / "eval" / "expert_diag.json"


def load_run(run_dir: Path) -> tuple[list[dict], list[dict]]:
    train: list[dict] = []
    evals: list[dict] = []
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return train, evals
    for line in p.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "update" in d:
            train.append(d)
        elif "eval_validation" in d:
            evals.append(d["eval_validation"])
    return train, evals


def paired(a: list[float], b: list[float]) -> tuple[float, float, float, int]:
    """配对差 a - b（正数 = a 更好），含标准误与 t。两边种子顺序必须一致。"""
    n = min(len(a), len(b))
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    diffs = [a[i] - b[i] for i in range(n)]
    md = sum(diffs) / n
    var = sum((d - md) ** 2 for d in diffs) / (n - 1)
    se = (var / n) ** 0.5
    return md, se, (md / se if se > 0 else float("nan")), n


def load_ref(path: Path, key: str) -> list[float] | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return d.get(key)


def main() -> None:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "r3"
    refs = sorted(
        p for p in MAIN.glob(f"outputs/{prefix}_*")
        if (p / "metrics.jsonl").exists()
    )
    if not refs:
        print(f"没有 outputs/{prefix}_*/metrics.jsonl")
        return

    runs: dict[str, tuple[list[dict], list[dict]]] = {}
    for p in refs:
        runs[p.name[len(prefix) + 1:]] = load_run(p)

    base = runs.get("base")
    base_seeds = base[1][-1].get("per_seed_success") if base and base[1] else None

    bc_seeds = load_ref(BC_BASELINE, "per_seed_success")
    ex_seeds = load_ref(EXPERT_DIAG, "success")

    print(f"组={prefix}  变体数={len(runs)}")
    if bc_seeds:
        print(f"BC 起点（同协议）：{sum(bc_seeds) / len(bc_seeds):.4f}  {len(bc_seeds)} 种子")
    else:
        print(f"（缺 {BC_BASELINE}）")
    if ex_seeds:
        print(f"专家同协议    ：{sum(ex_seeds) / len(ex_seeds):.4f}  {len(ex_seeds)} 种子")
    else:
        print(f"（缺 {EXPERT_DIAG}）")
    print()

    order = sorted(
        runs,
        key=lambda n: (runs[n][1][-1]["mean_success_rate"] if runs[n][1] else -1.0),
        reverse=True,
    )
    hdr = (f"{'变体':<12}{'轮':>4}{'熵首->末':>15}{'平均kl':>10}{'验证序列':>34}"
           f"{'末次':>8}{'对base':>16}{'t':>7}{'对BC':>16}{'t':>7}{'对专家':>16}{'t':>7}")
    print(hdr)
    print("-" * len(hdr))
    for name in order:
        train, ev = runs[name]
        if not ev:
            print(f"{name:<12}{'—':>4}  (无验证记录)")
            continue
        if train:
            e0 = train[0].get("entropy", 0.0)
            e1 = train[-1].get("entropy", 0.0)
            kl = sum(r.get("kl", 0.0) for r in train) / len(train)
            ent_s, kl_s = f"{e0:.2f}->{e1:.2f}", f"{kl:.5f}"
        else:
            ent_s = kl_s = "—"
        series = " ".join(f"{e['mean_success_rate']:.3f}" for e in ev)
        last = ev[-1]
        seeds = last.get("per_seed_success", [])
        cells = []
        if name != "base" and base_seeds and len(seeds) == len(base_seeds):
            md, se, t, _ = paired(seeds, base_seeds)
            cells.append((f"{md:+.4f}±{se:.4f}", f"{t:+.2f}"))
        else:
            cells.append(("—", "—"))
        for ref in (bc_seeds, ex_seeds):
            if ref and len(seeds) == len(ref):
                md, se, t, _ = paired(seeds, ref)
                cells.append((f"{md:+.4f}±{se:.4f}", f"{t:+.2f}"))
            else:
                cells.append(("—", "—"))
        print(f"{name:<12}{len(train):>4}{ent_s:>15}{kl_s:>10}{series:>34}"
              f"{last['mean_success_rate']:>8.4f}"
              + "".join(f"{c:>16}{t:>7}" for c, t in cells))

    print()
    print("口径：正数 = 该变体更好；判据看验证侧；|t|>=2 才算分得开。")
    print("SUMMARY_GROUP_DONE")


if __name__ == "__main__":
    main()