"""把各次运行的**逐种子验证成绩**抽成一个小 JSON，供本地留档与重算。

为什么不直接留 metrics.jsonl：那个文件里 90% 是每轮的瞬时指标（loss、梯度范数、
耗时），真正决定结论的只有 `eval_validation` 那几行 —— 而它的 `per_seed_success`
是**按种子配对**比较的唯一依据。抽出来之后，几 MB 的日志变成几十 KB 的矩阵，
以后想重判"哪个变体更好"不用再上服务器。

输出结构：
{
  "runs": {
    "<run 名>": {
      "seeds": [...],                    # 与每次验证的 per_seed 对齐
      "validations": [[...], [...]],     # 每次评估的逐种子成功率，按时间顺序
      "final": [...],                    # 最后一次（= 结论用的那一列）
      "final_mean": 0.8123
    }, ...
  },
  "baselines": {"bc": {...}, "expert": {...}}
}

用法（节点上）：
    cd /opt/qkd/graph_mappo && /opt/qkd/venv/bin/python .tmp/dump_perseed.py \
        outputs/r2_* outputs/repro_t* outputs/screen_*
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MAIN = Path("/opt/qkd/graph_mappo")


def extract(run_dir: Path) -> dict | None:
    mf = run_dir / "metrics.jsonl"
    if not mf.exists():
        return None
    seeds: list[int] = []
    validations: list[list[float]] = []
    for line in mf.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        ev = d.get("eval_validation")
        if not ev:
            continue
        per = ev.get("per_seed_success")
        if per is None:
            continue
        if not seeds:
            seeds = list(ev.get("seeds", []))
        validations.append([float(v) for v in per])
    if not validations:
        return None
    final = validations[-1]
    return {
        "seeds": seeds,
        "validations": validations,
        "final": final,
        "final_mean": sum(final) / len(final),
        "n_validations": len(validations),
    }


def main() -> None:
    patterns = sys.argv[1:] or ["outputs/r2_*", "outputs/repro_t*"]
    out: dict = {"runs": {}, "baselines": {}}
    for pattern in patterns:
        for run_dir in sorted(MAIN.glob(pattern)):
            if not run_dir.is_dir():
                continue
            data = extract(run_dir)
            if data is not None:
                out["runs"][run_dir.name] = data

    # 两个诊断场景的参照点也一并带上，它们决定"涨没涨"该怎么判。
    for name, path in (
        ("bc", MAIN / "outputs" / "eval" / "bc_diag_perseed.json"),
        ("expert", MAIN / "outputs" / "eval" / "expert_diag.json"),
    ):
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            out["baselines"][name] = {
                "seeds": d.get("seeds"),
                # 专家脚本用的键是 success，BC 探针用的是 per_seed_success
                "final": d.get("per_seed_success") or d.get("success"),
                "final_mean": d.get("mean_success_rate")
                or (sum(d["success"]) / len(d["success"]) if d.get("success") else None),
            }

    text = json.dumps(out, indent=1)
    Path("/tmp/perseed_matrix.json").write_text(text, encoding="utf-8")
    print(f"runs      : {len(out['runs'])}")
    print(f"baselines : {list(out['baselines'])}")
    print(f"bytes     : {len(text):,}")
    print(f"wrote     : /tmp/perseed_matrix.json")
    print("DUMP_PERSEED_DONE")


if __name__ == "__main__":
    main()
