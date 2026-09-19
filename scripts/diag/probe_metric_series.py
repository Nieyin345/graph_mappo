"""读已在记录的关键诊断量随轮次的变化，并**配对**比较多个 run。

`probe_metric_keys.py` 确认 `metrics.jsonl` 里已有：
  value_return_corr / value_std / return_std / mean_abs_advantage / kl / entropy
  actor_grad_norm / critic_grad_norm / actor_loss / critic_loss

这些是「critic 拟合好不好 → advantage 有没有信号 → actor 为什么不动」
这条链上**已经存在但没被看过**的证据。

用法（服务器上）：
  /opt/qkd/venv/bin/python /tmp/probe_metric_series.py \
      --runs mode_de_s42,ent01_t8_s42 --keys value_return_corr,actor_grad_norm,critic_grad_norm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
DEFAULT_KEYS = ["value_return_corr", "mean_abs_advantage", "kl", "entropy",
                "actor_grad_norm", "critic_grad_norm", "actor_loss", "critic_loss"]


def series(run: str, keys: list[str]) -> dict[int, dict[str, float]]:
    p = OUT / run / "metrics.jsonl"
    out: dict[int, dict[str, float]] = {}
    if not p.exists():
        return out
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
            last = int(o["update"])
        if last is None:
            continue
        for k in keys:
            v = o.get(k)
            if isinstance(v, (int, float)):
                out.setdefault(last, {})[k] = float(v)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--keys", default=",".join(DEFAULT_KEYS))
    a = ap.parse_args(argv[1:])
    runs = [r.strip() for r in a.runs.split(",") if r.strip()]
    keys = [k.strip() for k in a.keys.split(",") if k.strip()]

    for k in keys:
        print("=" * 96)
        print(f"  {k}")
        print("=" * 96)
        hdr = f"  {'u':>4}" + "".join(f"{r[:22]:>24}" for r in runs)
        print(hdr)
        allu = sorted({u for r in runs for u in series(r, [k])})
        data = {r: series(r, [k]) for r in runs}
        for u in allu:
            cells = []
            for r in runs:
                v = data[r].get(u, {}).get(k)
                cells.append(f"{v:>24.6f}" if v is not None else f"{'-':>24}")
            print(f"  {u:>4}" + "".join(cells))
        # 末段均值
        print(f"  {'末5均':>4}" + "".join(
            f"{_tail_mean(data[r], k):>24.6f}" for r in runs))
        print()
    return 0


def _tail_mean(d: dict[int, dict[str, float]], k: str, n: int = 5) -> float:
    us = sorted(u for u in d if k in d[u])
    sel = us[-n:]
    if not sel:
        return float("nan")
    return sum(d[u][k] for u in sel) / len(sel)


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
