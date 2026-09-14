"""Compact one-line-per-update view of a run's metrics.jsonl.

Usage:
    python show_metrics.py outputs/diag_srv/metrics.jsonl [--keys a,b,c]
    python show_metrics.py outputs/diag_srv/metrics.jsonl --last 5

Prints a header of the available keys with --keys help, otherwise a fixed
training-relevant column set, falling back gracefully when a key is absent.
"""
import argparse
import json
import sys

DEFAULT_COLS = [
    ("update", "{:>4}"),
    ("rollout_s", "{:>8.1f}"),
    ("update_s", "{:>8.1f}"),
    ("elapsed_s", "{:>8.1f}"),
    ("mean_success_rate", "{:>7.4f}"),
    ("eval_validation", "{:>7.4f}"),
    ("mean_served_keys", "{:>8.0f}"),
    ("value_std", "{:>7.4f}"),
    ("return_std", "{:>8.3f}"),
    ("value_return_corr", "{:>6.3f}"),
    ("mean_kl", "{:>8.5f}"),
    ("mean_ratio", "{:>7.4f}"),
    ("mean_entropy", "{:>7.3f}"),
    ("actor_grad_norm", "{:>7.3f}"),
    ("critic_grad_norm", "{:>7.3f}"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--keys", default=None, help="comma-separated column list")
    ap.add_argument("--last", type=int, default=0, help="only the last N rows")
    args = ap.parse_args()

    rows = []
    with open(args.path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("(no rows)")
        return
    if args.last:
        rows = rows[-args.last:]

    if args.keys:
        cols = [(k.strip(), "{:>10}") for k in args.keys.split(",") if k.strip()]
    else:
        cols = DEFAULT_COLS

    present = [(k, fmt) for k, fmt in cols if any(k in r for r in rows)]
    missing = [k for k, _ in cols if k not in {p[0] for p in present}]

    print("  ".join(k for k, _ in present))
    for r in rows:
        cells = []
        for k, fmt in present:
            v = r.get(k)
            if v is None:
                cells.append(f"{'-':>10}" if "{:>10}" == fmt else fmt.format(0))
            elif isinstance(v, (int, float)):
                cells.append(fmt.format(v))
            else:
                cells.append(f"{str(v):>10}")
        print("  ".join(cells))
    if missing:
        print(f"\n(absent from this run: {', '.join(missing)})", file=sys.stderr)


if __name__ == "__main__":
    main()
