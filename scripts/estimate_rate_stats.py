"""Write ``dataset/global/rate_stats.json``: the p99 reference for rate features.

``RateNormalizer`` normalizes link rates with ``log1p(rate) / log1p(p99)``. It
reads the p99 from ``rate_stats.json`` and silently falls back to a hardcoded
``p99 = 10.0`` when that file is missing -- which makes every normalized rate
feature land well above 1 for real links (their p99 is ~1e4 bps), so the rate
block dominates the edge feature vector. Run this once per dataset.

The reported ``p99`` is the 99th percentile over **positive** rates only.
Zero-rate samples are ~90% of the tensor (a link is visible only part of the
time), and including them would put ~10% of the serving links above 1.0. The
all-samples p99 is emitted alongside under ``p99_including_zeros`` for
reference.

Usage (from the project root)::

    conda run -n pytorch python scripts/estimate_rate_stats.py
    conda run -n pytorch python scripts/estimate_rate_stats.py --stride 60 --out dataset/global/rate_stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset-dir", default="dataset/global",
        help="Directory holding link_data.h5 and the registries (default: dataset/global).",
    )
    parser.add_argument("--rate-key", default="k_max", help="H5 dataset holding the secure key rate.")
    parser.add_argument(
        "--stride", type=int, default=60,
        help="Read one row every N slots (default 60 = one row per hour). "
             "The full tensor is 4 GB decompressed; a stride keeps the scan cheap.",
    )
    parser.add_argument("--out", default=None, help="Output JSON path (default: <dataset-dir>/rate_stats.json).")
    return parser.parse_args()


def main() -> None:
    import h5py

    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    h5_path = dataset_dir / "link_data.h5"
    out_path = Path(args.out) if args.out else dataset_dir / "rate_stats.json"

    if not h5_path.exists():
        raise SystemExit(f"link_data.h5 not found: {h5_path}")

    # link_type per column, from the CSV registry (falling back to the H5 one).
    link_registry = dataset_dir / "link_registry.csv"
    types: list[str] = []
    if link_registry.exists():
        with link_registry.open(encoding="utf-8", newline="") as handle:
            rows = sorted(csv.DictReader(handle), key=lambda r: int(r["link_id"]))
            types = [row["link_type"].strip().upper() for row in rows]
    else:
        with h5py.File(h5_path, "r") as f:
            for item in f["link_registry"][:]:
                types.append(bytes(item["link_type"]).decode("utf-8").strip().upper())

    with h5py.File(h5_path, "r") as f:
        ds = f[args.rate_key]
        n_steps, n_links = ds.shape
        if len(types) != n_links:
            raise SystemExit(
                f"registry has {len(types)} links but the dataset has {n_links} columns; "
                "regenerate the registry or pass --rate-key."
            )
        rows = np.arange(0, n_steps, max(1, args.stride))
        block = np.asarray(ds[rows, :], dtype=np.float32)
        print(f"read {block.shape[0]} of {n_steps} rows ({args.stride}-slot stride) "
              f"x {n_links} links")

    block = np.where(np.isnan(block), 0.0, np.maximum(block, 0.0))
    positive = block[block > 0.0]
    if positive.size == 0:
        raise SystemExit("no positive rates found; check --rate-key and the dataset.")

    stats = {
        "source": str(h5_path),
        "rate_key": args.rate_key,
        "stride": int(max(1, args.stride)),
        "rows_scanned": int(block.shape[0]),
        "n_links": int(n_links),
        "positive_fraction": float(positive.size / block.size),
        "global": {
            "p99": float(np.percentile(positive, 99)),
            "p99_including_zeros": float(np.percentile(block, 99)),
            "p50": float(np.percentile(positive, 50)),
            "max": float(positive.max()),
        },
        "per_link_type": {},
    }

    type_array = np.asarray(types)
    for link_type in sorted(set(types)):
        cols = np.flatnonzero(type_array == link_type)
        sub = block[:, cols]
        sub_pos = sub[sub > 0.0]
        if sub_pos.size == 0:
            print(f"  {link_type:8s}: all-zero over the scan, skipped")
            continue
        stats["per_link_type"][link_type] = {
            "n_links": int(len(cols)),
            "p99": float(np.percentile(sub_pos, 99)),
            "p50": float(np.percentile(sub_pos, 50)),
            "max": float(sub_pos.max()),
        }
        print(f"  {link_type:8s}: n_links={len(cols):4d}  "
              f"p99={stats['per_link_type'][link_type]['p99']:12.1f}  "
              f"p50={stats['per_link_type'][link_type]['p50']:10.1f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}")
    print(f"global p99 = {stats['global']['p99']:.1f} bps "
          f"(RateNormalizer log_p99 denominator becomes log1p(p99) = "
          f"{np.log1p(stats['global']['p99']):.3f}; without this file it is "
          f"log1p(10.0) = {np.log1p(10.0):.3f})")


if __name__ == "__main__":
    main()
