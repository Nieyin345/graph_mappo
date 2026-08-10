"""Estimate rate-normalization statistics from the real H5 dataset.

Writes ``dataset/global/rate_stats.json`` with the global and per-link-type
non-zero p99 / p99.9 / max of ``k_max`` (bps). ``RateNormalizer`` reads this
file so the ``log_p99`` normalization is calibrated to the real data instead of
a hard-coded value.

Usage (from the project root):

    conda run -n pytorch python scripts/estimate_rate_stats.py            # exact pass (slow, ~27 GB read)
    conda run -n pytorch python scripts/estimate_rate_stats.py --sample-every 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset" / "global" / "link_data.h5"


def quantiles(values: np.ndarray) -> dict[str, float]:
    nz = values[np.isfinite(values)]
    nz = np.maximum(nz, 0.0)
    frac_zero = float((nz == 0).mean()) if nz.size else 1.0
    nz = nz[nz > 0]
    if nz.size == 0:
        return {"p99": 0.0, "p99.9": 0.0, "max": 0.0, "frac_zero": 1.0}
    return {
        "p99": float(np.percentile(nz, 99)),
        "p99.9": float(np.percentile(nz, 99.9)),
        "max": float(nz.max()),
        "frac_zero": frac_zero,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sample-every", type=int, default=10, help="Rows to skip between samples (1 = exact).")
    parser.add_argument("--out", type=Path, default=ROOT / "dataset" / "global" / "rate_stats.json")
    args = parser.parse_args()

    t0 = time.time()
    # Stream in row chunks and keep only non-zero values so an exact pass does
    # not materialize the full (T, L) matrix in float64 (~18 GB). Values stay
    # float32; the final concatenation of non-zero samples is far smaller
    # (~17% of the matrix in the real dataset).
    _CHUNK_ROWS = 100000
    nonzero_parts: list[np.ndarray] = []
    with h5py.File(args.dataset, "r") as f:
        kmax = f["k_max"]
        T, L = kmax.shape
        rows = np.arange(0, T, max(1, args.sample_every))
        link_types = [bytes(item["link_type"]).decode("utf-8").strip().upper() for item in f["link_registry"][:]]
        theta_sec = int(f.attrs.get("theta_sec", 60))
        for start in range(0, rows.size, _CHUNK_ROWS):
            idx = rows[start : start + _CHUNK_ROWS]
            block = kmax[idx, :].astype(np.float32)
            nz = block[np.isfinite(block)]
            nz = np.maximum(nz, 0.0)
            nz = nz[nz > 0]
            if nz.size:
                nonzero_parts.append(nz)
    nonzero = np.concatenate(nonzero_parts) if nonzero_parts else np.zeros(0, dtype=np.float32)
    n_sampled = int(rows.size * L)
    frac_zero = 1.0 - (nonzero.size / n_sampled) if n_sampled else 1.0

    stats = {
        "meta": {
            "dataset": str(args.dataset),
            "total_rows": int(T),
            "sampled_rows": int(rows.size),
            "n_links": int(L),
            "theta_sec": theta_sec,
            "percentile_base": "nonzero k_max (bps), negative clipped to 0, NaN excluded",
        },
        "global": {**quantiles(nonzero), "frac_zero": frac_zero},
    }
    per_type: dict[str, dict] = {}
    type_names = sorted(set(link_types))
    for name in type_names:
        idx = np.array([i for i, t in enumerate(link_types) if t == name], dtype=np.int64)
        # Re-scan only the sampled rows for this link type, chunked as well.
        parts: list[np.ndarray] = []
        with h5py.File(args.dataset, "r") as f:
            kmax = f["k_max"]
            for start in range(0, rows.size, _CHUNK_ROWS):
                sel = rows[start : start + _CHUNK_ROWS]
                block = kmax[sel, :][:, idx].astype(np.float32)
                nz = block[np.isfinite(block)]
                nz = np.maximum(nz, 0.0)
                nz = nz[nz > 0]
                if nz.size:
                    parts.append(nz)
        typed = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        per_type[name] = {
            **quantiles(typed),
            "frac_zero": 1.0 - (typed.size / (rows.size * idx.size)) if idx.size else 1.0,
        }
    stats["per_link_type"] = per_type

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out} in {time.time() - t0:.1f}s")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
