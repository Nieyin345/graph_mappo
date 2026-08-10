# -*- coding: utf-8 -*-
"""Compact the global H5 dataset: keep only k_max (+ link_registry + attrs).

Drops los / distance / zenith (never read by training; los is derivable from
k_max with min_link_rate > 0), gzip-compresses k_max with row-friendly
full-width chunks so sequential training reads hit one chunk and random
day-start jumps decompress a single chunk.

Flow (safe, non-destructive):
  1. Preserve the original full file as ``link_data_full.h5`` (rename).
  2. Write the lean ``link_data.h5`` streaming row blocks from the preserved file.
  3. Verify k_max sample equality against the preserved file.
  4. Copy the lean file to ``link_data_compact_backup.h5`` (the "obtained" backup).

Usage:
    python scripts/compact_h5.py [--dataset-dir dataset/global]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np

CHUNK_ROWS = 2048
GZIP_LEVEL = 4
VERIFY_SAMPLES = 5000

# Datasets that are dropped from the lean file (never consumed by training).
_DROPPED = ("distance", "los", "zenith")


def compact_h5(dataset_dir: str | Path, verify_samples: int = VERIFY_SAMPLES) -> dict:
    """Compact ``link_data.h5`` in ``dataset_dir``; returns a status dict."""
    dataset_dir = Path(dataset_dir)
    source = dataset_dir / "link_data.h5"
    full = dataset_dir / "link_data_full.h5"
    backup = dataset_dir / "link_data_compact_backup.h5"
    if not source.exists():
        raise FileNotFoundError(f"source H5 not found: {source}")

    def dataset_keys(path: Path) -> set[str]:
        with h5py.File(path, "r") as f:
            return {name for name in f.keys() if isinstance(f[name], h5py.Dataset)}

    existing = dataset_keys(source)
    if not existing.intersection(_DROPPED):
        # Already lean: just ensure the backup copy exists.
        if not backup.exists():
            print(f"[compact] source is already lean; backing up to {backup.name} ...")
            shutil.copy2(source, backup)
        return {"status": "noop", "source": str(source), "backup": str(backup)}

    # 1. Preserve the original full file.
    if not full.exists():
        print(f"[compact] preserving original as {full.name} ...")
        source.replace(full)
    else:
        print(f"[compact] original already preserved as {full.name}; overwriting lean target.")
        if source.exists():
            source.unlink()

    # 2. Stream k_max from the preserved file into the lean target.
    with h5py.File(full, "r") as src:
        kmax_src = src["k_max"]
        T, L = kmax_src.shape
        attrs = dict(src.attrs)
        reg = src["link_registry"][:]
        chunk = (min(CHUNK_ROWS, T), L)
        print(f"[compact] streaming k_max ({T} x {L}) -> gzip(level={GZIP_LEVEL}) chunk={chunk} ...")
        t0 = time.perf_counter()
        with h5py.File(source, "w") as dst:
            dst.create_dataset(
                "k_max",
                shape=(T, L),
                dtype=kmax_src.dtype,
                chunks=chunk,
                compression="gzip",
                compression_opts=GZIP_LEVEL,
            )
            dst.create_dataset("link_registry", data=reg)
            for key, value in attrs.items():
                dst.attrs[key] = value
            rows_done = 0
            for t0_ in range(0, T, CHUNK_ROWS):
                t1_ = min(T, t0_ + CHUNK_ROWS)
                dst["k_max"][t0_:t1_, :] = kmax_src[t0_:t1_, :]
                rows_done = t1_
                if rows_done % (CHUNK_ROWS * 16) == 0:
                    print(f"    {rows_done}/{T} rows")
        elapsed = time.perf_counter() - t0
    size_mb = source.stat().st_size / 1e6

    # 3. Verify sampled equality.
    print(f"[compact] verifying {verify_samples} random samples ...")
    rng = np.random.RandomState(0)
    with h5py.File(full, "r") as src, h5py.File(source, "r") as dst:
        # h5py point selections require each index array to be sorted.
        n = min(verify_samples, T, L)
        rows = np.sort(rng.choice(T, size=n, replace=False))
        cols = np.sort(rng.choice(L, size=n, replace=False))
        a = src["k_max"][rows][:, cols]
        b = dst["k_max"][rows][:, cols]
        mismatches = int(np.count_nonzero(a != b))
        if mismatches:
            raise RuntimeError(f"k_max mismatch on {mismatches}/{verify_samples} samples")
        assert np.array_equal(src["link_registry"][:], dst["link_registry"][:]), "link_registry mismatch"

    # 4. Backup the obtained lean file.
    print(f"[compact] backing up lean file to {backup.name} ...")
    shutil.copy2(source, backup)

    print(
        f"[compact] done: {source.name} {size_mb:.1f} MB (gzip) in {elapsed:.1f}s; "
        f"original preserved as {full.name}"
    )
    return {
        "status": "compacted",
        "source": str(source),
        "full": str(full),
        "backup": str(backup),
        "size_mb": round(size_mb, 1),
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact dataset/global/link_data.h5 to k_max only.")
    parser.add_argument("--dataset-dir", default="dataset/global", help="directory containing link_data.h5")
    args = parser.parse_args()
    compact_h5(args.dataset_dir)


if __name__ == "__main__":
    main()
