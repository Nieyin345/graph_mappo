"""Align the ideal-channel build onto the real lean link set (journal exp (c)).

Usage (repo root, run once after `build_global_tensor.py --ideal-channel`):
    python tools/data_generation/align_ideal_channel.py
Used on 2026-09-28 to produce dataset/global_ideal_lean (configs/ideal_channel.yaml);
see docs/实验日志.md §11.19.

dataset/global_ideal (full build, --ideal-channel)  ->  dataset/global_ideal_lean
  * k_max columns reordered to the real lean link_registry (same 1978 links,
    matched by (node_u, node_v, link_type)); only rates differ.
  * link_registry / node_registry / rate_stats.json copied from the real set,
    so feature normalisation is identical (single-variable change).
Sanity report: node registry + HAP track identical; atmosphere-free links
(SAT-SAT, HAP-SAT, HAP-HAP) bit-identical; per-type rate change GS links.
"""
import csv, json, shutil, sys, filecmp
from pathlib import Path
import h5py, numpy as np

REAL, IDEAL, OUT = Path("dataset/global"), Path("dataset/global_ideal"), Path("dataset/global_ideal_lean")
REF = Path("dataset/global_real_ref")
if (OUT / "link_data.h5").exists():
    print("OUT exists, refusing to overwrite"); sys.exit(1)
OUT.mkdir(parents=True, exist_ok=True)

print("node_registry identical:", filecmp.cmp(REAL / "node_registry.csv", IDEAL / "node_registry.csv", shallow=False))
with h5py.File(REF / "hap_positions.h5") as a, h5py.File(IDEAL / "hap_positions.h5") as b:
    same = all(np.array_equal(a[k][()], b[k][()]) for k in a.keys()) and set(a.keys()) == set(b.keys())
    print("hap_positions identical:", same, list(a.keys()))

with h5py.File(IDEAL / "link_data.h5") as fi:
    print("ideal datasets:", {k: (fi[k].shape, str(fi[k].dtype)) for k in fi.keys()})
    ireg = fi["link_registry"][()]
    ikey = {(int(r["node_u"]), int(r["node_v"]), r["link_type"].decode()): i for i, r in enumerate(ireg)}
    with h5py.File(REAL / "link_data.h5") as fr:
        rreg = fr["link_registry"][()]
        types = np.array([r["link_type"].decode() for r in rreg])
        cols = np.array([ikey[(int(r["node_u"]), int(r["node_v"]), r["link_type"].decode())] for r in rreg])
        T, L = fr["k_max"].shape
        extra = sorted(set(range(len(ireg))) - set(cols.tolist()))
        print(f"real lean links {L}; ideal links {len(ireg)}; ideal-only links {len(extra)}")
        with h5py.File(OUT / "link_data.h5", "w") as fo:
            dk = fo.create_dataset("k_max", shape=(T, L), dtype="float32", chunks=fr["k_max"].chunks)
            fo.create_dataset("link_registry", data=rreg)
            for k, v in fr.attrs.items():
                fo.attrs[k] = v
            fo.attrs["channel"] = "ideal (clear sky, no solar background) -- build_global_tensor --ideal-channel"
            stats = {t: dict(n=0, same=0, sum_r=0.0, sum_i=0.0, pos_r=0, pos_i=0, maxdiff=0.0) for t in np.unique(types)}
            extra_pos = 0
            B = 1440 * 8
            for s in range(0, T, B):
                blk_i = fi["k_max"][s:s + B]
                out = blk_i[:, cols]
                dk[s:s + B] = out
                real = fr["k_max"][s:s + B]
                if extra:
                    extra_pos += int((blk_i[:, extra] >= 1.0).sum())
                for t in stats:
                    m = types == t
                    a, b = real[:, m], out[:, m]
                    st = stats[t]
                    st["n"] += a.size; st["same"] += int((a == b).sum())
                    st["sum_r"] += float(a.sum()); st["sum_i"] += float(b.sum())
                    st["pos_r"] += int((a >= 1).sum()); st["pos_i"] += int((b >= 1).sum())
                    st["maxdiff"] = max(st["maxdiff"], float(np.abs(a - b).max()))
    print(f"ideal-only links: positive link-steps {extra_pos}")
    print(f"{'type':8s} {'links':>5s} {'identical':>9s} {'maxdiff':>10s} {'mean_real':>10s} {'mean_ideal':>10s} {'ratio':>6s} {'up_real':>7s} {'up_ideal':>8s}")
    for t, st in stats.items():
        nl = int((types == t).sum())
        print(f"{t:8s} {nl:5d} {st['same']/st['n']:9.4f} {st['maxdiff']:10.3g} {st['sum_r']/st['n']:10.2f} {st['sum_i']/st['n']:10.2f} "
              f"{(st['sum_i']/max(st['sum_r'],1e-9)):6.3f} {st['pos_r']/st['n']:7.4f} {st['pos_i']/st['n']:8.4f}")

for f in ("link_registry.csv", "node_registry.csv", "rate_stats.json"):
    shutil.copy2(REAL / f, OUT / f)
for f in ("build_metadata.json", "weather_coverage.csv"):
    shutil.copy2(IDEAL / f, OUT / f)
print("MARK_ALIGN_DONE")
