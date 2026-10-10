"""Faster ERA5 friction-velocity download: one gridded request per cluster of nearby stations.

The gridded CDS queue cost is driven by the number of fields (variables x hours), not by the area, so a
box covering several stations costs about the same as a single point.  sshf / t2m / sp still come from
the point time-series dataset (fetch_era5_cds.py).  Output: the same .tmp/channel_v2/era5/<name>.csv.

  python tools/channel_v2/fetch_era5_boxes.py [--workers 3]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import fetch_era5_cds as F  # noqa: E402

CLUSTERS = {
    "eu_west": ["London", "Paris", "Madrid", "Rome", "Berlin", "Casablanca"],
    "eu_north": ["Stockholm", "Moscow"],
    "latam_n": ["MexicoCity", "Bogota"],
    "latam_s": ["SaoPaulo", "Santiago"],
    "africa_w": ["Lagos"],
    "asia_c": ["Almaty", "Mumbai"],
    "asia_se": ["Singapore", "Jakarta"],
    "asia_e": ["Beijing", "Tokyo"],
    "oceania": ["Perth", "Sydney", "Auckland"],
    "antarctica": ["McMurdo"],
    # stations added for the 45-station pool (tools/channel_v2/stations_45.csv)
    "mideast": ["Cairo", "Istanbul", "Tehran", "Riyadh"],
    "africa_e": ["AddisAbaba"],
    "africa_s": ["CapeTown"],
    "africa_w2": ["Dakar"],
    "asia_s": ["NewDelhi"],
    "asia_ne": ["Shanghai", "Seoul"],
    "na_c": ["Denver", "Houston"],
    "na_se": ["Miami"],
    "latam_s2": ["BuenosAires", "Brasilia"],
    "oceania2": ["Adelaide"],
}


def box(rows: pd.DataFrame) -> list[float]:
    """[N, W, S, E] on the 0.25 deg grid, containing every station's nearest grid point."""
    q = lambda x: np.round(np.asarray(x, float) * 4) / 4  # noqa: E731
    return [float(q(rows.lat).max()), float(q(rows.lon).min()),
            float(q(rows.lat).min()), float(q(rows.lon).max())]


def field_at(path: str, short_name: str, lat: float, lon: float) -> pd.Series:
    """Nearest ERA5 grid-cell time series for one NetCDF short name."""
    import h5py
    with h5py.File(path, "r") as f:
        la, lo = f["latitude"][()], f["longitude"][()]
        i = int(np.argmin(np.abs(la - lat)))
        j = int(np.argmin(np.abs(((lo - lon + 180) % 360) - 180)))
        tname = "valid_time" if "valid_time" in f else "time"
        units = f[tname].attrs.get("units", "seconds since 1970-01-01")
        units = units.decode() if isinstance(units, bytes) else str(units)
        step, _, origin = units.partition(" since ")
        t = pd.to_datetime(origin.strip()) + pd.to_timedelta(
            f[tname][()].astype("int64"), unit=step.strip()[0]
        )
        ds = f[short_name]
        v = ds[:, i, j].astype("float64")
        fill = ds.attrs.get("_FillValue", None)
        if fill is not None:
            v[v == float(np.asarray(fill).ravel()[0])] = np.nan
        v = v * float(np.asarray(ds.attrs.get("scale_factor", 1.0)).ravel()[0]) \
            + float(np.asarray(ds.attrs.get("add_offset", 0.0)).ravel()[0])
    return pd.Series(v, index=t)


def run_cluster(client, cname: str, rows: pd.DataFrame) -> str:
    todo = []
    for _, r in rows.iterrows():
        csv_path = os.path.join(F.OUT, f"{r['name']}.csv")
        ready = False
        if os.path.exists(csv_path):
            cached = pd.read_csv(csv_path)
            required = {"H_up_Wm2", "ustar", "t2m_K", "sp_Pa", "h_ERA5_m"}
            ready = (
                F.has_exact_2023_hourly_axis(cached)
                and required.issubset(cached.columns)
                and np.isfinite(cached[list(required)].to_numpy(dtype=float)).all()
            )
        if not ready:
            todo.append(r)
    if not todo:
        return f"cached {cname}"
    for r in todo:  # point time series (fast)
        tgt = os.path.join(F.OUT, "raw", f"{r['name']}_ts.zip")
        if not os.path.exists(tgt):
            F._retrieve(client, F.DATASET_TS, {"variable": F.VARS_TS,
                                               "location": {"longitude": float(r["lon"]), "latitude": float(r["lat"])},
                                               "date": [f"{F.YEAR}-01-01/{F.YEAR}-12-31"], "data_format": "csv"}, tgt)
    n, w, s, e = box(pd.DataFrame(todo))
    tgt = os.path.join(F.OUT, "raw", f"box_{cname}_grid025_zust_z.nc")
    if not os.path.exists(tgt):
        req = F._request(
            0.0, 0.0, list(range(1, 13)),
            variables=["friction_velocity", "geopotential"],
        )
        req["area"] = [n, w, s, e]
        F._retrieve(client, F.DATASET, req, tgt)
    msgs = []
    for r in todo:
        parts = F.read_ts_zip(os.path.join(F.OUT, "raw", f"{r['name']}_ts.zip"))
        parts["zust"] = field_at(tgt, "zust", float(r["lat"]), float(r["lon"]))
        parts["z"] = field_at(tgt, "z", float(r["lat"]), float(r["lon"]))
        df = F.to_table(parts)
        df.to_csv(os.path.join(F.OUT, f"{r['name']}.csv"), index=False)
        msgs.append(f"{r['name']} (u* p50 {np.nanmedian(df.ustar):.2f}, missing {int(df.isna().any(axis=1).sum())})")
    return f"ok {cname} box {[n, w, s, e]}: " + ", ".join(msgs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--stations", default="", help="station CSV (default: GS rows of dataset/global/node_registry.csv)")
    args = ap.parse_args()
    import cdsapi
    os.makedirs(os.path.join(F.OUT, "raw"), exist_ok=True)
    gs = F.load_stations(args.stations).set_index("name", drop=False)
    missing = set(gs.index) - {n for v in CLUSTERS.values() for n in v}
    client = cdsapi.Client()
    t0 = time.time()
    bad = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_cluster, client, c, gs.loc[[n for n in names if n in gs.index]]): c
                for c, names in CLUSTERS.items() if any(n in gs.index for n in names)}
        for fu in cf.as_completed(futs):
            try:
                print(fu.result(), f"({time.time() - t0:.0f}s)", flush=True)
            except Exception as exc:  # noqa: BLE001
                bad += 1
                print(f"FAILED {futs[fu]}: {exc}", flush=True)
    done = sum(os.path.exists(os.path.join(F.OUT, f"{n}.csv")) for n in gs.index)
    print(f"done, {bad} clusters failed; {done}/{len(gs)} stations have csv; not in any cluster: {sorted(missing)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
