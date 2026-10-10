"""Download ERA5 surface fluxes from the Copernicus CDS for the v2 rate model (validation only).

For every ground station in dataset/global/node_registry.csv, fetch hourly 2023 values of
  * surface_sensible_heat_flux (sshf, J m-2 accumulated over the 1 h interval ending at the timestamp; ECMWF sign: positive downward)
  * friction_velocity         (zust, m s-1)
  * 2m_temperature            (t2m, K)
  * surface_pressure          (sp, Pa)
  * geopotential              (z, m2 s-2; converted to ERA5 grid-orography h_ERA5_m)
at the nearest 0.25 deg grid point, and write .tmp/channel_v2/era5/<name>.csv with columns
  time, H_up_Wm2 (= -sshf / 3600, upward positive), ustar, t2m_K, sp_Pa, h_ERA5_m.
Does not touch weather/2023/, dataset/global/ or adapters/.

Setup (once):
  1. register at https://cds.climate.copernicus.eu and accept the licence of
     "ERA5 hourly data on single levels from 1940 to present" (dataset page -> Download -> Terms of use);
  2. put your personal access token in %USERPROFILE%\\.cdsapirc :
         url: https://cds.climate.copernicus.eu/api
         key: <your-personal-access-token>
  3. python tools/channel_v2/fetch_era5_cds.py --check      (one station, one day)
     python tools/channel_v2/fetch_era5_cds.py              (all stations; resumable)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, ".tmp", "channel_v2", "era5")
DATASET = "reanalysis-era5-single-levels"
DATASET_TS = "reanalysis-era5-single-levels-timeseries"   # point time series, fast; has no friction velocity
VARS_TS = ["surface_sensible_heat_flux", "2m_temperature", "surface_pressure"]
VARS = ["surface_sensible_heat_flux", "friction_velocity", "2m_temperature", "surface_pressure", "geopotential"]
SHORT = {"sshf": "sshf", "zust": "zust", "t2m": "t2m", "sp": "sp"}
OPTIONAL_SHORT = ("z",)  # surface geopotential [m^2 s^-2] -> ERA5 grid-orography height
YEAR = 2023
N_HOURS = 8760

def has_exact_2023_hourly_axis(df: pd.DataFrame) -> bool:
    if "time" not in df.columns:
        return False
    t = pd.to_datetime(df["time"], utc=True, errors="coerce")
    expected = pd.date_range(
        "2023-01-01T00:00:00Z", "2023-12-31T23:00:00Z", freq="h"
    )
    return (
        not t.isna().any()
        and len(t) == len(expected)
        and np.array_equal(
            t.astype("int64").to_numpy(),
            expected.astype("int64").to_numpy(),
        )
    )


def load_stations(path: str = "") -> pd.DataFrame:
    """Ground stations: explicit CSV or the v7 45-station source table by default."""
    if path:
        return pd.read_csv(path)
    return pd.read_csv(os.path.join(ROOT, "tools", "channel_v2", "stations_45.csv"))


def _request(lat: float, lon: float, months: list[int], days: list[int] | None = None,
             variables: list[str] | None = None) -> dict:
    # CDS can anchor the output grid at the requested area's north/west corner.
    # Snap the point itself; a station-centred +/-0.125 box shifts that grid.
    grid_lat = float(np.round(lat * 4) / 4)
    grid_lon = float(np.round(lon * 4) / 4)
    return {
        "product_type": ["reanalysis"],
        "variable": variables or VARS,
        "year": [str(YEAR)],
        "month": [f"{m:02d}" for m in months],
        "day": [f"{x:02d}" for x in (days or range(1, 32))],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [grid_lat, grid_lon, grid_lat, grid_lon],   # N, W, S, E
        "grid": [0.25, 0.25],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def _read_nc_bytes(raw: bytes) -> dict[str, pd.Series]:
    """Read one NetCDF4 (HDF5) file produced by the CDS; returns {short_name: Series indexed by UTC time}."""
    import h5py
    out = {}
    with h5py.File(io.BytesIO(raw), "r") as f:
        tname = "valid_time" if "valid_time" in f else "time"
        t = f[tname]
        units = t.attrs.get("units", b"seconds since 1970-01-01").decode() if isinstance(
            t.attrs.get("units", ""), bytes) else str(t.attrs.get("units", "seconds since 1970-01-01"))
        step, _, origin = units.partition(" since ")
        tt = pd.to_datetime(origin.strip()) + pd.to_timedelta(t[()].astype("int64"), unit=step.strip()[0])
        tt = tt.tz_localize(None) if getattr(tt, "tz", None) is not None else tt
        for k in tuple(SHORT) + OPTIONAL_SHORT:
            if k not in f:
                continue
            ds = f[k]
            v = ds[()].astype("float64")
            fill = ds.attrs.get("_FillValue", None)
            if fill is not None:
                v[v == float(np.asarray(fill).ravel()[0])] = np.nan
            v = v * float(np.asarray(ds.attrs.get("scale_factor", 1.0)).ravel()[0]) \
                + float(np.asarray(ds.attrs.get("add_offset", 0.0)).ravel()[0])
            v = v.reshape(v.shape[0], -1)[:, 0]          # single grid point
            out[k] = pd.Series(v, index=tt)
    return out


def read_download(path: str) -> dict[str, pd.Series]:
    """A CDS download is either one .nc or a zip of several .nc (accumulated and instantaneous streams)."""
    raw = open(path, "rb").read()
    parts: dict[str, pd.Series] = {}
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for n in z.namelist():
                if n.endswith(".nc"):
                    parts.update(_read_nc_bytes(z.read(n)))
    else:
        parts.update(_read_nc_bytes(raw))
    return parts


def read_ts_zip(path: str) -> dict[str, pd.Series]:
    """CSV (inside a zip) from the ERA5 point time-series dataset: valid_time,t2m,sp,sshf,latitude,longitude."""
    raw = open(path, "rb").read()
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            df = pd.concat([pd.read_csv(io.BytesIO(z.read(n))) for n in z.namelist() if n.endswith(".csv")])
    else:
        df = pd.read_csv(io.BytesIO(raw))
    t = pd.to_datetime(df["valid_time"])
    return {k: pd.Series(df[k].astype(float).values, index=t) for k in ("sshf", "t2m", "sp", "z") if k in df}


def to_table(parts: dict[str, pd.Series]) -> pd.DataFrame:
    missing = [k for k in SHORT if k not in parts]
    if missing:
        raise RuntimeError(f"variables missing in download: {missing}")
    df = pd.DataFrame(parts).sort_index()
    df = df[~df.index.duplicated()]
    full = pd.date_range(f"{YEAR}-01-01 00:00", f"{YEAR}-12-31 23:00", freq="h")
    df = df.reindex(full)
    h_era5_m = (df["z"].values / 9.80665) if "z" in df.columns else np.full(len(full), np.nan)
    return pd.DataFrame({
        "time": full.strftime("%Y-%m-%dT%H:%M"),
        # ERA5 single-level accumulated variables are one-hour accumulations ending at the timestamp; convert J/m2 to mean W/m2 and flip ECMWF downward-positive sign.
        "H_up_Wm2": -df["sshf"].values / 3600.0,
        "ustar": df["zust"].values,
        "t2m_K": df["t2m"].values,
        "sp_Pa": df["sp"].values,
        "h_ERA5_m": h_era5_m,
    })


def _retrieve(client, dataset: str, request: dict, target: str, tries: int = 30) -> None:
    """client.retrieve with back-off when the CDS rejects because too many requests are queued."""
    for i in range(tries):
        try:
            client.retrieve(dataset, request).download(target + ".part")
            os.replace(target + ".part", target)
            return
        except Exception as exc:  # noqa: BLE001
            if "temporarily limited" not in str(exc) or i == tries - 1:
                raise
            time.sleep(60)


def fetch_station(client, name: str, lat: float, lon: float, force: bool = False) -> str:
    csv = os.path.join(OUT, f"{name}.csv")
    if not force and os.path.exists(csv):
        cached = pd.read_csv(csv)
        required = {"H_up_Wm2", "ustar", "t2m_K", "sp_Pa", "h_ERA5_m"}
        if (
            has_exact_2023_hourly_axis(cached)
            and required.issubset(cached.columns)
            and np.isfinite(cached[list(required)].to_numpy(dtype=float)).all()
        ):
            return f"cached {name}"
    parts: dict[str, pd.Series] = {}
    # 1) sshf, t2m, sp from the point time-series dataset (one fast request per station)
    tgt = os.path.join(OUT, "raw", f"{name}_ts.zip")
    if not os.path.exists(tgt):
        _retrieve(client, DATASET_TS, {"variable": VARS_TS, "location": {"longitude": lon, "latitude": lat},
                                       "date": [f"{YEAR}-01-01/{YEAR}-12-31"], "data_format": "csv"}, tgt)
    parts.update(read_ts_zip(tgt))
    # 2) friction velocity from the gridded dataset (not in the time-series dataset)
    for months in ([list(range(1, 13))], [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]):
        try:
            for chunk in months:
                tgt = os.path.join(OUT, "raw", f"{name}_grid025_zust_z_{chunk[0]:02d}-{chunk[-1]:02d}.nc")
                if not os.path.exists(tgt):
                    _retrieve(client, DATASET, _request(
                        lat, lon, chunk, variables=["friction_velocity", "geopotential"]
                    ), tgt)
                for k, s in read_download(tgt).items():
                    parts[k] = pd.concat([parts[k], s]) if k in parts else s
            break
        except Exception as exc:  # noqa: BLE001  (cost limit on a full-year request -> quarters)
            if len(months) > 1:
                raise
            print(f"  {name}: full-year request failed ({str(exc)[:120]}), retrying by quarter", flush=True)
            parts.pop("zust", None)
            parts.pop("z", None)
    df = to_table(parts)
    n_nan = int(df[["H_up_Wm2", "ustar", "t2m_K", "sp_Pa", "h_ERA5_m"]].isna().any(axis=1).sum())
    df.to_csv(csv, index=False)
    return f"ok {name}: {len(df)} h, {n_nan} missing, H p50 {np.nanmedian(df.H_up_Wm2):.1f} W/m2, u* p50 {np.nanmedian(df.ustar):.2f} m/s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="credential/licence test: one station, one day")
    ap.add_argument("--workers", type=int, default=2, help="parallel CDS requests (CDS queues them anyway)")
    ap.add_argument("--only", default="", help="comma-separated station names")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stations", default="", help="station CSV (default: tools/channel_v2/stations_45.csv; pass an old registry explicitly for legacy reproduction)")
    args = ap.parse_args()
    try:
        import cdsapi
    except ImportError:
        print("pip install cdsapi h5py"); return 1
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    gs = load_stations(args.stations)
    if args.only:
        gs = gs[gs["name"].isin(args.only.split(","))]
    client = cdsapi.Client()
    if args.check:
        r = gs.iloc[0]
        tgt = os.path.join(OUT, "raw", "_check.nc")
        client.retrieve(DATASET, _request(r["lat"], r["lon"], [1], [1])).download(tgt)
        parts = read_download(tgt)
        for k, s in parts.items():
            print(f"  {k}: {len(s)} values, first {s.index[0]}, mean {np.nanmean(s.values):.4g}")
        required = tuple(SHORT) + OPTIONAL_SHORT
        print("CHECK OK" if all(k in parts for k in required) else "CHECK: variables missing")
        return 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_station, client, r["name"], float(r["lat"]), float(r["lon"]), args.force): r["name"]
                for _, r in gs.iterrows()}
        bad = 0
        for fu in cf.as_completed(futs):
            try:
                print(fu.result(), f"({time.time() - t0:.0f}s)", flush=True)
            except Exception as exc:  # noqa: BLE001
                bad += 1
                print(f"FAILED {futs[fu]}: {exc}", flush=True)
    print(f"done, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
