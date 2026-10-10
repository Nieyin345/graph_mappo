"""Fetch the 2023 weather inputs needed by the v7.5 rate model.

For every ground station in the station CSV:
  * Open-Meteo historical weather (ERA5): total cloud cover, 2 m temperature,
    surface pressure, and hourly **visibility** (the v7.5 weather-extinction driver)
Hourly, UTC, 2023-01-01 .. 2023-12-31.  Cached to .tmp/channel_v2/weather/<name>.csv.
Does not touch weather/2023/ or dataset/global/.

The legacy CAMS aerosol optical depth column is no longer required: the v7.5 baseline
derives tropospheric weather extinction (aerosol, haze, fog, precipitation) from a single
Koschmieder-Kim visibility term (spec §3.3). ``--with-aod`` can still fetch the AOD column
for legacy-v2 reproducibility.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, ".tmp", "channel_v2", "weather")

WEATHER_SOURCE_MODEL = "era5"
AOD_SOURCE_MODEL = "cams_global"

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
        and t.astype("int64").to_numpy().tolist()
        == expected.astype("int64").to_numpy().tolist()
    )

ARCHIVE = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
           "&start_date=2023-01-01&end_date=2023-12-31"
           "&hourly=cloud_cover,temperature_2m,surface_pressure,visibility&timezone=UTC"
           "&models=era5&elevation=nan&cell_selection=nearest")
AQ = ("https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}"
      "&start_date=2023-01-01&end_date=2023-12-31&hourly=aerosol_optical_depth"
      "&domains=cams_global&timezone=UTC")


def get_json(url: str, tries: int = 5) -> dict:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception as exc:  # noqa: BLE001
            print(f"  retry {i + 1}/{tries}: {exc}", flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError(url)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", default=os.path.join(ROOT, "tools", "channel_v2", "stations_45.csv"), help="station CSV (default: tools/channel_v2/stations_45.csv)")
    ap.add_argument("--with-aod", action="store_true",
                    help="also fetch the legacy CAMS aerosol_optical_depth column (v2 reproducibility)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-download even when a cache with the required columns exists")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    gs = pd.read_csv(args.stations)
    needed = ["visibility"] + (["aod550"] if args.with_aod else [])
    for _, row in gs.iterrows():
        path = os.path.join(OUT, f"{row['name']}.csv")
        if os.path.exists(path) and not args.overwrite:
            cached = pd.read_csv(path, nrows=16)
            has_cols = all(c in cached.columns for c in needed)
            tags_ok = (
                "weather_source_model" in cached.columns
                and set(pd.read_csv(path, usecols=["weather_source_model"])["weather_source_model"].dropna().astype(str)) == {WEATHER_SOURCE_MODEL}
                and has_exact_2023_hourly_axis(pd.read_csv(path))
            )
            aod_ok = (not args.with_aod) or (
                "aod_source_model" in cached.columns
                and "aod550" in cached.columns
            )
            if has_cols and tags_ok and aod_ok:
                print("cached", row["name"])
                continue
        a = get_json(ARCHIVE.format(lat=row["lat"], lon=row["lon"]))["hourly"]
        df = pd.DataFrame({
            "time": a["time"],
            "cloud_cover": a["cloud_cover"],
            "temperature_2m": a["temperature_2m"],
            "surface_pressure": a["surface_pressure"],
            "visibility": a["visibility"],
        })
        if args.with_aod:
            q = get_json(AQ.format(lat=row["lat"], lon=row["lon"]))["hourly"]
            aod = pd.Series(q["aerosol_optical_depth"], index=q["time"])
            df["aod550"] = df["time"].map(aod)
            df["aod_source_model"] = AOD_SOURCE_MODEL
        df["weather_source_model"] = WEATHER_SOURCE_MODEL
        df["source_api"] = "open-meteo"
        n_nan = int(df["visibility"].isna().sum())
        df.to_csv(path, index=False)
        print(f"{row['name']}: rows={len(df)} vis_nan={n_nan} "
              f"cloud_mean={df['cloud_cover'].mean():.1f}% vis_median_km={df['visibility'].median()/1000.0:.1f}",
              flush=True)
        time.sleep(1.0)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
