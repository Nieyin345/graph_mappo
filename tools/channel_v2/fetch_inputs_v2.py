"""Fetch the 2023 weather inputs needed by the v2 rate model (validation only).

For every ground station in dataset/global/node_registry.csv:
  * Open-Meteo historical weather (ERA5): total cloud cover, 2 m temperature, surface pressure
  * Open-Meteo air quality (CAMS global): aerosol optical depth at 550 nm
Hourly, UTC, 2023-01-01 .. 2023-12-31.  Cached to .tmp/channel_v2/weather/<name>.csv.
Does not touch weather/2023/ or dataset/global/.
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

ARCHIVE = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
           "&start_date=2023-01-01&end_date=2023-12-31"
           "&hourly=cloud_cover,temperature_2m,surface_pressure&timezone=UTC")
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
    ap.add_argument("--stations", default="", help="station CSV (default: GS rows of dataset/global/node_registry.csv)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.stations:
        gs = pd.read_csv(args.stations)
    else:
        nodes = pd.read_csv(os.path.join(ROOT, "dataset", "global", "node_registry.csv"))
        gs = nodes[nodes["type"] == "GS"]
    for _, row in gs.iterrows():
        path = os.path.join(OUT, f"{row['name']}.csv")
        if os.path.exists(path) and len(pd.read_csv(path)) == 8760:
            print("cached", row["name"])
            continue
        a = get_json(ARCHIVE.format(lat=row["lat"], lon=row["lon"]))["hourly"]
        q = get_json(AQ.format(lat=row["lat"], lon=row["lon"]))["hourly"]
        df = pd.DataFrame({
            "time": a["time"],
            "cloud_cover": a["cloud_cover"],
            "temperature_2m": a["temperature_2m"],
            "surface_pressure": a["surface_pressure"],
        })
        aod = pd.Series(q["aerosol_optical_depth"], index=q["time"])
        df["aod550"] = df["time"].map(aod)
        n_nan = int(df["aod550"].isna().sum())
        df.to_csv(path, index=False)
        print(f"{row['name']}: rows={len(df)} aod_nan={n_nan} "
              f"cloud_mean={df['cloud_cover'].mean():.1f}% aod_median={df['aod550'].median():.3f}",
              flush=True)
        time.sleep(1.0)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
