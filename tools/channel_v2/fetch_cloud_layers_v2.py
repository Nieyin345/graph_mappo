"""Fetch 2023 layered cloud cover (low / mid / high) for the coverage cloud model of build_dataset_v2.

Same source and request style as fetch_inputs_v2.py (Open-Meteo historical weather API, ERA5,
hourly, UTC, 2023-01-01 .. 2023-12-31).  ERA5 layer definitions (ECMWF IFS documentation):
low = sigma > 0.8, mid = 0.45 < sigma <= 0.8, high = sigma <= 0.45 (about 6 km and above).
Cached to .tmp/channel_v2/cloud_layers/<name>.csv; does not touch .tmp/channel_v2/weather/.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, ".tmp", "channel_v2", "cloud_layers")
COLS = ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"]
SOURCE_MODEL = "era5"

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
           "&start_date=2023-01-01&end_date=2023-12-31&hourly=" + ",".join(COLS)
           + "&timezone=UTC&models=era5&elevation=nan&cell_selection=nearest")


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
    ap.add_argument("--stations", default=os.path.join(ROOT, "tools", "channel_v2", "stations_45.csv"))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    gs = pd.read_csv(args.stations)
    for _, row in gs.iterrows():
        path = os.path.join(args.out, f"{row['name']}.csv")
        if os.path.exists(path):
            cached = pd.read_csv(path)
            if (
                has_exact_2023_hourly_axis(cached)
                and "source_model" in cached.columns
                and set(cached["source_model"].dropna().astype(str)) == {SOURCE_MODEL}
            ):
                print("cached", row["name"])
                continue
        a = get_json(ARCHIVE.format(lat=row["lat"], lon=row["lon"]))["hourly"]
        df = pd.DataFrame({"time": a["time"], **{c: a[c] for c in COLS}})
        df["source_model"] = SOURCE_MODEL
        df["source_api"] = "open-meteo_historical_weather"
        df.to_csv(path, index=False)
        print(f"{row['name']}: rows={len(df)} nan={int(df[COLS].isna().sum().sum())} "
              + " ".join(f"{c[12:] or 'total'}={df[c].mean():.1f}%" for c in COLS), flush=True)
        time.sleep(1.0)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
