"""Attach reproducible DEM terrain height to the synthetic GS coordinates.

The 45 v7 GS entries are city-coordinate modelling sites, not surveyed optical
terminals.  This helper queries Open-Meteo's Elevation API, which documents
Copernicus DEM 2021 GLO-90 (90 m resolution), and writes an enriched station
CSV with explicit altitude provenance.  The channel builder remains offline:
it consumes the resulting CSV and never fetches elevations implicitly.

Default output:
    tools/channel_v2/stations_45_with_elevation.csv

For a future real optical-station data set, replace alt_km/h_GS_m with the
surveyed site elevation and set elevation_source accordingly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "tools" / "channel_v2" / "stations_45.csv"
DEFAULT_OUT = ROOT / "tools" / "channel_v2" / "stations_45_with_elevation.csv"
ENDPOINT = "https://api.open-meteo.com/v1/elevation"
SOURCE = "copernicus_dem_2021_glo90_dsm_via_open_meteo"
RESOLUTION_M = 90.0


def fetch_elevations(rows: list[dict]) -> list[float]:
    lats = ",".join(str(float(r["lat"])) for r in rows)
    lons = ",".join(str(float(r["lon"])) for r in rows)
    query = urllib.parse.urlencode({"latitude": lats, "longitude": lons}, safe=",")
    with urllib.request.urlopen(f"{ENDPOINT}?{query}", timeout=90) as resp:
        payload = json.load(resp)
    elev = payload.get("elevation")
    if not isinstance(elev, list) or len(elev) != len(rows):
        raise RuntimeError(f"unexpected elevation response: {payload!r}")
    out = [float(x) for x in elev]
    if any((not math.isfinite(x)) or x < -500.0 or x > 9000.0 for x in out):
        raise RuntimeError(f"invalid DEM elevation(s): {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stations", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    src = Path(args.stations)
    dst = Path(args.out)
    with src.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"empty station table: {src}")
    if len(rows) > 100:
        raise SystemExit("Open-Meteo elevation endpoint accepts at most 100 coordinates per request")

    elevations = fetch_elevations(rows)
    fields = list(rows[0].keys())
    for name in ("alt_km", "h_GS_m", "elevation_source", "elevation_resolution_m"):
        if name not in fields:
            fields.append(name)

    for row, h_m in zip(rows, elevations):
        row["h_GS_m"] = f"{h_m:.3f}"
        row["alt_km"] = f"{h_m / 1000.0:.6f}"
        row["elevation_source"] = SOURCE
        row["elevation_resolution_m"] = f"{RESOLUTION_M:.0f}"

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} stations -> {dst}")
    print(f"source={SOURCE}, resolution={RESOLUTION_M:.0f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
