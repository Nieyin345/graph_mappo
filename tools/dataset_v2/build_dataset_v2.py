"""Build a link-rate dataset with the v7 (default) or legacy v2 rate model into a NEW directory.

    python tools/dataset_v2/build_dataset_v2.py --check-inputs          # are all weather files there?
    python tools/dataset_v2/build_dataset_v2.py --out dataset/global_v7 --constellation walker60 --workers 12   # v7 (default)
    python tools/dataset_v2/build_dataset_v2.py --out .tmp/ds_smoke --days 2 --allow-era5-fallback
    python tools/dataset_v2/build_dataset_v2.py --channel-model v2 --out dataset/global_v2e --workers 12       # legacy v2

What it does
------------
* nodes: the ground stations of ``tools/channel_v2/stations_45.csv`` (``base`` = 1 marks the 30-station
  base network, listed first), one HAP per station at 20 km (``--hap-mobility trajectory`` (default):
  each HAP drifts on the shared 24 h stratospheric-balloon track around its home station, altitude
  ~18.6-25 km; ``static``: pinned above the station, old v2 behavior), and the Walker-delta
  53:30/5/1 constellation at 550 km (``GlobalSatellites.walker_delta``, commit 98b3d2c);
* candidate links (eq. (1)): every SAT-GS, SAT-HAP and SAT-SAT pair, and the one-to-one HAP-GS home
  pairs (HAP_i <-> GS_i); HAP-GS availability is decided per slot from the (moving) HAP position and
  the ``alpha_min`` elevation mask, with no static prefilter;
* per slot (Delta = 60 s, 2023, t = minutes since 2023-01-01 00:00 UTC): geometry, visibility, mean
  transmittance and background probability from ``tools/channel_v2/rate_model_v2.RateModelV2``
  (ERA5 Monin-Obukhov C_n^2, CAMS AOD, SMF Monte-Carlo table; rate-model-v2 §3 upper atmosphere, §5.3 scintillation
  expectation (log-normal) and §1.3 ISL background tiers always on; no empirical excess loss), the finite-key length of
  ``finite_key.key_length`` and the cloud term of ``--cloud-model``:
    - ``binary_total`` (default, = dataset/global_v2): Lund-Shanklin indicator chi from the total cloud
      cover (one uniform per station and hour, shared by all links of that station, eq. (17));
      rate R = chi * ell / Delta (eq. (18));
    - ``coverage_layers`` (10-01, user decision): the low+mid cloud cover is read as the cloud-blocked
      fraction of the slot (advected cloud field: blocked time fraction = coverage), so the cloud-free
      fraction f = P_LS(alpha, C_lm), C_lm = 1-(1-C_low)(1-C_mid) (random overlap), and the key is the
      finite-key length of a block of f*N pulses; high (ice) cloud is a mean transmittance
      T_ci = 1 - C_high (1 - exp(-tau_ci m(alpha))) on eta (thin cirrus as attenuation at 1.55 um,
      Poulenard et al., Proc. SPIE 8971, 897107, 2014); R = ell(T_ci eta, Y0; f N) / Delta;
* links with zero rate over the whole period are dropped (as in the current dataset) and listed in
  ``link_registry_skipped.csv``;
* **v7.5 is the DEFAULT channel model** (``--channel-model v7``): static HAP, 15-degree
  ground elevation mask, 85 km ISL grazing height, bounded Beta two-moment collection efficiency,
  one station-scale persistent low/mid-cloud state per GS and scheduling slot, and a physical acquisition
  time ``t_acq`` (default 5 s) whose first-activation geometry is re-sampled over the actual usable
  window. Geometry sub-steps never re-sample the cloud state. The H5 stores conditional-clear rates plus
  compact QKD sufficient statistics so the runtime can aggregate logical QKD blocks across consecutive
  active slots and release inventory only when a block closes. ``--channel-model v2`` remains available
  only for reproducing legacy datasets. ``--domain-policy mark`` is diagnostic-only; finite-key outputs
  are expected-count performance estimates, not device security certificates.

Output (same layout as dataset/global, read by qkd_rl/link/rate_provider.py; v2 remains backward-compatible):
``node_registry.csv``, ``link_registry.csv``, ``link_registry_skipped.csv``, ``link_data.h5``,
``sat_positions.h5``, ``hap_positions.h5``, ``gs_positions.csv``, ``link_diag.csv``, and
``build_metadata.json``. For v7.5, ``link_data.h5`` stores:
- ``k_max`` / ``k_max_first``: 60 s finite-key control surfaces;
- ``k_proxy`` / ``k_proxy_first``: block-compatible asymptotic policy surfaces;
- compact ``qkd_gain``, ``qkd_y1``, ``qkd_usable_fraction`` sufficient statistics
  (plus first-activation variants) for runtime logical-block aggregation.
Generate ``rate_stats.json`` with ``scripts/estimate_rate_stats.py``; its auto mode follows the H5
``policy_rate_recommended`` attribute.

It never writes into dataset/global and refuses an existing output directory unless ``--overwrite`` is
given and the directory was created by this script.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import dataclasses
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# ① 仓库根先入 path：v7 模块用包式导入（from tools.channel_v2... / from tools.dataset_v2...），
#    只有根在 path 上才解析得到；缺了它 `--channel-model v7` 直接 ModuleNotFoundError。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ② 三个子目录也入 path：v2 路径用的是平铺导入（from finite_key import ...）。
for sub in ("tools/channel_v2", "tools/data_generation", "tools/dataset_v2"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from finite_key import FiniteKeyParams, slot_rate, slot_rate_states  # noqa: E402
from rate_model_v2 import (ParamsV2, RateModelV2, airmass_kasten_young, cn2_monin_obukhov,  # noqa: E402
                           p_clear_lund_shanklin, sun_elevation_deg, isl_illumination_class, MODEL_REVISION)

MARKER = ".built_by_build_dataset_v2"
MARKER_V7 = ".built_by_build_dataset_v7"
YEAR_HOURS = 8760
SLOT_MIN = 1                      # Delta = 60 s
BLOCK = 2048                      # rows per block = HDF5 chunk height (as in dataset/global)
LINK_BATCH = 256                  # links evaluated together inside a block (memory bound)
R_E = 6371.0
TYPES = ("HAP-GS", "SAT-GS", "SAT-HAP", "SAT-SAT")


@dataclasses.dataclass(frozen=True)
class BuildConfig:
    alpha_min_deg: float = 5.0    # eq. (1), ground-station elevation mask (as dataset/global)
    h_min_km: float = 15.0        # legacy v2 H-S/S-S cutoff; v7 maps this parameter only to SAT-SAT ISL grazing altitude
    hap_alt_km: float = 20.0
    # "static" = every HAP pinned 20 km above its home station (old v2 behavior);
    # "trajectory" = each HAP drifts on the shared balloon track around its home station (legacy).
    hap_mobility: str = "trajectory"
    sat_constellation: str = "walker"
    cloud_seed: int = 2023
    year: int = 2023
    protocol: str = "finite_key"  # "finite_key" (paper eq. (18)) | "asymptotic" (diagnostics only)
    cloud_model: str = "binary_total"   # legacy v2 default; v7.5 overrides to "station_markov"
    # visible optical depth of a high (ice) cloud layer; at 1.55 um the ice-crystal extinction efficiency
    # is ~2 as in the visible (geometric-optics limit), so the visible value is used. 0.3 = boundary
    # thin/opaque cirrus (Sassen & Cho 1992 classes); CALIPSO / ground-lidar annual means ~0.3.
    tau_cirrus: float = 0.3
    channel_model: str = "v2"
    domain_policy: str = "error"
    # v7.5: capture is a physical time inside the 60 s scheduling slot.
    capture_time_s: float = 5.0
    # station-scale cloud persistence; one C_g(t) per station and scheduling slot.
    cloud_tau_s: float = 3600.0
    # v7 intra-slot resolution: sub-samples per 60 s slot, at sub-interval MIDPOINTS.
    # The v7.5 paper baseline is n=4; n=1 is only a cheaper diagnostic.
    # Historical convergence checks found <1% change for n=4 vs 16 and <0.5% for n=8 vs 16.
    intra_slot_samples: int = 4


# ----------------------------------------------------------------------------- inputs
def _optional_float(row: dict, *keys: str):
    for key in keys:
        raw = row.get(key, "")
        if raw is not None and str(raw).strip() != "":
            value = float(raw)
            if not np.isfinite(value):
                raise SystemExit(f"non-finite station field {key} for {row.get('name', '')}")
            return value
    return None


def read_stations(path: Path) -> list[dict]:
    """Read GS coordinates without discarding vertical-reference metadata.

    Accepted station heights are alt_km, or h_GS / h_GS_m / alt_m /
    elevation_m in metres. ERA5 grid-orography height is read from h_ERA5 /
    h_ERA5_m / era5_alt_m. Missing values are retained as explicit flags
    instead of being silently confused with sea level.
    """
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        alt_km = _optional_float(r, "alt_km")
        h_gs_m = _optional_float(r, "h_GS", "h_GS_m", "h_gs_m", "alt_m", "elevation_m")
        gs_explicit = alt_km is not None or h_gs_m is not None
        if alt_km is not None and h_gs_m is not None:
            if abs(1000.0 * float(alt_km) - float(h_gs_m)) > 1.0:
                raise SystemExit(
                    f"{r['name']}: alt_km={alt_km} and h_GS={h_gs_m} m disagree by more than 1 m"
                )
        if alt_km is None:
            alt_km = 0.0 if h_gs_m is None else h_gs_m / 1000.0
        h_gs_m = 1000.0 * float(alt_km)

        h_era5_m = _optional_float(r, "h_ERA5", "h_ERA5_m", "h_era5_m", "era5_alt_m")
        era5_explicit = h_era5_m is not None
        if h_era5_m is None:
            h_era5_m = h_gs_m
        if not np.isfinite([alt_km, h_gs_m, h_era5_m]).all():
            raise SystemExit(f"{r['name']}: station/ERA5 elevations must be finite")

        out.append(dict(
            name=r["name"], lat=float(r["lat"]), lon=float(r["lon"]),
            alt_km=float(alt_km), h_GS_m=float(h_gs_m), h_ERA5_m=float(h_era5_m),
            gs_altitude_explicit=bool(gs_explicit),
            gs_altitude_source=(
                str(r.get("elevation_source") or r.get("gs_altitude_source") or "").strip()
                if gs_explicit else "missing"
            ),
            gs_altitude_resolution_m=_optional_float(r, "elevation_resolution_m"),
            era5_altitude_explicit=bool(era5_explicit),
            base=int(r.get("base", 1) or 0), continent=r.get("continent", ""),
        ))
    names = [r["name"] for r in out]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate station names in {path}")
    return out


def _read_csv_columns(
    path: Path,
    cols: list[str],
    *,
    require_2023_hourly_utc: bool = False,
) -> dict[str, np.ndarray]:
    import pandas as pd
    df = pd.read_csv(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    if require_2023_hourly_utc:
        if "time" not in df.columns:
            raise SystemExit(f"{path}: missing time column for hourly UTC alignment")
        parsed = pd.to_datetime(df["time"], utc=True, errors="coerce")
        expected = pd.date_range(
            "2023-01-01T00:00:00Z",
            "2023-12-31T23:00:00Z",
            freq="h",
        )
        if parsed.isna().any() or len(parsed) != len(expected):
            raise SystemExit(
                f"{path}: expected exactly {len(expected)} parseable hourly UTC timestamps"
            )
        actual_ns = parsed.astype("int64").to_numpy()
        expected_ns = expected.astype("int64").to_numpy()
        if not np.array_equal(actual_ns, expected_ns):
            bad = int(np.flatnonzero(actual_ns != expected_ns)[0])
            raise SystemExit(
                f"{path}: hourly UTC time axis mismatch at row {bad}: "
                f"got {parsed.iloc[bad]}, expected {expected[bad]}"
            )
    return {c: df[c].astype(float).to_numpy() for c in cols}


def _read_source_tags(path: Path, names: tuple[str, ...]) -> dict[str, str | None]:
    """Read constant provenance columns without coupling physics code to the fetcher."""
    import pandas as pd
    df = pd.read_csv(path, nrows=16)
    out = {}
    for name in names:
        if name not in df.columns:
            out[name] = None
            continue
        vals = {str(x).strip() for x in df[name].dropna().tolist() if str(x).strip()}
        out[name] = next(iter(vals)) if len(vals) == 1 else None
    return out


def _read_era5_station(path: Path):
    """Read required ERA5 surface series on the exact 2023 hourly UTC axis."""
    import pandas as pd
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise SystemExit(f"{path}: missing time column for ERA5 hourly alignment")
    parsed = pd.to_datetime(df["time"], utc=True, errors="coerce")
    expected = pd.date_range(
        "2023-01-01T00:00:00Z",
        "2023-12-31T23:00:00Z",
        freq="h",
    )
    if (
        parsed.isna().any()
        or len(parsed) != len(expected)
        or not np.array_equal(
            parsed.astype("int64").to_numpy(),
            expected.astype("int64").to_numpy(),
        )
    ):
        raise SystemExit(
            f"{path}: ERA5 rows must exactly match the 8760-hour 2023 UTC validity-time axis"
        )
    required = ["H_up_Wm2", "ustar", "t2m_K", "sp_Pa"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    data = {c: df[c].astype(float).to_numpy() for c in required}
    h_era5_m = None
    if "h_ERA5_m" in df.columns:
        h = df["h_ERA5_m"].astype(float).to_numpy()
        finite = h[np.isfinite(h)]
        if finite.size:
            h_era5_m = float(np.median(finite))
            if np.nanmax(np.abs(finite - h_era5_m)) > 1.0:
                raise SystemExit(
                    f"{path}: h_ERA5_m should be a station-static grid-orography height"
                )
    return data, h_era5_m


LAYER_COLS = ["cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"]


def check_inputs(stations, weather_dir: Path, era5_dir: Path, layers_dir: Path | None = None,
                 strict: bool = False) -> dict:
    """Per station: Open-Meteo weather (cloud, AOD) and ERA5 fluxes present, 8760 rows, NaN counts."""
    rep = {}
    for s in stations:
        r = {
            "h_GS_m": float(s["h_GS_m"]),
            "h_ERA5_m": float(s["h_ERA5_m"]),
            "gs_altitude_explicit": bool(s.get("gs_altitude_explicit", False)),
            "era5_altitude_explicit": bool(s.get("era5_altitude_explicit", False)),
        }
        w = weather_dir / f"{s['name']}.csv"
        e = era5_dir / f"{s['name']}.csv"
        if w.exists():
            # Strict v7 uses this file only for CAMS AOD. Low/mid/high cloud
            # comes exclusively from the ERA5 layered-cloud input below.
            weather_cols = ["aod550"] if strict else ["cloud_cover", "aod550"]
            d = _read_csv_columns(w, weather_cols, require_2023_hourly_utc=True)
            tags = _read_source_tags(w, ("weather_source_model", "aod_source_model"))
            r["weather_source_model"] = tags["weather_source_model"]
            r["aod_source_model"] = tags["aod_source_model"]
            r["weather_rows"] = int(len(d["aod550"]))
            r["aod_nan"] = int((~np.isfinite(d["aod550"])).sum())
            r["aod_negative"] = int(np.count_nonzero(
                np.isfinite(d["aod550"]) & (d["aod550"] < 0)
            ))
            if not strict:
                r["cloud_nan"] = int((~np.isfinite(d["cloud_cover"])).sum())
                r["cloud_out_of_range"] = int(np.count_nonzero(
                    np.isfinite(d["cloud_cover"])
                    & ((d["cloud_cover"] < 0) | (d["cloud_cover"] > 100))
                ))
        else:
            r["weather_rows"] = 0
        if e.exists():
            d, h_era5_file = _read_era5_station(e)
            r["era5_rows"] = int(len(d["ustar"]))
            r["era5_nan_hours"] = int((~np.isfinite(np.stack(list(d.values())))).any(axis=0).sum())
            if (not r["era5_altitude_explicit"]) and h_era5_file is not None:
                r["h_ERA5_m"] = h_era5_file
                r["era5_altitude_explicit"] = True
                r["era5_altitude_source"] = "era5_file_geopotential"
            elif r["era5_altitude_explicit"]:
                r["era5_altitude_source"] = "station_csv"
            else:
                r["era5_altitude_source"] = "missing"
        else:
            r["era5_rows"] = 0
            r["era5_altitude_source"] = "station_csv" if r["era5_altitude_explicit"] else "missing"
        r["ok"] = r["weather_rows"] == YEAR_HOURS and r["era5_rows"] == YEAR_HOURS
        if strict and (not r["gs_altitude_explicit"] or not r["era5_altitude_explicit"]):
            r["ok"] = False
        if strict and (
            r.get("aod_source_model") != "cams_global"
            or r.get("aod_nan", 0)
            or r.get("era5_nan_hours", 0)
            or r.get("aod_negative", 0)
        ):
            r["ok"] = False
        if layers_dir is not None:
            f = layers_dir / f"{s['name']}.csv"
            if f.exists():
                d = _read_csv_columns(f, LAYER_COLS, require_2023_hourly_utc=True)
                layer_tags = _read_source_tags(f, ("source_model",))
                r["layer_source_model"] = layer_tags["source_model"]
                r["layer_rows"] = int(len(d[LAYER_COLS[0]]))
                layer_values = np.stack(list(d.values()))
                r["layer_nan"] = int((~np.isfinite(layer_values)).sum())
                r["layer_out_of_range"] = int(np.count_nonzero(
                    np.isfinite(layer_values) & ((layer_values < 0) | (layer_values > 100))))
            else:
                r["layer_rows"] = 0
            r["ok"] = r["ok"] and r["layer_rows"] == YEAR_HOURS
            if strict and (
                r.get("layer_source_model") != "era5"
                or r.get("layer_nan", 0)
                or r.get("layer_out_of_range", 0)
            ):
                r["ok"] = False
        rep[s["name"]] = r
    return rep


def load_cloud_layers(stations, layers_dir: Path, *, strict: bool = False):
    """(G, 8760) low+mid combined cover C_lm = 1-(1-C_low)(1-C_mid) and high cover, fractions 0..1."""
    G = len(stations)
    c_lm = np.zeros((G, YEAR_HOURS))
    c_hi = np.zeros((G, YEAR_HOURS))
    fill = {}
    for gi, s in enumerate(stations):
        layer_path = layers_dir / f"{s['name']}.csv"
        d = _read_csv_columns(layer_path, LAYER_COLS, require_2023_hourly_utc=True)
        if strict:
            tags = _read_source_tags(layer_path, ("source_model",))
            if tags["source_model"] != "era5":
                raise SystemExit(
                    f"{s['name']}: v7.5 cloud layers require source_model=era5; "
                    "regenerate with fetch_cloud_layers_v2.py"
                )
        if len(d[LAYER_COLS[0]]) != YEAR_HOURS:
            raise SystemExit(f"{s['name']}: cloud-layer file has {len(d[LAYER_COLS[0]])} rows, need {YEAR_HOURS}")
        v = {}
        for c in LAYER_COLS:
            x = d[c]
            fill.setdefault(s["name"], {})[c + "_nan"] = int(np.isnan(x).sum())
            if strict and not np.isfinite(x).all():
                raise SystemExit(f"{s['name']}: {c} has missing/non-finite cloud-layer values; "
                                 "prefill from training-period or historical data")
            if strict and np.any((x < 0) | (x > 100)):
                raise SystemExit(f"{s['name']}: {c} contains cover outside [0, 100] percent")
            v[c] = np.clip(np.nan_to_num(x, nan=float(np.nanmean(x)) if np.isfinite(x).any() else 0.0) / 100.0, 0, 1)
        c_lm[gi] = 1.0 - (1.0 - v["cloud_cover_low"]) * (1.0 - v["cloud_cover_mid"])
        c_hi[gi] = v["cloud_cover_high"]
    return c_lm, c_hi, fill


def _most_regularization_report(H_up, ustar, T_K, P_Pa, *, z, cn2_min, zeta_max):
    """Count numerical regularization triggers using the same MOST equations as cn2_monin_obukhov."""
    H = np.asarray(H_up, dtype=float)
    us_raw = np.asarray(ustar, dtype=float)
    T = np.asarray(T_K, dtype=float)
    P = np.asarray(P_Pa, dtype=float)
    us = np.maximum(us_raw, 1e-3)
    rho_cp = P / (287.05 * T) * 1005.0
    t_star = -H / (rho_cp * us)
    # Same algebraic neutral-limit form as cn2_monin_obukhov().
    xi_raw = -z * 0.4 * 9.81 * H / (rho_cp * T * us ** 3)
    xi = np.clip(xi_raw, -1e3, zeta_max)
    f_t = np.where(xi < 0, 5.6 * (1.0 - 6.5 * np.minimum(xi, 0.0)) ** (-2.0 / 3.0),
                   5.5 * (1.0 + 1.1 * np.maximum(xi, 0.0) ** (2.0 / 3.0)))
    ct2 = t_star ** 2 * z ** (-2.0 / 3.0) * f_t
    cn2_raw = (79e-6 * (P / 100.0) / T ** 2) ** 2 * ct2
    n = int(H.size)
    denom = max(n, 1)
    count = lambda mask: int(np.count_nonzero(mask))
    return dict(
        source="ERA5 MOST-to-HV input",
        n_hours=n,
        ustar_floor_hours=count(us_raw < 1e-3),
        ustar_floor_fraction=count(us_raw < 1e-3) / denom,
        near_neutral_flux_hours=count(np.abs(H) < 1e-6),
        near_neutral_flux_fraction=count(np.abs(H) < 1e-6) / denom,
        zeta_low_clip_hours=count(xi_raw < -1e3),
        zeta_low_clip_fraction=count(xi_raw < -1e3) / denom,
        zeta_high_clip_hours=count(xi_raw > zeta_max),
        zeta_high_clip_fraction=count(xi_raw > zeta_max) / denom,
        zeta_nonfinite_hours=count(~np.isfinite(xi_raw)),
        zeta_nonfinite_fraction=count(~np.isfinite(xi_raw)) / denom,
        cn2_min_floor_hours=count((~np.isfinite(cn2_raw)) | (cn2_raw < cn2_min)),
        cn2_min_floor_fraction=count((~np.isfinite(cn2_raw)) | (cn2_raw < cn2_min)) / denom,
    )


def load_weather(stations, weather_dir: Path, era5_dir: Path, p: ParamsV2, allow_era5_fallback: bool,
                 *, strict: bool = False, apply_height_correction: bool = False,
                 allow_missing_station_elevation: bool = False):
    """Return hourly weather arrays used by the channel builder.

    Legacy v2 stores a near-ground C_n^2 value in a_cn2. Strict v7.5 first
    computes MOST C_{n,ref}^2 at z_ref=10 m and then stores the matched local
    AGL-HV coefficient A(t) in that historical a_cn2 array name. Callers must
    therefore treat it as the HV coefficient when channel_model=v7.
    """
    G = len(stations)
    cloud = np.zeros((G, YEAR_HOURS))
    aod = np.zeros((G, YEAR_HOURS))
    a_cn2 = np.full((G, YEAR_HOURS), np.nan)
    ok = np.zeros((G, YEAR_HOURS), dtype=bool)
    fill = {}
    for gi, s in enumerate(stations):
        weather_path = weather_dir / f"{s['name']}.csv"
        weather_cols = ["aod550"] if strict else ["cloud_cover", "aod550"]
        d = _read_csv_columns(weather_path, weather_cols, require_2023_hourly_utc=True)
        if strict:
            tags = _read_source_tags(weather_path, ("aod_source_model",))
            if tags["aod_source_model"] != "cams_global":
                raise SystemExit(
                    f"{s['name']}: v7.5 AOD requires aod_source_model=cams_global; "
                    "regenerate with fetch_inputs_v2.py"
                )
        if len(d["aod550"]) != YEAR_HOURS:
            raise SystemExit(f"{s['name']}: AOD file has {len(d['aod550'])} rows, need {YEAR_HOURS}")
        a = d["aod550"]
        if strict and not np.isfinite(a).all():
            raise SystemExit(
                f"{s['name']}: CAMS AOD contains missing values; "
                "prefill from training-period or historical data"
            )
        if strict and np.any(a < 0):
            raise SystemExit(f"{s['name']}: CAMS AOD must be nonnegative")
        fill[s["name"]] = {"aod_nan": int(np.isnan(a).sum())}
        if not strict:
            c = d["cloud_cover"]
            fill[s["name"]]["cloud_nan"] = int(np.isnan(c).sum())
            cloud[gi] = (
                np.nan_to_num(
                    c,
                    nan=float(np.nanmean(c)) if np.isfinite(c).any() else 0.0,
                )
                / 100.0
            )
        aod[gi] = np.nan_to_num(a, nan=float(np.nanmedian(a)) if np.isfinite(a).any() else 0.1)
        e = era5_dir / f"{s['name']}.csv"
        if not e.exists():
            if not allow_era5_fallback:
                raise SystemExit(f"no ERA5 file for {s['name']} ({e}); wait for the download or pass "
                                 "--allow-era5-fallback (smoke tests only)")
            fill[s["name"]]["era5"] = "missing -> day/night A for every hour"
            if strict:
                fill[s["name"]]["most_regularization"] = {"source": "explicit day/night baseline fallback", "n_hours": 0}
                fill[s["name"]]["hv_match_floor"] = {"source": "fallback A coefficients already use HV convention", "n_hours": 0}
            continue
        f, h_era5_file = _read_era5_station(e)
        if len(f["ustar"]) != YEAR_HOURS:
            raise SystemExit(f"{s['name']}: ERA5 file has {len(f['ustar'])} rows, need {YEAR_HOURS}")
        h_gs_m = float(s.get("h_GS_m", 1000.0 * float(s.get("alt_km", 0.0))))
        if s.get("era5_altitude_explicit", False):
            h_era5_m = float(s["h_ERA5_m"])
            height_source = "station_csv"
        elif h_era5_file is not None:
            h_era5_m = float(h_era5_file)
            height_source = "era5_file_geopotential"
            s["era5_altitude_explicit"] = True
        elif allow_missing_station_elevation:
            h_era5_m = h_gs_m
            height_source = "smoke_fallback_equal_to_gs"
        else:
            raise SystemExit(
                f"{s['name']}: v7.5 requires ERA5 grid-orography height; "
                "provide h_ERA5 in the station CSV or regenerate the ERA5 CSV with h_ERA5_m"
            )
        # Persist the resolved ERA5 reference surface so geometry/output
        # metadata and the atmospheric correction use the same height.
        s["h_ERA5_m"] = float(h_era5_m)
        s["era5_altitude_source"] = height_source
        delta_h_m = h_gs_m - h_era5_m
        lapse = 6.5e-3
        t_era5 = np.asarray(f["t2m_K"], dtype=float)
        p_era5 = np.asarray(f["sp_Pa"], dtype=float)
        if apply_height_correction:
            t_gs = t_era5 - lapse * delta_h_m
            t_bar = 0.5 * (t_era5 + t_gs)
            p_gs = p_era5 * np.exp(-9.81 * delta_h_m / (287.05 * t_bar))
            height_method = "surface_lapse_hypsometric"
        else:
            # Legacy v2 reproducibility: keep ERA5 surface P/T untouched.
            t_gs = t_era5
            p_gs = p_era5
            height_method = "legacy_unadjusted"

        valid = np.isfinite(np.stack([f["H_up_Wm2"], f["ustar"], t_gs, p_gs])).all(axis=0)
        if strict and not valid.all():
            raise SystemExit(f"{s['name']}: ERA5 has missing/non-finite rows after station-height correction; "
                             "use a documented historical fill or an explicit smoke-test fallback")
        if strict and (np.any(t_gs <= 0) or np.any(p_gs <= 0)):
            raise SystemExit(f"{s['name']}: height-corrected ERA5 temperature and pressure must be positive")
        fill[s["name"]]["station_height"] = {
            "h_GS_m": h_gs_m,
            "h_ERA5_m": h_era5_m,
            "delta_h_m": delta_h_m,
            "method": height_method,
            "h_ERA5_source": height_source,
            "temperature_lapse_K_per_m": lapse,
            "gs_altitude_explicit": bool(s.get("gs_altitude_explicit", False)),
            "era5_altitude_explicit": bool(s.get("era5_altitude_explicit", False)),
        }
        cn2 = cn2_monin_obukhov(f["H_up_Wm2"], f["ustar"], t_gs, p_gs,
                                z=p.z_ref, cn2_min=p.cn2_min, zeta_max=p.zeta_max)
        if strict:
            fill[s["name"]]["most_regularization"] = _most_regularization_report(
                f["H_up_Wm2"], f["ustar"], t_gs, p_gs,
                z=p.z_ref, cn2_min=p.cn2_min, zeta_max=p.zeta_max,
            )
            from tools.channel_v7.model import match_hv
            A, clipped = match_hv(cn2, z=p.z_ref, wind=p.hv_wind, return_clipped=True)
            n_clipped = int(np.count_nonzero(clipped & valid))
            fill[s["name"]]["hv_match_floor"] = {
                "n_hours": int(valid.sum()),
                "background_exceeds_reference_hours": n_clipped,
                "background_exceeds_reference_fraction": n_clipped / max(int(valid.sum()), 1),
            }
        else:
            A = cn2
        a_cn2[gi] = np.where(valid, A, np.nan)
        ok[gi] = valid
        fill[s["name"]]["era5_nan_hours"] = int((~valid).sum())
    return cloud, aod, a_cn2, ok, fill


# ----------------------------------------------------------------------------- nodes and candidate links
def ecef(lat, lon, alt):
    la, lo = np.radians(lat), np.radians(lon)
    r = R_E + np.asarray(alt, dtype=float)
    return r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)


def geometry(c1, c2):
    """Distance [km], elevation at the lower endpoint [deg], Earth not blocking, lowest ray height [km]."""
    (la1, lo1, a1), (la2, lo2, a2) = c1, c2
    x1, y1, z1 = ecef(la1, lo1, a1)
    x2, y2, z2 = ecef(la2, lo2, a2)
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    d2 = dx * dx + dy * dy + dz * dz
    d = np.sqrt(d2)
    t = -(x1 * dx + y1 * dy + z1 * dz) / np.maximum(d2, 1e-12)
    inside = (t > 0) & (t < 1)
    cx, cy, cz = x1 + t * dx, y1 + t * dy, z1 + t * dz
    minr = np.sqrt(cx * cx + cy * cy + cz * cz)
    los = ~(inside & (minr < R_E))
    a1 = np.broadcast_to(np.asarray(a1, dtype=float), d.shape)
    a2 = np.broadcast_to(np.asarray(a2, dtype=float), d.shape)
    graze = np.where(inside, minr - R_E, np.minimum(a1, a2))
    low1 = a1 <= a2
    ola = np.where(low1, la1, la2)
    olo = np.where(low1, lo1, lo2)
    oal = np.where(low1, a1, a2)
    ox, oy, oz = ecef(ola, olo, oal)
    tx, ty, tz = np.where(low1, x2, x1), np.where(low1, y2, y1), np.where(low1, z2, z1)
    lx, ly, lz = tx - ox, ty - oy, tz - oz
    nrm = np.maximum(np.sqrt(lx * lx + ly * ly + lz * lz), 1e-12)
    la_r, lo_r = np.radians(ola), np.radians(olo)
    sin_e = (lx * np.cos(la_r) * np.cos(lo_r) + ly * np.cos(la_r) * np.sin(lo_r) + lz * np.sin(la_r)) / nrm
    return d, np.degrees(np.arcsin(np.clip(sin_e, -1.0, 1.0))), los, graze


def haversine_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(math.radians, (la1, lo1, la2, lo2))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_E * math.asin(math.sqrt(h))


def satellite_configs(name: str) -> list[dict]:
    from legacy_data import GlobalSatellites
    return GlobalSatellites.get_satellites(name)


_GM = 398600.4418
_OMEGA_E = 7.2921159e-5


def propagate(cfg: dict, t_s):
    """Circular Keplerian orbit with Earth rotation; identical to
    build_global_tensor._vectorized_sat_positions for e = 0 (checked in tests/test_dataset_v2.py).
    Kept here so this builder does not import the legacy module chain (cdsapi, ...)."""
    t_s = np.asarray(t_s, dtype=float)
    inc = float(cfg.get("inclination", 0.0))
    a = R_E + float(cfg["alt_km"])
    i_rad = math.radians(inc)
    if "init_u_deg" in cfg:
        m0 = math.radians(float(cfg["init_u_deg"]))
    elif inc == 0:
        m0 = 0.0
    else:
        m0 = math.asin(max(-1.0, min(1.0, math.sin(math.radians(float(cfg.get("init_lat", 0.0)))) / math.sin(i_rad))))
    n = math.sqrt(_GM / a ** 3)
    raan = math.radians(float(cfg.get("init_lon", 0.0))) - _OMEGA_E * t_s
    u = m0 + n * t_s
    x_orb, y_orb = a * np.cos(u), a * np.sin(u)
    X = x_orb * np.cos(raan) - y_orb * math.cos(i_rad) * np.sin(raan)
    Y = x_orb * np.sin(raan) + y_orb * math.cos(i_rad) * np.cos(raan)
    Z = y_orb * math.sin(i_rad)
    lat = np.degrees(np.arcsin(Z / a))
    lon = (np.degrees(np.arctan2(Y, X)) + 180.0) % 360.0 - 180.0
    return lat, lon, np.full_like(lat, a - R_E)


def sat_positions(cfgs, t_min):
    """(S, B) lat, lon, alt of every satellite at slot times t_min (minutes)."""
    out = [propagate(c, np.asarray(t_min, dtype=float) * 60.0) for c in cfgs]
    return (np.stack([o[0] for o in out]), np.stack([o[1] for o in out]), np.stack([o[2] for o in out]))


# ----------------------------------------------------------------------------- HAP mobility
# Ported from tools/data_generation/legacy_physics.py (HapMobilityModel + GeoMath.shift_trajectory)
# so the v2 generator can drift HAPs along the shared stratospheric-balloon track instead of
# pinning each HAP statically above its station. Self-contained (no legacy_physics import, which
# would pull in cdsapi/xarray).
HAP_CYCLE_STEPS = 1440          # slots per 24 h drift cycle (theta = 60 s)
HAP_BLEND_STEPS = 240           # final 4 h blended back to the start so the cycle closes


def _latlon_to_cartesian(lat, lon, alt_km=0.0):
    """ECEF unit-sphere-agnostic direction scaled by R_E + alt (matches legacy GeoMath)."""
    la, lo = np.radians(lat), np.radians(lon)
    r = R_E + alt_km
    return np.array([r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)])


def _cartesian_to_latlon(x, y, z):
    lat = np.degrees(np.arcsin(z / R_E))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon + 360.0


def _rotation_matrix_from_vectors(a, b):
    """Rotation matrix taking unit vector a onto unit vector b (legacy GeoMath)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.isclose(c, 1.0):
        return np.eye(3)
    if np.isclose(c, -1.0):
        perp = np.array([1.0, 0.0, 0.0]) if not np.allclose(a, [1, 0, 0]) else np.array([0.0, 1.0, 0.0])
        v = np.cross(a, perp)
        v = v / np.linalg.norm(v)
        return -np.eye(3) + 2.0 * np.outer(v, v)
    s = float(np.linalg.norm(v))
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * ((1.0 - c) / (s ** 2))


def _shift_trajectory(lats, lons, new_lat0, new_lon0):
    """Rigidly rotate a shared track so its first point lands on (new_lat0, new_lon0)."""
    lat0, lon0 = lats[0], lons[0]
    if lat0 == new_lat0 and lon0 == new_lon0:
        return np.asarray(lats), np.asarray(lons)
    traj = np.array([_latlon_to_cartesian(la, lo) for la, lo in zip(lats, lons)])
    rm = _rotation_matrix_from_vectors(traj[0], _latlon_to_cartesian(new_lat0, new_lon0))
    rot = traj @ rm.T
    out = np.array([_cartesian_to_latlon(x, y, z) for x, y, z in rot])
    return out[:, 0], out[:, 1]


def build_hap_positions(stations, cfg: BuildConfig, track_path: Path) -> np.ndarray:
    """(HAP_CYCLE_STEPS, G, 3) per-slot HAP [lat, lon, alt_km] drifting on the shared balloon track.

    Each HAP follows the same measured 24 h stratospheric-balloon trajectory, rigidly rotated so
    slot 0 sits at that HAP's home station. The last HAP_BLEND_STEPS slots are blended back to the
    start so the 24 h cycle is continuous when tiled across the year (as in the legacy pipeline).
    """
    cols = _read_csv_columns(track_path, ["Time_s", "Longitude_deg", "Latitude_deg", "Altitude_m"])
    order = np.argsort(cols["Time_s"])
    t_src = cols["Time_s"][order]
    lon_src = cols["Longitude_deg"][order]
    lat_src = cols["Latitude_deg"][order]
    alt_src = cols["Altitude_m"][order] / 1000.0
    # uniform 60 s grid over one drift cycle, then tile the source track to fill it if shorter
    grid = np.arange(HAP_CYCLE_STEPS) * 60.0
    span = max(t_src[-1], 1.0)
    grid_src = np.mod(grid, span)
    lat_c = np.interp(grid_src, t_src, lat_src)
    lon_c = np.interp(grid_src, t_src, lon_src)
    alt_c = np.interp(grid_src, t_src, alt_src)
    # close the loop: blend the tail back onto the head within the last HAP_BLEND_STEPS slots
    blend_start = HAP_CYCLE_STEPS - HAP_BLEND_STEPS
    dlat = lat_c[-1] - lat_c[0]
    dlon = lon_c[-1] - lon_c[0]
    dalt = alt_c[-1] - alt_c[0]
    w = np.clip((np.arange(HAP_CYCLE_STEPS) - blend_start) / HAP_BLEND_STEPS, 0.0, 1.0)
    lat_c = lat_c - w * dlat
    lon_c = lon_c - w * dlon
    alt_c = alt_c - w * dalt
    gs_lat = np.array([s["lat"] for s in stations])
    gs_lon = np.array([s["lon"] for s in stations])
    pos = np.empty((HAP_CYCLE_STEPS, len(stations), 3), dtype=np.float32)
    for i, (la0, lo0) in enumerate(zip(gs_lat, gs_lon)):
        la_i, lo_i = _shift_trajectory(list(lat_c), list(lon_c), float(la0), float(lo0))
        pos[:, i, 0] = la_i
        pos[:, i, 1] = lo_i
        pos[:, i, 2] = alt_c
    return pos


def build_nodes(stations, cfg: BuildConfig, sat_cfgs):
    G, S = len(stations), len(sat_cfgs)
    rows = []
    for i, s in enumerate(stations):
        rows.append(dict(
            node_id=i, type="GS", name=s["name"], lat=s["lat"], lon=s["lon"],
            alt_km=float(s["alt_km"]), base=s["base"]
        ))
    for i, s in enumerate(stations):
        rows.append(dict(node_id=G + i, type="HAP", name=f"HAP_{s['name']}", lat=s["lat"], lon=s["lon"],
                         alt_km=cfg.hap_alt_km, base=s["base"]))
    la, lo, al = sat_positions(sat_cfgs, np.array([0.0]))
    for i, c in enumerate(sat_cfgs):
        rows.append(dict(node_id=2 * G + i, type="SAT", name=c["name"], lat=float(la[i, 0]), lon=float(lo[i, 0]),
                         alt_km=float(al[i, 0]), base=1))
    return rows, G, S


def candidate_links(nodes, G, S, cfg: BuildConfig):
    """Candidate pairs (u < v) in node order plus genuinely rejected pairs.

    HAPs are paired one-to-one with their home ground station (HAP_i <-> GS_i, node ids G+i and i);
    a HAP never serves a neighbouring station. Availability is decided per slot from the HAP's
    (moving or static) position and the elevation mask, not by a static prefilter.
    """
    links, skipped = [], []
    gs = nodes[:G]
    hap = nodes[G:2 * G]
    for i in range(G):                             # HAP-GS: one-to-one home pairing
        g, h = gs[i], hap[i]
        links.append((g["node_id"], h["node_id"], "HAP-GS"))
    for g in gs:                                   # SAT-GS
        for k in range(S):
            links.append((g["node_id"], 2 * G + k, "SAT-GS"))
    for h in hap:                                  # SAT-HAP
        for k in range(S):
            links.append((h["node_id"], 2 * G + k, "SAT-HAP"))
    for a in range(S):                             # SAT-SAT
        for b in range(a + 1, S):
            links.append((2 * G + a, 2 * G + b, "SAT-SAT"))
    links.sort(key=lambda x: (x[0], x[1]))
    return links, skipped


# ----------------------------------------------------------------------------- worker
_W: dict = {}


def _init_worker(state: dict) -> None:
    _W.clear()
    _W.update(state)
    if state["cfg"].channel_model == "v7":
        from tools.channel_v7.model import RateModelV7
        _W["model"] = RateModelV7(
            state["params"], sat_alt_km=state["sat_alt_km"], hap_alt_km=state["cfg"].hap_alt_km
        )
    else:
        # Legacy v2 keeps its documented HAP-trajectory atmosphere approximation.
        _W["model"] = RateModelV2(
            state["params"], sat_alt_km=state["sat_alt_km"], hap_alt_km=state["cfg"].hap_alt_km
        )


def compute_block(b0: int, b1: int):
    """Rates for a time block, dispatched to the selected channel-model implementation."""
    W = _W
    if W["cfg"].channel_model == "v7":
        from tools.channel_v7.dataset import compute_block as compute_block_v7
        return compute_block_v7(W, b0, b1, geometry, sat_positions)
    cfg: BuildConfig = W["cfg"]
    p: ParamsV2 = W["params"]
    m: RateModelV2 = W["model"]
    fk: FiniteKeyParams = W["fk"]
    G = W["G"]
    t = np.arange(b0, b1, dtype=np.int64)
    B = len(t)
    hours = np.minimum(t // 60, YEAR_HOURS - 1)
    s_la, s_lo, s_al = sat_positions(W["sat_cfgs"], t)
    gla, glo = W["gs_lat"], W["gs_lon"]
    sun = sun_elevation_deg(t[None, :], gla[:, None], glo[:, None])       # (G, B), GS = HAP position
    L = len(W["link_u"])
    out = np.zeros((B, L), dtype=np.float32)
    n_vis = np.zeros(L, dtype=np.int64)
    n_pos = np.zeros(L, dtype=np.int64)
    n_cloud = np.zeros(L, dtype=np.int64)

    def coords(ids):
        ids = np.asarray(ids)
        la = np.empty((len(ids), B))
        lo = np.empty((len(ids), B))
        al = np.empty((len(ids), B))
        is_sat = ids >= 2 * G
        is_hap = (ids >= G) & ~is_sat
        is_gs = ~is_sat & ~is_hap
        k = ids[is_sat] - 2 * G
        la[is_sat], lo[is_sat], al[is_sat] = s_la[k], s_lo[k], s_al[k]
        la[is_gs] = gla[ids[is_gs]][:, None]
        lo[is_gs] = glo[ids[is_gs]][:, None]
        al[is_gs] = 0.0
        h_idx = ids[is_hap] - G                       # HAP -> its home station index
        if W["hap_mob"] == "trajectory":
            hp = W["hap_pos"]                         # (HAP_CYCLE_STEPS, G, 3) [lat, lon, alt_km]
            hap_t = (t % HAP_CYCLE_STEPS)[None, :]    # (1, B) per-slot cycle index
            la[is_hap] = hp[hap_t, h_idx[:, None], 0]
            lo[is_hap] = hp[hap_t, h_idx[:, None], 1]
            al[is_hap] = hp[hap_t, h_idx[:, None], 2]
        else:
            la[is_hap] = gla[h_idx][:, None]
            lo[is_hap] = glo[h_idx][:, None]
            al[is_hap] = cfg.hap_alt_km
        return la, lo, al

    for lt in TYPES:
        cols_all = np.flatnonzero(W["link_type"] == lt)
        for c0 in range(0, len(cols_all), LINK_BATCH):
            cols = cols_all[c0:c0 + LINK_BATCH]
            u, v = W["link_u"][cols], W["link_v"][cols]
            dist, elev, los, graze = geometry(coords(u), coords(v))
            if lt in ("HAP-GS", "SAT-GS"):
                vis = los & (elev >= cfg.alpha_min_deg)
            else:
                vis = los & (graze >= cfg.h_min_km)
            n_vis[cols] += vis.sum(axis=1)
            li, ti = np.nonzero(vis)
            if li.size == 0:
                continue
            kw = {}
            chi = np.ones(li.size)
            cov_f = None
            if lt in ("HAP-GS", "SAT-GS"):
                g = u[li]                                   # GS is always the lower node id
                hr = hours[ti]
                s_rx = sun[g, ti]
                a = W["a_cn2"][g, hr]
                a_fb = np.where(s_rx > 0.0, p.a_day, p.a_night)
                kw.update(sun_rx_deg=s_rx, aod550=W["aod"][g, hr], a_cn2=np.where(W["era5_ok"][g, hr], a, a_fb))
                if cfg.cloud_model == "coverage_layers":
                    cov_f = p_clear_lund_shanklin(elev[li, ti], W["c_lm"][g, hr])
                    t_ci = 1.0 - W["c_hi"][g, hr] * (1.0 - np.exp(-cfg.tau_cirrus
                                                               * airmass_kasten_young(elev[li, ti])))
                else:
                    pclr = p_clear_lund_shanklin(elev[li, ti], W["cloud"][g, hr])
                    chi = (W["u_cloud"][g, hr] < pclr).astype(float)
            elif lt == "SAT-HAP":
                kw.update(sun_rx_deg=sun[u[li] - G, ti])
            elif lt == "SAT-SAT":
                # §1.3 / §6：光照档由两星星下点太阳高度角分类；掠射高度给上层大气切层
                la_u, lo_u, _ = coords(u); la_v, lo_v, _ = coords(v)
                s_u = sun_elevation_deg(t[ti], la_u[li, ti], lo_u[li, ti])
                s_v = sun_elevation_deg(t[ti], la_v[li, ti], lo_v[li, ti])
                kw.update(isl_illumination=isl_illumination_class(s_v, s_u), grazing_alt_km=graze[li, ti])
            r_asym, parts = m.link_rate(lt, dist[li, ti] * 1e3, elev[li, ti], return_parts=True, **kw)
            # 有限码长路径（规范 v6.7 §3.2）：块内先汇总分基/分强度计数（对闪烁 Gauss–Hermite
            # 状态 j 求和，每个状态携带 N_j = w_j * N_use 个脉冲），再应用一次 Lim/Sidhu 界。
            # 不再使用"逐 GH 节点算 ell 再加权"的近似。
            sc_h = parts.get("scint_h"); sc_w = parts.get("scint_w")

            def fk_block_rate(eta_arr, y0_arr, n_use, sel):
                """eta_arr, y0_arr, n_use: shape (n,) per-slot 量（已按 sel 选过）；sel 为取用 sc_h 的掩码。"""
                if sc_h is None:
                    return slot_rate(eta_arr, y0_arr, fk, n_pulses=n_use)
                # 组装 (J, n) 状态数组：J = GH 节点数
                eta_s = np.stack([eta_arr * sc_h[k][sel] for k in range(len(sc_w))], axis=0)
                y0_s = np.broadcast_to(y0_arr, eta_s.shape)
                n_s = np.stack([sc_w[k] * n_use for k in range(len(sc_w))], axis=0)
                return slot_rate_states(eta_s, y0_s, n_s, fk)

            if cov_f is not None:
                # coverage_layers：低中云晴空比例 f = cov_f 视为每槽可用时间比例 ⟹ N_use = floor(f*N)
                eta_c = parts["eta"] * t_ci
                r = np.zeros(li.size)
                ok = cov_f > 0
                if cfg.protocol == "asymptotic":
                    if sc_h is None:
                        r[ok] = m.skr(eta_c[ok], np.broadcast_to(parts["y0"], eta_c.shape)[ok]) * cov_f[ok]
                    else:
                        r[ok] = sum(sc_w[k] * m.skr(eta_c[ok] * sc_h[k][ok], np.broadcast_to(parts["y0"], eta_c.shape)[ok]) for k in range(len(sc_w))) * cov_f[ok]
                else:
                    n_use = np.floor(cov_f[ok] * fk.n_pulses)
                    r[ok] = fk_block_rate(eta_c[ok], np.broadcast_to(parts["y0"], eta_c.shape)[ok], n_use, ok)
                blocked = ~ok
            else:
                # binary_total：整块按 chi 掷骰；chi=0 零产钥，chi=1 满块 N
                if cfg.protocol == "asymptotic":
                    r = np.asarray(r_asym, dtype=float) * chi
                else:
                    r = fk_block_rate(parts["eta"], np.broadcast_to(parts["y0"], np.shape(parts["eta"])),
                                      chi * fk.n_pulses, slice(None))
                blocked = chi == 0
            out[ti, cols[li]] = r.astype(np.float32)
            np.add.at(n_pos, cols[li], (r > 0).astype(np.int64))
            np.add.at(n_cloud, cols[li], blocked.astype(np.int64))
    return b0, out, n_vis, n_pos, n_cloud


# ----------------------------------------------------------------------------- main
def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def prepare_out(out: Path, overwrite: bool, marker: str = MARKER) -> None:
    protected = (ROOT / "dataset" / "global").resolve()
    if out.resolve() == protected or protected in out.resolve().parents:
        raise SystemExit("refusing to write into dataset/global")
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite (only for directories built by this script)")
        if not (out / marker).exists():
            raise SystemExit(f"{out} does not carry this builder's marker; refusing to overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / marker).write_text(f"created by {marker}\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="new output directory; defaults to dataset/global_v7 (v7) or dataset/global_v2d (legacy v2)")
    ap.add_argument("--channel-model", default="v7", choices=["v2", "v7"],
                    help="v7 = the specification baseline (DEFAULT, 2026-10-07); v2 = legacy model, "
                         "kept only for reproducing pre-v7 datasets")
    ap.add_argument("--stations", default=None, help="station CSV; v7 default = stations_45_with_elevation.csv, legacy v2 default = stations_45.csv")
    ap.add_argument(
        "--allow-missing-station-elevation", action="store_true",
        help=("v7 smoke-test escape hatch only: allow missing h_GS/h_ERA5 and fall back to sea-level/"
              "coincident-surface assumptions. Main v7 builds should leave this disabled."),
    )
    ap.add_argument("--weather-dir", default=".tmp/channel_v2/weather")
    ap.add_argument("--era5-dir", default=".tmp/channel_v2/era5")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--cloud-seed", type=int, default=BuildConfig.cloud_seed)
    ap.add_argument("--constellation", default=BuildConfig.sat_constellation, choices=["walker", "walker60", "legacy"],
                    help="diagnostics: legacy = the hand-written list of dataset/global")
    ap.add_argument("--protocol", default=BuildConfig.protocol, choices=["finite_key", "asymptotic"],
                    help="asymptotic is a training diagnostic; published comparisons use finite_key")
    ap.add_argument("--cloud-model", default=None,
                    choices=["station_markov", "binary_total", "coverage_layers"],
                    help="v7.5 baseline = station_markov; binary_total/coverage_layers are legacy v2 modes")
    ap.add_argument("--tau-cirrus", type=float, default=BuildConfig.tau_cirrus)
    ap.add_argument("--hap-mobility", default=None, choices=["trajectory", "static"],
                    help="v7 baseline requires static; v2 defaults to trajectory")
    ap.add_argument("--alpha-min-deg", type=float, default=None)
    ap.add_argument("--h-min-km", type=float, default=None)
    ap.add_argument(
        "--domain-policy", default=None, choices=["error", "extrapolate", "mark"],
        help=("v7 weak-fluctuation domain handling: error aborts on V_R>=1; "
              "extrapolate keeps the documented bounded engineering closure while flagging model_domain; "
              "mark writes diagnostic NaNs and is not trainable. v7 default = extrapolate.")
    )
    ap.add_argument("--capture-time-s", type=float, default=None,
                    help="v7.5 acquisition/PAT time inside a 60 s slot; default 5 s")
    ap.add_argument("--capture-gamma", type=float, default=None,
                    help="deprecated v7 compatibility input; gamma is converted to t_acq=60*(1-gamma)")
    ap.add_argument("--cloud-tau-s", type=float, default=BuildConfig.cloud_tau_s,
                    help="v7.5 station-cloud correlation time tau_cld in seconds")
    ap.add_argument(
        "--intra-slot-samples", type=int, default=BuildConfig.intra_slot_samples,
        help=("v7 geometry sub-samples per 60 s slot, sampled at sub-interval midpoints. "
              "Paper baseline = 4; use 1 only as a cheaper diagnostic.")
    )
    ap.add_argument("--eps-sec-total", type=float, default=None,
                    help="composable security budget: total epsilon_sec over the declared service "
                         "period; per-block eps_sec = eps_sec_total / n_max_blocks (spec §5.2). "
                         "Omit for the fixed per-block 1e-9 protocol comparison.")
    ap.add_argument("--eps-cor-total", type=float, default=None,
                    help="composable correctness budget: total epsilon_cor over the period; per-block "
                         "eps_cor = eps_cor_total / n_max_blocks. Omit for the fixed per-block 1e-15.")
    ap.add_argument("--n-max-blocks", type=int, default=None,
                    help="block-count upper bound N_max for the eps allocation (required with "
                         "--eps-sec-total/--eps-cor-total); the declared service period's QKD blocks.")
    ap.add_argument("--hap-track", default="dataset/balloon_sim_data.csv",
                    help="24 h stratospheric-balloon track (Time_s, Longitude_deg, Latitude_deg, Altitude_m)")
    ap.add_argument("--cloud-layers-dir", default=".tmp/channel_v2/cloud_layers")
    ap.add_argument("--allow-era5-fallback", action="store_true",
                    help="explicit smoke-test fallback for stations without ERA5 files")
    ap.add_argument("--keep-zero-links", action="store_true", help="do not drop links that are zero all period")
    ap.add_argument("--no-stats", action="store_true", help="skip scripts/estimate_rate_stats.py")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check-inputs", action="store_true", help="only report which input files are ready")
    args = ap.parse_args(argv)

    channel_model = args.channel_model
    cloud_model = args.cloud_model or ("station_markov" if channel_model == "v7" else BuildConfig.cloud_model)
    hap_mobility = args.hap_mobility or ("static" if channel_model == "v7" else BuildConfig.hap_mobility)
    alpha_min = args.alpha_min_deg if args.alpha_min_deg is not None else (15.0 if channel_model == "v7" else BuildConfig.alpha_min_deg)
    h_min = args.h_min_km if args.h_min_km is not None else (85.0 if channel_model == "v7" else BuildConfig.h_min_km)
    out_arg = args.out or ("dataset/global_v7" if channel_model == "v7" else "dataset/global_v2d")
    stations_arg = args.stations or (
        "tools/channel_v2/stations_45_with_elevation.csv"
        if channel_model == "v7"
        else "tools/channel_v2/stations_45.csv"
    )
    domain_policy = args.domain_policy or ("extrapolate" if channel_model == "v7" else "error")
    if args.capture_time_s is not None and args.capture_gamma is not None:
        ap.error("use only one of --capture-time-s and deprecated --capture-gamma")
    if args.capture_time_s is not None:
        capture_time_s = float(args.capture_time_s)
    elif args.capture_gamma is not None:
        if not np.isfinite(args.capture_gamma) or not 0.0 <= args.capture_gamma <= 1.0:
            ap.error("--capture-gamma must be finite in [0, 1]")
        capture_time_s = 60.0 * (1.0 - float(args.capture_gamma))
    else:
        capture_time_s = float(BuildConfig.capture_time_s)
    if not np.isfinite(capture_time_s) or not 0.0 <= capture_time_s <= 60.0:
        ap.error("--capture-time-s must be finite in [0, 60]")
    if channel_model == "v2" and (args.capture_time_s is not None or args.capture_gamma is not None):
        ap.error("capture-time settings are only supported by --channel-model v7")
    if channel_model == "v7" and cloud_model != "station_markov":
        ap.error("v7.5 baseline requires --cloud-model station_markov")
    if not np.isfinite(args.cloud_tau_s) or args.cloud_tau_s <= 0:
        ap.error("--cloud-tau-s must be finite and positive")
    if channel_model == "v2" and domain_policy != "error":
        ap.error("--domain-policy is only supported by --channel-model v7")
    if channel_model == "v7" and hap_mobility != "static":
        ap.error("the v7 baseline requires --hap-mobility static")
    if channel_model == "v7" and domain_policy == "mark" and not args.no_stats:
        ap.error("--domain-policy mark is diagnostic-only; also pass --no-stats")
    if ((args.eps_sec_total is not None) or (args.eps_cor_total is not None)) and args.n_max_blocks is None:
        ap.error("--eps-sec-total/--eps-cor-total require --n-max-blocks (composable budget "
                 "eps_block = eps_total / N_max, spec §5.2)")
    if args.eps_cor_total is not None and args.eps_sec_total is None:
        ap.error("--eps-cor-total requires --eps-sec-total as well")

    rel = lambda x: (ROOT / x) if not Path(x).is_absolute() else Path(x)  # noqa: E731
    stations_path = rel(stations_arg)
    if channel_model == "v7" and not stations_path.exists():
        ap.error(
            f"v7.5 station table not found: {stations_path}. Generate DEM-backed "
            "site elevations first with: python tools/channel_v2/fetch_station_elevations.py"
        )
    stations = read_stations(stations_path)
    if channel_model == "v7" and not args.allow_missing_station_elevation and not args.check_inputs:
        missing_gs = [s["name"] for s in stations if not s.get("gs_altitude_explicit", False)]
        if missing_gs:
            ap.error(
                "v7.5 main build requires explicit GS site/terrain elevation in the station CSV "
                "(alt_km or h_GS/h_GS_m). Missing GS="
                f"{missing_gs[:8]}{'...' if len(missing_gs) > 8 else ''}. "
                "For the synthetic city-coordinate baseline generate Copernicus GLO-90 heights with "
                "python tools/channel_v2/fetch_station_elevations.py. "
                "ERA5 grid-orography may come from h_ERA5 in the station CSV or from "
                "h_ERA5_m in the regenerated ERA5 file. "
                "Use --allow-missing-station-elevation only for smoke tests."
            )
    wdir, edir, out = rel(args.weather_dir), rel(args.era5_dir), rel(out_arg)
    ldir = rel(args.cloud_layers_dir)
    # v7 always needs the layered cloud input: coverage_layers reads f from C_lm,
    # and the binary_total control uses the SAME C_lm so the comparison isolates
    # the time interpretation (spec §4.1). v2 keeps layers only for coverage_layers.
    need_layers = cloud_model == "coverage_layers" or channel_model == "v7"
    if args.check_inputs:
        rep = check_inputs(stations, wdir, edir, ldir if need_layers else None,
                           strict=(channel_model == "v7"))
        bad = [n for n, r in rep.items() if not r["ok"]]
        for n, r in rep.items():
            print(f"{n:14s} {'OK ' if r['ok'] else 'MISSING'} {r}")
        print(f"{len(rep) - len(bad)}/{len(rep)} stations ready" + (f"; missing: {bad}" if bad else ""))
        return 1 if bad else 0

    import h5py

    t_start = time.time()
    if not np.isfinite([alpha_min, h_min, args.tau_cirrus]).all() or alpha_min < 0 or h_min < 0 or args.tau_cirrus < 0:
        ap.error("elevation/grazing thresholds and tau-cirrus must be finite and nonnegative")
    if args.intra_slot_samples < 1:
        ap.error("--intra-slot-samples must be a positive integer")
    cfg = BuildConfig(
        cloud_seed=args.cloud_seed, sat_constellation=args.constellation, protocol=args.protocol,
        cloud_model=cloud_model, tau_cirrus=args.tau_cirrus, hap_mobility=hap_mobility,
        alpha_min_deg=alpha_min, h_min_km=h_min, channel_model=channel_model,
        domain_policy=domain_policy, capture_time_s=capture_time_s,
        cloud_tau_s=float(args.cloud_tau_s), intra_slot_samples=args.intra_slot_samples,
    )
    if channel_model == "v7":
        from tools.channel_v7.model import ParamsV7
        params = ParamsV7(elev_min_deg=alpha_min, isl_min_grazing_alt_km=h_min)
    else:
        params = ParamsV2()
    eps_totals = (args.eps_sec_total is not None) or (args.eps_cor_total is not None)
    if eps_totals and args.n_max_blocks is None:
        ap.error("--eps-sec-total/--eps-cor-total require --n-max-blocks (the composable budget "
                 "eps_block = eps_total / N_max, spec §5.2)")
    fk = FiniteKeyParams(
        eps_sec_total=args.eps_sec_total,
        # 不填 eps_cor_total 时**保持 None** → per_block_eps 回退到固定 1e-15（与 --help 一致）。
        # 曾经这里写成 `... else args.eps_sec_total`，把正确性预算错绑到保密预算，
        # 使 -log2(2/eps_cor) 从 50.6 掉到 40.9 bit（200 轮审计 Finding 1）。
        eps_cor_total=args.eps_cor_total,
        n_max_blocks=args.n_max_blocks,
    )
    T = args.days * 1440 // SLOT_MIN
    if not 1 <= args.days <= 365:
        raise SystemExit("--days must be in 1..365 (weather covers 2023)")
    strict_inputs = channel_model == "v7"
    cloud, aod, a_cn2, era5_ok, fill = load_weather(
        stations, wdir, edir, params, args.allow_era5_fallback,
        strict=strict_inputs,
        apply_height_correction=(channel_model == "v7"),
        allow_missing_station_elevation=args.allow_missing_station_elevation,
    )
    c_lm = c_hi = None
    if need_layers:
        c_lm, c_hi, lfill = load_cloud_layers(stations, ldir, strict=strict_inputs)
        for k, v in lfill.items():
            fill.setdefault(k, {}).update(v)
    prepare_out(out, args.overwrite, marker=MARKER_V7 if channel_model == "v7" else MARKER)

    sat_cfgs = satellite_configs(cfg.sat_constellation)
    nodes, G, S = build_nodes(stations, cfg, sat_cfgs)
    links, skipped = candidate_links(nodes, G, S, cfg)
    L = len(links)
    sat_alt = float(np.median([c["alt_km"] for c in sat_cfgs]))
    u_cloud = None
    cloud_state = None
    cloud_rho = None
    if channel_model == "v7":
        from tools.channel_v7.dataset import generate_station_cloud_state
        cloud_state, cloud_rho = generate_station_cloud_state(
            c_lm, T, seed=cfg.cloud_seed, slot_s=fk.slot_s, tau_cld_s=cfg.cloud_tau_s
        )
    else:
        rng = np.random.default_rng(cfg.cloud_seed)
        u_cloud = rng.random((G, YEAR_HOURS))
    print(f"nodes GS={G} HAP={G} SAT={S}; candidate links {L} "
          f"({', '.join(f'{t}={sum(1 for x in links if x[2] == t)}' for t in TYPES)}); T={T}", flush=True)

    state = dict(cfg=cfg, params=params, fk=fk, G=G, sat_cfgs=sat_cfgs, sat_alt_km=sat_alt,
                 gs_lat=np.array([s["lat"] for s in stations]), gs_lon=np.array([s["lon"] for s in stations]),
                 gs_alt_km=np.array([s["alt_km"] for s in stations], dtype=float),
                 hap_mob=cfg.hap_mobility,
                 hap_pos=build_hap_positions(stations, cfg, rel(args.hap_track)) if cfg.hap_mobility == "trajectory" else None,
                 link_u=np.array([x[0] for x in links]), link_v=np.array([x[1] for x in links]),
                 link_type=np.array([x[2] for x in links]), cloud=cloud, aod=aod, a_cn2=a_cn2,
                 era5_ok=era5_ok, u_cloud=u_cloud, cloud_state=cloud_state,
                 c_lm=c_lm, c_hi=c_hi)

    tmp_path = out / "_link_data_all_candidates.h5"
    n_vis = np.zeros(L, dtype=np.int64)
    n_pos = np.zeros(L, dtype=np.int64)
    n_proxy_pos = np.zeros(L, dtype=np.int64)
    n_cloud = np.zeros(L, dtype=np.int64)
    r_sum = np.zeros(L)
    proxy_sum = np.zeros(L)
    v7_stat_sum = {}
    if cfg.channel_model == "v7":
        from tools.channel_v7.dataset import STAT_FIELDS
        v7_stat_sum = {key: np.zeros((L,) + shape, dtype=np.float64) for key, shape in STAT_FIELDS.items()}
    blocks = [(b, min(b + BLOCK, T)) for b in range(0, T, BLOCK)]
    with h5py.File(tmp_path, "w") as fa, h5py.File(out / "sat_positions.h5", "w") as fsat:
        ds = fa.create_dataset("k_max", shape=(T, L), dtype=np.float32, chunks=(min(BLOCK, T), L),
                               compression="gzip", compression_opts=4)
        ds_first = None
        ds_proxy = ds_proxy_first = None
        ds_los = None
        ds_qgain = ds_qy1 = ds_qfrac = None
        ds_qgain_first = ds_qy1_first = ds_qfrac_first = None
        if cfg.channel_model == "v7":
            ds_first = fa.create_dataset("k_max_first", shape=(T, L), dtype=np.float32,
                                         chunks=(min(BLOCK, T), L), compression="gzip", compression_opts=4)
            ds_proxy = fa.create_dataset("k_proxy", shape=(T, L), dtype=np.float32,
                                         chunks=(min(BLOCK, T), L), compression="gzip", compression_opts=4)
            ds_proxy_first = fa.create_dataset("k_proxy_first", shape=(T, L), dtype=np.float32,
                                               chunks=(min(BLOCK, T), L), compression="gzip", compression_opts=4)
            ds_los = fa.create_dataset("los", shape=(T, L), dtype=np.uint8,
                                       chunks=(min(BLOCK, T), L), compression="gzip", compression_opts=4)
            qt, ql = min(256, T), min(256, L)
            ds_qgain = fa.create_dataset("qkd_gain", shape=(T, L, len(fk.mu)), dtype=np.float32,
                                         chunks=(qt, ql, len(fk.mu)), compression="gzip", compression_opts=4)
            ds_qy1 = fa.create_dataset("qkd_y1", shape=(T, L), dtype=np.float32,
                                       chunks=(qt, ql), compression="gzip", compression_opts=4)
            ds_qfrac = fa.create_dataset("qkd_usable_fraction", shape=(T, L), dtype=np.float32,
                                         chunks=(qt, ql), compression="gzip", compression_opts=4)
            ds_qgain_first = fa.create_dataset("qkd_gain_first", shape=(T, L, len(fk.mu)), dtype=np.float32,
                                               chunks=(qt, ql, len(fk.mu)), compression="gzip", compression_opts=4)
            ds_qy1_first = fa.create_dataset("qkd_y1_first", shape=(T, L), dtype=np.float32,
                                             chunks=(qt, ql), compression="gzip", compression_opts=4)
            ds_qfrac_first = fa.create_dataset("qkd_usable_fraction_first", shape=(T, L), dtype=np.float32,
                                               chunks=(qt, ql), compression="gzip", compression_opts=4)
        dsat = fsat.create_dataset("positions", shape=(T, S, 3), dtype=np.float32, chunks=(min(BLOCK, T), S, 3))
        fsat.create_dataset("node_ids", data=np.arange(2 * G, 2 * G + S, dtype=np.int32))
        fsat.attrs["theta_sec"] = 60
        fsat.attrs["constellation"] = cfg.sat_constellation

        def consume(res):
            if cfg.channel_model == "v7":
                b0, blk, first, geometry_visible, nv, npos, ncl, stats, first_qkd = res
                end = b0 + blk.shape[0]
                ds_first[b0:end] = first
                ds_los[b0:end] = geometry_visible
                proxy = (fk.f_rep * stats["usable_fraction"] * stats["proxy_per_pulse"]).astype(np.float32)
                proxy_first = (fk.f_rep * first_qkd["usable_fraction"] * first_qkd["proxy_per_pulse"]).astype(np.float32)
                ds_proxy[b0:end] = proxy
                ds_proxy_first[b0:end] = proxy_first
                n_proxy_pos[:] += (np.nan_to_num(proxy, nan=0.0) > 0.0).sum(axis=0)
                proxy_sum[:] += np.nansum(proxy, axis=0, dtype=np.float64)
                ds_qgain[b0:end] = stats["gain"].astype(np.float32)
                ds_qy1[b0:end] = stats["single_photon_yield"].astype(np.float32)
                ds_qfrac[b0:end] = stats["usable_fraction"].astype(np.float32)
                ds_qgain_first[b0:end] = first_qkd["gain"].astype(np.float32)
                ds_qy1_first[b0:end] = first_qkd["single_photon_yield"].astype(np.float32)
                ds_qfrac_first[b0:end] = first_qkd["usable_fraction"].astype(np.float32)
                for key, values in stats.items():
                    v7_stat_sum[key] += np.nansum(values, axis=0, dtype=np.float64)
            else:
                b0, blk, nv, npos, ncl = res
            ds[b0:b0 + blk.shape[0]] = blk
            n_vis[:] += nv
            n_pos[:] += npos
            n_cloud[:] += ncl
            r_sum[:] += (np.nansum(blk, axis=0, dtype=np.float64) if cfg.channel_model == "v7"
                         else blk.sum(axis=0, dtype=np.float64))

        done = 0
        if args.workers <= 1:
            _init_worker(state)
            it = (compute_block(b0, b1) for b0, b1 in blocks)
            for res in it:
                consume(res)
                done += 1
                print(f"  block {done}/{len(blocks)} ({time.time() - t_start:.0f}s)", flush=True)
        else:
            # ★ 有界在飞窗口（2026-10-07 修内存泄漏）：**不能**一次 submit 全部 257 块。
            # 每块结果约 1 GB（v7 的 stats 字典是 float64 (B,L,shape)），旧写法
            # `futs = [submit(...) for b in blocks]` 把全部结果同时留在父进程里
            # ⟹ 全年构建父进程涨到 143 GB 被 OOM 杀，worker 变孤儿空转。
            # 改为最多 `window = 2*workers` 块在飞：按序 consume 后立即释放。
            with cf.ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                        initargs=(state,)) as ex:
                window = max(1, 2 * args.workers)
                it = iter(blocks)
                pending: list = []
                for b0, b1 in it:
                    pending.append(ex.submit(compute_block, b0, b1))
                    if len(pending) >= window:
                        break
                while pending:
                    fu = pending.pop(0)
                    consume(fu.result())
                    done += 1
                    if done % 10 == 0 or done == len(blocks):
                        print(f"  block {done}/{len(blocks)} ({time.time() - t_start:.0f}s)", flush=True)
                    nxt = next(it, None)
                    if nxt is not None:
                        pending.append(ex.submit(compute_block, *nxt))
        for b0, b1 in blocks:
            la, lo, al = sat_positions(sat_cfgs, np.arange(b0, b1))
            dsat[b0:b1] = np.stack([la.T, lo.T, al.T], axis=-1).astype(np.float32)

    keep_signal = n_proxy_pos if cfg.channel_model == "v7" else n_pos
    keep = np.ones(L, dtype=bool) if args.keep_zero_links else (keep_signal > 0)
    for j in np.flatnonzero(~keep):
        u, v, lt = links[j]
        if cfg.channel_model == "v7":
            moment_visible = float(v7_stat_sum["moment_visible_substeps"][j])
            moment_valid = float(v7_stat_sum["moment_compatible_substeps"][j])
            domain_visible = float(v7_stat_sum["domain_visible_substeps"][j])
            domain_valid = float(v7_stat_sum["domain_valid_substeps"][j])
            if moment_visible > 0.0 and moment_valid <= 0.0:
                reason = "bounded_moment_incompatible_all_visible_substeps"
            elif cfg.domain_policy == "mark" and domain_visible > 0.0 and domain_valid <= 0.0:
                reason = "weak_fluctuation_domain_invalid_all_visible_substeps"
            else:
                reason = "zero_proxy_rate_all_period"
        else:
            reason = "zero_rate_all_period"
        skipped.append(dict(node_u=u, node_v=v, link_type=lt, reason=reason,
                            value=int(n_vis[j]), threshold=""))
    kept = np.flatnonzero(keep)
    ground_station_all = np.full(L, -1, dtype=np.int32)
    for j, (u, v, lt) in enumerate(links):
        if lt in ("HAP-GS", "SAT-GS"):
            if u < G:
                ground_station_all[j] = int(u)
            elif v < G:
                ground_station_all[j] = int(v)
            else:
                raise ValueError(f"ground link {j} has no GS endpoint")
    reg = np.zeros(len(kept), dtype=[("link_id", "<i4"), ("node_u", "<i4"), ("node_v", "<i4"), ("link_type", "S10")])
    for new_id, j in enumerate(kept):
        u, v, lt = links[j]
        reg[new_id] = (new_id, u, v, lt.encode())
    with h5py.File(tmp_path, "r") as fa, h5py.File(out / "link_data.h5", "w") as fo:
        src = fa["k_max"]
        dst = fo.create_dataset("k_max", shape=(T, len(kept)), dtype=np.float32, chunks=(min(BLOCK, T), len(kept)),
                                compression="gzip", compression_opts=4)
        dst_first = None
        dst_proxy = dst_proxy_first = None
        dst_los = None
        qkd_out = {}
        if cfg.channel_model == "v7":
            src_first = fa["k_max_first"]
            src_proxy = fa["k_proxy"]
            src_proxy_first = fa["k_proxy_first"]
            src_los = fa["los"]
            dst_first = fo.create_dataset("k_max_first", shape=(T, len(kept)), dtype=np.float32,
                                          chunks=(min(BLOCK, T), len(kept)), compression="gzip", compression_opts=4)
            dst_proxy = fo.create_dataset("k_proxy", shape=(T, len(kept)), dtype=np.float32,
                                          chunks=(min(BLOCK, T), len(kept)), compression="gzip", compression_opts=4)
            dst_proxy_first = fo.create_dataset("k_proxy_first", shape=(T, len(kept)), dtype=np.float32,
                                                chunks=(min(BLOCK, T), len(kept)), compression="gzip", compression_opts=4)
            dst_los = fo.create_dataset("los", shape=(T, len(kept)), dtype=np.uint8,
                                        chunks=(min(BLOCK, T), len(kept)), compression="gzip", compression_opts=4)
            qt, ql = min(256, T), min(256, len(kept))
            for name in ("qkd_y1", "qkd_usable_fraction", "qkd_y1_first", "qkd_usable_fraction_first"):
                qkd_out[name] = fo.create_dataset(name, shape=(T, len(kept)), dtype=np.float32,
                                                  chunks=(qt, ql), compression="gzip", compression_opts=4)
            for name in ("qkd_gain", "qkd_gain_first"):
                qkd_out[name] = fo.create_dataset(name, shape=(T, len(kept), len(fk.mu)), dtype=np.float32,
                                                  chunks=(qt, ql, len(fk.mu)), compression="gzip", compression_opts=4)
        for b0, b1 in blocks:
            dst[b0:b1] = src[b0:b1][:, kept]
            if dst_first is not None:
                dst_first[b0:b1] = src_first[b0:b1][:, kept]
                dst_proxy[b0:b1] = src_proxy[b0:b1][:, kept]
                dst_proxy_first[b0:b1] = src_proxy_first[b0:b1][:, kept]
                dst_los[b0:b1] = src_los[b0:b1][:, kept]
                for name, out_ds in qkd_out.items():
                    src_q = fa[name][b0:b1]
                    out_ds[b0:b1] = src_q[:, kept]
        fo.create_dataset("link_registry", data=reg)
        fo.attrs.update(dict(
            theta_sec=60,
            period_steps=T,
            period_hours=T * 60.0 / 3600.0,
            period_seconds=T * 60,
            min_elevation_deg=cfg.alpha_min_deg,
            static_hap_gs_prefilter=0,
            hap_gs_topology="one_to_one_home",
            hap_mobility=cfg.hap_mobility,
            rate_model=f"{cfg.channel_model}_{cfg.protocol}",
            cloud_model=cfg.cloud_model,
            domain_policy=cfg.domain_policy,
        ))
        if cfg.channel_model == "v7":
            fo.attrs["isl_min_grazing_alt_km"] = cfg.h_min_km
            fo.attrs["capture_time_s"] = cfg.capture_time_s
            fo.attrs["capture_gamma"] = 1.0 - cfg.capture_time_s / float(fk.slot_s)
            fo.attrs["cloud_tau_s"] = cfg.cloud_tau_s
            fo.attrs["cloud_rho"] = float(cloud_rho)
            fo.create_dataset(
                "station_cloud_state",
                data=cloud_state.T.astype(np.uint8),
                compression="gzip", compression_opts=4,
            )
            slot_hour = np.minimum(
                (np.arange(T, dtype=np.float64) * float(fk.slot_s) // 3600.0).astype(np.int64),
                c_lm.shape[1] - 1,
            )
            clear_prob = np.clip(1.0 - c_lm[:, slot_hour], 0.0, 1.0).T.astype(np.float32)
            fo.create_dataset(
                "station_cloud_clear_probability",
                data=clear_prob,
                compression="gzip", compression_opts=4,
            )
            fo.create_dataset("link_ground_station", data=ground_station_all[kept])
            fo.create_dataset(
                "station_elevation_m",
                data=np.asarray([s["h_GS_m"] for s in stations], dtype=np.float64),
            )
            fo.create_dataset(
                "station_era5_elevation_m",
                data=np.asarray([s["h_ERA5_m"] for s in stations], dtype=np.float64),
            )
            fo.attrs["station_height_correction"] = "T_GS=T_ERA5-0.0065*deltah;P_GS=P_ERA5*exp(-g*deltah/(Rd*Tbar))"
            fo.attrs["rate_semantics"] = "k_max=60s_finite_key_control;k_proxy=asymptotic_training_proxy;both_conditional_on_low_mid_cloud_clear"
            fo.attrs["policy_rate_recommended"] = "k_proxy"
            fo.attrs["policy_activation_rate_recommended"] = "k_proxy_first"
            fo.attrs["finite_key_control_rate"] = "k_max"
            fo.attrs["finite_key_control_activation_rate"] = "k_max_first"
            fo.attrs["qkd_error_gain_reconstruction"] = "T_mu=e_d*Q_mu+(0.5-e_d)*Y0;Y0=Q_vacuum"
            fo.attrs["qkd_stats_semantics"] = "conditional_clear_block_sufficient_stats_v7_5"
            fo.attrs["model_domain_semantics"] = "fraction_of_visible_geometry_substeps_with_V_R_lt_1"
            fo.attrs["bounded_moment_semantics"] = "U_in_[0,1]:SI<=(1-m_c)/m_c;incompatible_slots_are_NaN_and_unavailable"
            fo.attrs["invalid_rate_semantics"] = "NaN marks an explicitly invalid physical sample, not zero key capacity"
            fo.attrs["station_elevation_semantics"] = "synthetic_city_GS_surface_height;Copernicus_DEM_2021_GLO90_DSM_via_Open-Meteo_unless_overridden"
            fo.attrs["hourly_time_axis"] = "exact_UTC_validity_hour;hour_value_held_for_60_minute_slots"
            fo.attrs["sshf_time_semantics"] = "backward_1h_accumulation_ending_at_validity_time;converted_to_mean_Wm2_and_carried_forward_causally"
            fo.attrs["los_semantics"] = "pure_geometry_contact:LOS_and_elevation_or_grazing_distance_limits;no_cloud_or_rate_threshold"
            fo.attrs["finite_key_params_json"] = json.dumps(dataclasses.asdict(fk))
        else:
            fo.attrs["h_min_km"] = cfg.h_min_km
    tmp_path.unlink()

    with h5py.File(out / "hap_positions.h5", "w") as fh:
        if cfg.hap_mobility == "trajectory":
            pos = state["hap_pos"].astype(np.float32)          # (HAP_CYCLE_STEPS, G, 3)
            period = HAP_CYCLE_STEPS
            static = 0
        else:
            pos = np.zeros((1440, G, 3), dtype=np.float32)
            pos[:, :, 0] = state["gs_lat"][None, :]
            pos[:, :, 1] = state["gs_lon"][None, :]
            pos[:, :, 2] = cfg.hap_alt_km
            period = 1440
            static = 1
        fh.create_dataset("positions", data=pos)
        fh.create_dataset("node_ids", data=np.arange(G, 2 * G, dtype=np.int32))
        fh.attrs.update(dict(
            theta_sec=60,
            period_steps=period,
            period_hours=period * 60.0 / 3600.0,
            static=static,
            mobility=cfg.hap_mobility,
        ))

    write_csv(out / "node_registry.csv", nodes, ["node_id", "type", "name", "lat", "lon", "alt_km", "base"])
    write_csv(
        out / "gs_positions.csv",
        [
            dict(
                node_id=n["node_id"],
                name=n["name"],
                lat=n["lat"],
                lon=n["lon"],
                alt_km=n["alt_km"],
                h_GS_m=stations[i]["h_GS_m"],
                elevation_source=stations[i].get("gs_altitude_source", ""),
                elevation_resolution_m=stations[i].get("gs_altitude_resolution_m", ""),
                h_ERA5_m=stations[i]["h_ERA5_m"],
                h_ERA5_source=stations[i].get("era5_altitude_source", ""),
            )
            for i, n in enumerate(nodes[:G])
        ],
        [
            "node_id", "name", "lat", "lon", "alt_km", "h_GS_m",
            "elevation_source", "elevation_resolution_m",
            "h_ERA5_m", "h_ERA5_source",
        ],
    )
    write_csv(out / "link_registry.csv", [dict(link_id=i, node_u=int(r["node_u"]), node_v=int(r["node_v"]),
                                               link_type=r["link_type"].decode()) for i, r in enumerate(reg)],
              ["link_id", "node_u", "node_v", "link_type"])
    old_id = {j: i for i, j in enumerate(kept)}
    write_csv(out / "link_registry_skipped.csv", skipped,
              ["node_u", "node_v", "link_type", "reason", "value", "threshold"])
    diag = []
    for j, (u, v, lt) in enumerate(links):
        row = dict(candidate=j, link_id=old_id.get(j, ""), node_u=u, node_v=v, link_type=lt,
                   visible_frac=n_vis[j] / T, finite_key_positive_frac=n_pos[j] / T,
                   proxy_positive_frac=n_proxy_pos[j] / T if cfg.channel_model == "v7" else n_pos[j] / T,
                   cloud_blocked_frac_of_visible=(n_cloud[j] / n_vis[j]) if n_vis[j] else 0.0,
                   mean_finite_key_rate_bps=r_sum[j] / T,
                   mean_finite_key_rate_when_positive_bps=(r_sum[j] / n_pos[j]) if n_pos[j] else 0.0,
                   mean_proxy_rate_bps=(proxy_sum[j] / T if cfg.channel_model == "v7" else r_sum[j] / T))
        if cfg.channel_model == "v7":
            denom = max(int(n_vis[j]), 1)
            domain_visible = float(v7_stat_sum["domain_visible_substeps"][j])
            domain_valid = float(v7_stat_sum["domain_valid_substeps"][j])
            moment_visible = float(v7_stat_sum["moment_visible_substeps"][j])
            moment_valid = float(v7_stat_sum["moment_compatible_substeps"][j])
            row.update(
                # Exact fractions across all participating visible geometry sub-steps.
                domain_valid_frac=(domain_valid / domain_visible if domain_visible > 0.0 else 0.0),
                moment_compatible_frac=(moment_valid / moment_visible if moment_visible > 0.0 else 0.0),
                # Retain the slot-averaged diagnostic under an explicit name.
                mean_visible_substep_domain_valid_fraction=float(
                    v7_stat_sum["model_domain"][j] / denom
                ),
                mean_visible_substep_moment_compatible_fraction=float(
                    v7_stat_sum["moment_compatible"][j] / denom
                ),
                mean_rytov=float(v7_stat_sum["rytov"][j] / denom),
                mean_collection_tail=float(v7_stat_sum["collection_tail"][j] / denom),
                mean_collection=float(v7_stat_sum["collection_mean"][j] / denom),
                mean_collection_truncated=float(v7_stat_sum["collection_mean_truncated"][j] / denom),
                mean_collection_bias=float(v7_stat_sum["collection_mean_bias"][j] / denom),
                mean_collection_second_moment=float(v7_stat_sum["collection_second_moment"][j] / denom),
                mean_usable_fraction=float(v7_stat_sum["usable_fraction"][j] / denom),
                mean_proxy_per_pulse=float(v7_stat_sum["proxy_per_pulse"][j] / denom),
            )
            for i, label in enumerate(("mu1", "mu2", "mu3")):
                row[f"mean_gain_{label}"] = float(v7_stat_sum["gain"][j, i] / denom)
                row[f"mean_error_gain_{label}"] = float(v7_stat_sum["error_gain"][j, i] / denom)
        diag.append(row)
    write_csv(out / "link_diag.csv", diag, list(diag[0].keys()))

    summary = {}
    for lt in TYPES:
        idx = [j for j, x in enumerate(links) if x[2] == lt]
        if not idx:
            continue
        idx = np.array(idx)
        summary[lt] = dict(
            candidates=int(len(idx)), kept=int(keep[idx].sum()),
            visible_frac=float(n_vis[idx].sum() / (T * len(idx))),
            finite_key_positive_frac=float(n_pos[idx].sum() / (T * len(idx))),
            finite_key_positive_given_visible=float(n_pos[idx].sum() / max(n_vis[idx].sum(), 1)),
            mean_finite_key_rate_when_positive_bps=float(r_sum[idx].sum() / max(n_pos[idx].sum(), 1)),
            finite_key_network_mean_bps_per_slot=float(r_sum[idx].sum() / T),
            proxy_positive_frac=(float(n_proxy_pos[idx].sum() / (T * len(idx)))
                                 if cfg.channel_model == "v7" else float(n_pos[idx].sum() / (T * len(idx)))),
            proxy_positive_given_visible=(float(n_proxy_pos[idx].sum() / max(n_vis[idx].sum(), 1))
                                          if cfg.channel_model == "v7" else float(n_pos[idx].sum() / max(n_vis[idx].sum(), 1))),
            mean_proxy_rate_when_positive_bps=(float(proxy_sum[idx].sum() / max(n_proxy_pos[idx].sum(), 1))
                                               if cfg.channel_model == "v7" else float(r_sum[idx].sum() / max(n_pos[idx].sum(), 1))),
            proxy_network_mean_bps_per_slot=(float(proxy_sum[idx].sum() / T)
                                             if cfg.channel_model == "v7" else float(r_sum[idx].sum() / T)),
        )
    inputs = {}
    # v7 无论 cloud_model 取哪个都读分层云（binary_total 对照也用同一 C_lm），
    # 所以来源校验和必须把分层云也算进去，否则对照数据集的 inputs_md5 缺一项。
    _need_layers_md5 = cfg.channel_model == "v7" or cfg.cloud_model == "coverage_layers"
    for s in stations:
        for kind, d in (("weather", wdir), ("era5", edir)) + ((("cloud_layers", ldir),)
                                                              if _need_layers_md5 else ()):
            f = d / f"{s['name']}.csv"
            if f.exists():
                inputs[f"{kind}/{s['name']}.csv"] = md5(f)
    if cfg.channel_model == "v7":
        from tools.channel_v7.model import MODEL_REVISION as model_revision
        model_path = ROOT / "tools/channel_v7/model.py"
        collection_model_path = ROOT / "tools/channel_v7/dataset.py"
    else:
        model_revision = MODEL_REVISION
        model_path = ROOT / "tools/channel_v2/rate_model_v2.py"
        collection_model_path = None
    meta = dict(
        builder="tools/dataset_v2/build_dataset_v2.py", git_head=git_head(),
        channel_model=cfg.channel_model, model_revision=model_revision,
        hap_mobility=cfg.hap_mobility,
        scint_expectation=("bounded_beta_two_moment" if cfg.channel_model == "v7" else "lognormal_gauss_hermite"),
        scint_quadrature=("gauss_legendre_beta_quantile" if cfg.channel_model == "v7" else "gauss_hermite"),
        scint_quadrature_order=params.scint_gh_nodes,
        scint_gh_nodes=params.scint_gh_nodes,
        upper_atm_theta_per_km=[params.theta_strat_per_km, params.theta_meso_per_km], isl_nb=params.isl_nb,
        created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t_start, 1),
        theta_sec=60, t_steps=T, days=args.days, year=cfg.year, t0="2023-01-01T00:00Z",
        num_nodes=len(nodes), num_gs=G, num_hap=G, num_sat=S, num_links=int(len(kept)),
        num_candidate_links=L, num_skipped_links=len(skipped),
        base_nodes="node_registry.csv column base = 1 (30-station base network, listed first)",
        build_config=dataclasses.asdict(cfg), rate_model_params=dataclasses.asdict(params),
        finite_key_params=fk.as_dict(), sat_alt_km=sat_alt, stations_file=str(stations_arg),
        stations_md5=md5(stations_path),
        station_vertical_reference=[
            {
                "name": s["name"],
                "h_GS_m": s["h_GS_m"],
                "h_ERA5_m": s["h_ERA5_m"],
                "delta_h_m": s["h_GS_m"] - s["h_ERA5_m"],
                "gs_altitude_explicit": s["gs_altitude_explicit"],
                "gs_altitude_source": s.get("gs_altitude_source", ""),
                "gs_altitude_resolution_m": s.get("gs_altitude_resolution_m"),
                "era5_altitude_explicit": s["era5_altitude_explicit"],
                "era5_altitude_source": s.get("era5_altitude_source", "station_csv" if s["era5_altitude_explicit"] else "missing"),
            }
            for s in stations
        ],
        station_height_correction_method=(
            "surface_lapse_hypsometric" if cfg.channel_model == "v7" else "legacy_unadjusted"
        ),
        station_height_temperature_lapse_K_per_m=(6.5e-3 if cfg.channel_model == "v7" else None),
        allow_missing_station_elevation=bool(args.allow_missing_station_elevation),
        weather_fill=fill,
        turbulence_state_semantics=(
            "a_cn2 array stores matched AGL-HV coefficient A(t) after MOST Cn2(z_ref) matching"
            if cfg.channel_model == "v7"
            else "a_cn2 array stores the legacy near-ground Cn2 input"
        ),
        inputs_md5=inputs,
        input_sources=(
            {
                "cloud_layers": "ERA5 via Open-Meteo Historical Weather API (models=era5, elevation=nan)",
                "aod550": "CAMS global via Open-Meteo Air Quality API",
                "surface_most": "ERA5 single-levels via CDS",
            }
            if cfg.channel_model == "v7" else {}
        ),
        hourly_time_alignment=(
            {
                "axis": "exact 2023 UTC hourly validity timestamps; all input CSV time columns are validated before use",
                "minute_slots": "hour h values are held for scheduling slots [h:00, h:59]",
                "cloud_layers": "ERA5 hourly value at validity hour h, held over that UTC hour",
                "aod550": "CAMS hourly value at UTC hour h, held over that UTC hour",
                "t2m_sp_ustar": "ERA5 values at validity hour h, held over that UTC hour",
                "sshf": (
                    "ERA5 one-hour accumulation ending at validity hour h, converted to W/m2 "
                    "and causally carried forward over [h:00,h:59]; therefore MOST heat flux "
                    "is a backward-looking one-hour mean rather than a future-hour mean"
                ),
            }
            if cfg.channel_model == "v7" else {}
        ),
        allow_era5_fallback=bool(args.allow_era5_fallback), per_type=summary,
        rate_model_md5=md5(model_path),
        collection_model_md5=(md5(collection_model_path) if collection_model_path else ""),
        finite_key_md5=md5(ROOT / "tools/dataset_v2/finite_key.py"),
    )
    if cfg.channel_model == "v7":
        visible_slot_total = int(n_vis.sum())
        domain_visible_substeps = float(v7_stat_sum["domain_visible_substeps"].sum())
        domain_valid_substeps = float(v7_stat_sum["domain_valid_substeps"].sum())
        moment_visible_substeps = float(v7_stat_sum["moment_visible_substeps"].sum())
        moment_compatible_substeps = float(v7_stat_sum["moment_compatible_substeps"].sum())
        meta.update(
            capture_time_s=cfg.capture_time_s,
            capture_gamma=1.0 - cfg.capture_time_s / float(fk.slot_s),
            cloud_tau_s=cfg.cloud_tau_s,
            cloud_rho=float(cloud_rho),
            cloud_model="station_markov",
            policy_rate_dataset="k_proxy",
            finite_key_control_dataset="k_max",
            intra_slot_samples=cfg.intra_slot_samples,
            domain_policy=cfg.domain_policy,
            aperture_averaging=bool(getattr(params, "aperture_averaging", False)),
            saturate_scint=bool(getattr(params, "saturate_scint", False)),
            model_domain_visible_slots=visible_slot_total,
            model_domain_visible_substeps=domain_visible_substeps,
            model_domain_valid_substeps=domain_valid_substeps,
            model_domain_valid_visible_fraction=(
                domain_valid_substeps / domain_visible_substeps
                if domain_visible_substeps > 0.0 else 1.0
            ),
            bounded_moment_visible_substeps=moment_visible_substeps,
            bounded_moment_compatible_substeps=moment_compatible_substeps,
            bounded_moment_compatible_visible_fraction=(
                moment_compatible_substeps / moment_visible_substeps
                if moment_visible_substeps > 0.0 else 1.0
            ),
            finite_key_scope=("expected-count finite-key performance estimator; not an actual-count device security certificate"
                              if cfg.protocol == "finite_key" else "asymptotic diagnostic; no finite-key security claim"),
            phase_randomization_assumed=(cfg.protocol == "finite_key"),
            per_block_eps_only=(cfg.protocol == "finite_key"),
        )
    with open(out / "build_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)
    print(json.dumps(summary, indent=1), flush=True)

    if not args.no_stats:
        stride = 60 if T >= 60 * 100 else 1
        stats_rate_key = "k_proxy" if cfg.channel_model == "v7" else "k_max"
        subprocess.run([sys.executable, str(ROOT / "scripts/estimate_rate_stats.py"), "--dataset-dir", str(out),
                        "--rate-key", stats_rate_key, "--stride", str(stride)], cwd=ROOT, check=True)
    print(f"done in {time.time() - t_start:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
