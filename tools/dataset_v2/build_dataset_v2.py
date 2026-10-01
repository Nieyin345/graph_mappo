"""Build a link-rate dataset with the v2 rate model (paper Sec. III-B + App. C) into a NEW directory.

    python tools/dataset_v2/build_dataset_v2.py --check-inputs          # are all weather files there?
    python tools/dataset_v2/build_dataset_v2.py --out dataset/global_v2 --workers 8
    python tools/dataset_v2/build_dataset_v2.py --out .tmp/ds_v2_smoke --days 2 --allow-era5-fallback

What it does
------------
* nodes: the ground stations of ``tools/channel_v2/stations_45.csv`` (``base`` = 1 marks the 30-station
  base network, listed first), one HAP 20 km above every station (static, paper III-A, user decision
  09-30: option A), and the Walker-delta 53:30/5/1 constellation at 550 km
  (``GlobalSatellites.walker_delta``, commit 98b3d2c);
* candidate links (eq. (1)): every SAT-GS, SAT-HAP and SAT-SAT pair, and HAP-GS pairs whose ground
  distance is at most ``d_max`` and whose (static) elevation reaches ``alpha_min``;
* per slot (Delta = 60 s, 2023, t = minutes since 2023-01-01 00:00 UTC): geometry, visibility, mean
  transmittance and background probability from ``tools/channel_v2/rate_model_v2.RateModelV2``
  (ERA5 Monin-Obukhov C_n^2, CAMS AOD, SMF Monte-Carlo table, kappa_x), the finite-key length of
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
  ``link_registry_skipped.csv``.

Output (same layout as dataset/global, read by qkd_rl/link/rate_provider.py without changes):
``node_registry.csv``, ``link_registry.csv``, ``link_registry_skipped.csv``, ``link_data.h5`` (``k_max``
(T, L) float32 bit/s + ``link_registry``), ``sat_positions.h5``, ``hap_positions.h5``,
``gs_positions.csv``, ``link_diag.csv`` (per-link visibility / rate statistics), ``build_metadata.json``
(every parameter, input checksums) and ``rate_stats.json`` (scripts/estimate_rate_stats.py).

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
for sub in ("tools/channel_v2", "tools/data_generation", "tools/dataset_v2"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from finite_key import FiniteKeyParams, slot_rate  # noqa: E402
from rate_model_v2 import (ParamsV2, RateModelV2, airmass_kasten_young, cn2_monin_obukhov,  # noqa: E402
                           p_clear_lund_shanklin, sun_elevation_deg)

MARKER = ".built_by_build_dataset_v2"
YEAR_HOURS = 8760
SLOT_MIN = 1                      # Delta = 60 s
BLOCK = 2048                      # rows per block = HDF5 chunk height (as in dataset/global)
LINK_BATCH = 256                  # links evaluated together inside a block (memory bound)
R_E = 6371.0
TYPES = ("HAP-GS", "SAT-GS", "SAT-HAP", "SAT-SAT")


@dataclasses.dataclass(frozen=True)
class BuildConfig:
    alpha_min_deg: float = 5.0    # eq. (1), ground-station elevation mask (as dataset/global)
    h_min_km: float = 15.0        # eq. (1), lowest ray height of H-S / S-S links (as the current ISL code)
    d_max_km: float = 700.0       # HAP-GS candidate ground distance (as dataset/global)
    hap_alt_km: float = 20.0
    sat_constellation: str = "walker"
    cloud_seed: int = 2023
    year: int = 2023
    protocol: str = "finite_key"  # "finite_key" (paper eq. (18)) | "asymptotic" (diagnostics only)
    cloud_model: str = "binary_total"   # "binary_total" (dataset/global_v2) | "coverage_layers" (10-01)
    # visible optical depth of a high (ice) cloud layer; at 1.55 um the ice-crystal extinction efficiency
    # is ~2 as in the visible (geometric-optics limit), so the visible value is used. 0.3 = boundary
    # thin/opaque cirrus (Sassen & Cho 1992 classes); CALIPSO / ground-lidar annual means ~0.3.
    tau_cirrus: float = 0.3


# ----------------------------------------------------------------------------- inputs
def read_stations(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = [dict(name=r["name"], lat=float(r["lat"]), lon=float(r["lon"]),
                base=int(r.get("base", 1) or 0), continent=r.get("continent", "")) for r in rows]
    names = [r["name"] for r in out]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate station names in {path}")
    return out


def _read_csv_columns(path: Path, cols: list[str]) -> dict[str, np.ndarray]:
    import pandas as pd
    df = pd.read_csv(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    return {c: df[c].astype(float).to_numpy() for c in cols}


LAYER_COLS = ["cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"]


def check_inputs(stations, weather_dir: Path, era5_dir: Path, layers_dir: Path | None = None) -> dict:
    """Per station: Open-Meteo weather (cloud, AOD) and ERA5 fluxes present, 8760 rows, NaN counts."""
    rep = {}
    for s in stations:
        r = {}
        w = weather_dir / f"{s['name']}.csv"
        e = era5_dir / f"{s['name']}.csv"
        if w.exists():
            d = _read_csv_columns(w, ["cloud_cover", "aod550"])
            r["weather_rows"] = int(len(d["cloud_cover"]))
            r["cloud_nan"] = int(np.isnan(d["cloud_cover"]).sum())
            r["aod_nan"] = int(np.isnan(d["aod550"]).sum())
        else:
            r["weather_rows"] = 0
        if e.exists():
            d = _read_csv_columns(e, ["H_up_Wm2", "ustar", "t2m_K", "sp_Pa"])
            r["era5_rows"] = int(len(d["ustar"]))
            r["era5_nan_hours"] = int(np.isnan(np.stack(list(d.values()))).any(axis=0).sum())
        else:
            r["era5_rows"] = 0
        r["ok"] = r["weather_rows"] == YEAR_HOURS and r["era5_rows"] == YEAR_HOURS
        if layers_dir is not None:
            f = layers_dir / f"{s['name']}.csv"
            if f.exists():
                d = _read_csv_columns(f, LAYER_COLS)
                r["layer_rows"] = int(len(d[LAYER_COLS[0]]))
                r["layer_nan"] = int(np.isnan(np.stack(list(d.values()))).sum())
            else:
                r["layer_rows"] = 0
            r["ok"] = r["ok"] and r["layer_rows"] == YEAR_HOURS
        rep[s["name"]] = r
    return rep


def load_cloud_layers(stations, layers_dir: Path):
    """(G, 8760) low+mid combined cover C_lm = 1-(1-C_low)(1-C_mid) and high cover, fractions 0..1."""
    G = len(stations)
    c_lm = np.zeros((G, YEAR_HOURS))
    c_hi = np.zeros((G, YEAR_HOURS))
    fill = {}
    for gi, s in enumerate(stations):
        d = _read_csv_columns(layers_dir / f"{s['name']}.csv", LAYER_COLS)
        if len(d[LAYER_COLS[0]]) != YEAR_HOURS:
            raise SystemExit(f"{s['name']}: cloud-layer file has {len(d[LAYER_COLS[0]])} rows, need {YEAR_HOURS}")
        v = {}
        for c in LAYER_COLS:
            x = d[c]
            fill.setdefault(s["name"], {})[c + "_nan"] = int(np.isnan(x).sum())
            v[c] = np.clip(np.nan_to_num(x, nan=float(np.nanmean(x)) if np.isfinite(x).any() else 0.0) / 100.0, 0, 1)
        c_lm[gi] = 1.0 - (1.0 - v["cloud_cover_low"]) * (1.0 - v["cloud_cover_mid"])
        c_hi[gi] = v["cloud_cover_high"]
    return c_lm, c_hi, fill


def load_weather(stations, weather_dir: Path, era5_dir: Path, p: ParamsV2, allow_era5_fallback: bool):
    """(G, 8760) arrays: cloud fraction, AOD550, ERA5 C_n^2(0) and a mask of hours with valid ERA5."""
    G = len(stations)
    cloud = np.zeros((G, YEAR_HOURS))
    aod = np.zeros((G, YEAR_HOURS))
    a_cn2 = np.full((G, YEAR_HOURS), np.nan)
    ok = np.zeros((G, YEAR_HOURS), dtype=bool)
    fill = {}
    for gi, s in enumerate(stations):
        d = _read_csv_columns(weather_dir / f"{s['name']}.csv", ["cloud_cover", "aod550"])
        if len(d["cloud_cover"]) != YEAR_HOURS:
            raise SystemExit(f"{s['name']}: weather file has {len(d['cloud_cover'])} rows, need {YEAR_HOURS}")
        c, a = d["cloud_cover"], d["aod550"]
        fill[s["name"]] = {"cloud_nan": int(np.isnan(c).sum()), "aod_nan": int(np.isnan(a).sum())}
        cloud[gi] = np.nan_to_num(c, nan=float(np.nanmean(c)) if np.isfinite(c).any() else 0.0) / 100.0
        aod[gi] = np.nan_to_num(a, nan=float(np.nanmedian(a)) if np.isfinite(a).any() else 0.1)
        e = era5_dir / f"{s['name']}.csv"
        if not e.exists():
            if not allow_era5_fallback:
                raise SystemExit(f"no ERA5 file for {s['name']} ({e}); wait for the download or pass "
                                 "--allow-era5-fallback (smoke tests only)")
            fill[s["name"]]["era5"] = "missing -> day/night A for every hour"
            continue
        f = _read_csv_columns(e, ["H_up_Wm2", "ustar", "t2m_K", "sp_Pa"])
        if len(f["ustar"]) != YEAR_HOURS:
            raise SystemExit(f"{s['name']}: ERA5 file has {len(f['ustar'])} rows, need {YEAR_HOURS}")
        valid = np.isfinite(np.stack([f["H_up_Wm2"], f["ustar"], f["t2m_K"], f["sp_Pa"]])).all(axis=0)
        cn2 = cn2_monin_obukhov(f["H_up_Wm2"], f["ustar"], f["t2m_K"], f["sp_Pa"],
                                z=p.z_ref, cn2_min=p.cn2_min, zeta_max=p.zeta_max)
        a_cn2[gi] = np.where(valid, cn2, np.nan)
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


def build_nodes(stations, cfg: BuildConfig, sat_cfgs):
    G, S = len(stations), len(sat_cfgs)
    rows = []
    for i, s in enumerate(stations):
        rows.append(dict(node_id=i, type="GS", name=s["name"], lat=s["lat"], lon=s["lon"], alt_km=0.0, base=s["base"]))
    for i, s in enumerate(stations):
        rows.append(dict(node_id=G + i, type="HAP", name=f"HAP_{s['name']}", lat=s["lat"], lon=s["lon"],
                         alt_km=cfg.hap_alt_km, base=s["base"]))
    la, lo, al = sat_positions(sat_cfgs, np.array([0.0]))
    for i, c in enumerate(sat_cfgs):
        rows.append(dict(node_id=2 * G + i, type="SAT", name=c["name"], lat=float(la[i, 0]), lon=float(lo[i, 0]),
                         alt_km=float(al[i, 0]), base=1))
    return rows, G, S


def candidate_links(nodes, G, S, cfg: BuildConfig):
    """Candidate pairs (u < v) in node order, plus the statically rejected HAP-GS pairs."""
    links, skipped = [], []
    gs = nodes[:G]
    hap = nodes[G:2 * G]
    for g in gs:                                   # HAP-GS
        for h in hap:
            dist = haversine_km(g["lat"], g["lon"], h["lat"], h["lon"])
            _, el, _, _ = geometry((np.array(g["lat"]), np.array(g["lon"]), np.array(0.0)),
                                   (np.array(h["lat"]), np.array(h["lon"]), np.array(h["alt_km"])))
            el = float(el)
            if dist > cfg.d_max_km:
                skipped.append(dict(node_u=g["node_id"], node_v=h["node_id"], link_type="HAP-GS",
                                    reason="hap_gs_static_distance_prefilter", value=dist, threshold=cfg.d_max_km))
            elif el < cfg.alpha_min_deg:
                skipped.append(dict(node_u=g["node_id"], node_v=h["node_id"], link_type="HAP-GS",
                                    reason="hap_gs_static_elevation_below_alpha_min", value=el,
                                    threshold=cfg.alpha_min_deg))
            else:
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
    _W["model"] = RateModelV2(state["params"], sat_alt_km=state["sat_alt_km"], hap_alt_km=state["cfg"].hap_alt_km)


def compute_block(b0: int, b1: int):
    """Rates (b1-b0, L) float32 and per-link visible / positive slot counts for slots [b0, b1)."""
    W = _W
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
        g_of = np.where(is_hap, ids - G, ids)
        k = ids[is_sat] - 2 * G
        la[is_sat], lo[is_sat], al[is_sat] = s_la[k], s_lo[k], s_al[k]
        ng = ~is_sat
        la[ng] = gla[g_of[ng]][:, None]
        lo[ng] = glo[g_of[ng]][:, None]
        al[ng] = np.where(is_hap[ng], cfg.hap_alt_km, 0.0)[:, None]
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
            r_asym, parts = m.link_rate(lt, dist[li, ti] * 1e3, elev[li, ti], return_parts=True, **kw)
            if cov_f is not None:
                eta_c = parts["eta"] * t_ci
                r = np.zeros(li.size)
                ok = cov_f > 0
                if cfg.protocol == "asymptotic":
                    r[ok] = m.skr(eta_c[ok], np.broadcast_to(parts["y0"], eta_c.shape)[ok]) * cov_f[ok]
                else:
                    r[ok] = slot_rate(eta_c[ok], np.broadcast_to(parts["y0"], eta_c.shape)[ok], fk,
                                      n_pulses=cov_f[ok] * fk.n_pulses)
                blocked = ~ok
            else:
                if cfg.protocol == "asymptotic":
                    r = np.asarray(r_asym, dtype=float) * chi
                else:
                    r = slot_rate(parts["eta"], parts["y0"], fk) * chi
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


def prepare_out(out: Path, overwrite: bool) -> None:
    protected = (ROOT / "dataset" / "global").resolve()
    if out.resolve() == protected or protected in out.resolve().parents:
        raise SystemExit("refusing to write into dataset/global")
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite (only for directories built by this script)")
        if not (out / MARKER).exists():
            raise SystemExit(f"{out} was not created by build_dataset_v2.py; not overwriting")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / MARKER).write_text("created by tools/dataset_v2/build_dataset_v2.py\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="dataset/global_v2")
    ap.add_argument("--stations", default="tools/channel_v2/stations_45.csv")
    ap.add_argument("--weather-dir", default=".tmp/channel_v2/weather")
    ap.add_argument("--era5-dir", default=".tmp/channel_v2/era5")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--cloud-seed", type=int, default=BuildConfig.cloud_seed)
    ap.add_argument("--constellation", default=BuildConfig.sat_constellation, choices=["walker", "legacy"],
                    help="diagnostics: legacy = the hand-written list of dataset/global")
    ap.add_argument("--protocol", default=BuildConfig.protocol, choices=["finite_key", "asymptotic"],
                    help="diagnostics: asymptotic = GLLP rate of rate_model_v2 instead of eq. (18)")
    ap.add_argument("--cloud-model", default=BuildConfig.cloud_model, choices=["binary_total", "coverage_layers"],
                    help="binary_total = hourly Lund-Shanklin indicator on total cover (dataset/global_v2); "
                         "coverage_layers = low+mid cover as blocked slot fraction, high cloud as attenuation")
    ap.add_argument("--tau-cirrus", type=float, default=BuildConfig.tau_cirrus)
    ap.add_argument("--cloud-layers-dir", default=".tmp/channel_v2/cloud_layers")
    ap.add_argument("--allow-era5-fallback", action="store_true",
                    help="stations without an ERA5 file use the day/night A of Liorni 2019 (smoke tests only)")
    ap.add_argument("--keep-zero-links", action="store_true", help="do not drop links that are zero all period")
    ap.add_argument("--no-stats", action="store_true", help="skip scripts/estimate_rate_stats.py")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check-inputs", action="store_true", help="only report which input files are ready")
    args = ap.parse_args(argv)

    rel = lambda x: (ROOT / x) if not Path(x).is_absolute() else Path(x)  # noqa: E731
    stations = read_stations(rel(args.stations))
    wdir, edir, out = rel(args.weather_dir), rel(args.era5_dir), rel(args.out)
    ldir = rel(args.cloud_layers_dir)
    if args.check_inputs:
        rep = check_inputs(stations, wdir, edir, ldir if args.cloud_model == "coverage_layers" else None)
        bad = [n for n, r in rep.items() if not r["ok"]]
        for n, r in rep.items():
            print(f"{n:14s} {'OK ' if r['ok'] else 'MISSING'} {r}")
        print(f"{len(rep) - len(bad)}/{len(rep)} stations ready" + (f"; missing: {bad}" if bad else ""))
        return 1 if bad else 0

    import h5py

    t_start = time.time()
    cfg = BuildConfig(cloud_seed=args.cloud_seed, sat_constellation=args.constellation, protocol=args.protocol,
                      cloud_model=args.cloud_model, tau_cirrus=args.tau_cirrus)
    params = ParamsV2()
    fk = FiniteKeyParams()
    T = args.days * 1440 // SLOT_MIN
    if not 1 <= args.days <= 365:
        raise SystemExit("--days must be in 1..365 (weather covers 2023)")
    cloud, aod, a_cn2, era5_ok, fill = load_weather(stations, wdir, edir, params, args.allow_era5_fallback)
    c_lm = c_hi = None
    if cfg.cloud_model == "coverage_layers":
        c_lm, c_hi, lfill = load_cloud_layers(stations, ldir)
        for k, v in lfill.items():
            fill.setdefault(k, {}).update(v)
    prepare_out(out, args.overwrite)

    sat_cfgs = satellite_configs(cfg.sat_constellation)
    nodes, G, S = build_nodes(stations, cfg, sat_cfgs)
    links, skipped = candidate_links(nodes, G, S, cfg)
    L = len(links)
    sat_alt = float(np.median([c["alt_km"] for c in sat_cfgs]))
    rng = np.random.default_rng(cfg.cloud_seed)
    u_cloud = rng.random((G, YEAR_HOURS))
    print(f"nodes GS={G} HAP={G} SAT={S}; candidate links {L} "
          f"({', '.join(f'{t}={sum(1 for x in links if x[2] == t)}' for t in TYPES)}); T={T}", flush=True)

    state = dict(cfg=cfg, params=params, fk=fk, G=G, sat_cfgs=sat_cfgs, sat_alt_km=sat_alt,
                 gs_lat=np.array([s["lat"] for s in stations]), gs_lon=np.array([s["lon"] for s in stations]),
                 link_u=np.array([x[0] for x in links]), link_v=np.array([x[1] for x in links]),
                 link_type=np.array([x[2] for x in links]), cloud=cloud, aod=aod, a_cn2=a_cn2,
                 era5_ok=era5_ok, u_cloud=u_cloud, c_lm=c_lm, c_hi=c_hi)

    tmp_path = out / "_link_data_all_candidates.h5"
    n_vis = np.zeros(L, dtype=np.int64)
    n_pos = np.zeros(L, dtype=np.int64)
    n_cloud = np.zeros(L, dtype=np.int64)
    r_sum = np.zeros(L)
    blocks = [(b, min(b + BLOCK, T)) for b in range(0, T, BLOCK)]
    with h5py.File(tmp_path, "w") as fa, h5py.File(out / "sat_positions.h5", "w") as fsat:
        ds = fa.create_dataset("k_max", shape=(T, L), dtype=np.float32, chunks=(min(BLOCK, T), L),
                               compression="gzip", compression_opts=4)
        dsat = fsat.create_dataset("positions", shape=(T, S, 3), dtype=np.float32, chunks=(min(BLOCK, T), S, 3))
        fsat.create_dataset("node_ids", data=np.arange(2 * G, 2 * G + S, dtype=np.int32))
        fsat.attrs["theta_sec"] = 60
        fsat.attrs["constellation"] = cfg.sat_constellation

        def consume(res):
            b0, blk, nv, npos, ncl = res
            ds[b0:b0 + blk.shape[0]] = blk
            n_vis[:] += nv
            n_pos[:] += npos
            n_cloud[:] += ncl
            r_sum[:] += blk.sum(axis=0, dtype=np.float64)

        done = 0
        if args.workers <= 1:
            _init_worker(state)
            it = (compute_block(b0, b1) for b0, b1 in blocks)
            for res in it:
                consume(res)
                done += 1
                print(f"  block {done}/{len(blocks)} ({time.time() - t_start:.0f}s)", flush=True)
        else:
            with cf.ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                        initargs=(state,)) as ex:
                futs = [ex.submit(compute_block, b0, b1) for b0, b1 in blocks]
                for fu in futs:                    # in order, so writes stay sequential
                    consume(fu.result())
                    done += 1
                    if done % 10 == 0 or done == len(blocks):
                        print(f"  block {done}/{len(blocks)} ({time.time() - t_start:.0f}s)", flush=True)
        for b0, b1 in blocks:
            la, lo, al = sat_positions(sat_cfgs, np.arange(b0, b1))
            dsat[b0:b1] = np.stack([la.T, lo.T, al.T], axis=-1).astype(np.float32)

    keep = np.ones(L, dtype=bool) if args.keep_zero_links else (n_pos > 0)
    for j in np.flatnonzero(~keep):
        u, v, lt = links[j]
        skipped.append(dict(node_u=u, node_v=v, link_type=lt, reason="zero_rate_all_period",
                            value=int(n_vis[j]), threshold=""))
    kept = np.flatnonzero(keep)
    reg = np.zeros(len(kept), dtype=[("link_id", "<i4"), ("node_u", "<i4"), ("node_v", "<i4"), ("link_type", "S10")])
    for new_id, j in enumerate(kept):
        u, v, lt = links[j]
        reg[new_id] = (new_id, u, v, lt.encode())
    with h5py.File(tmp_path, "r") as fa, h5py.File(out / "link_data.h5", "w") as fo:
        src = fa["k_max"]
        dst = fo.create_dataset("k_max", shape=(T, len(kept)), dtype=np.float32, chunks=(min(BLOCK, T), len(kept)),
                                compression="gzip", compression_opts=4)
        for b0, b1 in blocks:
            dst[b0:b1] = src[b0:b1][:, kept]
        fo.create_dataset("link_registry", data=reg)
        fo.attrs.update(dict(theta_sec=60, period_steps=T, period_hours=T, min_elevation_deg=cfg.alpha_min_deg,
                             h_min_km=cfg.h_min_km, max_hap_gs_distance_km=cfg.d_max_km, static_hap_gs_prefilter=1,
                             rate_model="v2_finite_key", cloud_model=cfg.cloud_model))
    tmp_path.unlink()

    with h5py.File(out / "hap_positions.h5", "w") as fh:
        pos = np.zeros((1440, G, 3), dtype=np.float32)
        pos[:, :, 0] = state["gs_lat"][None, :]
        pos[:, :, 1] = state["gs_lon"][None, :]
        pos[:, :, 2] = cfg.hap_alt_km
        fh.create_dataset("positions", data=pos)
        fh.create_dataset("node_ids", data=np.arange(G, 2 * G, dtype=np.int32))
        fh.attrs.update(dict(theta_sec=60, period_steps=1440, period_hours=1440, static=1))

    write_csv(out / "node_registry.csv", nodes, ["node_id", "type", "name", "lat", "lon", "alt_km", "base"])
    write_csv(out / "gs_positions.csv", [dict(node_id=n["node_id"], name=n["name"], lat=n["lat"], lon=n["lon"])
                                         for n in nodes[:G]], ["node_id", "name", "lat", "lon"])
    write_csv(out / "link_registry.csv", [dict(link_id=i, node_u=int(r["node_u"]), node_v=int(r["node_v"]),
                                               link_type=r["link_type"].decode()) for i, r in enumerate(reg)],
              ["link_id", "node_u", "node_v", "link_type"])
    old_id = {j: i for i, j in enumerate(kept)}
    write_csv(out / "link_registry_skipped.csv", skipped,
              ["node_u", "node_v", "link_type", "reason", "value", "threshold"])
    diag = []
    for j, (u, v, lt) in enumerate(links):
        diag.append(dict(candidate=j, link_id=old_id.get(j, ""), node_u=u, node_v=v, link_type=lt,
                         visible_frac=n_vis[j] / T, positive_frac=n_pos[j] / T,
                         cloud_blocked_frac_of_visible=(n_cloud[j] / n_vis[j]) if n_vis[j] else 0.0,
                         mean_rate_bps=r_sum[j] / T,
                         mean_rate_when_positive_bps=(r_sum[j] / n_pos[j]) if n_pos[j] else 0.0))
    write_csv(out / "link_diag.csv", diag, list(diag[0].keys()))

    summary = {}
    for lt in TYPES:
        idx = [j for j, x in enumerate(links) if x[2] == lt]
        if not idx:
            continue
        idx = np.array(idx)
        summary[lt] = dict(candidates=int(len(idx)), kept=int(keep[idx].sum()),
                           visible_frac=float(n_vis[idx].sum() / (T * len(idx))),
                           positive_frac=float(n_pos[idx].sum() / (T * len(idx))),
                           positive_given_visible=float(n_pos[idx].sum() / max(n_vis[idx].sum(), 1)),
                           mean_rate_when_positive_bps=float(r_sum[idx].sum() / max(n_pos[idx].sum(), 1)),
                           network_mean_bps_per_slot=float(r_sum[idx].sum() / T))
    inputs = {}
    for s in stations:
        for kind, d in (("weather", wdir), ("era5", edir)) + ((("cloud_layers", ldir),)
                                                              if cfg.cloud_model == "coverage_layers" else ()):
            f = d / f"{s['name']}.csv"
            if f.exists():
                inputs[f"{kind}/{s['name']}.csv"] = md5(f)
    meta = dict(
        builder="tools/dataset_v2/build_dataset_v2.py", git_head=git_head(),
        created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t_start, 1),
        theta_sec=60, t_steps=T, days=args.days, year=cfg.year, t0="2023-01-01T00:00Z",
        num_nodes=len(nodes), num_gs=G, num_hap=G, num_sat=S, num_links=int(len(kept)),
        num_candidate_links=L, num_skipped_links=len(skipped),
        base_nodes="node_registry.csv column base = 1 (30-station base network, listed first)",
        build_config=dataclasses.asdict(cfg), rate_model_params=dataclasses.asdict(params),
        finite_key_params=fk.as_dict(), sat_alt_km=sat_alt, stations_file=str(args.stations),
        stations_md5=md5(rel(args.stations)), weather_fill=fill, inputs_md5=inputs,
        allow_era5_fallback=bool(args.allow_era5_fallback), per_type=summary,
        rate_model_md5=md5(ROOT / "tools/channel_v2/rate_model_v2.py"),
        finite_key_md5=md5(ROOT / "tools/dataset_v2/finite_key.py"),
    )
    with open(out / "build_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)
    print(json.dumps(summary, indent=1), flush=True)

    if not args.no_stats:
        stride = 60 if T >= 60 * 100 else 1
        subprocess.run([sys.executable, str(ROOT / "scripts/estimate_rate_stats.py"), "--dataset-dir", str(out),
                        "--stride", str(stride)], cwd=ROOT, check=True)
    print(f"done in {time.time() - t_start:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
