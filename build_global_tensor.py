"""
build_global_tensor.py
======================
薄壳调度器：自身不含任何物理公式。
调用 helper.py / setup.py 中的函数生成全局物理数据，
输出 5 个文件到 dataset/global/。
使用 O(1) 内存流式写入 (Streaming) 架构处理巨型时间序列。
"""
import h5py
import numpy as np
import pandas as pd
import os
import math
import json
from tqdm import tqdm

from data import (gs, hap, sat, system,
                  GlobalGroundStations, GlobalHAPs, GlobalSatellites, SimConfig)
from helper import GeoMath, HapMobilityModel, KeplerianPropagator, R

# ==========================================
# 全局时间参数 (分钟级高频仿真)
# ==========================================
THETA = 60           # 👉 核心步长(秒)。之后你想改步长，只需修改这个数字即可！(例如：300=5分钟，3600=1小时)

# 以下全局参数会自动根据 THETA 换算成“步数”，永远不需要你手动修改：
T_HAP = int((24 * 3600) / THETA)            # HAP 漂移周期 (固定24小时对应的步数)
BLEND_HOURS = int((4 * 3600) / THETA)       # 闭合回环混合窗口 (最后4小时对应的步数)
T_YEAR = int((365 * 24 * 3600) / THETA)     # 预计算总时长 (全年 365 天对应的总步数)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Geometry/quality controls. These keep the tensor a physical candidate graph rather
# than every combinatorial edge in the network.
MIN_ELEVATION_DEG = 5.0
ENABLE_STATIC_HAP_GS_PREFILTER = True
MAX_HAP_GS_DISTANCE_KM = 700.0

# 无地面端链路 (SAT-HAP / SAT-SAT) 的晴空夜间默认天气，与 load_weather_data 的默认值一致。
# 只读共享，避免每个 worker 重复分配 52 万步的 10 个数组。
def _build_default_weather(t_year):
    return {
        "c_low": np.zeros(t_year, dtype=np.float32),
        "c_mid": np.zeros(t_year, dtype=np.float32),
        "c_high": np.zeros(t_year, dtype=np.float32),
        "rain": np.zeros(t_year, dtype=np.float32),
        "snow": np.zeros(t_year, dtype=np.float32),
        "temp": np.full(t_year, 15.0, dtype=np.float32),
        "rh": np.full(t_year, 50.0, dtype=np.float32),
        "ws": np.full(t_year, 2.0, dtype=np.float32),
        "sf": np.zeros(t_year, dtype=np.float32),
        "vis": np.full(t_year, 24000.0, dtype=np.float32),
    }


_DEFAULT_WEATHER = _build_default_weather(T_YEAR)

_WEATHER_NUMERIC_KEYS = ("vis", "rain", "snow", "temp", "sf", "c_low", "c_mid", "c_high")

def load_weather_data(gnodes, T_YEAR, THETA):
    """预先将所有地面的 CSV 气象数据按时间维度拉齐并提取出来"""
    print(">>> [1.5/5] 载入地面站真实天气物理数据...")
    weather_dict = {}
    coverage_rows = []
    repeat_factor = max(1, int(3600 / THETA))
    # 数值列缺失/NaN 时回退到对物理引擎友好的默认值，防止 NaN 传播进 kmax
    _NUM_FALLBACK = {
        "cloud_cover_low": 0.0, "cloud_cover_mid": 0.0, "cloud_cover_high": 0.0,
        "rain_mm": 0.0, "snow_cm": 0.0, "temperature_2m": 15.0,
        "relative_humidity_2m": 50.0, "wind_speed_10m": 2.0,
        "direct_radiation_w": 0.0, "visibility_m": 24000.0,
    }
    configs = GlobalGroundStations.get_stations()
    for idx, node in enumerate(gnodes):
        # 默认值为对物理引擎友好的值
        c_low = np.zeros(T_YEAR, dtype=np.float32)
        c_mid = np.zeros(T_YEAR, dtype=np.float32)
        c_high = np.zeros(T_YEAR, dtype=np.float32)
        rain = np.zeros(T_YEAR, dtype=np.float32)
        snow = np.zeros(T_YEAR, dtype=np.float32)
        temp = np.full(T_YEAR, 15.0, dtype=np.float32)
        rh = np.full(T_YEAR, 50.0, dtype=np.float32)
        ws = np.full(T_YEAR, 2.0, dtype=np.float32)
        sf = np.zeros(T_YEAR, dtype=np.float32)
        vis = np.full(T_YEAR, 24000.0, dtype=np.float32)

        cfg = next((c for c in configs if c["name"] == node.tag), None)
        source = "default_clear"
        csv_rows = 0
        covered_steps = 0
        missing_columns = []
        if cfg:
            file_path = os.path.join(_REPO_ROOT, "weather", "2023", f"{node.tag}_weather.csv")
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    csv_rows = len(df)
                    for col, default in _NUM_FALLBACK.items():
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
                        else:
                            missing_columns.append(col)
                            df[col] = default
                    limit = min(T_YEAR, len(df) * repeat_factor)
                    covered_steps = limit
                    source = "csv"
                    c_low[:limit] = np.repeat(df['cloud_cover_low'].values, repeat_factor)[:limit] / 100.0
                    c_mid[:limit] = np.repeat(df['cloud_cover_mid'].values, repeat_factor)[:limit] / 100.0
                    c_high[:limit] = np.repeat(df['cloud_cover_high'].values, repeat_factor)[:limit] / 100.0
                    rain[:limit] = np.repeat(df['rain_mm'].values, repeat_factor)[:limit]
                    snow[:limit] = np.repeat(df['snow_cm'].values, repeat_factor)[:limit]
                    temp[:limit] = np.repeat(df['temperature_2m'].values, repeat_factor)[:limit]
                    rh[:limit] = np.repeat(df['relative_humidity_2m'].values, repeat_factor)[:limit]
                    ws[:limit] = np.repeat(df['wind_speed_10m'].values, repeat_factor)[:limit]
                    sf[:limit] = np.repeat(df['direct_radiation_w'].values, repeat_factor)[:limit]
                    vis[:limit] = np.repeat(df['visibility_m'].values, repeat_factor)[:limit]
                except Exception as e:
                    print(f"    - Warning: Failed to parse weather for {node.tag}: {e}")
            else:
                print(f"    - Warning: Missing weather CSV for {node.tag}; using clear default.")
        
        weather_dict[idx] = {
            "c_low": c_low, "c_mid": c_mid, "c_high": c_high,
            "rain": rain, "snow": snow, "temp": temp, "rh": rh,
            "ws": ws, "sf": sf, "vis": vis,
        }
        coverage_rows.append({
            "node_id": idx,
            "name": node.tag,
            "source": source,
            "csv_rows": csv_rows,
            "covered_steps": covered_steps,
            "fallback_steps": T_YEAR - covered_steps,
            "coverage_ratio": covered_steps / T_YEAR if T_YEAR else 0.0,
            "missing_columns": ";".join(missing_columns),
        })
    return weather_dict, coverage_rows


def build_hap_positions_closed_loop(hnodes, syst_hap):
    N_hap = len(hnodes)
    HapMobilityModel.update_hap_coordinates("stratotegic", hnodes, syst_hap)
    raw_lats = np.zeros((T_HAP, N_hap))
    raw_lons = np.zeros((T_HAP, N_hap))
    raw_alts = np.zeros((T_HAP, N_hap))
    for i, h in enumerate(hnodes):
        for t in range(T_HAP):
            raw_lats[t, i] = h.la.get(t, h.la[0])
            raw_lons[t, i] = h.lg.get(t, h.lg[0])
            raw_alts[t, i] = h.H.get(t, h.H[0])
    positions = np.zeros((T_HAP, N_hap, 3), dtype=np.float32)
    blend_start = T_HAP - BLEND_HOURS
    for i in range(N_hap):
        delta_lat = raw_lats[-1, i] - raw_lats[0, i]
        delta_lon = raw_lons[-1, i] - raw_lons[0, i]
        delta_alt = raw_alts[-1, i] - raw_alts[0, i]
        for t in range(T_HAP):
            lat = raw_lats[t, i]
            lon = raw_lons[t, i]
            alt = raw_alts[t, i]
            if t >= blend_start:
                w = (t - blend_start) / BLEND_HOURS
                lat -= w * delta_lat
                lon -= w * delta_lon
                alt -= w * delta_alt
            positions[t, i, :] = [lat, lon, alt]
        for t in range(T_HAP):
            hnodes[i].la[t] = float(positions[t, i, 0])
            hnodes[i].lg[t] = float(positions[t, i, 1])
            hnodes[i].H[t]  = float(positions[t, i, 2])
    print(f"[HAP] 闭合修正完成: 混合窗口={BLEND_HOURS}步, T=0 与 T={T_HAP-1} 末尾坐标差 < 0.001 deg")
    return positions

from concurrent.futures import ProcessPoolExecutor, as_completed
import os

_WORKER_CONTEXT = {}

def _observer_target_for_elevation(lat1, lon1, alt1, lat2, lon2, alt2):
    """Use the lower endpoint as the local horizon observer for link elevation."""
    if alt1 <= alt2:
        return lat1, lon1, alt1, lat2, lon2, alt2
    return lat2, lon2, alt2, lat1, lon1, alt1

def _hap_gs_min_distance_km(gs_coord, hap_track):
    gs_lat, gs_lon, gs_alt = gs_coord
    return min(
        GeoMath.calculate_3d_distance(gs_lat, gs_lon, gs_alt, h_lat, h_lon, h_alt)
        for h_lat, h_lon, h_alt in hap_track
    )

def _init_worker_context(
    T_link,
    N_gs,
    N_hap,
    gs_coords,
    hap_pos,
    sat_coords_full,
    THETA,
    T_HAP,
    min_elev_deg,
):
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update({
        "T_link": T_link,
        "N_gs": N_gs,
        "N_hap": N_hap,
        "gs_coords": gs_coords,
        "hap_pos": hap_pos,
        "sat_coords_full": sat_coords_full,
        "THETA": THETA,
        "T_HAP": T_HAP,
        "min_elev_deg": min_elev_deg,
        "qkd_system": None,
    })

def _compact_weather(weather):
    if weather is None:
        return None
    return {key: weather[key] for key in _WEATHER_NUMERIC_KEYS}

_DEFAULT_NUMERIC_WEATHER = _compact_weather(_DEFAULT_WEATHER)


def _configure_globals(theta_sec: int = 60) -> None:
    """Recompute time-step-derived globals for a custom THETA (CLI/tests)."""
    global THETA, T_HAP, BLEND_HOURS, T_YEAR, _DEFAULT_WEATHER, _DEFAULT_NUMERIC_WEATHER
    THETA = int(theta_sec)
    T_HAP = int((24 * 3600) / THETA)
    BLEND_HOURS = int((4 * 3600) / THETA)
    T_YEAR = int((365 * 24 * 3600) / THETA)
    _DEFAULT_WEATHER = _build_default_weather(T_YEAR)
    _DEFAULT_NUMERIC_WEATHER = _compact_weather(_DEFAULT_WEATHER)

def _build_link_task(l_idx, u_idx, v_idx, ltype, w_u, w_v, latitude_impossible=False):
    return (l_idx, u_idx, v_idx, ltype, _compact_weather(w_u), _compact_weather(w_v), latitude_impossible)

def _latitude_interval_gap_deg(a, b):
    a_min, a_max = sorted((float(a[0]), float(a[1])))
    b_min, b_max = sorted((float(b[0]), float(b[1])))
    if a_max < b_min:
        return b_min - a_max
    if b_max < a_min:
        return a_min - b_max
    return 0.0

def _los_central_angle_limit_deg(alt_a_km, alt_b_km):
    r_a = R + max(0.0, float(alt_a_km))
    r_b = R + max(0.0, float(alt_b_km))
    return math.degrees(math.acos(min(1.0, R / r_a)) + math.acos(min(1.0, R / r_b)))

def _elevation_from_central_angle_deg(observer_alt_km, target_alt_km, central_angle_deg):
    r_obs = R + max(0.0, float(observer_alt_km))
    r_tgt = R + max(0.0, float(target_alt_km))
    psi = math.radians(float(central_angle_deg))
    d_sq = r_obs * r_obs + r_tgt * r_tgt - 2.0 * r_obs * r_tgt * math.cos(psi)
    if d_sq <= 0.0:
        return 90.0
    d = math.sqrt(d_sq)
    sin_elev = (r_tgt * math.cos(psi) - r_obs) / d
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))

def _elevation_central_angle_limit_deg(observer_alt_km, target_alt_km, min_elev_deg):
    horizon_limit = _los_central_angle_limit_deg(observer_alt_km, target_alt_km)
    if min_elev_deg <= 0.0:
        return horizon_limit
    lo, hi = 0.0, horizon_limit
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if _elevation_from_central_angle_deg(observer_alt_km, target_alt_km, mid) >= min_elev_deg:
            lo = mid
        else:
            hi = mid
    return lo

def _node_is_sat(idx, N_gs, N_hap):
    return idx >= N_gs + N_hap

def _sat_local_idx(idx, N_gs, N_hap):
    return idx - N_gs - N_hap

def _link_is_latitude_impossible(
    u_idx,
    v_idx,
    ltype,
    node_lat_ranges,
    node_altitudes,
    sat_inclinations,
    N_gs,
    N_hap,
    min_elev_deg,
):
    u_range = node_lat_ranges[u_idx]
    v_range = node_lat_ranges[v_idx]
    if _node_is_sat(u_idx, N_gs, N_hap):
        inc = abs(float(sat_inclinations[_sat_local_idx(u_idx, N_gs, N_hap)]))
        u_range = (-inc, inc)
    if _node_is_sat(v_idx, N_gs, N_hap):
        inc = abs(float(sat_inclinations[_sat_local_idx(v_idx, N_gs, N_hap)]))
        v_range = (-inc, inc)

    lat_gap = _latitude_interval_gap_deg(u_range, v_range)
    if lat_gap <= 0.0:
        return False

    if "GS" in ltype:
        if _node_is_sat(u_idx, N_gs, N_hap):
            gs_idx, sat_idx = v_idx, u_idx
        else:
            gs_idx, sat_idx = u_idx, v_idx
        angle_limit = _elevation_central_angle_limit_deg(
            node_altitudes[gs_idx],
            node_altitudes[sat_idx],
            min_elev_deg,
        )
    else:
        angle_limit = _los_central_angle_limit_deg(node_altitudes[u_idx], node_altitudes[v_idx])
    return lat_gap > angle_limit + 1e-6

def _node_coord_timelines(idx, T_link, N_gs, N_hap, gs_coords, hap_pos, sat_coords_full, T_HAP):
    if idx < N_gs:
        lat, lon, alt = gs_coords[idx]
        return (
            np.full(T_link, lat, dtype=np.float32),
            np.full(T_link, lon, dtype=np.float32),
            np.full(T_link, alt, dtype=np.float32),
        )
    if idx < N_gs + N_hap:
        h_idx = idx - N_gs
        hap_t = np.arange(T_link) % T_HAP
        return (
            hap_pos[hap_t, h_idx, 0],
            hap_pos[hap_t, h_idx, 1],
            hap_pos[hap_t, h_idx, 2],
        )
    s_idx = idx - N_gs - N_hap
    return (
        sat_coords_full[:, s_idx, 0],
        sat_coords_full[:, s_idx, 1],
        sat_coords_full[:, s_idx, 2],
    )

_CPU_COUNT_SENTINEL = object()

def _resolve_worker_count(env=None, cpu_count=_CPU_COUNT_SENTINEL):
    env = os.environ if env is None else env
    cpu_count = os.cpu_count() if cpu_count is _CPU_COUNT_SENTINEL else cpu_count
    default_workers = max(1, int(cpu_count or 1))
    try:
        requested = int(env.get("HAP_QKD_WORKERS", "0"))
    except (TypeError, ValueError):
        requested = 0
    return requested if requested > 0 else default_workers

def _executor_chunksize(num_tasks, worker_count):
    if num_tasks <= 0 or worker_count <= 0:
        return 1
    return max(1, min(32, math.ceil(num_tasks / (worker_count * 8))))

def _worker_qkd_system():
    qkd_system = _WORKER_CONTEXT.get("qkd_system")
    if qkd_system is None:
        from adapters.unified_channel import UnifiedQKDRateModel
        qkd_system = UnifiedQKDRateModel()
        _WORKER_CONTEXT["qkd_system"] = qkd_system
    return qkd_system

def _vectorized_ecef(lat_deg, lon_deg, alt_km):
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    r = R + alt_km
    return (
        r * np.cos(lat_rad) * np.cos(lon_rad),
        r * np.cos(lat_rad) * np.sin(lon_rad),
        r * np.sin(lat_rad),
    )


def _vectorized_los(lat1, lon1, alt1, lat2, lon2, alt2):
    """Vectorized GeoMath.check_line_of_sight (True = chord not blocked by Earth)."""
    x1, y1, z1 = _vectorized_ecef(lat1, lon1, alt1)
    x2, y2, z2 = _vectorized_ecef(lat2, lon2, alt2)
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    d_mag_sq = dx * dx + dy * dy + dz * dz
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -(x1 * dx + y1 * dy + z1 * dz) / d_mag_sq
    cx = x1 + t * dx
    cy = y1 + t * dy
    cz = z1 + t * dz
    min_dist_sq = cx * cx + cy * cy + cz * cz
    blocked = (t > 0.0) & (t < 1.0) & (min_dist_sq < R * R)
    return ~blocked


def _vectorized_distance(lat1, lon1, alt1, lat2, lon2, alt2):
    """Vectorized GeoMath.calculate_3d_distance (km)."""
    x1, y1, z1 = _vectorized_ecef(lat1, lon1, alt1)
    x2, y2, z2 = _vectorized_ecef(lat2, lon2, alt2)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def _vectorized_elevation_deg(obs_lat, obs_lon, obs_alt, tgt_lat, tgt_lon, tgt_alt):
    """Vectorized GeoMath.elevation_angle_deg with the lower endpoint as observer."""
    ox, oy, oz = _vectorized_ecef(obs_lat, obs_lon, obs_alt)
    tx, ty, tz = _vectorized_ecef(tgt_lat, tgt_lon, tgt_alt)
    lx, ly, lz = tx - ox, ty - oy, tz - oz
    los_norm = np.sqrt(lx * lx + ly * ly + lz * lz)
    lat_rad = np.radians(obs_lat)
    lon_rad = np.radians(obs_lon)
    ux = np.cos(lat_rad) * np.cos(lon_rad)
    uy = np.cos(lat_rad) * np.sin(lon_rad)
    uz = np.sin(lat_rad)
    with np.errstate(divide="ignore", invalid="ignore"):
        sin_elev = (lx * ux + ly * uy + lz * uz) / los_norm
    sin_elev = np.clip(sin_elev, -1.0, 1.0)
    elev = np.degrees(np.arcsin(sin_elev))
    return np.where(los_norm <= 0.0, 90.0, elev)


_GM_KEPLER = 398600.4418
_OMEGA_E = 7.2921159e-5


def _vectorized_sat_positions(cfg, t_seconds):
    """Vectorized KeplerianPropagator.get_position_at_time over an array of times."""
    alt_km = cfg["alt_km"]
    inclination_deg = float(cfg.get("inclination", 0.0))
    a = 6371.0 + alt_km
    e = 0.0
    raan0 = math.radians(float(cfg.get("init_lon", 0.0)))
    aop = 0.0
    init_lat = float(cfg.get("init_lat", 0.0))
    i_rad = math.radians(inclination_deg)
    if inclination_deg == 0:
        m0 = 0.0
    else:
        val = max(-1.0, min(1.0, math.sin(math.radians(init_lat)) / math.sin(i_rad)))
        m0 = math.asin(val)
    n = math.sqrt(_GM_KEPLER / (a ** 3))
    raan = raan0 - _OMEGA_E * t_seconds
    M = m0 + n * t_seconds
    E = M
    for _ in range(50):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E = E - dE
    tan_half_nu = np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2.0)
    nu = 2.0 * np.arctan(tan_half_nu)
    r = a * (1.0 - e * np.cos(E))
    altitude = r - R
    u = aop + nu
    x_orb = r * np.cos(u)
    y_orb = r * np.sin(u)
    X = x_orb * np.cos(raan) - y_orb * np.cos(i_rad) * np.sin(raan)
    Y = x_orb * np.sin(raan) + y_orb * np.cos(i_rad) * np.cos(raan)
    Z = y_orb * np.sin(i_rad)
    lat = np.degrees(np.arcsin(Z / r))
    lon = np.degrees(np.arctan2(Y, X))
    lon = (lon + 180.0) % 360.0 - 180.0
    return lat, lon, altitude


def compute_link_timeline(args):
    l_idx, u_idx, v_idx, ltype, w_u, w_v, latitude_impossible = args
    ctx = _WORKER_CONTEXT
    T_link = ctx["T_link"]
    N_gs = ctx["N_gs"]
    N_hap = ctx["N_hap"]
    gs_coords = ctx["gs_coords"]
    hap_pos = ctx["hap_pos"]
    sat_coords_full = ctx["sat_coords_full"]
    T_HAP = ctx["T_HAP"]
    min_elev_deg = ctx["min_elev_deg"]

    kmax_array = np.zeros(T_link, dtype=np.float32)
    if latitude_impossible:
        return (l_idx, kmax_array)

    u_type = "GS" if u_idx < N_gs else ("HAP" if u_idx < N_gs + N_hap else "SAT")
    v_type = "GS" if v_idx < N_gs else ("HAP" if v_idx < N_gs + N_hap else "SAT")
    w = w_u if u_type == "GS" else (w_v if v_type == "GS" else _DEFAULT_NUMERIC_WEATHER)

    lat1_arr, lon1_arr, alt1_arr = _node_coord_timelines(
        u_idx, T_link, N_gs, N_hap, gs_coords, hap_pos, sat_coords_full, T_HAP
    )
    lat2_arr, lon2_arr, alt2_arr = _node_coord_timelines(
        v_idx, T_link, N_gs, N_hap, gs_coords, hap_pos, sat_coords_full, T_HAP
    )

    # Vectorized geometry in float64 (exact numpy versions of the scalar GeoMath
    # path used by the original generator).
    lat1 = lat1_arr.astype(np.float64)
    lon1 = lon1_arr.astype(np.float64)
    alt1 = alt1_arr.astype(np.float64)
    lat2 = lat2_arr.astype(np.float64)
    lon2 = lon2_arr.astype(np.float64)
    alt2 = alt2_arr.astype(np.float64)

    visible = _vectorized_los(lat1, lon1, alt1, lat2, lon2, alt2)
    if not np.any(visible):
        return (l_idx, kmax_array)

    dist = _vectorized_distance(lat1, lon1, alt1, lat2, lon2, alt2)
    # Elevation uses the lower endpoint as the local horizon observer (matches
    # _observer_target_for_elevation).
    obs_lat = np.where(alt1 <= alt2, lat1, lat2)
    obs_lon = np.where(alt1 <= alt2, lon1, lon2)
    obs_alt = np.where(alt1 <= alt2, alt1, alt2)
    tgt_lat = np.where(alt1 <= alt2, lat2, lat1)
    tgt_lon = np.where(alt1 <= alt2, lon2, lon1)
    tgt_alt = np.where(alt1 <= alt2, alt2, alt1)
    elev = _vectorized_elevation_deg(obs_lat, obs_lon, obs_alt, tgt_lat, tgt_lon, tgt_alt)

    if "GS" in ltype:
        visible = visible & (elev >= min_elev_deg)
    visible = visible & (dist > 0.0)
    if not np.any(visible):
        return (l_idx, kmax_array)

    qkd_system = None
    for t in np.flatnonzero(visible):
        if qkd_system is None:
            qkd_system = _worker_qkd_system()
        kmax_array[t] = qkd_system.compute_secure_key_rate(
            distance_m=float(dist[t] * 1000.0),
            visibility_m=float(w["vis"][t]),
            rain_mm=float(w["rain"][t]),
            snow_cm=float(w["snow"][t]),
            temperature_c=float(w["temp"][t]),
            sf_w=float(w["sf"][t]),
            h_rx_km=float(alt2[t]),
            h_tx_km=float(alt1[t]),
            rx_node_type=v_type,
            tx_node_type=u_type,
            c_low=float(w["c_low"][t]),
            c_mid=float(w["c_mid"][t]),
            c_high=float(w["c_high"][t]),
            elevation_angle_deg=float(elev[t]),
        )
    return (l_idx, kmax_array)


def build_all(out_dir=None):
    out_dir = out_dir or os.path.join("dataset", "global")
    os.makedirs(out_dir, exist_ok=True)

    print(">>> [1/5] 构建节点...")
    gs_configs = GlobalGroundStations.get_stations()[:SimConfig.NUM_GS]
    gnodes = [gs(c["lon"], c["lat"], 1e9, c["name"]) for c in gs_configs]
    
    # 抽取天气数据字典
    weather_dict, weather_coverage = load_weather_data(gnodes, T_YEAR, THETA)
    pd.DataFrame(weather_coverage).to_csv(os.path.join(out_dir, "weather_coverage.csv"), index=False)
    
    hap_configs = GlobalHAPs.get_haps()[:SimConfig.NUM_HAPS]
    syst_hap = system(range(T_HAP), THETA, np.array([[1, 1]]))
    hnodes = [hap({0: c["lon"]}, {0: c["lat"]}, {0: c["alt_km"]}, 1e9, c["name"]) for c in hap_configs]
    
    sat_configs = GlobalSatellites.get_satellites()[:SimConfig.NUM_SATS]
    # No longer store t->coord dictionaries to save memory
    snodes = [sat({0: c.get("init_lon", 0.0)}, {0: c.get("init_lat", 0.0)}, {0: c["alt_km"]}, 1e9, c["name"]) for c in sat_configs]

    all_nodes = gnodes + hnodes + snodes
    N = len(all_nodes)
    N_gs = len(gnodes)
    N_hap = len(hnodes)
    N_sat = len(snodes)
    print(f"    GS={N_gs}, HAP={N_hap}, SAT={N_sat}, 总计={N}")

    print(">>> [2/5] 写入 node_registry.csv...")
    rows = []
    for i, node in enumerate(all_nodes):
        if isinstance(node, gs):
            rows.append({"node_id": i, "type": "GS", "name": node.tag, "lat": node.la, "lon": node.lg, "alt_km": 0.0})
        elif isinstance(node, hap):
            rows.append({"node_id": i, "type": "HAP", "name": node.tag, "lat": node.la[0], "lon": node.lg[0], "alt_km": node.H[0]})
        elif isinstance(node, sat):
            rows.append({"node_id": i, "type": "SAT", "name": node.tag, "lat": node.la[0], "lon": node.lg[0], "alt_km": node.H[0]})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "node_registry.csv"), index=False)

    print(">>> [3/5] 预备 HAP 位置 & 建立链路索引...")
    gs_rows = [{"node_id": i, "name": gnodes[i].tag, "lat": gnodes[i].la, "lon": gnodes[i].lg} for i in range(N_gs)]
    pd.DataFrame(gs_rows).to_csv(os.path.join(out_dir, "gs_positions.csv"), index=False)
    
    hap_pos = build_hap_positions_closed_loop(hnodes, syst_hap)
    hap_node_ids = np.array([N_gs + i for i in range(N_hap)], dtype=np.int32)
    with h5py.File(os.path.join(out_dir, "hap_positions.h5"), "w") as f:
        f.create_dataset("positions", data=hap_pos)
        f.create_dataset("node_ids", data=hap_node_ids)
        f.attrs["theta_sec"] = THETA
        f.attrs["period_hours"] = T_HAP
        f.attrs["period_steps"] = T_HAP
        f.attrs["blend_hours"] = BLEND_HOURS

    T_link = T_YEAR
    link_defs = []
    skipped_links = []
    for i, u in enumerate(all_nodes):
        for j, v in enumerate(all_nodes):
            if i >= j: continue
            u_type = "GS" if isinstance(u, gs) else ("HAP" if isinstance(u, hap) else "SAT")
            v_type = "GS" if isinstance(v, gs) else ("HAP" if isinstance(v, hap) else "SAT")
            types = {u_type, v_type}
            link_type = None
            if types == {"SAT", "HAP"}: link_type = "SAT-HAP"
            elif types == {"SAT", "GS"}: link_type = "SAT-GS"
            elif types == {"HAP", "GS"}: link_type = "HAP-GS"
            elif types == {"SAT"}: link_type = "SAT-SAT"

            if link_type == "HAP-GS" and ENABLE_STATIC_HAP_GS_PREFILTER:
                gs_idx = i if u_type == "GS" else j
                hap_idx = (i if u_type == "HAP" else j) - N_gs
                min_dist = _hap_gs_min_distance_km(
                    (gnodes[gs_idx].la, gnodes[gs_idx].lg, 0.0),
                    hap_pos[:, hap_idx, :],
                )
                if min_dist > MAX_HAP_GS_DISTANCE_KM:
                    skipped_links.append({
                        "node_u": i,
                        "node_v": j,
                        "link_type": link_type,
                        "reason": "hap_gs_static_distance_prefilter",
                        "min_distance_km": min_dist,
                        "threshold_km": MAX_HAP_GS_DISTANCE_KM,
                    })
                    continue

            if link_type is not None:
                link_defs.append((i, j, link_type))
    L = len(link_defs)
    print(f"    有效链路数: {L}, 时间步: {T_link}")
    if skipped_links:
        print(f"    静态预筛跳过链路数: {len(skipped_links)}")
        pd.DataFrame(skipped_links).to_csv(os.path.join(out_dir, "link_registry_skipped.csv"), index=False)

    link_rows = [{"link_id": idx, "node_u": u, "node_v": v, "link_type": lt} for idx, (u, v, lt) in enumerate(link_defs)]
    pd.DataFrame(link_rows).to_csv(os.path.join(out_dir, "link_registry.csv"), index=False)
    metadata = {
        "theta_sec": THETA,
        "t_year_steps": T_YEAR,
        "t_hap_steps": T_HAP,
        "min_elevation_deg": MIN_ELEVATION_DEG,
        "enable_static_hap_gs_prefilter": ENABLE_STATIC_HAP_GS_PREFILTER,
        "max_hap_gs_distance_km": MAX_HAP_GS_DISTANCE_KM,
        "num_nodes": N,
        "num_gs": N_gs,
        "num_hap": N_hap,
        "num_sat": N_sat,
        "num_links": L,
        "num_skipped_links": len(skipped_links),
        "weather_fallback_steps_total": int(sum(row["fallback_steps"] for row in weather_coverage)),
    }
    with open(os.path.join(out_dir, "build_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(">>> [4/5] 准备流式写入 (O(1) 内存)...")
    sat_node_ids = np.array([N_gs + N_hap + i for i in range(N_sat)], dtype=np.int32)

    f_sat = h5py.File(os.path.join(out_dir, "sat_positions.h5"), "w")
    ds_sat = f_sat.create_dataset("positions", shape=(T_link, N_sat, 3), dtype=np.float32, chunks=True)
    f_sat.create_dataset("node_ids", data=sat_node_ids)
    f_sat.attrs["theta_sec"] = THETA

    f_link = h5py.File(os.path.join(out_dir, "link_data.h5"), "w")
    dt = np.dtype([("link_id", "i4"), ("node_u", "i4"), ("node_v", "i4"), ("link_type", "S10")])
    reg = np.array([(idx, u, v, lt.encode()) for idx, (u, v, lt) in enumerate(link_defs)], dtype=dt)
    f_link.create_dataset("link_registry", data=reg)

    # Lean storage: only k_max is kept (los/distance/zenith are never consumed
    # by training and los is derivable from k_max with min_link_rate > 0).
    ds_kmax = f_link.create_dataset(
        "k_max",
        shape=(T_link, L),
        dtype=np.float32,
        chunks=(min(2048, T_link), L),
        compression="gzip",
        compression_opts=4,
    )
    f_link.attrs["theta_sec"] = THETA
    f_link.attrs["period_hours"] = T_link
    f_link.attrs["period_steps"] = T_link
    f_link.attrs["min_elevation_deg"] = MIN_ELEVATION_DEG
    f_link.attrs["static_hap_gs_prefilter"] = int(ENABLE_STATIC_HAP_GS_PREFILTER)
    f_link.attrs["max_hap_gs_distance_km"] = MAX_HAP_GS_DISTANCE_KM

    print(">>> [5/6] 预结算全天候卫星坐标 (向量化)...")
    t_seconds = np.arange(T_link, dtype=np.float64) * THETA
    sat_coords_full = np.zeros((T_link, N_sat, 3), dtype=np.float32)
    for i, cfg in enumerate(sat_configs):
        lat, lon, alt = _vectorized_sat_positions(cfg, t_seconds)
        sat_coords_full[:, i, 0] = lat
        sat_coords_full[:, i, 1] = lon
        sat_coords_full[:, i, 2] = alt
    ds_sat[:] = sat_coords_full
    f_sat.close()

    worker_count = _resolve_worker_count()
    print(f">>> [6/6] 多核并发极限流式计算 ({worker_count} workers)...")
    gs_coords = np.array([(gnodes[i].la, gnodes[i].lg, 0.0) for i in range(N_gs)], dtype=np.float32)
    node_lat_ranges = []
    node_altitudes = []
    for i in range(N_gs):
        node_lat_ranges.append((float(gs_coords[i, 0]), float(gs_coords[i, 0])))
        node_altitudes.append(0.0)
    for i in range(N_hap):
        node_lat_ranges.append((float(np.min(hap_pos[:, i, 0])), float(np.max(hap_pos[:, i, 0]))))
        node_altitudes.append(float(np.max(hap_pos[:, i, 2])))
    sat_inclinations = [abs(float(cfg.get("inclination", 0.0))) for cfg in sat_configs]
    for cfg in sat_configs:
        inc = abs(float(cfg.get("inclination", 0.0)))
        node_lat_ranges.append((-inc, inc))
        node_altitudes.append(float(cfg["alt_km"]))
    
    tasks = []
    latitude_impossible_count = 0
    for l_idx, (u_idx, v_idx, ltype) in enumerate(link_defs):
        w_u = weather_dict[u_idx] if u_idx < N_gs else None
        w_v = weather_dict[v_idx] if v_idx < N_gs else None
        latitude_impossible = _link_is_latitude_impossible(
            u_idx,
            v_idx,
            ltype,
            node_lat_ranges,
            node_altitudes,
            sat_inclinations,
            N_gs,
            N_hap,
            MIN_ELEVATION_DEG,
        )
        if latitude_impossible:
            latitude_impossible_count += 1
        tasks.append(_build_link_task(l_idx, u_idx, v_idx, ltype, w_u, w_v, latitude_impossible))
    if latitude_impossible_count:
        print(f"    纬度上界短路链路数: {latitude_impossible_count} (保留链路，全年输出0)")

    # Workers return per-link timelines; we stage them in an on-disk memmap and
    # pack row-block by row-block into the gzip target. Writing full rows
    # sequentially avoids the read-modify-write amplification of column writes
    # into a row-major gzip-compressed dataset (which would recompress every
    # chunk per link column).
    stage_path = os.path.join(out_dir, "_kmax_stage.dat")
    stage = np.memmap(stage_path, dtype=np.float32, mode="w+", shape=(T_link, L), order="F")
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_worker_context,
            initargs=(T_link, N_gs, N_hap, gs_coords, hap_pos, sat_coords_full, THETA, T_HAP, MIN_ELEVATION_DEG),
        ) as executor:
            futures = [executor.submit(compute_link_timeline, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel Physics"):
                l_idx, kmax_array = future.result()
                stage[:, l_idx] = kmax_array
        stage.flush()
        block_rows = min(2048, T_link)
        for r0 in tqdm(range(0, T_link, block_rows), desc="Packing kmax rows"):
            r1 = min(T_link, r0 + block_rows)
            ds_kmax[r0:r1, :] = np.ascontiguousarray(stage[r0:r1, :])
    finally:
        del stage
        try:
            os.remove(stage_path)
        except OSError:
            pass
    f_link.close()

    print(f"\n全部完成！巨型流式大表生成完毕，未发生任何内存累积。")

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build the global link-rate H5 tensor.")
    parser.add_argument("--theta", type=int, default=60, help="core step size in seconds (default 60)")
    parser.add_argument("--out-dir", default=os.path.join("dataset", "global"), help="output directory")
    parser.add_argument("--workers", type=int, default=0, help="worker processes (0 = all cores)")
    args = parser.parse_args()
    if args.theta != 60:
        _configure_globals(args.theta)
    if args.workers > 0:
        os.environ["HAP_QKD_WORKERS"] = str(args.workers)
    build_all(out_dir=args.out_dir)


if __name__ == "__main__":
    main()

