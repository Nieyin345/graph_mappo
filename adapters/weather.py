from __future__ import annotations

"""Weather layer — per link type, each aligned to its own source paper.

Two weather models coexist because the two source papers model weather differently:

  * **HAP-GS (StratoQ)** — the analytical **Kim** fog/rain/snow model below (this file);
  * **SAT-GS (TNSM)** — a **libRadtran transmittance table** (cloud cover x elevation, 785 nm),
    handled inside ``TNSMSatelliteCapacityProvider`` (weather is the ``cc`` axis of its channel
    table, not a multiplier here). ``WeatherModel.cloud_cover_at`` feeds it the cloud percent;
  * **SAT-HAP** — immune (both endpoints sit above the weather layer).

``WeatherModel`` therefore exposes the weather in two coupled views: ``condition_at`` (Kim,
for HAP-GS) and ``cloud_cover_at`` (percent, for the TNSM SAT-GS table).

--- StratoQ Kim model (HAP-GS) ---

Same model as StratoQ's `helper.py` (`plob` method; = the Q. Zhang slide). Total link loss (dB):

    L_t = L_geo + L_ma + L_fog·fog[t] + L_snow·snow[t] + L_rain·rain[t]

`L_geo` (geometric) and `L_ma` (molecular/atmospheric absorption) are already inside the
clear-sky capacity produced by the channel providers, so this layer adds only the
**weather** terms on top. Each weather loss is an attenuation rate (dB/km) times the slant
path through the weather layer, `R = H / sin(α)` (α = elevation angle, H = layer thickness):

    Fog (Kim):  L_fog  = 4.343 · (3.91 / V) · (λ/550)^(−U) · (H_C / sin α)   [dB]
                U = 1.6 (V>50), 1.3 (6<V≤50), 0.585·V^(1/3) (V≤6)
    Rain:       L_rain = 1.076 · R_mm^0.67 · (H_W / sin α)                    [Carbonneau, dB]
    Snow:       L_snow = a(λ) · S_mm^b · (H_W / sin α)                        [dry/wet, dB]

    (v2 revision, rate-model-v2.md §4: the 4.343 factor converts Kim's natural-log extinction
    coefficient to dB/km; rain/snow switched from visibility-based to rate-based FSO formulas.
    StratoQ's original 3.91/V-as-dB shorthand underestimated fog by 4.34x.)

(StratoQ has only fog/rain/snow — a cloud is optically dense fog, so it is not a separate
condition; use the fog term with cloud-appropriate visibility.)

V = visibility (km), λ = wavelength (nm). Transmittance = 10^(−L/10).

Elevation makes the ground link types differ **from the geometry, not by hand**:
  - HAP-GS : HAP at a fixed moderate elevation → steady loss (Kim, this file);
  - SAT-GS : handled by the TNSM table, not here (cloud x elevation, 785 nm);
  - SAT-HAP: both endpoints (25 km / 567 km) are **above the weather layer** → the path never
             enters it → R = 0 → **no weather loss (immune)**.

Cloud fields are synthetic (per-GS, deterministic) or supplied as a series (hook for real
cloud/visibility data, cf. TNSM Visual Crossing / libRadtran).
"""

import hashlib
import math
from typing import Dict, List, Optional, Union

# visibility V (km) and weather-layer altitude H (km) per condition — StratoQ helper.py
# values (V_fog=0.5, V_rain_snow=4, H_C=0.5 fog, H_W=5 rain/snow). StratoQ has only
# fog/rain/snow (no cloud) — a cloud is optically dense fog, covered by the same Kim model,
# so it is not a separate condition. Override per-GS/slot via the dict form. FSO snow
# attenuation (58/V) is very large — heavy snow can fully block the link (physically correct).
_VISIBILITY_KM = {"fog": 0.5, "rain": 4.0, "snow": 4.0}
_LAYER_KM = {"fog": 2.0, "rain": 3.0, "snow": 3.0}   # v2: fog=boundary layer, rain/snow=precip layer
_MIN_SIN = math.sin(math.radians(5.0))  # clamp so sin(α) doesn't blow up near the horizon
_DEFAULT_RAIN_RATE_MM_H = 10.0
_DEFAULT_SNOW_RATE_MM_H = 5.0
_CLEAR_VISIBILITY_KM = 50.0

def _gamma_rain_db_per_km(rate_mm_h: float) -> float:
    """Carbonneau & Wisely: gamma = 1.076 R^0.67 dB/km (FSO, wavelength-flat)."""
    return 1.076 * max(0.0, rate_mm_h) ** 0.67


def _gamma_snow_db_per_km(rate_mm_h: float, wavelength_nm: float, dry: bool = True) -> float:
    """Dry/wet snow gamma = a(lambda) S^b dB/km (lambda in nm, S in mm/h)."""
    if dry:
        a, b = 5.42e-5 * wavelength_nm + 5.4958776, 1.38
    else:
        a, b = 1.023e-4 * wavelength_nm + 3.7855466, 0.72
    return a * max(0.0, rate_mm_h) ** b


def _fog_U(V: float) -> float:
    if V > 50:
        return 1.6
    if V > 6:
        return 1.3
    return 0.585 * V ** (1.0 / 3.0)


def residual_weather_loss_db(
    elevation_deg: float,
    wavelength_nm: float = 1550.0,
    visibility_m: float = 24000.0,
    rain_rate_mm_h: float = 0.0,
    snow_rate_mm_h: float = 0.0,
    temperature_c: float = 15.0,
    boundary_layer_km: float = 2.0,
    precip_layer_km: float = 3.0,
) -> float:
    """v2 residual-extinction weather loss for a ground-facing optical link.

    Visibility gives total 550 nm extinction. Rain and snow are computed explicitly;
    the remaining 550 nm extinction is treated as residual fog/haze and converted to
    the operating wavelength with Kim/Kruse's wavelength exponent.
    """

    rain_rate = max(0.0, float(rain_rate_mm_h))
    snow_rate = max(0.0, float(snow_rate_mm_h))
    visibility = float(visibility_m)
    clear_visibility = visibility >= 24000.0

    vis_km = _CLEAR_VISIBILITY_KM if clear_visibility else max(visibility / 1000.0, 0.001)
    gamma_total_550 = 4.343 * 3.91 / vis_km
    gamma_base_550 = 4.343 * 3.91 / _CLEAR_VISIBILITY_KM
    dry_snow = float(temperature_c) <= 0.0

    gamma_rain = _gamma_rain_db_per_km(rain_rate)
    gamma_snow_550 = _gamma_snow_db_per_km(snow_rate, 550.0, dry_snow)
    gamma_snow_lambda = _gamma_snow_db_per_km(snow_rate, wavelength_nm, dry_snow)

    gamma_fog_residual_550 = max(gamma_base_550, gamma_total_550 - gamma_rain - gamma_snow_550)
    visibility_residual_km = 4.343 * 3.91 / max(gamma_fog_residual_550, 1e-12)
    q = _fog_U(visibility_residual_km)
    gamma_fog_lambda = gamma_fog_residual_550 * (wavelength_nm / 550.0) ** (-q)

    sin_elev = max(math.sin(math.radians(elevation_deg)), _MIN_SIN)
    boundary_slant = boundary_layer_km / sin_elev
    precip_slant = precip_layer_km / sin_elev
    return gamma_fog_lambda * boundary_slant + (gamma_rain + gamma_snow_lambda) * precip_slant


def residual_weather_transmittance(
    elevation_deg: float,
    wavelength_nm: float = 1550.0,
    visibility_m: float = 24000.0,
    rain_rate_mm_h: float = 0.0,
    snow_rate_mm_h: float = 0.0,
    temperature_c: float = 15.0,
) -> float:
    """Capacity factor in (0, 1] from ``residual_weather_loss_db``."""

    return 10.0 ** (
        -residual_weather_loss_db(
            elevation_deg,
            wavelength_nm,
            visibility_m,
            rain_rate_mm_h,
            snow_rate_mm_h,
            temperature_c,
        )
        / 10.0
    )


def weather_loss_db(
    condition: str,
    elevation_deg: float,
    wavelength_nm: float = 1550.0,
    V: Optional[float] = None,
    H: Optional[float] = None,
    rain_rate_mm_h: Optional[float] = None,
    snow_rate_mm_h: Optional[float] = None,
    temperature_c: Optional[float] = None,
) -> float:
    """Weather-only loss (dB) for one condition on a ground-facing link at elevation α."""
    if condition in (None, "clear"):
        return 0.0
    V = V if V is not None else _VISIBILITY_KM.get(condition, 10.0)
    H = H if H is not None else _LAYER_KM.get(condition, 1.0)
    R = H / max(math.sin(math.radians(elevation_deg)), _MIN_SIN)   # slant path through layer
    if condition == "fog":
        # v2 fix: Kim's sigma = 3.91/V is a natural-log extinction coefficient (1/km);
        # dB/km needs the 4.343 factor (V=1 km -> ~13.8 dB/km @785 nm, literature value).
        U = _fog_U(V)
        return 4.343 * (3.91 / V) * (wavelength_nm / 550.0) ** (-U) * R
    if condition == "snow":
        dry = True if temperature_c is None else float(temperature_c) <= 0.0
        snow_rate = _DEFAULT_SNOW_RATE_MM_H if snow_rate_mm_h is None else float(snow_rate_mm_h)
        return _gamma_snow_db_per_km(snow_rate, wavelength_nm, dry) * R
    if condition == "rain":
        rain_rate = _DEFAULT_RAIN_RATE_MM_H if rain_rate_mm_h is None else float(rain_rate_mm_h)
        return _gamma_rain_db_per_km(rain_rate) * R
    return 0.0


def weather_transmittance(
    condition: str,
    elevation_deg: float,
    wavelength_nm: float = 1550.0,
    V: Optional[float] = None,
    H: Optional[float] = None,
    rain_rate_mm_h: Optional[float] = None,
    snow_rate_mm_h: Optional[float] = None,
    temperature_c: Optional[float] = None,
) -> float:
    """Capacity factor in (0, 1] = 10^(−L_weather/10)."""
    return 10.0 ** (
        -weather_loss_db(
            condition,
            elevation_deg,
            wavelength_nm,
            V,
            H,
            rain_rate_mm_h,
            snow_rate_mm_h,
            temperature_c,
        )
        / 10.0
    )


# condition (StratoQ Kim, for HAP-GS) <-> cloud fraction (TNSM table, for SAT-GS): both are
# views of the SAME weather. A cloudy sky is a dense-fog downlink for the HAP and a high
# cloud-cover column for the satellite. Used to derive one from the other for the dict form.
_COND_TO_CLOUD = {"clear": 0.0, "fog": 0.9, "rain": 0.6, "snow": 0.7}


class WeatherModel:
    """Per-GS, per-slot weather, in **two coupled representations** (one per source paper):

      * ``condition_at(gs,t)`` -> {"clear","fog","rain","snow"} : StratoQ Kim model, HAP-GS;
      * ``cloud_cover_at(gs,t)`` -> cloud cover **percent** (0-100) : TNSM Transmittance table,
        SAT-GS.

    They are two views of the same underlying weather field, so a cloudy spell attenuates both
    the HAP-GS link (as fog) and the SAT-GS link (as cloud cover) at the same slots.

    condition:
      - "clear"  : clear everywhere;
      - "cloudy" : synthetic per-GS cloud spells (continuous cloud fraction; timing varies per GS);
      - dict ``{gs_id: [condition_per_slot]}`` : supplied condition series (e.g. from real data),
        each entry one of {"clear","fog","rain","snow"}; cloud fraction is derived from it.
    """

    def __init__(self, condition: Union[str, dict] = "clear", n_slots: int = 12,
                 gs_ids: Optional[List[str]] = None, seed: int = 0, coverage: float = 0.6):
        self.condition = condition
        self.n_slots = n_slots
        self._series: Dict[str, List[str]] = {}       # gs -> [condition]  (HAP-GS Kim)
        self._cloud: Dict[str, List[float]] = {}       # gs -> [cloud fraction 0-1] (SAT-GS table)
        if isinstance(condition, dict):
            self._series = {g: list(v) for g, v in condition.items()}
            self._cloud = {g: [_COND_TO_CLOUD.get(c, 0.0) for c in v] for g, v in condition.items()}
        elif condition == "cloudy":
            for g in (gs_ids or []):
                cloud = self._synth_cloud(g, n_slots, seed, coverage)
                self._cloud[g] = cloud
                # HAP-GS sees the same spell as fog (a dense cloud is optically fog to the FSO link)
                self._series[g] = ["fog" if c > 0.3 else "clear" for c in cloud]

    def _synth_cloud(self, gs_id: str, n: int, seed: int, coverage: float) -> List[float]:
        """Continuous cloud fraction [0,1] over the day; a diurnal wave crossing (1-coverage)."""
        h = int(hashlib.md5(f"{gs_id}|{seed}".encode()).hexdigest(), 16)
        phase = (h % 24) / 24.0
        out = []
        for i in range(n):
            frac = i / max(n, 1)
            wave = 0.5 + 0.5 * math.sin(2 * math.pi * (frac + phase))
            cloud = max(0.0, (wave - (1.0 - coverage)) / max(coverage, 1e-6))
            out.append(min(1.0, cloud))
        return out

    def condition_at(self, gs_id: str, t: int) -> str:
        series = self._series.get(gs_id)
        if not series:
            return "clear"
        return series[t] if t < len(series) else series[-1]

    def cloud_cover_at(self, gs_id: str, t: int) -> float:
        """Cloud cover **percent** (0-100) at (gs,t) for the TNSM SAT-GS transmittance table."""
        cloud = self._cloud.get(gs_id)
        if not cloud:
            return 0.0
        c = cloud[t] if t < len(cloud) else cloud[-1]
        return c * 100.0


def apply_weather(capacity_by_time: Dict[int, float], gs_id: str, link_type: str,
                  weather: WeatherModel, elevation_by_time, wavelength_nm: float = 1550.0) -> Dict[int, float]:
    """Apply the weather transmittance per slot to a ground link's capacity.

    ``elevation_by_time``: {slot: elevation_deg} (or a constant for a fixed-geometry link).
    SAT-HAP links are immune (path never enters the weather layer) → returned unchanged.
    """
    if link_type == "SAT-HAP":
        return dict(capacity_by_time)
    out = {}
    for t, cap in capacity_by_time.items():
        alpha = elevation_by_time.get(t, 90.0) if isinstance(elevation_by_time, dict) else elevation_by_time
        cond = weather.condition_at(gs_id, t)
        out[t] = cap * weather_transmittance(cond, alpha, wavelength_nm)
    return out
