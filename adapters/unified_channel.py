from __future__ import annotations

"""Unified link-rate model v2 — the single front-end implementing ``rate-model-v2.md`` (repo root).

``UnifiedLinkModel(link_type).skr_bps(...)`` is the ONE entry point for all four links; each
routes to its canonical source so this model, the scenario builders, and the visualizations
all return the same number:

  SAT-GS  : TNSM finite-key table (v2 §1.4) — skeleton effective-eta → rate_fk lookup.
  HAP-GS  : analytic skeleton eta = h_PE·eta_tropo·eta_upper·eta_wx·eta_coll → GLLP/DW protocol;
            the real StratoQ K_MAX traces (hap_capacity) are the calibration ground truth (v2 §4.2).
  SAT-HAP : DELEGATES to ``HAPSatelliteChannel`` (JOCN diffraction × thin-stratosphere
            Beer-Lambert × pointing → DW) — the production physical channel.
  SAT-SAT : DELEGATES to ``InterSatelliteChannel`` (JOCN vacuum, Eqs. 5-7) — production physical channel.

Shared skeleton (v2 §0): eta_link = h_PE(geometry+pointing) · eta_upper · eta_wx · eta_coll.
Protocol  (v2 §1): GLLP asymptotic decoy BB84 (Ma et al., PRA 72, 012326) with background p_b,
                   or DW closed form; SAT-HAP/SAT-SAT canonical = DW (JOCN convention).
Turbulence(v2 §5): statistics only — sigma_I^2 from an HV-profile slant integral feeds an
                   outage probability / fade margin; it never multiplies the mean capacity.
eta_coll  (v2 §5.4): HAP-GS turbulence-limited SMF/AO collection efficiency, zenith-referenced
                   −20 dB (calm night) / −35.2 dB (turbulent day) × sin^{6/5}α, StratoQ-calibrated.

Per-link hardware follows the source papers (v2 §7): SAT-GS 785 nm (TNSM), HAP-GS 1550 nm
(StratoQ), SAT-HAP & SAT-SAT 843.9 nm (JOCN).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from adapters.inter_satellite_channel import binary_entropy, diffraction_efficiency
from adapters.weather import residual_weather_loss_db, weather_loss_db

R_EARTH_KM = 6371.0
H_TROPO_TOP_KM = 12.0
H_STRATO_TOP_KM = 50.0
H_MESO_TOP_KM = 85.0
THETA_STRATO_PER_KM = 1e-4   # v2 §3 (denser layer -> larger coefficient)
THETA_MESO_PER_KM = 1e-5

# clear-sky zenith transmittance of the troposphere (molecular+aerosol), per wavelength.
# 785 nm value fitted from TNSM FS_loss raw data (docs/unified_link_modeling.md §4.1);
# 1550 nm from StratoQ Table-2-era assumptions. Applied as T_zen^(1/sin(elev)).
T_ZEN_TROPO = {785.0: 0.89, 1550.0: 0.91, 843.9: 0.90}

# v2 §1.3 — sky spectral radiance N_B (W m^-2 sr^-1 nm^-1) per illumination scenario
N_B_BY_SCENE = {"night": 1e-5, "day": 5e-3, "umbra": 1e-6, "penumbra": 1e-4, "full_sun": 5e-3}
# v2 §1.4 / JOCN — ISL misalignment QBER by illumination (used when no p_b model is wanted)
QBER_BY_ILLUMINATION = {"umbra": 0.015, "penumbra": 0.03, "full_sun": 0.06}


@dataclass
class LinkHardware:
    """One per-link parameter set (v2 §7)."""
    wavelength_nm: float
    w0_m: float                    # transmit beam waist (GS-facing links)
    rx_radius_m: float             # receiver aperture radius
    divergence_rad: float          # full-angle divergence used by the (D_r/(theta d)) form
    obs_transmissivity: float = 1.0
    pointing_xi: Optional[float] = None      # quasi-static links: xi = w_z / (2 sigma_s)
    pointing_jitter_rad: Optional[float] = None  # jitter-driven links: sigma_s = jitter * d
    source_rate_hz: float = 80e6
    source_eff: float = 0.01
    detector_eff: float = 0.85
    rx_optics_eff: float = 0.8
    mu: float = 0.5
    p_dark: float = 1e-6
    e_tot: float = 0.015
    q_sift: float = 0.5
    f_ec: float = 1.2
    fov_rad: float = 20e-6         # narrow QKD FOV (v2 §1.3 note)
    b_opt_nm: float = 1.0
    gate_window_s: float = 1e-9
    eta_coll_db: Dict[str, float] = field(default_factory=dict)  # {"night": dB, "day": dB}


PRESETS: Dict[str, LinkHardware] = {
    # TNSM hardware; the finite-key table already folds pointing/system losses into eta_sys.
    # rx_optics_eff = eta_sys = 0.01 (FS_loss CSV) so the analytic-GLLP branch is comparable.
    "SAT-GS": LinkHardware(wavelength_nm=785.0, w0_m=0.15, rx_radius_m=0.5,
                           divergence_rad=1.22 * 785e-9 / 0.30, pointing_xi=6.0,
                           source_rate_hz=100e6, source_eff=1.0, rx_optics_eff=0.01,
                           e_tot=0.02, p_dark=1e-5),
    # StratoQ Table 2 hardware + the eta_coll term the closed form needs (v2 §5.4).
    # Values are ZENITH-referenced; eta_coll() applies the r0 scaling sin(alpha)^{6/5}.
    # day: fitted against skr_results.pkl with the scaling law (gm ratio 1.04x over 20-54 km);
    # night: coarse calibration against the K_MAX calm-config band.
    "HAP-GS": LinkHardware(wavelength_nm=1550.0, w0_m=0.15, rx_radius_m=0.375,
                           divergence_rad=1.22 * 1550e-9 / 0.40, pointing_xi=6.0,
                           source_rate_hz=80e6, source_eff=0.01, e_tot=0.04,
                           eta_coll_db={"night": -5.0, "day": -5.0}),
    # JOCN hardware (843.9 nm). NOTE: SAT-HAP/SAT-SAT skr_bps DELEGATES to the production
    # physical channels (HAPSatelliteChannel / InterSatelliteChannel); these presets are kept
    # only for the wavelength used by the turbulence/scintillation side and for reference.
    "SAT-HAP": LinkHardware(wavelength_nm=843.9, w0_m=0.15, rx_radius_m=0.30,
                            divergence_rad=17.2e-6, obs_transmissivity=0.91, pointing_xi=6.0,
                            source_rate_hz=575e6, source_eff=0.01, e_tot=0.04),
    "SAT-SAT": LinkHardware(wavelength_nm=843.9, w0_m=0.15, rx_radius_m=0.30,
                            divergence_rad=17.2e-6, obs_transmissivity=0.91,
                            pointing_jitter_rad=1e-6,
                            source_rate_hz=575e6, source_eff=0.01, e_tot=0.015),
}

_NODE_ALT_KM = {"GS": 0.0, "HAP": 20.0, "SAT": 567.0}
ISL_TROPOSPHERE_BLOCK_KM = 15.0


@dataclass(frozen=True)
class RateModelInput:
    """Main-project input shape for the unified adapter rate model."""

    distance_m: float
    tx_node_type: str
    rx_node_type: str
    elevation_angle_deg: float = 90.0
    visibility_m: float = 24000.0
    rain_mm: float = 0.0
    snow_cm: float = 0.0
    temperature_c: float = 15.0
    sf_w: float = 0.0
    c_low: float = 0.0
    c_mid: float = 0.0
    c_high: float = 0.0
    h_rx_km: float = 0.0
    h_tx_km: float = 600.0
    illumination: Optional[str] = None


def normalize_node_type(node_type: str) -> str:
    node = str(node_type).upper()
    aliases = {
        "GROUND": "GS",
        "GROUND_STATION": "GS",
        "SATELLITE": "SAT",
        "HIGH_ALTITUDE_PLATFORM": "HAP",
        "AIR_PLATFORM": "HAP",
    }
    return aliases.get(node, node)


def infer_link_type(tx_node_type: str, rx_node_type: str) -> str:
    tx = normalize_node_type(tx_node_type)
    rx = normalize_node_type(rx_node_type)
    pair = frozenset((tx, rx))
    if pair == frozenset(("SAT", "GS")):
        return "SAT-GS"
    if pair == frozenset(("HAP", "GS")):
        return "HAP-GS"
    if pair == frozenset(("SAT", "HAP")):
        return "SAT-HAP"
    if tx == "SAT" and rx == "SAT":
        return "SAT-SAT"
    raise ValueError(f"Unsupported QKD link type: {tx_node_type!r}->{rx_node_type!r}")


def cloud_fraction_to_percent(c_low: float, c_mid: float, c_high: float) -> float:
    """Main-project weather fields use fractions; TNSM adapter expects percent."""

    return max(0.0, min(100.0, (float(c_low) + float(c_mid) + float(c_high)) * 100.0))


def infer_weather_condition(
    visibility_m: float,
    rain_mm: float,
    snow_cm: float,
    c_low: float,
    c_mid: float,
    c_high: float,
    include_cloud_as_fog: bool = False,
) -> str:
    """Map detailed weather inputs to the reference adapter's condition names."""

    if float(snow_cm) > 0.0:
        return "snow"
    if float(rain_mm) > 0.0:
        return "rain"
    if float(visibility_m) < 10000.0 or (
        include_cloud_as_fog and cloud_fraction_to_percent(c_low, c_mid, c_high) > 30.0
    ):
        return "fog"
    return "clear"


def infer_illumination(sf_w: float, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if float(sf_w) > 100.0:
        return "full_sun"
    if float(sf_w) > 0.0:
        return "penumbra"
    return "umbra"


def slant_range_km(elev_deg: float, h_hi_km: float, h_lo_km: float = 0.0) -> float:
    """Spherical-Earth slant range from the lower node at elevation elev_deg."""
    e = math.radians(max(0.5, elev_deg))
    r_lo = R_EARTH_KM + h_lo_km
    r_hi = R_EARTH_KM + h_hi_km
    return math.sqrt(r_hi ** 2 - (r_lo * math.cos(e)) ** 2) - r_lo * math.sin(e)


def hv57_cn2(h_km: float, cn2_ground: float = 1.7e-14, wind_rms: float = 21.0) -> float:
    """Hufnagel-Valley C_n^2(h) profile (ITU-R P.1621); h in km."""
    h = max(0.0, h_km) * 1000.0
    return (0.00594 * (wind_rms / 27.0) ** 2 * (1e-5 * h) ** 10 * math.exp(-h / 1000.0)
            + 2.7e-16 * math.exp(-h / 1500.0)
            + cn2_ground * math.exp(-h / 100.0))


def grazing_altitude_km(distance_km: float, h_tx_km: float, h_rx_km: float) -> float:
    """Minimum altitude of the straight optical path between two nodes."""

    d = float(distance_km)
    r_tx = R_EARTH_KM + float(h_tx_km)
    r_rx = R_EARTH_KM + float(h_rx_km)
    if d <= 0.0:
        return min(r_tx, r_rx) - R_EARTH_KM

    d_tx_to_projection = (r_tx**2 + d**2 - r_rx**2) / (2.0 * d)
    if 0.0 < d_tx_to_projection < d:
        return math.sqrt(max(0.0, r_tx**2 - d_tx_to_projection**2)) - R_EARTH_KM
    return min(r_tx, r_rx) - R_EARTH_KM


def isl_upper_atmosphere_transmittance(distance_km: float, h_tx_km: float, h_rx_km: float) -> float:
    """ISL upper-atmosphere chord loss from v2 §6.

    Returns 0 when Earth/troposphere blocks the path, 1 for a fully vacuum chord,
    and Beer-Lambert loss for chords through 15-85 km altitude.
    """

    h_graz = grazing_altitude_km(distance_km, h_tx_km, h_rx_km)
    if h_graz < ISL_TROPOSPHERE_BLOCK_KM:
        return 0.0
    if h_graz >= H_MESO_TOP_KM:
        return 1.0

    r_graz = R_EARTH_KM + h_graz

    def chord_length_below(h_top_km: float) -> float:
        if h_graz >= h_top_km:
            return 0.0
        r_top = R_EARTH_KM + h_top_km
        return 2.0 * math.sqrt(max(0.0, r_top**2 - r_graz**2))

    l_meso_total = chord_length_below(H_MESO_TOP_KM)
    l_strato_total = chord_length_below(H_STRATO_TOP_KM)
    l_block_total = chord_length_below(ISL_TROPOSPHERE_BLOCK_KM)
    l_meso = max(0.0, l_meso_total - l_strato_total)
    l_strato = max(0.0, l_strato_total - l_block_total)
    return math.exp(-(THETA_STRATO_PER_KM * l_strato + THETA_MESO_PER_KM * l_meso))


class UnifiedLinkModel:
    """v2 canonical model for one link type. All returns are mean quantities unless noted."""

    def __init__(self, link_type: str, hardware: Optional[LinkHardware] = None,
                 use_finite_key_sat_gs: bool = True, finite_key_peak_bps: Optional[float] = None):
        if link_type not in PRESETS:
            raise ValueError(f"unknown link type {link_type}")
        self.link_type = link_type
        self.hw = hardware or PRESETS[link_type]
        self._fk = None
        if link_type == "SAT-GS" and use_finite_key_sat_gs:
            from adapters.satellite_capacity import TNSMSatelliteCapacityProvider
            prov = TNSMSatelliteCapacityProvider()
            if finite_key_peak_bps is not None:
                prov.peak_bps = finite_key_peak_bps
            self._fk = prov

    # ---------------- geometry + pointing (v2 §2) ----------------
    def beam_radius_m(self, distance_m: float) -> float:
        theta_half = self.hw.divergence_rad / 2.0   # divergence_rad is the full angle throughout
        return math.sqrt(self.hw.w0_m ** 2 + (theta_half * distance_m) ** 2)

    def h_pe_bar(self, distance_m: float) -> float:
        """Mean geometry+pointing efficiency: A0 · xi^2/(1+xi^2) (v2 §2, h_th→0 mean)."""
        if self.link_type in ("SAT-HAP", "SAT-SAT"):
            # JOCN parameterization shared with the repo ISL/SAT-HAP models
            eta_geo = diffraction_efficiency(2 * self.hw.rx_radius_m, self.hw.obs_transmissivity,
                                             self.hw.divergence_rad, distance_m / 1000.0)
        else:
            w_z = self.beam_radius_m(distance_m)
            eta_geo = min(1.0, 2.0 * self.hw.rx_radius_m ** 2 / w_z ** 2)
        if self.hw.pointing_xi is not None:
            xi2 = self.hw.pointing_xi ** 2
        else:
            sigma_s = (self.hw.pointing_jitter_rad or 1e-6) * distance_m
            xi2 = (self.beam_radius_m(distance_m) / (2.0 * sigma_s)) ** 2 if sigma_s > 0 else 1e9
        return eta_geo * xi2 / (1.0 + xi2)

    # ---------------- upper atmosphere (v2 §3) ----------------
    def eta_upper(self, elev_deg: float, h_lo_km: float, h_hi_km: float) -> float:
        s = 1.0 / max(math.sin(math.radians(max(0.5, elev_deg))), 1e-2)

        def seg(lo, hi):
            a, b = max(h_lo_km, lo), min(h_hi_km, hi)
            return max(0.0, b - a) * s

        L_s = seg(H_TROPO_TOP_KM, H_STRATO_TOP_KM)
        L_m = seg(H_STRATO_TOP_KM, H_MESO_TOP_KM)
        return math.exp(-(THETA_STRATO_PER_KM * L_s + THETA_MESO_PER_KM * L_m))

    def eta_tropo_clear(self, elev_deg: float) -> float:
        """Clear-sky tropospheric transmittance T_zen^(csc a) for links touching the ground."""
        t_zen = T_ZEN_TROPO.get(self.hw.wavelength_nm, 0.9)
        airmass = 1.0 / max(math.sin(math.radians(max(0.5, elev_deg))), 1e-2)
        return t_zen ** airmass

    # ---------------- collection term (v2 §5.4) ----------------
    def eta_coll(self, daytime: bool = False, elev_deg: float = 90.0) -> float:
        """Zenith-referenced turbulence collection efficiency with the Fried-parameter
        scaling law eta_coll(alpha) = eta_ref * sin(alpha)^{6/5}: lower elevation ->
        longer boundary-layer slant -> larger int Cn2 dl -> r0 ~ (.)^{-3/5} smaller ->
        single-mode coupling (r0/D)^2 worse. Validated against StratoQ trajectory data
        (gm ratio 1.04x over 20-54 km, tools/validate_link_models_refdata.py)."""
        db = self.hw.eta_coll_db.get("day" if daytime else "night", 0.0)
        if db == 0.0:
            return 1.0
        scale = math.sin(math.radians(max(5.0, elev_deg))) ** 1.2
        return 10.0 ** (db / 10.0) * scale

    # ---------------- background (v2 §1.3) ----------------
    def p_b(self, scene: str = "night") -> float:
        n_b = N_B_BY_SCENE.get(scene, 0.0)
        area = math.pi * self.hw.rx_radius_m ** 2
        omega = math.pi * (self.hw.fov_rad / 2.0) ** 2
        p_bg_w = n_b * self.hw.rx_optics_eff * area * omega * self.hw.b_opt_nm
        lam_m = self.hw.wavelength_nm * 1e-9
        mu_b = p_bg_w * lam_m / (6.626e-34 * 3.0e8) * self.hw.gate_window_s
        return 1.0 - math.exp(-self.hw.detector_eff * mu_b)

    # ---------------- protocol layer (v2 §1) ----------------
    def gllp_skr_per_pulse(self, eta_link: float, p_b: float = 0.0,
                           e_tot: Optional[float] = None) -> float:
        hw = self.hw
        eta = hw.detector_eff * hw.rx_optics_eff * eta_link
        y0 = hw.p_dark + p_b
        mu = hw.mu
        q1 = math.exp(-mu) * mu * (eta + y0)
        q_mu = y0 + 1.0 - math.exp(-eta * mu)
        if q_mu <= 0 or q1 <= 0:
            return 0.0
        e0, et = 0.5, (hw.e_tot if e_tot is None else e_tot)
        e1 = min(0.499, max(1e-9, (e0 * y0 + et * eta) / (y0 + eta)))
        e_mu = min(0.499, max(1e-9, (e0 * y0 + et * (1.0 - math.exp(-eta * mu))) / q_mu))
        r = hw.q_sift * (q1 * (1.0 - _h2(e1)) - q_mu * hw.f_ec * _h2(e_mu))
        return max(0.0, r)

    def dw_skr_per_pulse(self, eta_link: float, qber: Optional[float] = None) -> float:
        """Devetak-Winter closed form (ideal single-photon source, asymptotic):
        R = eta (1 - 2 H2(Q)). This is the optimistic reference layer used by
        StratoQ/JOCN (and by the K_MAX traces); GLLP with mu=0.5 sits ~12-15x below it.
        Provided so the unified channel can be compared against DW-based references
        at the same protocol layer."""
        hw = self.hw
        eta = hw.detector_eff * hw.rx_optics_eff * eta_link
        q = hw.e_tot if qber is None else qber
        return max(0.0, eta * (1.0 - 2.0 * _h2(q)))

    # ---------------- turbulence statistics (v2 §5 — never in the mean) ----------------
    def scintillation_index(self, elev_deg: float, cn2_ground: float = 1.7e-14,
                            h_lo_km: float = 0.0) -> float:
        """Downlink sigma_I^2 = 2.25 k^{7/6} sec^{11/6}(zeta) ∫ C_n^2(h)(h-h0)^{5/6} dh."""
        lam_m = self.hw.wavelength_nm * 1e-9
        k = 2.0 * math.pi / lam_m
        zeta = math.radians(90.0 - max(0.5, min(89.5, elev_deg)))
        n, h_top = 400, 30.0  # C_n^2 above 30 km is negligible
        dh = (h_top - h_lo_km) / n
        integ = sum(hv57_cn2(h_lo_km + (i + 0.5) * dh) * ((i + 0.5) * dh * 1000.0) ** (5.0 / 6.0)
                    for i in range(n)) * dh * 1000.0
        return 2.25 * k ** (7.0 / 6.0) * (1.0 / math.cos(zeta)) ** (11.0 / 6.0) * integ

    def outage_probability(self, elev_deg: float, fade_margin_db: float = 3.0, **kw) -> float:
        s2 = min(self.scintillation_index(elev_deg, **kw), 1.0)  # weak-fluctuation clamp
        if s2 <= 1e-12:
            return 0.0
        sigma = math.sqrt(math.log(1.0 + s2))
        thr = -fade_margin_db / 10.0 * math.log(10.0)
        z = (thr + 0.5 * sigma ** 2) / sigma
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    # ---------------- top-level per-link SKR (bps) ----------------
    def skr_bps(self, elev_deg: Optional[float] = None, distance_km: Optional[float] = None,
                condition: str = "clear", cloud_pct: float = 0.0,
                illumination: str = "umbra", daytime: bool = False,
                protocol: str = "gllp", h_rx_km: Optional[float] = None,
                h_tx_km: Optional[float] = None) -> float:
        """protocol: "gllp" (v2 default, conservative) or "dw" (optimistic reference layer,
        for like-for-like comparison against StratoQ/JOCN/K_MAX numbers)."""
        hw = self.hw

        def proto(eta, p_b=0.0, qber=None):
            if protocol == "dw":
                return self.dw_skr_per_pulse(eta, qber)
            return self.gllp_skr_per_pulse(eta, p_b, e_tot=qber)
        if self.link_type == "SAT-GS":
            elev = elev_deg if elev_deg is not None else 90.0
            if self._fk is not None:                      # v2 §1.4 finite-key table (default)
                base = self._fk.skr_at(elev, cloud_pct)
                wx = 10 ** (-weather_loss_db(condition, elev, hw.wavelength_nm) / 10.0) \
                    if condition not in (None, "clear") else 1.0
                return base * wx
            d = slant_range_km(elev, _NODE_ALT_KM["SAT"]) * 1e3
            eta = (self.h_pe_bar(d) * self.eta_tropo_clear(elev)
                   * self.eta_upper(elev, 0.0, _NODE_ALT_KM["SAT"])
                   * 10 ** (-weather_loss_db(condition, elev, hw.wavelength_nm) / 10.0))
            scene = "day" if daytime else "night"
            return hw.source_rate_hz * hw.source_eff * proto(eta, self.p_b(scene))
        if self.link_type == "HAP-GS":
            d_km = distance_km if distance_km is not None else 40.0
            elev = elev_deg if elev_deg is not None else math.degrees(
                math.asin(min(1.0, _NODE_ALT_KM["HAP"] / max(d_km, _NODE_ALT_KM["HAP"])))
            )
            h_lo = min(0.0 if h_rx_km is None else float(h_rx_km), 20.0 if h_tx_km is None else float(h_tx_km))
            h_hi = max(0.0 if h_rx_km is None else float(h_rx_km), 20.0 if h_tx_km is None else float(h_tx_km))
            eta = (self.h_pe_bar(d_km * 1e3) * self.eta_tropo_clear(elev)
                   * self.eta_upper(elev, h_lo, h_hi)
                   * 10 ** (-weather_loss_db(condition, elev, hw.wavelength_nm) / 10.0)
                   * self.eta_coll(daytime, elev))
            scene = "day" if daytime else "night"
            return hw.source_rate_hz * hw.source_eff * proto(eta, self.p_b(scene))
        if self.link_type == "SAT-HAP":
            # Canonical SAT-HAP = the production physical channel (JOCN diffraction ×
            # thin-stratosphere Beer-Lambert × pointing → DW). unified_channel delegates to it
            # so the scenario, this model, and the visualization all share one number.
            ch = self._sat_hap_channel()
            if distance_km is not None:
                cos_z = (ch.h_sat_km - ch.h_hap_km) / max(distance_km, ch.h_sat_km - ch.h_hap_km)
                zen = math.degrees(math.acos(min(1.0, cos_z)))
            else:
                zen = 90.0 - (elev_deg if elev_deg is not None else 90.0)
            return ch.skr_bps(zen)
        # SAT-SAT / ISL: canonical = the production InterSatelliteChannel (JOCN vacuum, Eq.5-7).
        ch = self._isl_channel()
        d_km = distance_km if distance_km is not None else 2000.0
        h_tx = _NODE_ALT_KM["SAT"] if h_tx_km is None else float(h_tx_km)
        h_rx = _NODE_ALT_KM["SAT"] if h_rx_km is None else float(h_rx_km)
        return ch.skr_bps(d_km, illumination) * isl_upper_atmosphere_transmittance(d_km, h_tx, h_rx)

    def _sat_hap_channel(self):
        if not hasattr(self, "_sh_ch"):
            from adapters.hap_satellite_channel import HAPSatelliteChannel
            self._sh_ch = HAPSatelliteChannel()
        return self._sh_ch

    def _isl_channel(self):
        if not hasattr(self, "_isl_ch"):
            from adapters.inter_satellite_channel import InterSatelliteChannel
            self._isl_ch = InterSatelliteChannel()
        return self._isl_ch


class UnifiedQKDRateModel:
    """Stable public front-end over the v2 adapter link models.

    This is the main project's unified rate API. It returns bits per second,
    matching the existing ``k_max`` convention.
    """

    def __init__(
        self,
        protocol: str = "gllp",
        use_finite_key_sat_gs: bool = True,
        finite_key_peak_bps: Optional[float] = None,
    ) -> None:
        self.protocol = protocol
        self.use_finite_key_sat_gs = use_finite_key_sat_gs
        self.finite_key_peak_bps = finite_key_peak_bps
        self._models: dict[str, UnifiedLinkModel] = {}

    def _model_for(self, link_type: str) -> UnifiedLinkModel:
        model = self._models.get(link_type)
        if model is None:
            model = UnifiedLinkModel(
                link_type,
                use_finite_key_sat_gs=self.use_finite_key_sat_gs,
                finite_key_peak_bps=self.finite_key_peak_bps,
            )
            self._models[link_type] = model
        return model

    def _weather_loss_db(self, link_type: str, request: RateModelInput, elev_deg: float) -> float:
        if link_type in ("SAT-HAP", "SAT-SAT"):
            return 0.0
        return residual_weather_loss_db(
            elev_deg,
            self._model_for(link_type).hw.wavelength_nm,
            visibility_m=request.visibility_m,
            rain_rate_mm_h=request.rain_mm,
            snow_rate_mm_h=request.snow_cm * 10.0,
            temperature_c=request.temperature_c,
        )

    def compute_rate_bps(self, request: RateModelInput) -> float:
        link_type = infer_link_type(request.tx_node_type, request.rx_node_type)
        model = self._model_for(link_type)
        distance_km = max(0.0, float(request.distance_m) / 1000.0)
        elev = max(0.0, min(90.0, float(request.elevation_angle_deg)))
        cloud_pct = cloud_fraction_to_percent(request.c_low, request.c_mid, request.c_high)
        illumination = request.illumination or (
            "penumbra" if link_type == "SAT-SAT" else infer_illumination(request.sf_w)
        )
        daytime = float(request.sf_w) > 100.0
        weather_factor = 10.0 ** (-self._weather_loss_db(link_type, request, elev) / 10.0)

        base_rate = model.skr_bps(
            elev_deg=elev,
            distance_km=distance_km,
            condition="clear",
            cloud_pct=cloud_pct,
            illumination=illumination,
            daytime=daytime,
            protocol=self.protocol,
            h_rx_km=request.h_rx_km,
            h_tx_km=request.h_tx_km,
        )
        return float(base_rate * weather_factor)

    def compute_secure_key_rate(
        self,
        distance_m: float,
        visibility_m: float,
        rain_mm: float,
        snow_cm: float,
        temperature_c: float,
        current_time_str: str = "2023-01-01T12:00",
        sunrise_str: str = "2023-01-01T06:00",
        sunset_str: str = "2023-01-01T18:00",
        rh: float = 50.0,
        ws: float = 2.0,
        sf_w: float = 0.0,
        h_rx_km: float = 0.0,
        h_tx_km: float = 600.0,
        rx_node_type: str = "GS",
        tx_node_type: str = "SAT",
        c_low: float = 0.0,
        c_mid: float = 0.0,
        c_high: float = 0.0,
        elevation_angle_deg: float = 90.0,
        illumination: Optional[str] = None,
    ) -> float:
        del current_time_str, sunrise_str, sunset_str, rh, ws
        return self.compute_rate_bps(
            RateModelInput(
                distance_m=distance_m,
                tx_node_type=tx_node_type,
                rx_node_type=rx_node_type,
                elevation_angle_deg=elevation_angle_deg,
                visibility_m=visibility_m,
                rain_mm=rain_mm,
                snow_cm=snow_cm,
                temperature_c=temperature_c,
                sf_w=sf_w,
                c_low=c_low,
                c_mid=c_mid,
                c_high=c_high,
                h_rx_km=h_rx_km,
                h_tx_km=h_tx_km,
                illumination=illumination,
            )
        )


def _h2(x: float) -> float:
    return binary_entropy(x)
