from __future__ import annotations

"""HAP <-> Satellite free-space-optical (FSO) channel model.

A LEO→HAP downlink is a **space-FSO link**, so it shares the SAME diffraction geometry as the
inter-satellite link (``inter_satellite_channel``); the ONLY difference is that its path still
crosses a thin slice of stratosphere. The model is therefore

    η = η_diff(d) · η_atm(d) · E[h_p]
        η_diff : diffraction-limited efficiency (D_r·η_obs)²/(θ·d)²  — JOCN 2026, the SAME
                 formula ``inter_satellite_channel.diffraction_efficiency`` uses (the earlier
                 Gaussian-beam / Friis forms were the same 1/d² physics, differently parameterised),
        η_atm  : Beer-Lambert stratospheric transmittance (thin — HAP sits at ~25 km); this is
                 the ONE term ISL drops (vacuum). See docs/HAP-Satellite链路建模调研.md §2.2,
        E[h_p] : mean pointing-error loss (Farid-Hranilovic, §4) — HAP quasi-static ⇒ large ξ.

Turbulence (survey §3) is normalised out of the *mean* capacity (E[h_a]=1); it only drives
outage/variance, exposed via ``outage_probability``.

SKR = R_eff · η_det · η · (1 - 2·H(QBER)). Because the HAP is closer to the satellite than the
ground (η_diff larger) and above most of the atmosphere (η_atm≈1), the SAT-HAP downlink is
stronger than SAT-GS and its higher vantage gives a wider window — both fall out of the geometry.
The source rate is calibrated so the overhead peak ≈ 8 kbps (network balance; cf. the SAT-GS peak
calibration). Optics/detector parameters follow JOCN Table 1 (D_r=0.6 m, η_obs=0.91, θ=17.2 µrad).
"""

import math
from dataclasses import dataclass

from adapters.inter_satellite_channel import diffraction_efficiency


def binary_entropy(q: float) -> float:
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return -q * math.log2(q) - (1 - q) * math.log2(1 - q)


@dataclass
class HAPSatelliteChannel:
    # geometry
    h_sat_km: float = 567.0
    h_hap_km: float = 25.0
    # optics — JOCN 2026 diffraction model (shared with InterSatelliteChannel / ISL)
    rx_diameter_m: float = 0.6          # D_r receiver telescope
    obs_transmissivity: float = 0.91    # η_obs (central obstruction)
    beam_divergence_rad: float = 17.2e-6  # θ (full-angle)
    wavelength_m: float = 843.9e-9      # JOCN λ (used by the turbulence/outage side only)
    # Beer-Lambert stratospheric attenuation (§2.2) — the ONLY term ISL drops
    theta1: float = 1e-5                # mesospheric
    theta2: float = 1e-4                # stratospheric (1e-4 medium … 4e-3 volcanic)
    rho: float = 0.1                    # fraction affected by mesosphere
    # pointing error (§4.1): xi = w_eq / (2 σ_s); HAP quasi-static -> small jitter -> large xi
    pe_xi: float = 6.0
    pe_A0: float = 1.0
    # BB84 / source — source rate calibrated so the overhead peak ≈ 8 kbps (network balance)
    source_rate_hz: float = 575e6
    source_eff: float = 0.01
    detector_eff: float = 0.85
    qber: float = 0.04
    # turbulence (for outage only). Above ~30 km Cn²≈0, so only a thin layer near the HAP
    # is turbulent; integrating over the full slant would grossly overestimate scintillation.
    rx_aperture_m: float = 0.6          # for aperture averaging (= D_r)
    cn2: float = 1e-18                  # stratospheric (weak)
    turbulent_layer_km: float = 5.0     # effective turbulent path thickness near the HAP

    # --- geometry ---
    def slant_range_km(self, zenith_deg: float) -> float:
        return (self.h_sat_km - self.h_hap_km) / math.cos(math.radians(min(zenith_deg, 89.5)))

    # --- deterministic loss h_l ---
    def geometric_efficiency(self, slant_km: float) -> float:
        """Diffraction-limited efficiency (D_r·η_obs)²/(θ·d)² — shared JOCN space-FSO model."""
        return diffraction_efficiency(self.rx_diameter_m, self.obs_transmissivity,
                                      self.beam_divergence_rad, slant_km)

    def stratospheric_transmittance(self, slant_km: float) -> float:
        """Beer-Lambert (§2.2): g = exp[-(θ1 ρ + θ2 (1-ρ)) L]."""
        return math.exp(-(self.theta1 * self.rho + self.theta2 * (1 - self.rho)) * slant_km)

    def pointing_efficiency_mean(self) -> float:
        """Mean of Farid-Hranilovic h_p: E[h_p] = A0 · ξ²/(1+ξ²)."""
        xi2 = self.pe_xi ** 2
        return self.pe_A0 * xi2 / (1.0 + xi2)

    # --- mean channel efficiency & SKR ---
    def channel_efficiency(self, zenith_deg: float) -> float:
        L = self.slant_range_km(zenith_deg)
        return (self.geometric_efficiency(L)
                * self.stratospheric_transmittance(L)
                * self.pointing_efficiency_mean())

    def skr_bps(self, zenith_deg: float) -> float:
        eta = self.channel_efficiency(zenith_deg) * self.detector_eff
        r_eff = self.source_rate_hz * self.source_eff
        skr = r_eff * eta * (1.0 - 2.0 * binary_entropy(self.qber))
        return max(skr, 0.0)

    # --- turbulence / outage (statistical side, §3 & §5; not used by the mean-capacity trace) ---
    def scintillation_index(self, zenith_deg: float) -> float:
        """Downlink scintillation index with aperture averaging (§3.3, reduced form).

        Plane-wave Rytov variance over the *turbulent layer near the HAP* (not the full
        vacuum slant), times an aperture-averaging factor (Andrews & Phillips). Small for
        this link — turbulence is weak above 20 km — consistent with the survey (§7).
        """
        k = 2 * math.pi / self.wavelength_m
        sec = 1.0 / math.cos(math.radians(min(zenith_deg, 89.5)))
        L_turb = self.turbulent_layer_km * 1e3 * sec       # slant path through the turbulent layer
        L_full = self.slant_range_km(zenith_deg) * 1e3
        rytov = 1.23 * self.cn2 * k ** (7 / 6) * L_turb ** (11 / 6)
        d2 = self.rx_aperture_m ** 2
        a_avg = 1.0 / (1.0 + 1.1 * (d2 / (self.wavelength_m * L_full)) ** (7 / 6))
        return rytov * a_avg

    def outage_probability(self, zenith_deg: float, fade_margin_db: float = 3.0) -> float:
        """Outage indicator P(fade > margin) under weak turbulence (log-normal, §3.1).

        For the weak scintillation of this high-altitude link a log-normal irradiance model
        is appropriate and numerically robust: with log-irradiance variance
        σ_l² = ln(1 + σ_I²), P_out = 0.5·erfc[(0.5σ_l² + m) / (σ_l·√2)] where m is the fade
        margin in nepers. Returns a probability in [0, 1].
        """
        si = max(self.scintillation_index(zenith_deg), 1e-12)
        sigma_l2 = math.log(1.0 + si)
        sigma_l = math.sqrt(sigma_l2)
        m = fade_margin_db * math.log(10.0) / 10.0
        return 0.5 * math.erfc((0.5 * sigma_l2 + m) / (sigma_l * math.sqrt(2.0)))
