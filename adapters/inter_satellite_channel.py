from __future__ import annotations

"""Inter-satellite (ISL / ISQL) free-space-optical channel model — JOCN 2026.

Implements the inter-satellite QKD link (ISQL) transmission-efficiency model of

    Zhang et al., "Adaptive QKD link establishment for single- and dual-layer LEO quantum
    satellite networks," J. Opt. Commun. Netw. 18(5), 2026, Eqs. (5)-(9), Table 1.

An ISL operates in **vacuum**, so relative to a satellite-ground link (SGQL) the total
efficiency drops the atmospheric and turbulence terms:

    SGQL:  η = η_diff · η_atm · η_turb · η_pde · η_rx · η_cpl · η_pat
    ISQL:  η = η_diff ·                 η_pde · η_rx · η_cpl · η_pat     (no η_atm, no η_turb)

with the diffraction-limited efficiency (Eq. 7)

    η_diff = (D_r · η_obs)² / (θ · d)²

where D_r = receiver-telescope diameter, η_obs = telescope transmissivity (central
obstruction), θ = full-angle beam divergence, and d = inter-satellite link distance. So an
ISL is **distance-limited (η ∝ 1/d²) but weather-immune** — the natural high-altitude relay.

The BB84/decoy secret-key rate (Eq. 5) is used in the same linearised (SKR ∝ η) convention as
the other channels in this repo: SKR = R_eff · η · (1 - 2·H(QBER)). Different **illumination
conditions** (umbra / penumbra / full-sun, JOCN §3) raise the background yield Y0 and hence the
QBER; we expose that as a 3-level QBER since ISLs on the sun-unlit side (umbra) are the best case.

Baseline optics/detector parameters are JOCN Table 1 (λ=843.9 nm, D_r=0.6 m, η_obs=0.91,
θ=17.2 µrad, η_pde=0.58, η_rx=0.753, η_cpl=0.56, η_pat=0.912).
"""

import math
from dataclasses import dataclass, field
from typing import Dict


def binary_entropy(q: float) -> float:
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return -q * math.log2(q) - (1 - q) * math.log2(1 - q)


def diffraction_efficiency(rx_diameter_m: float, obs_transmissivity: float,
                           beam_divergence_rad: float, distance_km: float) -> float:
    """Diffraction-limited free-space efficiency η_diff = (D_r·η_obs)² / (θ·d)²  (JOCN Eq. 7).

    The single geometric model shared by both space-FSO links in this repo — inter-satellite
    (``InterSatelliteChannel``) and satellite→HAP (``HAPSatelliteChannel``). It is the far-field
    (∝1/d²) diffraction law; the Friis ``(λ/4πd)²`` and Gaussian-beam ``1−exp(−2a²/w²)`` forms
    are the SAME physics in a different parameterisation (antenna gains / beam-waist+aperture)."""
    d = max(distance_km, 1e-6) * 1e3
    return (rx_diameter_m * obs_transmissivity) ** 2 / (beam_divergence_rad * d) ** 2


# QBER per illumination condition (JOCN §3): background yield Y0 grows umbra < penumbra <
# full-sun, raising the error rate. Umbra ~ misalignment floor (1.5%, Table 1).
_QBER_BY_ILLUMINATION: Dict[str, float] = {"umbra": 0.015, "penumbra": 0.03, "full_sun": 0.06}


@dataclass
class InterSatelliteChannel:
    # receiver optics / beam (JOCN Table 1)
    rx_diameter_m: float = 0.6            # D_r
    obs_transmissivity: float = 0.91      # η_obs (central obstruction)
    beam_divergence_rad: float = 17.2e-6  # θ (full-angle)
    # receiver-side component efficiencies (JOCN Table 1)
    eta_pde: float = 0.58                 # photon detection efficiency (= detector QE)
    eta_rx: float = 0.753                 # receiver optical efficiency
    eta_cpl: float = 0.56                 # coupling efficiency
    eta_pat: float = 0.912                # pointing & tracking efficiency
    # BB84 / source (linearised SKR ∝ η, same convention as the other channels)
    source_rate_hz: float = 1e9           # GHz-class source (Jinan-1-like real-time QKD)
    source_eff: float = 0.1
    wavelength_m: float = 843.9e-9        # JOCN λ (record only; η_diff uses θ directly)

    def diffraction_efficiency(self, distance_km: float) -> float:
        """η_diff = (D_r·η_obs)² / (θ·d)²  (Eq. 7) — the shared space-FSO diffraction model."""
        return diffraction_efficiency(self.rx_diameter_m, self.obs_transmissivity,
                                      self.beam_divergence_rad, distance_km)

    def _rx_chain(self) -> float:
        """Receiver-side product η_pde·η_rx·η_cpl·η_pat (distance-independent)."""
        return self.eta_pde * self.eta_rx * self.eta_cpl * self.eta_pat

    def channel_efficiency(self, distance_km: float) -> float:
        """Total ISQL efficiency η (Eq. 6 without η_atm, η_turb) — vacuum, weather-immune."""
        return min(1.0, self.diffraction_efficiency(distance_km) * self._rx_chain())

    def skr_bps(self, distance_km: float, illumination: str = "umbra") -> float:
        """ISL secret-key rate (bps). ``illumination`` in {umbra, penumbra, full_sun}."""
        eta = self.channel_efficiency(distance_km)
        qber = _QBER_BY_ILLUMINATION.get(illumination, 0.015)
        r_eff = self.source_rate_hz * self.source_eff
        return max(0.0, r_eff * eta * (1.0 - 2.0 * binary_entropy(qber)))
