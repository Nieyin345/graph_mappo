"""Channel / key-rate model v2 (paper §2 second draft), standalone.

This module is a *validation implementation* of the rate model written in
docs/论文草稿_第2节_系统与信道模型.md (second draft, 2026-09-29).  It does NOT touch
the production rate code (adapters/unified_channel.py) or dataset/global/.

Conventions
-----------
* Links are directed from the higher node to the lower node (downlink convention):
  SAT->GS, HAP->GS, SAT->HAP, SAT<->SAT.
* All functions are vectorised over numpy arrays (one link, many time steps).
* Units: metres, radians unless the name says otherwise; rates in bit/s.

Literature (location in the source; local copy under 文献库/<short name>/)
------------------------------------------------------------------------
Each function below repeats the reference it implements.  Full table:
docs/论文草稿_第2节_系统与信道模型.md, "引用位置对照表".
  eta_bp      Andrews & Phillips 2005, ch. 12 (chapter-level; doi 10.1117/3.626196.ch12) downlink W_LT and T;
              same form: Shivkant et al. 2025 arXiv:2507.20908 eq. (25)
              Vasylyev 2019 PRA 99 053830, eq. (41)-(43)           Vasylyev-2019-sat-atmospheric-links
              Farid & Hranilovic 2007 JLT, eq. (9)-(11) (consistency) Farid-2007-pointing-errors
  r0          Fidler & Wallner 2008 EURASIP JWCN 2008:864031 eq. (3); Shivkant 2025 eq. (14)
  eta_phi     Karakosta 2025, App. C.1 eq. (C11)-(C12) (eta0)      Karakosta-2025-balloon-qkd-model
              Noll 1976 JOSA 66(3), Table IV (Delta_3 = 0.134); Fidler & Wallner 2008 eq. (8)
              eta_phi/eta0 (default, 09-30): Monte-Carlo table of the pupil-plane overlap integral
              (Fidler & Wallner 2008 eq. (1)), Kolmogorov screens with subharmonics (Lane et al. 1992),
              tip/tilt removed; generator tools/channel_v2/gen_smf_table.py (3000 screens, 1024^2)
              old closed form (smf_model="core_halo"): core exp(-s2) Fidler 2008 eq. (7), halo ibid. eq. (6),
              split Tokovinin AO tutorial part 2 eq. (1)-(2)
  HVIntegrals Hufnagel-Valley with C_n^2(0) exp(-h/100 m) ground term: Fidler & Wallner 2008 eq. (4);
              Liorni 2019 NJP 21 093055, App. E eq. (E.1)
  eta_mol     Hansen & Travis 1974 (Rayleigh; as quoted in Bodhaine et al. 1999 JAOT 16:1854, eq. (15));
              Kasten & Young 1989 (air mass); Chu 2020 §4.2 (0.01 dB/km at 1550 nm) x H_m -> tau_abs
  eta_aer     Open-Meteo / CAMS AOD550 (Zippenfenig 2023), Angstrom law
  p_clear_*   Lund & Shanklin 1973 JAM 12:28; matrix from Lund, Grantham & Elam 1978 (AFGL-TR-78-0276) Table 1
  y0          2025-wavelength-sat-qkd eq. (7)-(8); Liorni 2019 App. C eq. (C.2)-(C.3);
              Haeusler et al. 2024 Opt. Eng. 63(4) 041211 eqs. (4)-(8) (SMF etendue beta^2 lambda^2)
  skr         GLLP decoy BB84 (Lo 2005; Ma 2005), same form as 2025-wavelength-sat-qkd eq. (9)-(14)
  slot mean   Vasylyev 2019 eq. (67)-(69) (average counts first; D1)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8

# Lund & Shanklin (1973) C-matrix, values from Lund, Grantham & Elam (1978), Table 1.
LS_ELEV = np.array([10.0, 30.0, 90.0])
LS_TABLE = np.array([
    [0.97, 0.86, 0.76, 0.65, 0.55, 0.47, 0.39, 0.32, 0.24, 0.16, 0.03],   # 10 deg
    [0.98, 0.93, 0.86, 0.80, 0.73, 0.66, 0.57, 0.50, 0.38, 0.24, 0.06],   # 30 deg
    [1.00, 0.97, 0.92, 0.87, 0.81, 0.77, 0.70, 0.62, 0.48, 0.31, 0.08],   # 90 deg
])


@dataclass
class ParamsV2:
    """Table 2 initial values of the second draft (all are assumptions, to be tuned)."""
    lam: float = 1550e-9
    # W0, D: Karakosta 2025 Table 2 (10 cm; 0.4/0.3 m), StratoQ 2026 Table 2 (15 cm; 0.75/0.4 m)
    # 09-29 decision: HAP/SAT receiving aperture 0.3 -> 0.5 m so that SAT-HAP and SAT-SAT links are
    # usable (12-day check: positive fraction SAT-HAP .20 -> .72, SAT-SAT .05 -> 1.00); W0 kept at 0.10 m
    # so the SAT-GS calibration against Micius / Sidhu 2022 is unchanged.
    w0: Dict[str, float] = field(default_factory=lambda: {"HAP": 0.10, "SAT": 0.10})
    d_rx: Dict[str, float] = field(default_factory=lambda: {"GS": 0.6, "HAP": 0.5, "SAT": 0.5})
    # Micius tracking 1.2 urad (Liao 2017 Fig. 2b; Vasylyev 2019 text ref. [99]);
    # Karakosta 2025 Table 1: 1 urad; Chu 2020 Table I: HAP 5 urad
    sigma_theta: Dict[str, float] = field(default_factory=lambda: {"HAP": 2.0e-6, "SAT": 1.2e-6})
    eta_tx_opt: float = 0.8         # Farid & Hranilovic 2007 JLT, parameter table (eta_T = eta_R = 0.8)
    # receiver optics incl. filters: 2025-wavelength-sat-qkd, text after eq. (1) and §5 (eta_opt = 0.45)
    eta_rx_opt: float = 0.45
    # (efficiency, dark count rate [Hz]); SNSPD .85 / SPAD .25: Karakosta 2025 Table 1;
    # MoSi 87 %/100 cps: 2025-wavelength-sat-qkd §5; SPAD 25 %: Chu 2020 §4.5;
    # 200 Hz overall dark count: JOCN 2026 Table 1 (09-29 decision: SPAD 1 kHz -> 200 Hz,
    # cooled free-running InGaAs class)
    detector: Dict[str, tuple] = field(default_factory=lambda: {
        "GS": (0.85, 100.0), "HAP": (0.25, 200.0), "SAT": (0.25, 200.0)})
    eta0_smf: float = 0.81          # Karakosta 2025 App. C.1 eq. (C11), beta = 1.12 (no obstruction)
    delta3: float = 0.134           # Noll 1976 Table IV: residual after piston + tip/tilt (Fidler 2008 eq. (8))
    # 09-30: "mc_table" (default) = overlap-integral table SMF_TT_TABLE; "core_halo" = old eq. (7a)
    smf_model: str = "mc_table"
    # zenith optical depth = 0.01 dB/km (Chu 2020 §4.2, 1550 nm) x H_m 8.4 km / 4.343 = 0.0193
    tau_abs: float = 0.0193
    h_mol_km: float = 8.4
    angstrom: float = 1.3           # Angstrom 1964 Tellus 16:64 average for natural atmospheres (1.3 +- 0.5)
    hv_wind: float = 21.0           # HV-5/7 (Andrews 2005 ch. 12; Liorni 2019 text after eq. (E.1))
    a_day: float = 2.75e-14         # fallback near-ground Cn2 (Liorni 2019 App. E, after eq. (E.1)) when no ERA5 flux
    a_night: float = 1.10e-14
    # Monin-Obukhov near-ground Cn2 from ERA5 fluxes, paper eqs. (14)-(16)
    z_ref: float = 10.0             # ERA5 lowest model level ~10 m (ECMWF FUG §2.1.5.2); sensitivity 5 / 20 m
    cn2_min: float = 1e-16          # numerical floor for H -> 0; < 3 % of the HV background path integral
    zeta_max: float = 10.0          # stable-side validity limit of the fitted f_T (CASES-99 range, Hartogensis & De Bruin 2005)
    lb_day: float = 6.85e-5         # W m^-2 sr^-1 nm^-1, clear day sky at 1550 nm (Khanna 2024 Table 1)
    lb_night: float = 1.5e-7        # moonlit clear night (Khanna 2024 Table 1; cf. Liorni 2019 Table E3)
    # sky radiance above 20 km / ground: molecular column fraction p(20 km)/p0 = 5474.9/101325
    # (US Standard Atmosphere 1976); aerosol above 20 km neglected -> upper bound
    kappa_strat: float = 0.054
    dlam_nm: float = 1.0
    gate_s: float = 1e-9
    fov_rad: float = 100e-6         # Liorni 2019 Table E3; Miao 2005 §2.3
    # f_rep, mu, nu, e_d, f_ec, q: 2025-wavelength-sat-qkd §5; Micius 100 MHz (Liao 2017)
    f_rep: float = 100e6
    mu: float = 0.7
    e_d: float = 0.01
    f_ec: float = 1.22
    q_sift: float = 0.5
    n_det: int = 4
    # Lumped elevation-dependent excess loss [dB per unit air mass above zenith], paper eq. (11a):
    # L_x = kappa_x * max(m(alpha) - 1, 0), applied to G-S and G-H only.
    #   form: extinction proportional to air mass, Khmelev et al. 2023 (Entropy 25(4) 670, eqs. (5)-(6));
    #   anchor: Liao 2017 main text, 1200 km (elev 19.8 deg, m = 2.93): atmospheric absorption + turbulence 3-8 dB;
    #   site: Xinglong winter AOD440 = 0.19, EAE 1.07 (Zhu et al. 2014, Atmos. Environ. 84, 54-64, abstract);
    #   this model at 848.6 nm with that AOD (tau_abs 0.0193, AE 1.07): 1.66 dB -> residual 1.34 / 3.84 / 6.34 dB
    #   -> kappa(848.6 nm) = 0.70 / 1.99 / 3.29 dB per air mass;
    #   1550 nm: x (848.6/1550)^(7/6) = 0.495 (Rytov variance sigma_R^2 = 1.23 Cn2 k^(7/6) L^(11/6),
    #   Liorni 2019 App. A) -> 0.34 / 0.99 / 1.63. Adopted 1.0 (user decision 09-29); sensitivity 0 / 0.34 / 1.63.
    #   (before 09-29 night: tau_abs 0.01 -> 0.38 / 1.02 / 1.66)
    kappa_x_db: float = 1.0

    @property
    def k(self) -> float:
        return 2.0 * math.pi / self.lam


# ----------------------------------------------------------------------------- SMF coupling table
# Mean SMF coupling of a tip/tilt-corrected receiver normalised to the turbulence-free value,
# eta_phi/eta0 versus D/r0. Pupil-plane overlap integral (Fidler & Wallner 2008 eq. (1)),
# Gaussian fibre mode with the optimum coupling parameter (eta0 = 0.8145, Karakosta 2025 eq. (C11)),
# Kolmogorov FFT phase screens with 3 subharmonic levels (Lane, Glindemann & Dainty 1992),
# piston/tip/tilt removed by least squares (residual variance / Noll 0.134 (D/r0)^(5/3) = 1.014).
# 3000 screens on 1024^2 (pupil 256 px); relative standard error <= 2 %; 512^2 agrees within it.
# Generated by tools/channel_v2/gen_smf_table.py (python gen_smf_table.py 3000 1024), 2026-09-30.
SMF_TT_TABLE = (
    (0.25, 9.8789e-01),
    (0.5, 9.6214e-01),
    (0.75, 9.2717e-01),
    (1, 8.8547e-01),
    (1.25, 8.3898e-01),
    (1.5, 7.8932e-01),
    (2, 6.8581e-01),
    (2.5, 5.8356e-01),
    (3, 4.8821e-01),
    (3.5, 4.0298e-01),
    (4, 3.2921e-01),
    (5, 2.1543e-01),
    (6, 1.3963e-01),
    (7, 9.1267e-02),
    (8, 6.1109e-02),
    (10, 3.0712e-02),
    (12, 1.8164e-02),
    (14, 1.2101e-02),
    (16, 8.6766e-03),
    (20, 5.1105e-03),
    (24, 3.3967e-03),
    (28, 2.4222e-03),
    (32, 1.7762e-03),
    (40, 1.0935e-03),
    (48, 7.3323e-04),
    (56, 5.2162e-04),
    (64, 3.9498e-04),
)
_SMF_LD = np.log(np.array([0.0 + r[0] for r in SMF_TT_TABLE]))
_SMF_LV = np.log(np.array([r[1] for r in SMF_TT_TABLE]))


def smf_tt_ratio(x):
    """eta_phi/eta0 at D/r0 = x: log-log interpolation of SMF_TT_TABLE.
    Below the first node: Marechal on the tip/tilt residual, exp(-0.134 x^(5/3)) (s2 < 0.01, exact limit);
    above the last node (D/r0 > 64): (r0/D)^2 tail, the fitted local slope of the last nodes is -1.9."""
    x = np.asarray(x, dtype=float)
    lx = np.log(np.maximum(x, 1e-12))
    out = np.exp(np.interp(lx, _SMF_LD, _SMF_LV))
    lo = x < SMF_TT_TABLE[0][0]
    hi = x > SMF_TT_TABLE[-1][0]
    out = np.where(lo, np.exp(-0.134 * np.maximum(x, 0.0) ** (5 / 3)), out)
    out = np.where(hi, SMF_TT_TABLE[-1][1] * (SMF_TT_TABLE[-1][0] / np.maximum(x, 1e-12)) ** 2, out)
    return out


# ----------------------------------------------------------------------------- helpers
def h2(x):
    x = np.clip(x, 1e-12, 0.5)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def rayleigh_tau(lam_um: float) -> float:
    """Hansen & Travis (1974) Rayleigh zenith optical depth at sea level, paper eq. (9);
    formula as quoted in Bodhaine et al. 1999 (JAOT 16:1854), eq. (15)."""
    return 0.008569 * lam_um ** -4 * (1 + 0.0113 * lam_um ** -2 + 0.00013 * lam_um ** -4)


def cn2_monin_obukhov(H_up, ustar, T_K, P_Pa, z=10.0, cn2_min=1e-16, zeta_max=10.0):
    """Near-ground Cn2 [m^-2/3] from surface fluxes, paper eqs. (14)-(16).

    H_up: sensible heat flux, upward positive [W m-2] (ERA5 sshf is downward positive: H_up = -sshf/3600).
    f_T coefficients: Kooijmans & Hartogensis 2016 (Boundary-Layer Meteorol. 160, 501-527), Table 2,
      unstable 5.6 (1 - 6.5 zeta)^(-2/3), stable 5.5 (1 + 1.1 zeta^(2/3)); base forms their eqs. (19)-(20),
      after Wyngaard et al. 1971. Cn2 = (79e-6 P[hPa] / T^2)^2 CT2 (e.g. Solignac et al. 2009 AMT eq. (4),
      humidity term neglected).
    """
    H = np.asarray(H_up, dtype=float)
    us = np.maximum(np.asarray(ustar, dtype=float), 1e-3)
    T = np.asarray(T_K, dtype=float)
    P = np.asarray(P_Pa, dtype=float)
    rho_cp = P / (287.05 * T) * 1005.0
    t_star = -H / (rho_cp * us)
    Hs = np.where(np.abs(H) < 1e-6, 1e-6, H)
    L_o = -rho_cp * T * us ** 3 / (0.4 * 9.81 * Hs)
    zeta = np.clip(z / L_o, -1e3, zeta_max)
    f = np.where(zeta < 0, 5.6 * (1.0 - 6.5 * np.minimum(zeta, 0.0)) ** (-2.0 / 3.0),
                 5.5 * (1.0 + 1.1 * np.maximum(zeta, 0.0) ** (2.0 / 3.0)))
    ct2 = t_star ** 2 * z ** (-2.0 / 3.0) * f
    cn2 = (79e-6 * (P / 100.0) / T ** 2) ** 2 * ct2
    return np.maximum(np.nan_to_num(cn2, nan=cn2_min), cn2_min)


def airmass_kasten_young(elev_deg):
    """Kasten & Young (1989) Appl. Opt. 28(22) approximation formula; paper eq. (8)."""
    e = np.maximum(np.asarray(elev_deg, dtype=float), 0.0)
    return 1.0 / (np.sin(np.radians(e)) + 0.50572 * (e + 6.07995) ** -1.6364)


def sun_elevation_deg(t_min, lat_deg, lon_deg):
    """NOAA approximation. t_min = minutes since 2023-01-01 00:00 UTC."""
    t_min = np.asarray(t_min, dtype=float)
    day = t_min / 1440.0
    g = 2 * np.pi / 365.0 * (day - 0.5)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    minute_utc = np.mod(t_min, 1440.0)
    tst = minute_utc + eqt + 4.0 * np.asarray(lon_deg, dtype=float)
    ha = np.radians(tst / 4.0 - 180.0)
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    cz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    return np.degrees(np.arcsin(np.clip(cz, -1, 1)))


def p_clear_lund_shanklin(elev_deg, cloud_frac):
    """Eq. (12): (1-C) * C_LS(alpha, 10C) / C_LS(90, 10C).
    Elevation dependence from Lund & Shanklin (1973); matrix: Lund et al. (1978) Table 1."""
    elev = np.clip(np.asarray(elev_deg, dtype=float), 10.0, 90.0)
    c = np.clip(np.asarray(cloud_frac, dtype=float), 0.0, 1.0)
    c10 = c * 10.0
    i0 = np.clip(np.floor(c10).astype(int), 0, 9)
    wc = c10 - i0
    col = lambda row: LS_TABLE[row, i0] * (1 - wc) + LS_TABLE[row, i0 + 1] * wc
    v10, v30, v90 = col(0), col(1), col(2)
    lo = elev <= 30.0
    w_lo = (elev - 10.0) / 20.0
    w_hi = (elev - 30.0) / 60.0
    v = np.where(lo, v10 * (1 - w_lo) + v30 * w_lo, v30 * (1 - w_hi) + v90 * w_hi)
    return (1.0 - c) * v / v90


# ----------------------------------------------------------------------------- turbulence
class HVIntegrals:
    """Linear-in-A decomposition of the HV integrals used by r0 and the long-term beam.

    HV profile, paper eq. (17): Fidler & Wallner 2008 eq. (4) (with C_n^2(0) e^{-h/100} ground term);
    Liorni 2019 App. E eq. (E.1).
    mu_2d weight ((h-h0)/(H-h0))^{5/3} is the Andrews downlink weight (zero at the receiver).

    For a receiver at h0 = 0 and a transmitter at altitude H:
        int_0^H Cn2 dh                        = A*c_a + c_b
        int_0^H Cn2 (h/H)^{5/3} dh (mu_2d)    = A*m_a + m_b
    """

    def __init__(self, h_top_m: float, wind: float = 21.0, n: int = 200001):
        h = np.linspace(0.0, min(h_top_m, 30e3), n)
        rest = (0.00594 * (wind / 27.0) ** 2 * (1e-5 * h) ** 10 * np.exp(-h / 1000.0)
                + 2.7e-16 * np.exp(-h / 1500.0))
        a_term = np.exp(-h / 100.0)
        w = (h / h_top_m) ** (5.0 / 3.0)
        self.c_a = np.trapezoid(a_term, h)
        self.c_b = np.trapezoid(rest, h)
        self.m_a = np.trapezoid(a_term * w, h)
        self.m_b = np.trapezoid(rest * w, h)
        self.h_top = h_top_m


# ----------------------------------------------------------------------------- model
class RateModelV2:
    def __init__(self, p: ParamsV2 | None = None, sat_alt_km: float = 567.0, hap_alt_km: float = 20.0):
        self.p = p or ParamsV2()
        self.hv_sat = HVIntegrals(sat_alt_km * 1e3, self.p.hv_wind)
        self.hv_hap = HVIntegrals(hap_alt_km * 1e3, self.p.hv_wind)
        self.tau_r = rayleigh_tau(self.p.lam * 1e6)
        self.h_nu = H_PLANCK * C_LIGHT / self.p.lam

    # --- beam propagation, paper eq. (2)-(4)
    # W_LT^2 = W_d^2 (1+T), T = 4.35 mu_2d Lam^{5/6} k^{7/6} H^{5/6} sec^{11/6}:
    #   Andrews & Phillips 2005 ch. 12 (downlink; chapter-level), same form Shivkant et al. 2025
    #   arXiv:2507.20908 eq. (25) with Lambda = 2L/(k W^2), Z = h0 + L
    # <eta> = 1 - exp(-2a^2/W^2): Vasylyev 2019 eq. (43); pointing adds 4 sigma_r^2 to W^2
    #   (mean of Farid & Hranilovic 2007 eq. (9)-(11) for a << W)
    def eta_bp(self, L, tx, rx, elev_deg=None, A=None, hv: HVIntegrals | None = None):
        p = self.p
        k = p.k
        w0 = p.w0[tx]
        a = p.d_rx[rx] / 2.0
        lam0 = 2.0 * L / (k * w0 ** 2)
        wd2 = w0 ** 2 * (1.0 + lam0 ** 2)
        T = 0.0
        if rx == "GS":
            Lam = 2.0 * L / (k * wd2)
            sec = 1.0 / np.sin(np.radians(np.maximum(elev_deg, 1.0)))
            mu2d = A * hv.m_a + hv.m_b
            T = 4.35 * mu2d * Lam ** (5 / 6) * k ** (7 / 6) * hv.h_top ** (5 / 6) * sec ** (11 / 6)
        wlt2 = wd2 * (1.0 + T)
        sig = p.sigma_theta[tx] * L
        return 1.0 - np.exp(-2.0 * a ** 2 / (wlt2 + 4.0 * sig ** 2)), T

    def r0(self, elev_deg, A, hv: HVIntegrals):
        """Plane-wave Fried parameter, paper eq. (6): Fidler & Wallner 2008 eq. (3)."""
        sec = 1.0 / np.sin(np.radians(np.maximum(elev_deg, 1.0)))
        integ = A * hv.c_a + hv.c_b
        return (0.423 * self.p.k ** 2 * sec * integ) ** (-3.0 / 5.0)

    def eta_phi(self, elev_deg, A, hv):
        """SMF coupling after tip/tilt correction, eq. (7).

        Default (smf_model="mc_table"): eta0 * table value, see SMF_TT_TABLE.
        smf_model="core_halo" (paper draft eq. (7a) until 09-30), coherent core + uncorrected halo:
            S = exp(-s2) + (1 - exp(-s2)) [1 + (D/r0)^{5/3}]^{-6/5},  s2 = Delta_3 (D/r0)^{5/3}.
        The pure exp(-s2) form is only valid for s2 < ~1 and gives unphysical
        -30..-60 dB at low elevation; the halo term keeps S ~ (r0/D)^2 there.
        Refs: eta0 Karakosta 2025 App. C.1 eq. (C11); Delta_3 Noll 1976 Table IV;
        core Fidler & Wallner 2008 eqs. (7)(8); halo ibid. eq. (6);
        split Tokovinin AO tutorial part 2 eqs. (1)(2); paper eq. (7), (7a).
        (Until 09-29 the halo used 1/(1+(D/r0)^2): same limits, < 0.3 dB different.)
        """
        r0 = self.r0(elev_deg, A, hv)
        x = self.p.d_rx["GS"] / r0
        if self.p.smf_model == "mc_table":
            # 09-30 default: overlap-integral table (see SMF_TT_TABLE). The closed form below is
            # within 0.5 dB for D/r0 <= 4 but 1.6-4.6 dB too pessimistic for D/r0 = 6-12.
            return self.p.eta0_smf * smf_tt_ratio(x), r0
        if self.p.smf_model != "core_halo":
            raise ValueError(self.p.smf_model)
        s2 = self.p.delta3 * x ** (5 / 3)
        core = np.exp(-s2)
        strehl = core + (1.0 - core) * (1.0 + x ** (5 / 3)) ** (-6 / 5)
        return self.p.eta0_smf * strehl, r0

    def eta_mol(self, elev_deg, h_top_km):
        """Paper eq. (10): (tau_R + tau_abs) * column fraction * Kasten-Young air mass."""
        p = self.p
        tau = (self.tau_r + p.tau_abs) * (1.0 - math.exp(-h_top_km / p.h_mol_km))
        return np.exp(-tau * airmass_kasten_young(elev_deg))

    def eta_aer(self, elev_deg, aod550):
        """Paper eq. (11): CAMS AOD550 (Open-Meteo AQ API) scaled by the Angstrom law."""
        p = self.p
        tau = np.asarray(aod550) * (p.lam * 1e9 / 550.0) ** (-p.angstrom)
        return np.exp(-tau * airmass_kasten_young(elev_deg))

    # --- background, paper eq. (20)-(21)
    # Y0 = eta_rx eta_det N_b + n_det f_dark dt: 2025-wavelength-sat-qkd eq. (8)
    # N_b = L_b G dlam dt / h nu with G = Omega pi a^2 (free space; Liorni 2019 eq. (C.2)-(C.3))
    #   or G = beta^2 lambda^2, beta = 1.12 (SMF ground receiver, independent of D;
    #   Haeusler et al. 2024 Opt. Eng. 63(4) 041211, eqs. (4)-(8); same beta as eta0)
    def y0(self, rx, lb):
        p = self.p
        eta_det, f_dark = p.detector[rx]
        if rx == "GS":
            etendue = (1.12 * p.lam) ** 2
        else:
            etendue = math.pi * (p.fov_rad / 2) ** 2 * math.pi * (p.d_rx[rx] / 2) ** 2
        n_b = np.asarray(lb) * etendue * p.dlam_nm * p.gate_s / self.h_nu
        return p.eta_rx_opt * eta_det * n_b + p.n_det * f_dark * p.gate_s

    # --- protocol, paper eq. (22): asymptotic decoy BB84 (GLLP; Lo 2005, Ma 2005),
    #     same form as 2025-wavelength-sat-qkd eq. (9)-(14)
    def skr(self, eta, y0):
        p = self.p
        eta = np.asarray(eta, dtype=float)
        y0 = np.broadcast_to(np.asarray(y0, dtype=float), eta.shape)
        mu = p.mu
        gain = 1.0 - np.exp(-eta * mu)
        q_mu = y0 + gain
        e_mu = (0.5 * y0 + p.e_d * gain) / q_mu
        q1 = mu * np.exp(-mu) * (y0 + eta)
        e1 = (0.5 * y0 + p.e_d * eta) / (y0 + eta)
        r = p.q_sift * (q1 * (1 - h2(e1)) - p.f_ec * q_mu * h2(e_mu))
        return p.f_rep * np.maximum(r, 0.0)

    # --- one link over many steps
    def link_rate(self, ltype, L_m, elev_deg, *, sun_rx_deg=None, aod550=None, a_cn2=None,
                  return_parts=False):
        """Clear-sky slot rate (cloud handled by the caller). ltype in
        {'SAT-GS','HAP-GS','SAT-HAP','SAT-SAT'}; transmitter = higher node."""
        p = self.p
        parts = {}
        if ltype in ("SAT-GS", "HAP-GS"):
            tx = "SAT" if ltype == "SAT-GS" else "HAP"
            hv = self.hv_sat if tx == "SAT" else self.hv_hap
            h_top_km = 1e3 if tx == "SAT" else hv.h_top / 1e3
            bp, T = self.eta_bp(L_m, tx, "GS", elev_deg, a_cn2, hv)
            phi, r0 = self.eta_phi(elev_deg, a_cn2, hv)
            mol = self.eta_mol(elev_deg, h_top_km)
            aer = self.eta_aer(elev_deg, aod550)
            sys_ = p.eta_tx_opt * p.eta_rx_opt * p.detector["GS"][0]
            xs = 10.0 ** (-p.kappa_x_db * np.maximum(airmass_kasten_young(elev_deg) - 1.0, 0.0) / 10.0)
            eta = bp * phi * mol * aer * xs * sys_
            lb = np.where(np.asarray(sun_rx_deg) > 0.0, p.lb_day, p.lb_night)
            y0 = self.y0("GS", lb)
            parts.update(bp=bp, T=T, phi=phi, r0=r0, mol=mol, aer=aer, xs=xs, sys=sys_)
        elif ltype == "SAT-HAP":
            bp, _ = self.eta_bp(L_m, "SAT", "HAP")
            sys_ = p.eta_tx_opt * p.eta_rx_opt * p.detector["HAP"][0]
            eta = bp * sys_
            lb = np.where(np.asarray(sun_rx_deg) > 0.0, p.lb_day * p.kappa_strat, 0.0)
            y0 = self.y0("HAP", lb)
            parts.update(bp=bp, sys=sys_)
        elif ltype == "SAT-SAT":
            bp, _ = self.eta_bp(L_m, "SAT", "SAT")
            sys_ = p.eta_tx_opt * p.eta_rx_opt * p.detector["SAT"][0]
            eta = bp * sys_
            y0 = self.y0("SAT", 0.0)
            parts.update(bp=bp, sys=sys_)
        else:
            raise ValueError(ltype)
        rate = self.skr(eta, y0)
        if return_parts:
            parts.update(eta=eta, y0=y0)
            return rate, parts
        return rate
