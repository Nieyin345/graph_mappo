# -*- coding: utf-8 -*-
"""Review fixes for the v7 channel model (2026-10-07 external review).

Covers: (1) aperture averaging (Karakosta 2025 B3/B4/B5) in HAP-GS; (2) intra-slot
sub-stepping geometry resolution. Runs without weather files or datasets.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("tools/channel_v2", "tools/dataset_v2"):
    sys.path.insert(0, str(ROOT / sub))
sys.path.insert(0, str(ROOT))

from tools.channel_v7.model import (ParamsV7, RateModelV7, bounded_collection,
                                    correlation_width, aperture_averaged_scint, saturating_scint)


def _slant(elev_deg, h_m):
    R_E = 6371e3
    e = np.radians(np.asarray(elev_deg, float))
    return np.sqrt((R_E + h_m) ** 2 - (R_E * np.cos(e)) ** 2) - R_E * np.sin(e)


def _states(lt, elev, h_m, p=None):
    p = p or ParamsV7()
    m = RateModelV7(p, sat_alt_km=550.0, hap_alt_km=20.0)
    L = _slant(elev, h_m)
    return m.optical_states(lt, L, np.asarray(elev, float), A=np.full_like(np.asarray(elev, float), p.a_night),
                            aod=np.full_like(np.asarray(elev, float), 0.1),
                            sun=np.full_like(np.asarray(elev, float), -10.0),
                            high_cloud=np.zeros_like(np.asarray(elev, float)), tau_cirrus=0.0), p


def test_correlation_width_is_sqrt_lambda_L_only():
    # ρ_c = √(λL) (Karakosta B3). The strong-branch B4 is deliberately NOT used (it is
    # discontinuous with B3 at V_R=1 and would make the aperture decision jump).
    lam = ParamsV7().lam
    for L in (20e3, 77e3, 550e3, 1518e3):
        rho = correlation_width(np.array([L]), lam)[0]
        assert rho == pytest.approx(np.sqrt(lam * L), rel=1e-12)


def test_sigma_I2_is_continuous_across_V_R_eq_1():
    # The previous B3/B4 hard switch made σ_I² jump at V_R=1 for SAT-GS geometry.
    # With ρ_c=√(λL) fixed, the composite σ_I² must be CONTINUOUS everywhere
    # (the point-receiver branch, ap=False, must also be monotone non-decreasing).
    lam, D = ParamsV7().lam, 0.6
    for L in (20e3, 550e3, 1518e3):
        rho = float(correlation_width(np.array([L]), lam)[0])
        ap = D > rho
        vrs = np.linspace(0.02, 60, 6000)
        si = np.array([float(np.asarray(aperture_averaged_scint(np.array([v]), np.array([L]), D, lam))[0])
                       if ap else min(v, 1.0) for v in vrs])
        d = np.diff(si)
        assert np.abs(d).max() < 0.02, f"σ_I² jumps at V_R~{vrs[np.argmax(np.abs(d))+1]:.3f} (L={L})"
        if not ap:                        # point-receiver saturation closure is monotone
            assert d.min() > -1e-6, f"point σ_I² drops at V_R~{vrs[np.argmin(d)+1]:.3f}"


def test_aperture_averaged_b5_reproduces_karakosta_formula():
    # Karakosta 2025 eq. (B5)/(B6): manual recomputation
    lam, L, D, vr = 1550e-9, 20e3, 0.6, 0.04
    k = 2 * np.pi / lam
    d2 = k * D ** 2 / (4 * L)
    b0 = vr
    t1 = 0.49 * b0 / (1 + 0.18 * d2 + 0.56 * b0 ** (6 / 5)) ** (7 / 6)
    t2 = 0.51 * b0 * (1 + 0.69 * b0 ** (6 / 5)) ** (-5 / 6) / (1 + 0.90 * d2 + 0.62 * d2 * b0 ** (6 / 5))
    want = np.exp(t1 + t2) - 1
    got = aperture_averaged_scint(np.array([vr]), np.array([L]), D, lam)[0]
    assert got == pytest.approx(want, rel=1e-12)
    assert got < vr  # aperture averaging reduces the index


def test_beta_collection_preserves_mean_after_aperture_averaging():
    # v7.5: PDT must preserve m_c exactly; aperture averaging changes SI, not the mean.
    m_c = np.array([0.95])
    si_ap = aperture_averaged_scint(np.array([0.04]), np.array([20e3]), 0.6, 1550e-9)
    nodes, weights, _, e_ap, e2_ap = bounded_collection(m_c, np.asarray(si_ap), 128)
    assert float(e_ap[0]) == pytest.approx(0.95)
    assert float(e2_ap[0]) == pytest.approx(float(m_c[0] ** 2 * (1 + si_ap[0])))
    assert np.sum(weights[:, 0] * nodes[:, 0]) == pytest.approx(0.95, rel=2e-5)


def test_hap_gs_uses_aperture_averaging_sat_gs_does_not():
    st_hap, p = _states("HAP-GS", [90.0], 20e3)
    assert bool(np.ravel(st_hap["aperture_active"])[0])          # D=0.6 > rho_c~0.18
    si = float(np.ravel(st_hap["scint_index"])[0])
    vr = float(np.ravel(st_hap["rytov"])[0])
    assert si < vr                                               # aperture-averaged < point Rytov

    st_sat, _ = _states("SAT-GS", [90.0], 550e3)
    assert not bool(np.ravel(st_sat["aperture_active"])[0])      # D=0.6 < rho_c~0.92
    assert float(np.ravel(st_sat["scint_index"])[0]) == float(np.ravel(st_sat["rytov"])[0])


def test_aperture_averaging_flag_reverts_to_point_rytov():
    p = ParamsV7(aperture_averaging=False)
    st, _ = _states("HAP-GS", [90.0], 20e3, p=p)
    assert not bool(np.ravel(st["aperture_active"])[0])
    # with the flag off, HAP-GS scint index is just the saturated point Rytov (== V_R here, V_R<1)
    assert float(np.ravel(st["scint_index"])[0]) == float(np.ravel(st["rytov"])[0])


def test_saturating_scint_is_only_strong_regime_closure():
    vr = np.array([0.01, 0.9, 1.0, 5.0, 38.0])
    si = saturating_scint(vr)
    np.testing.assert_allclose(si[:3], vr[:3])                   # weak regime unchanged
    np.testing.assert_allclose(si[3:], 1.0)                      # capped at the point-receiver limit
