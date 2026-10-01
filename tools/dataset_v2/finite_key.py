"""Finite-key secret key length of efficient two-decoy BB84, vectorised (paper App. C, eqs. (C2)-(C7)).

Security bounds: Lim, Curty, Walenta, Xu & Zbinden, PRA 89, 022307 (2014), eqs. (1)-(5), with the
multiplicative Chernoff bounds of Sidhu et al. 2022 (Sidhu-2022-sat-qkd-finite-key, eq. (7)) in place
of Hoeffding.  One block = one slot: N = f_rep * Delta pulses; every protocol parameter is fixed for all
links (paper Table I; optimised once at a -36 dB / Y0 = 8e-7 reference link, q/fk/fk_fixed2.py).

All functions work element-wise on numpy arrays of mean channel transmittance ``eta`` and background
(dark + stray light) detection probability ``y0``.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FiniteKeyParams:
    f_rep: float = 100e6          # pulse rate [Hz] (Liao 2017)
    slot_s: float = 60.0          # block = one slot (paper III-A)
    mu: tuple = (0.62, 0.24, 0.0)  # signal / decoy / vacuum intensities
    p_mu: tuple = (0.49, 0.40, 0.11)
    p_x: float = 0.71             # X-basis probability, same on both sides
    e_d: float = 0.01             # misalignment error
    f_ec: float = 1.16            # error-correction efficiency (Sidhu 2022)
    eps_sec: float = 1e-9
    eps_cor: float = 1e-15

    @property
    def n_pulses(self) -> float:
        return self.f_rep * self.slot_s

    def as_dict(self) -> dict:
        d = asdict(self)
        d["n_pulses"] = self.n_pulses
        return d


def h2(x):
    x = np.clip(x, 1e-15, 1.0 - 1e-15)
    return -x * np.log2(x) - (1.0 - x) * np.log2(1.0 - x)


def key_length(eta, y0, fk: FiniteKeyParams = FiniteKeyParams(), n_pulses=None):
    """Secret key length ell [bit] of one block (eq. (C7)); zero where no key can be extracted.

    ``n_pulses`` (scalar or array broadcastable to ``eta``) overrides the block size N = f_rep * Delta;
    the coverage cloud model passes N_eff = f * N, f = cloud-free fraction of the slot.  None keeps
    the original scalar N (bit-identical to the version used for dataset/global_v2)."""
    eta = np.asarray(eta, dtype=np.float64)
    y0 = np.broadcast_to(np.asarray(y0, dtype=np.float64), eta.shape)
    mu1, mu2, mu3 = fk.mu
    if mu3 != 0.0:
        raise NotImplementedError("the closed forms below assume a vacuum decoy (mu3 = 0)")
    N = fk.n_pulses if n_pulses is None else np.broadcast_to(np.asarray(n_pulses, dtype=np.float64), eta.shape)
    beta = math.log(21.0 / fk.eps_sec)
    tau0 = sum(p * math.exp(-m) for p, m in zip(fk.p_mu, fk.mu))
    tau1 = sum(p * math.exp(-m) * m for p, m in zip(fk.p_mu, fk.mu))
    wx, wz = fk.p_x ** 2, (1.0 - fk.p_x) ** 2

    def plus(y):   # Chernoff upper deviation, Sidhu 2022 eq. (7)
        return beta + np.sqrt(2.0 * beta * y + beta ** 2)

    def minus(y):  # Chernoff lower deviation
        return beta / 2.0 + np.sqrt(2.0 * beta * y + beta ** 2 / 4.0)

    n_x, n_z, m_x, m_z = [], [], [], []
    for p, m in zip(fk.p_mu, fk.mu):
        gain = 1.0 - np.exp(-eta * m)
        d = 1.0 - (1.0 - y0) * np.exp(-eta * m)       # detection probability D_k
        e = fk.e_d * gain + y0 / 2.0                    # error probability E_k
        n_x.append(N * wx * p * d)                      # eq. (C2)
        n_z.append(N * wz * p * d)
        m_x.append(N * wx * p * e)
        m_z.append(N * wz * p * e)

    def bounds(n):  # eq. (C3): n^- and n^+ per intensity, rescaled by e^k / p_k
        lo = [math.exp(m) / p * (x - minus(x)) for x, p, m in zip(n, fk.p_mu, fk.mu)]
        hi = [math.exp(m) / p * (x + plus(x)) for x, p, m in zip(n, fk.p_mu, fk.mu)]
        return lo, hi

    def s01(n):     # eq. (C4) with mu3 = 0
        lo, hi = bounds(n)
        s0 = tau0 * lo[2]    # Lim 2014 eq. (2); s1 uses this bound as is (eq. (3))
        s1 = tau1 * mu1 * (lo[1] - hi[2] - mu2 ** 2 / mu1 ** 2 * (hi[0] - s0 / tau0)) / (mu1 * mu2 - mu2 ** 2)
        return np.maximum(s0, 0.0), s1

    sx0, sx1 = s01(n_x)
    _, sz1 = s01(n_z)
    mlo, mhi = bounds(m_z)
    vz1 = tau1 * (mhi[1] - mlo[2]) / mu2                # eq. (C5)

    ok = (sx1 > 0) & (sz1 > 0)
    sx1s, sz1s = np.where(ok, sx1, 1.0), np.where(ok, sz1, 1.0)
    b = vz1 / sz1s
    okb = ok & (b > 0) & (b < 0.5)
    bs = np.where(okb, b, 0.25)
    # eq. (C6): Upsilon(eps_sec, b, s_Z1, s_X1)
    arg = (sz1s + sx1s) / (sz1s * sx1s * (1.0 - bs) * bs) * 21.0 ** 2 / fk.eps_sec ** 2
    ups = np.sqrt(np.maximum((sz1s + sx1s) * (1.0 - bs) * bs / (sz1s * sx1s * math.log(2.0)) * np.log2(arg), 0.0))
    phi = bs + ups
    okp = okb & (phi < 0.5)

    nx = sum(n_x)
    ex = sum(m_x) / nx
    lam_ec = fk.f_ec * nx * h2(ex)
    ell = (sx0 + sx1s * (1.0 - h2(np.where(okp, phi, 0.25))) - lam_ec
           - 6.0 * math.log2(21.0 / fk.eps_sec) - math.log2(2.0 / fk.eps_cor))
    return np.where(okp, np.maximum(np.floor(ell), 0.0), 0.0)


def slot_rate(eta, y0, fk: FiniteKeyParams = FiniteKeyParams(), n_pulses=None):
    """Slot key rate ell / Delta [bit/s] before the cloud indicator (paper eq. (18))."""
    return key_length(eta, y0, fk, n_pulses) / fk.slot_s
