"""Finite-key secret key length of efficient two-decoy BB84, vectorised (paper App. C, eqs. (C2)-(C7)).

Security bounds: Lim, Curty, Walenta, Xu & Zbinden, PRA 89, 022307 (2014), eqs. (1)-(5), with the
multiplicative Chernoff bounds of Sidhu et al. 2022 (Sidhu-2022-sat-qkd-finite-key, eq. (7)) in place
of Hoeffding. The 60 s scheduling slot is only a sampling/action interval. A QKD block may
aggregate sufficient statistics across multiple consecutive scheduling slots; the finite-key bound is
applied once when that logical block closes. Every protocol parameter is fixed for all links.

All functions work element-wise on numpy arrays of mean channel transmittance ``eta`` and background
(dark + stray light) detection probability ``y0``. When v7 supplies model-expected gains/counts,
the returned finite-key value is a performance estimate, not an actual-count security bound or device
security certificate; full composable security requires observed counts and a network-wide budget.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np


@dataclass(frozen=True)
class FiniteKeyParams:
    f_rep: float = 100e6          # pulse rate [Hz] (Liao 2017)
    slot_s: float = 60.0          # scheduling-slot duration; QKD block duration is independent
    mu: tuple = (0.62, 0.24, 0.0)  # signal / decoy / vacuum intensities
    p_mu: tuple = (0.49, 0.40, 0.11)
    p_x: float = 0.71             # X-basis probability, same on both sides
    e_d: float = 0.01             # misalignment error
    f_ec: float = 1.16            # error-correction efficiency (Sidhu 2022)
    eps_sec: float = 1e-9         # per-block epsilon_sec (fixed protocol comparison; spec §5.2)
    eps_cor: float = 1e-15        # per-block epsilon_cor (fixed protocol comparison; spec §5.2)
    # Spec §5.2 composable allocation over the declared service period: with a
    # block-count upper bound N_max, eps_{sec,b} = eps_sec_tot / N_max and
    # eps_{cor,b} = eps_cor_tot / N_max. When these are None the fixed
    # per-block pair above is used (the spec's permitted protocol comparison).
    eps_sec_total: float | None = None
    eps_cor_total: float | None = None
    n_max_blocks: int | None = None

    def __post_init__(self):
        if not np.isfinite(self.f_rep) or self.f_rep <= 0.0:
            raise ValueError("f_rep must be finite and positive")
        if not np.isfinite(self.slot_s) or self.slot_s <= 0.0:
            raise ValueError("slot_s must be finite and positive")

        mu = np.asarray(self.mu, dtype=np.float64)
        p_mu = np.asarray(self.p_mu, dtype=np.float64)
        if mu.shape != (3,) or p_mu.shape != (3,):
            raise ValueError("the implemented decoy bound requires exactly 3 intensities/probabilities")
        if np.any(~np.isfinite(mu)) or np.any(mu < 0.0):
            raise ValueError("mu must contain finite nonnegative intensities")
        if not (mu[0] > mu[1] > mu[2] and np.isclose(mu[2], 0.0, rtol=0.0, atol=1e-15)):
            raise ValueError("requires ordered signal > decoy > vacuum=0 intensities")
        if np.any(~np.isfinite(p_mu)) or np.any(p_mu <= 0.0) or not np.isclose(
            p_mu.sum(), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("p_mu must contain three positive probabilities summing to 1")
        if not np.isfinite(self.p_x) or not 0.0 < self.p_x < 1.0:
            raise ValueError("p_x must be finite in (0, 1)")
        if not np.isfinite(self.e_d) or not 0.0 <= self.e_d < 0.5:
            raise ValueError("e_d must be finite in [0, 0.5)")
        if not np.isfinite(self.f_ec) or self.f_ec < 1.0:
            raise ValueError("f_ec must be finite and >= 1")
        for name in ("eps_sec", "eps_cor"):
            v = float(getattr(self, name))
            if not 0.0 < v < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")

        if self.n_max_blocks is not None:
            if (
                isinstance(self.n_max_blocks, (bool, np.bool_))
                or int(self.n_max_blocks) != self.n_max_blocks
                or int(self.n_max_blocks) < 1
            ):
                raise ValueError("n_max_blocks must be a positive integer")
        for name in ("eps_sec_total", "eps_cor_total"):
            v = getattr(self, name)
            if v is not None and not (0.0 < float(v) < 1.0):
                raise ValueError(f"{name} must be in (0, 1) when set")
        if (
            self.eps_sec_total is not None or self.eps_cor_total is not None
        ) and self.n_max_blocks is None:
            raise ValueError(
                "a total composable security budget requires n_max_blocks"
            )

    @property
    def n_pulses(self) -> float:
        return self.f_rep * self.slot_s

    def per_block_eps(self) -> tuple[float, float]:
        """(eps_sec, eps_cor) applied to each block.

        Fixed per-block values by default; the spec §5.2 allocation
        ``eps_tot / N_max`` when ``eps_sec_total`` and ``n_max_blocks`` are set.
        """
        if self.eps_sec_total is None and self.eps_cor_total is None:
            return self.eps_sec, self.eps_cor
        if self.n_max_blocks is None:  # guarded by __post_init__; defensive only
            raise ValueError("total security budgets require n_max_blocks")
        n = int(self.n_max_blocks)
        es = (
            float(self.eps_sec_total) / n
            if self.eps_sec_total is not None else self.eps_sec
        )
        ec = (
            float(self.eps_cor_total) / n
            if self.eps_cor_total is not None else self.eps_cor
        )
        return max(es, 1e-300), max(ec, 1e-300)

    def effective(self) -> "FiniteKeyParams":
        """This parameter set with the per-block security budget resolved.

        Call this once at a public entry point so the Lim/Sidhu bounds use the
        allocated (or fixed) per-block epsilons.
        """
        es, ec = self.per_block_eps()
        if es == self.eps_sec and ec == self.eps_cor:
            return self
        return replace(self, eps_sec=es, eps_cor=ec)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["n_pulses"] = self.n_pulses
        d["eps_sec_per_block"], d["eps_cor_per_block"] = self.per_block_eps()
        d["eps_budget_source"] = (
            "allocated_tot_over_n_max"
            if (self.eps_sec_total is not None or self.eps_cor_total is not None)
            else "fixed_per_block"
        )
        return d


def h2(x):
    x = np.clip(x, 1e-15, 1.0 - 1e-15)
    return -x * np.log2(x) - (1.0 - x) * np.log2(1.0 - x)


def key_length(eta, y0, fk: FiniteKeyParams = FiniteKeyParams(), n_pulses=None):
    """Secret key length ell [bit] of one block (eq. (C7)); zero where no key can be extracted.

    ``n_pulses`` (scalar or array broadcastable to ``eta``) overrides the
    legacy one-slot block size N=f_rep*Delta. This entry point is retained for
    historical/standalone calculations; the v7.5 main path aggregates
    qkd_gain/error_gain statistics and calls ``key_length_gains`` at logical
    block closure.
    """
    fk = fk.effective()
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
        d = 1.0 - (1.0 - y0) * np.exp(-eta * m)       # Q_mu
        e = y0 / 2.0 + fk.e_d * (1.0 - y0) * gain     # T_mu, same convention as v7.5 spec §5.1
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
    ex = sum(m_x) / np.where(nx > 0, nx, 1.0)   # n=0 blocks: placeholder, masked by okp below
    lam_ec = fk.f_ec * nx * h2(ex)
    ell = (sx0 + sx1s * (1.0 - h2(np.where(okp, phi, 0.25))) - lam_ec
           - 6.0 * math.log2(21.0 / fk.eps_sec) - math.log2(2.0 / fk.eps_cor))
    return np.where(okp, np.maximum(np.floor(ell), 0.0), 0.0)


def slot_rate(eta, y0, fk: FiniteKeyParams = FiniteKeyParams(), n_pulses=None):
    """Legacy one-slot convenience wrapper; cloud/block timing is handled outside in v7.5."""
    return key_length(eta, y0, fk, n_pulses) / fk.slot_s


def key_length_states(eta_states, y0_states, n_states, fk: FiniteKeyParams = FiniteKeyParams()):
    """Secret key length ell [bit] of ONE block whose pulses are split into states.

    Legacy state-array interface: within a block, aggregate the per-basis/per-intensity expected
    counts over the block's states j (the Gauss-Hermite scintillation nodes, each
    carrying ``n_states[j]`` pulses at ``eta_states[j]`` and ``y0_states[j]``),
    then apply the Lim/Sidhu bounds ONCE on the aggregated statistics.  This is
    NOT the same as averaging per-state finite-key lengths (that path, still used
    by older code, is the ``sum_j w_j * ell(eta_j)`` approximation).

        nbar_{b,k} = p_b^2 p_mu_k * sum_j N_j Q_mu_k(eta_j, Y0_j)
        ell        = bounds( sum_j n_{X,k}, sum_j m_{X,k}, ... )

    ``eta_states`` / ``y0_states`` / ``n_states`` are arrays over states j sharing a
    leading axis of length J (any remaining axes are independent blocks).  Returns
    ell [bit] with the same trailing shape as the per-state arrays.
    """
    fk = fk.effective()
    eta_s = np.asarray(eta_states, dtype=np.float64)
    n_s = np.asarray(n_states, dtype=np.float64)
    y0_s = np.broadcast_to(np.asarray(y0_states, dtype=np.float64), eta_s.shape)
    mu1, mu2, mu3 = fk.mu
    if mu3 != 0.0:
        raise NotImplementedError("the closed forms below assume a vacuum decoy (mu3 = 0)")
    beta = math.log(21.0 / fk.eps_sec)
    tau0 = sum(p * math.exp(-m) for p, m in zip(fk.p_mu, fk.mu))
    tau1 = sum(p * math.exp(-m) * m for p, m in zip(fk.p_mu, fk.mu))
    wx, wz = fk.p_x ** 2, (1.0 - fk.p_x) ** 2

    def plus(y):
        return beta + np.sqrt(2.0 * beta * y + beta ** 2)

    def minus(y):
        return beta / 2.0 + np.sqrt(2.0 * beta * y + beta ** 2 / 4.0)

    # aggregate expected counts over the block's states j (sum over leading axis)
    n_x, n_z, m_x, m_z = [], [], [], []
    for p, m in zip(fk.p_mu, fk.mu):
        gain = 1.0 - np.exp(-eta_s * m)
        d = 1.0 - (1.0 - y0_s) * np.exp(-eta_s * m)   # Q_mu
        e = y0_s / 2.0 + fk.e_d * (1.0 - y0_s) * gain # T_mu, same convention as v7.5 spec §5.1
        n_x.append(np.sum(n_s * wx * p * d, axis=0))    # eq. (C2), summed over j
        n_z.append(np.sum(n_s * wz * p * d, axis=0))
        m_x.append(np.sum(n_s * wx * p * e, axis=0))
        m_z.append(np.sum(n_s * wz * p * e, axis=0))

    def bounds(n):
        lo = [math.exp(m) / p * (x - minus(x)) for x, p, m in zip(n, fk.p_mu, fk.mu)]
        hi = [math.exp(m) / p * (x + plus(x)) for x, p, m in zip(n, fk.p_mu, fk.mu)]
        return lo, hi

    def s01(n):
        lo, hi = bounds(n)
        s0 = tau0 * lo[2]
        s1 = tau1 * mu1 * (lo[1] - hi[2] - mu2 ** 2 / mu1 ** 2 * (hi[0] - s0 / tau0)) / (mu1 * mu2 - mu2 ** 2)
        return np.maximum(s0, 0.0), s1

    sx0, sx1 = s01(n_x)
    _, sz1 = s01(n_z)
    mlo, mhi = bounds(m_z)
    vz1 = tau1 * (mhi[1] - mlo[2]) / mu2

    ok = (sx1 > 0) & (sz1 > 0)
    sx1s, sz1s = np.where(ok, sx1, 1.0), np.where(ok, sz1, 1.0)
    b = vz1 / sz1s
    okb = ok & (b > 0) & (b < 0.5)
    bs = np.where(okb, b, 0.25)
    arg = (sz1s + sx1s) / (sz1s * sx1s * (1.0 - bs) * bs) * 21.0 ** 2 / fk.eps_sec ** 2
    ups = np.sqrt(np.maximum((sz1s + sx1s) * (1.0 - bs) * bs / (sz1s * sx1s * math.log(2.0)) * np.log2(arg), 0.0))
    phi = bs + ups
    okp = okb & (phi < 0.5)

    nx = sum(n_x)
    ex = sum(m_x) / np.where(nx > 0, nx, 1.0)
    lam_ec = fk.f_ec * nx * h2(ex)
    ell = (sx0 + sx1s * (1.0 - h2(np.where(okp, phi, 0.25))) - lam_ec
           - 6.0 * math.log2(21.0 / fk.eps_sec) - math.log2(2.0 / fk.eps_cor))
    return np.where(okp, np.maximum(np.floor(ell), 0.0), 0.0)


def slot_rate_states(eta_states, y0_states, n_states, fk: FiniteKeyParams = FiniteKeyParams()):
    """Legacy one-slot convenience wrapper; v7.5 main evaluation uses logical blocks."""
    return key_length_states(eta_states, y0_states, n_states, fk) / fk.slot_s


def key_length_gains(gains, errors, n_pulses, fk=FiniteKeyParams()):
    """v7 sufficient-statistic interface, (..., intensity); legacy entry points unchanged.

    This applies the finite-key formula to expected gains/error gains and possibly
    non-integer state-weighted counts. It is a performance estimator, not an
    actual-count security bound or device security certificate. Security
    budgets are resolved through ``FiniteKeyParams.effective()``: fixed
    per-block values are used by default, or declared total budgets are divided
    by ``n_max_blocks`` before applying the bound.
    """
    fk = fk.effective()
    gains, errors = np.broadcast_arrays(np.asarray(gains, float), np.asarray(errors, float))
    if gains.shape[-1] != len(fk.mu):
        raise ValueError("last dimension must enumerate intensities")
    if (not np.isfinite(gains).all() or not np.isfinite(errors).all()
            or np.any((gains < 0) | (gains > 1)) or np.any((errors < 0) | (errors > gains))):
        raise ValueError("invalid gains/error gains")
    N = np.broadcast_to(np.asarray(n_pulses, float), gains.shape[:-1])
    if np.any(~np.isfinite(N)) or np.any(N < 0):
        raise ValueError("invalid pulse count")
    if not (0 < fk.eps_sec < 1 and 0 < fk.eps_cor < 1):
        raise ValueError("invalid security budgets")
    mu1, mu2, mu3 = fk.mu
    if not (mu1 > mu2 > mu3 == 0 and mu1 > mu2 + mu3):
        raise ValueError("requires ordered signal, decoy, vacuum")
    beta = math.log(21.0 / fk.eps_sec)
    tau0 = sum(p * math.exp(-m) for p, m in zip(fk.p_mu, fk.mu))
    tau1 = sum(p * math.exp(-m) * m for p, m in zip(fk.p_mu, fk.mu))
    wx, wz = fk.p_x ** 2, (1.0-fk.p_x) ** 2
    def plus(y):
        return beta + np.sqrt(2*beta*y + beta**2)
    def minus(y):
        return beta/2 + np.sqrt(2*beta*y + beta**2/4)
    n_x = [N*wx*p*gains[...,i] for i,p in enumerate(fk.p_mu)]
    n_z = [N*wz*p*gains[...,i] for i,p in enumerate(fk.p_mu)]
    m_x = [N*wx*p*errors[...,i] for i,p in enumerate(fk.p_mu)]
    m_z = [N*wz*p*errors[...,i] for i,p in enumerate(fk.p_mu)]
    def bounds(n):
        lo = [math.exp(m) / p * (x - minus(x)) for x, p, m in zip(n, fk.p_mu, fk.mu)]
        hi = [math.exp(m) / p * (x + plus(x)) for x, p, m in zip(n, fk.p_mu, fk.mu)]
        return lo, hi

    def s01(n):
        lo, hi = bounds(n)
        s0 = tau0 * lo[2]
        s1 = tau1 * mu1 * (lo[1] - hi[2] - mu2 ** 2 / mu1 ** 2 * (hi[0] - s0 / tau0)) / (mu1 * mu2 - mu2 ** 2)
        return np.maximum(s0, 0.0), s1

    sx0, sx1 = s01(n_x)
    _, sz1 = s01(n_z)
    mlo, mhi = bounds(m_z)
    vz1 = tau1 * (mhi[1] - mlo[2]) / mu2

    ok = (sx1 > 0) & (sz1 > 0)
    sx1s, sz1s = np.where(ok, sx1, 1.0), np.where(ok, sz1, 1.0)
    b = vz1 / sz1s
    okb = ok & (b > 0) & (b < 0.5)
    bs = np.where(okb, b, 0.25)
    arg = (sz1s + sx1s) / (sz1s * sx1s * (1.0 - bs) * bs) * 21.0 ** 2 / fk.eps_sec ** 2
    ups = np.sqrt(np.maximum((sz1s + sx1s) * (1.0 - bs) * bs / (sz1s * sx1s * math.log(2.0)) * np.log2(arg), 0.0))
    phi = bs + ups
    okp = okb & (phi < 0.5)

    nx = sum(n_x)
    ex = sum(m_x) / np.where(nx > 0, nx, 1.0)
    lam_ec = fk.f_ec * nx * h2(ex)
    ell = (sx0 + sx1s * (1.0 - h2(np.where(okp, phi, 0.25))) - lam_ec
           - 6.0 * math.log2(21.0 / fk.eps_sec) - math.log2(2.0 / fk.eps_cor))
    return np.where(okp, np.maximum(np.floor(ell), 0.0), 0.0)


def asymptotic_key_from_stats(gains, errors, y1, n_pulses, fk: FiniteKeyParams = FiniteKeyParams()) -> float:
    """Block-aggregated asymptotic decoy proxy from v7.5 sufficient statistics."""
    gains = np.asarray(gains, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if gains.shape != (len(fk.mu),) or errors.shape != gains.shape:
        raise ValueError("gains/errors must have one value per intensity")
    n = float(n_pulses)
    y1 = float(y1)
    if n <= 0.0:
        return 0.0
    if not np.isfinite(y1) or not 0.0 <= y1 <= 1.0:
        raise ValueError("single-photon yield must be finite in [0, 1]")
    mu = np.asarray(fk.mu, dtype=np.float64)
    vac = np.flatnonzero(np.isclose(mu, 0.0, rtol=0.0, atol=1e-15))
    if len(vac) != 1:
        raise ValueError(
            f"asymptotic decoy proxy requires exactly one vacuum intensity; mu={mu.tolist()}"
        )
    y0 = float(gains[int(vac[0])])
    t1 = y0 / 2.0 + fk.e_d * (y1 - y0)
    e1 = t1 / y1 if y1 > 0.0 else 0.0
    eq = np.divide(errors, gains, out=np.zeros_like(errors), where=gains > 0.0)
    terms = (
        np.exp(-mu) * y0
        + mu * np.exp(-mu) * y1 * (1.0 - h2(e1))
        - fk.f_ec * gains * h2(eq)
    )
    per_pulse = fk.p_x**2 * max(float(np.dot(np.asarray(fk.p_mu), terms)), 0.0)
    return n * per_pulse


class QKDBlockAccumulator:
    """Accumulate expected sufficient statistics and release only on block close."""

    def __init__(self, fk: FiniteKeyParams = FiniteKeyParams()):
        self.fk = fk
        self.reset()

    def reset(self):
        self.n_pulses = 0.0
        self.gain_counts = np.zeros(len(self.fk.mu), dtype=np.float64)
        self.error_counts = np.zeros(len(self.fk.mu), dtype=np.float64)
        self.y1_counts = 0.0
        self.has_y1 = True
        self.slot_count = 0

    @property
    def is_open(self) -> bool:
        return self.slot_count > 0

    def add_slot(self, gains, errors, n_pulses, *, y1=None) -> None:
        gains, errors = np.broadcast_arrays(
            np.asarray(gains, dtype=np.float64),
            np.asarray(errors, dtype=np.float64),
        )
        if gains.shape != (len(self.fk.mu),):
            raise ValueError("slot gains/errors must have one value per intensity")
        n = float(n_pulses)
        if not np.isfinite(n) or n < 0.0:
            raise ValueError("slot pulse count must be finite and nonnegative")
        if (
            not np.isfinite(gains).all()
            or not np.isfinite(errors).all()
            or np.any((gains < 0.0) | (gains > 1.0))
            or np.any((errors < 0.0) | (errors > gains))
        ):
            raise ValueError("invalid slot gains/error gains")
        if y1 is None:
            self.has_y1 = False
        else:
            y1 = float(y1)
            if not np.isfinite(y1) or not 0.0 <= y1 <= 1.0:
                raise ValueError("slot y1 must be finite in [0, 1]")
            self.y1_counts += n * y1
        self.n_pulses += n
        self.gain_counts += n * gains
        self.error_counts += n * errors
        self.slot_count += 1

    def aggregate(self):
        if self.n_pulses <= 0.0:
            z = np.zeros(len(self.fk.mu), dtype=np.float64)
            return z, z.copy(), 0.0
        return (
            self.gain_counts / self.n_pulses,
            self.error_counts / self.n_pulses,
            self.n_pulses,
        )

    def aggregate_y1(self) -> float:
        if self.n_pulses <= 0.0:
            return 0.0
        if not self.has_y1:
            raise ValueError("asymptotic block release requires y1 for every slot")
        return self.y1_counts / self.n_pulses

    def close_finite_key(self) -> float:
        gains, errors, n = self.aggregate()
        released = 0.0 if n <= 0.0 else float(key_length_gains(gains, errors, n, self.fk))
        self.reset()
        return released

    def close_asymptotic(self) -> float:
        gains, errors, n = self.aggregate()
        released = 0.0 if n <= 0.0 else asymptotic_key_from_stats(
            gains, errors, self.aggregate_y1(), n, self.fk
        )
        self.reset()
        return float(released)
