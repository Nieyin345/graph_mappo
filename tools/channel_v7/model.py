"""v7 optical channel; legacy v2 remains available for existing datasets.

Karakosta et al., PR Research 7, 023199, Appendix B2/B8/B19.
This implements the documented approximations, not an experimental calibration.
"""
from dataclasses import dataclass

import numpy as np

from tools.channel_v2.rate_model_v2 import ParamsV2, RateModelV2, HVIntegrals
from tools.dataset_v2.finite_key import FiniteKeyParams, h2

MODEL_REVISION = "v7_5_beta_station_cloud_block_proxy_20261008"


@dataclass
class ParamsV7(ParamsV2):
    scint_gh_nodes: int = 41  # Gauss-Legendre nodes in Beta CDF coordinates
    isl_background: float = 1.5e-7
    # 强起伏饱和开关（2026-10-07）：V_R<1 用 Rytov 近似 σ_I²≃V_R（不变）；
    # V_R≥1 时该近似失效，改用饱和闭合 σ_I²→1（点接收强湍流极限，Andrews &
    # Phillips 2005 ch.9；Karakosta 2025 §B.1）。False = 旧的裸 Rytov（会触发域外中止）。
    saturate_scint: bool = True
    # 孔径平均（2026-10-07，外部审查第 1 条）：当接收口径 D_Rx 大于辐照相关宽度 ρ_c
    # （Karakosta 2025 式 B3/B4）时，用孔径平均球面波闪烁指数式 B5 取代点接收 σ_I²。
    # 对 HAP-GS（D=0.6 m > ρ_c≈0.18 m）生效，对 SAT-GS（D<ρ_c≈0.92 m）不生效。
    aperture_averaging: bool = True


def correlation_width(L, lam):
    """Irradiance correlation width ρ_c = √(λL)（Karakosta et al. 2025, PRR 7 023199, 式 B3）。

    仅用作**孔径平均的判据宽度** D_Rx>ρ_c。**不**切换到式 B4 的强起伏形式
    ρ_c^(strong)=0.36·σ_R^{−3/5}·√(λL)=0.36·V_R^{−3/10}·√(λL)（因 V_R=σ_R²）：
    B3 与 B4 在 V_R=1 处不连续（相差 1/0.36≈2.8×），以 V_R 为键切换会让**建模的收集口径状态
    在弱/强分界处跳变**，而相关宽度本身是连续物理量、不是开关。基线在弱起伏（V_R<1）
    工作，ρ_c^(weak) 是该区的正确宽度；B4 只在域外（外推）区才有意义，故不作判据。
    依据：Karakosta 2025 附录 B.1；Andrews & Phillips 2005 ch.9。
    """
    return np.sqrt(np.asarray(lam, float) * np.asarray(L, dtype=float))


def aperture_averaged_scint(vr, L, D, lam):
    """孔径平均球面波闪烁指数 σ_I²(D_Rx)（Karakosta et al. 2025 式 B5）。

    σ_I² = exp[ 0.49β₀²/(1+0.18d²+0.56β₀^{12/5})^{7/6}
               + 0.51β₀²(1+0.69β₀^{12/5})^{-5/6}/(1+0.90d²+0.62d²β₀^{12/5}) ] - 1,
    其中 d=√(kD_Rx²/(4z))（式 B6）。v7.5 当前实现接口定义 β₀² := V_R，
    其中 V_R 由 B2 下行球面波路径积分得到。该接口只保证实现路径一致，**不等于
    B2/B5/B7 的物理参数转换已经完成文献核实**；投稿前仍须按规范 §7 逐式核对
    Karakosta B2/B5/B7 与原始 Andrews/Phillips 定义。代码不得在该接口之后再次
    乘 0.4065，以免在未核实前引入第二套混杂口径。
    仅当 D_Rx>ρ_c=√(λL) 时使用（弱起伏区）。不对上界做数值裁剪：
    域外或有界矩不相容由调用层显式标记，避免把异常值静默压成 1。
    依据：Karakosta et al. 2025 附录 B.1 式 B5–B7；物理口径状态见规范 §7。
    """
    k = 2.0 * np.pi / float(lam)
    b0 = np.asarray(vr, dtype=float)
    d2 = k * float(D) ** 2 / (4.0 * np.asarray(L, dtype=float))
    t1 = 0.49 * b0 / (1.0 + 0.18 * d2 + 0.56 * b0 ** (6.0 / 5.0)) ** (7.0 / 6.0)
    t2 = (0.51 * b0 * (1.0 + 0.69 * b0 ** (6.0 / 5.0)) ** (-5.0 / 6.0)
          / (1.0 + 0.90 * d2 + 0.62 * d2 * b0 ** (6.0 / 5.0)))
    return np.maximum(np.expm1(t1 + t2), 0.0)


def saturating_scint(vr, cap=1.0):
    """点接收闪烁指数的**经验外推闭合**：取 σ_I² = min(V_R, 1)。

    Rytov 近似 σ_I²≃V_R 仅在其适用域 V_R<1 成立。V_R≥1 时本模块**没有**经验证的强起伏
    统计模型，只用 min(V_R,1) 作**经验外推**（在 V_R=1 连续、单调、恒 ≤1，且给出一条可解析、
    有界的取值），**不是**已验证的强起伏模型；域外样本的数值结论须与域内分开报告（§7）。
    对 SAT–GS（D=0.6 m < ρ_c≈0.92–1.5 m，孔径平均不生效）此项在 V_R≥1 触发；对 HAP–GS
    （D>ρ_c）走式 B5。依据：Andrews & Phillips 2005 ch.9（强湍流趋于饱和，值趋于有限极限）。
    """
    return np.minimum(np.asarray(vr, dtype=float), float(cap))


def match_hv(cn_ref, z=10.0, wind=21.0, *, return_clipped=False):
    """Match a MOST reference Cn2 value to the HV near-ground coefficient A.

    Returns ``A=max(0, (Cn2_ref-B_HV(z))*exp(z/100))``. With
    ``return_clipped=True``, also return the mask where the HV background alone
    exceeds the reference value (the required matching-floor diagnostic).
    """
    if not np.isfinite(z) or z < 0 or not np.isfinite(wind) or wind < 0:
        raise ValueError("HV matching height and wind must be finite and nonnegative")
    cn_ref = np.asarray(cn_ref, dtype=float)
    if np.any(~np.isfinite(cn_ref)) or np.any(cn_ref < 0):
        raise ValueError("reference Cn2 must be finite and nonnegative")
    background = (0.00594 * (wind / 27) ** 2 * (1e-5 * z) ** 10
                  * np.exp(-z / 1000) + 2.7e-16 * np.exp(-z / 1500))
    raw = (cn_ref - background) * np.exp(z / 100)
    clipped = raw < 0
    matched = np.maximum(0.0, raw)
    return (matched, clipped) if return_clipped else matched


def bounded_collection(mean, scint_index, order=41):
    """v7.5 bounded two-moment collection model.

    ``mean`` is the long-term collection efficiency m_c and ``scint_index`` is
    Var(U)/E[U]^2. For 0 < SI < (1-m)/m the baseline is Beta(alpha,beta) with
    matched first two moments. SI=0 is deterministic. The physical upper bound
    SI=(1-m)/m is represented by the endpoint Bernoulli law
    P(U=1)=m, P(U=0)=1-m. Values above the bound are rejected.

    The continuous Beta law is integrated in CDF coordinates with
    Gauss-Legendre nodes. ``weights``has the same trailing shape as ``mean``
    because the Bernoulli boundary needs per-sample weights.

    ``tail`` is retained for dataset-schema compatibility and is identically
    zero because the baseline distribution is natively supported on [0, 1].
    """
    from scipy.special import betaincinv

    mean, scint_index = np.broadcast_arrays(
        np.asarray(mean, dtype=float), np.asarray(scint_index, dtype=float)
    )
    if np.any(~np.isfinite(mean)) or np.any((mean < 0.0) | (mean > 1.0)):
        raise ValueError("collection mean must be finite in [0, 1]")
    if np.any(~np.isfinite(scint_index)) or np.any(scint_index < 0.0):
        raise ValueError("scintillation index must be finite and nonnegative")
    if isinstance(order, (bool, np.bool_)) or int(order) != order or int(order) < 2:
        raise ValueError("quadrature order must be an integer >= 2")
    order = int(order)

    endpoint = (mean <= 0.0) | (mean >= 1.0)
    if np.any(endpoint & (scint_index > 0.0)):
        raise ValueError("endpoint collection means require zero scintillation index")

    max_si = np.full_like(mean, np.inf, dtype=float)
    interior = (mean > 0.0) & (mean < 1.0)
    max_si[interior] = (1.0 - mean[interior]) / mean[interior]
    tol = 64.0 * np.finfo(float).eps * np.maximum(1.0, max_si)
    if np.any(interior & (scint_index > max_si + tol)):
        raise ValueError("scintillation index exceeds the physical bounded-moment limit")

    deterministic = (scint_index == 0.0) | endpoint
    boundary = interior & ~deterministic & (np.abs(scint_index - max_si) <= tol)
    continuous = interior & ~deterministic & ~boundary

    x, w = np.polynomial.legendre.leggauss(order)
    q = ((x + 1.0) / 2.0).reshape((order,) + (1,) * mean.ndim)
    base_w = (w / 2.0).reshape((order,) + (1,) * mean.ndim)
    nodes = np.broadcast_to(mean, (order,) + mean.shape).astype(float, copy=True)
    weights = np.broadcast_to(base_w, nodes.shape).astype(float, copy=True)

    if np.any(continuous):
        kappa = np.ones_like(mean)
        kappa[continuous] = (
            (1.0 - mean[continuous])
            / (mean[continuous] * scint_index[continuous])
            - 1.0
        )
        alpha = np.where(continuous, mean * kappa, 1.0)
        beta = np.where(continuous, (1.0 - mean) * kappa, 1.0)
        beta_nodes = betaincinv(alpha[None, ...], beta[None, ...], q)
        nodes = np.where(continuous[None, ...], beta_nodes, nodes)

    if np.any(boundary):
        flat_nodes = nodes.reshape(order, -1)
        flat_weights = weights.reshape(order, -1)
        flat_mean = mean.reshape(-1)
        for idx in np.flatnonzero(boundary.reshape(-1)):
            flat_nodes[:, idx] = 0.0
            flat_weights[:, idx] = 0.0
            flat_nodes[1, idx] = 1.0
            flat_weights[0, idx] = 1.0 - flat_mean[idx]
            flat_weights[1, idx] = flat_mean[idx]

    actual_mean = mean.copy()
    actual_second = mean ** 2 * (1.0 + scint_index)
    actual_second = np.where(deterministic, mean ** 2, actual_second)
    tail = np.zeros_like(mean)
    return nodes, weights, tail, actual_mean, actual_second


def state_gains(eta, y0, weights, fk=FiniteKeyParams()):
    """Average gains/errors over optical states; output shape (..., intensity)."""
    eta, y0 = np.broadcast_arrays(np.asarray(eta, float), np.asarray(y0, float))
    weights = np.asarray(weights, float)
    if weights.ndim == 1:
        weights = weights.reshape((-1,) + (1,) * (eta.ndim - 1))
    weights = np.broadcast_to(weights, eta.shape)
    if np.any(~np.isfinite(eta)) or np.any((eta < 0) | (eta > 1)):
        raise ValueError("invalid optical efficiency")
    if np.any(~np.isfinite(y0)) or np.any((y0 < 0) | (y0 > 1)):
        raise ValueError("invalid background probability")
    if np.any(weights < 0) or not np.allclose(weights.sum(axis=0), 1):
        raise ValueError("state weights must sum to one")
    ps = -np.expm1(-eta[..., None] * np.asarray(fk.mu))
    q = y0[..., None] + (1-y0[..., None]) * ps
    err = y0[..., None]/2 + fk.e_d*(1-y0[..., None])*ps
    return (weights[..., None]*q).sum(axis=0), (weights[..., None]*err).sum(axis=0)


def asymptotic_components(eta, y0, weights, fk=FiniteKeyParams()):
    """Return block-aggregatable asymptotic sufficient statistics.

    All outputs are averaged over the optical/high-cloud state axis. q/err
    have one value per decoy intensity; vacuum, y1 and t1 are the quantities
    required by the v7.5 block-aggregated training proxy in spec §5.3.
    """
    q, err = state_gains(eta, y0, weights, fk)
    eta, y0 = np.broadcast_arrays(np.asarray(eta, float), np.asarray(y0, float))
    w = np.asarray(weights, float)
    if w.ndim == 1:
        w = w.reshape((-1,) + (1,) * (eta.ndim - 1))
    w = np.broadcast_to(w, eta.shape)
    vacuum = (w * y0).sum(axis=0)
    y1 = (w * (y0 + (1 - y0) * eta)).sum(axis=0)
    t1 = (w * (y0 / 2 + fk.e_d * (1 - y0) * eta)).sum(axis=0)
    return q, err, vacuum, y1, t1


def asymptotic_from_components(q, err, vacuum, y1, t1, fk=FiniteKeyParams()):
    """v7.5 asymptotic proxy per emitted pulse from aggregated statistics."""
    q = np.asarray(q, float)
    err = np.asarray(err, float)
    vacuum, y1, t1 = np.broadcast_arrays(
        np.asarray(vacuum, float), np.asarray(y1, float), np.asarray(t1, float)
    )
    e1 = np.divide(t1, y1, out=np.zeros_like(y1), where=y1 > 0)
    eq = np.divide(err, q, out=np.zeros_like(q), where=q > 0)
    mu = np.asarray(fk.mu)
    terms = (
        np.exp(-mu) * vacuum[..., None]
        + mu * np.exp(-mu) * y1[..., None] * (1 - h2(e1[..., None]))
        - fk.f_ec * q * h2(eq)
    )
    return fk.p_x**2 * np.maximum((np.asarray(fk.p_mu) * terms).sum(axis=-1), 0)


def asymptotic_states(eta, y0, weights, fk=FiniteKeyParams()):
    q, err, vacuum, y1, t1 = asymptotic_components(eta, y0, weights, fk)
    return asymptotic_from_components(q, err, vacuum, y1, t1, fk)


class RateModelV7(RateModelV2):
    def __init__(self, p=None, sat_alt_km=550.0, hap_alt_km=20.0):
        p = p if p is not None else ParamsV7()
        order = p.scint_gh_nodes
        if isinstance(order, (bool, np.bool_)) or int(order) != order or order < 2:
            raise ValueError("scint_gh_nodes must be an integer >= 2")
        if not np.isfinite(p.isl_background) or p.isl_background < 0:
            raise ValueError("isl_background must be finite and nonnegative")
        self.sat_alt_km = float(sat_alt_km)
        self.hap_alt_km = float(hap_alt_km)
        self._ground_hv_cache = {}
        super().__init__(p, sat_alt_km, hap_alt_km)

    def _ground_hv(self, tx: str, h_gs_km: float) -> HVIntegrals:
        """Local AGL-HV integrals from the real GS surface to the transmitter altitude."""
        h_gs_km = float(h_gs_km)
        top_km = self.hap_alt_km if tx == "HAP" else self.sat_alt_km
        rel_m = (top_km - h_gs_km) * 1e3
        if not np.isfinite(rel_m) or rel_m <= 0.0:
            raise ValueError(f"invalid ground/transmitter altitude ordering: GS={h_gs_km} km, {tx}={top_km} km")
        key = (tx, round(h_gs_km, 6))
        hv = self._ground_hv_cache.get(key)
        if hv is None:
            hv = HVIntegrals(rel_m, self.p.hv_wind)
            self._ground_hv_cache[key] = hv
        return hv

    def eta_bp(self, L, tx, rx, elev_deg=None, A=None, hv=None):
        L = np.asarray(L, float)
        wd2 = self.p.w0[tx]**2 + (self.p.lam*L/(np.pi*self.p.w0[tx]))**2
        vr = self.scint_var(elev_deg, A, hv) if rx == "GS" else 0.0
        extra = 3.26*L/self.p.k*np.asarray(vr)**(6/5)
        width = wd2 + extra + 4*(self.p.sigma_theta[tx]*L)**2
        return -np.expm1(-2*(self.p.d_rx[rx]/2)**2/width), extra/wd2

    def y0(self, rx, lb):
        return -np.expm1(-super().y0(rx, lb))

    def optical_states(self, ltype, L, elev, *, A=None, vis_km=0.0, sun=0.0,
                       high_cloud=0.0, tau_cirrus=0.3, h_gs_km=0.0):
        """Caller applies LOS/elevation. Returns state statistics and domain diagnostics.

        ``vis_km`` is the station-hour visibility (km) driving the v7.5 Koschmieder-Kim
        weather extinction via ``eta_wx`` (replaces the earlier AOD column)."""
        from tools.channel_v2.rate_model_v2 import airmass_kasten_young
        if not np.isfinite(tau_cirrus) or tau_cirrus < 0:
            raise ValueError("tau_cirrus must be finite and nonnegative")
        if ltype not in ("HAP-GS", "SAT-GS", "SAT-HAP", "SAT-SAT"):
            raise ValueError(f"unsupported link type: {ltype!r}")
        L, elev = np.broadcast_arrays(np.asarray(L, float), np.asarray(elev, float))
        if np.any(~np.isfinite(L)) or np.any(L <= 0):
            raise ValueError("slant distance must be finite and positive")
        if np.any(~np.isfinite(elev)):
            raise ValueError("link elevation must be finite")
        ground = ltype in ("HAP-GS", "SAT-GS")
        tx = "HAP" if ltype == "HAP-GS" else "SAT"
        if ground:
            if A is None or sun is None:
                raise ValueError("ground links require matched HV coefficient A and solar elevation")
            A, vis_km = np.asarray(A, float), np.asarray(vis_km, float)
            if np.any(~np.isfinite(A)) or np.any(A < 0):
                raise ValueError("matched HV coefficient A must be finite and nonnegative")
            if np.any(~np.isfinite(vis_km)) or np.any(vis_km <= 0):
                raise ValueError("visibility must be finite and positive (km)")
        rx = "GS" if ground else ("HAP" if ltype == "SAT-HAP" else "SAT")
        if ground:
            if np.ndim(h_gs_km) != 0:
                raise ValueError("optical_states expects scalar h_gs_km; caller must group samples by GS")
            h_gs_km = float(h_gs_km)
            hv = self._ground_hv(tx, h_gs_km)
            tx_top_km = self.hap_alt_km if tx == "HAP" else self.sat_alt_km
        else:
            hv = self.hv_hap if tx == "HAP" else self.hv_sat
            tx_top_km = self.hap_alt_km if tx == "HAP" else self.sat_alt_km
        bp, _ = self.eta_bp(L, tx, rx, elev, A, hv)
        if ground:
            vr = self.scint_var(elev, A, hv)
            # 收集效率的闪烁涨落幅度（2026-10-07，外部审查第 1/2/3 条修订）：
            #  1) 孔径平均：D_Rx>ρ_c=√(λL)（式 B3，**不切 B4 强分支**——那会在 V_R=1 处跳变）。
            #     HAP-GS（D=0.6>ρ_c≈0.18）生效；SAT-GS（D<ρ_c≈0.92）不生效。
            #  2) 点接收（未孔径平均）且 V_R≥1：用经验外推闭合 σ_I²=min(V_R,1)。此时它**真正触发**
            #     （孔径平均生效时走 B5，不走此项）。
            rho_c = correlation_width(L, self.p.lam)
            ap_active = np.asarray(self.p.d_rx["GS"], float) > rho_c
            si_ap = aperture_averaged_scint(vr, L, self.p.d_rx["GS"], self.p.lam)
            if self.p.aperture_averaging:
                si = np.where(ap_active, si_ap, saturating_scint(vr) if self.p.saturate_scint else vr)
            else:
                si = saturating_scint(vr) if self.p.saturate_scint else vr
                ap_active = np.zeros_like(vr, dtype=bool)
            # A bounded U in [0,1] with mean m_c can have at most
            # Var(U)/E[U]^2 = (1-m_c)/m_c.  Keep this diagnostic separate
            # from the weak-fluctuation domain flag: V_R can be outside the
            # weak domain yet still yield admissible bounded moments, and vice versa.
            mean_bp = np.asarray(bp, dtype=float)
            interior = (mean_bp > 0.0) & (mean_bp < 1.0)
            max_si = np.zeros_like(mean_bp, dtype=float)
            max_si[interior] = (1.0 - mean_bp[interior]) / mean_bp[interior]
            tol = 64.0 * np.finfo(float).eps * np.maximum(1.0, max_si)
            moment_compatible = np.where(
                interior,
                np.asarray(si) <= max_si + tol,
                np.asarray(si) <= tol,
            )
            # Do not clip an incompatible target SI into the Beta family.
            # A deterministic placeholder is used only to keep the vectorized
            # optical calculation finite; dataset.py marks every affected slot
            # invalid/NaN and never exposes this placeholder as a usable rate.
            si_beta = np.where(moment_compatible, si, 0.0)
            nodes, weights, tail, actual_mean, actual_second = bounded_collection(
                bp, si_beta, self.p.scint_gh_nodes
            )
            # Diagnostics preserve the requested second moment even when it is
            # physically incompatible with U in [0,1].  The placeholder Beta
            # closure above is never exposed as a usable rate for those samples.
            target_second = mean_bp ** 2 * (1.0 + np.asarray(si, dtype=float))
            actual_second = np.where(moment_compatible, actual_second, target_second)
            phi, _ = self.eta_phi(elev, A, hv)
            # Molecular column uses the absolute GS/transmitter altitude
            # engineering reference (DEM/geopotential-height baseline), while
            # turbulence uses the locally shifted AGL-HV profile above.
            clear = self.eta_mol(elev, tx_top_km, h_gs_km) * self.eta_wx(elev, vis_km)
            lb = np.where(np.asarray(sun)>0, self.p.lb_day, self.p.lb_night)
        else:
            vr = np.zeros_like(L)
            moment_compatible = np.ones_like(L, dtype=bool)
            nodes, weights, tail, actual_mean, actual_second = (
                bp[None, ...], np.ones(1), np.zeros_like(L), bp, bp ** 2
            )
            phi = 1.0
            clear = self.eta_mol(elev, self.hv_sat.h_top/1000, self.hv_hap.h_top/1000) if rx == "HAP" else 1.0
            lb = (np.where(np.asarray(sun)>0, self.p.lb_day*self.p.kappa_strat, 0.0)
                  if rx == "HAP" else self.p.isl_background)
        eta = nodes*clear*phi*self.p.eta_tx_opt*self.p.eta_rx_opt*self.p.detector[rx][0]
        y0 = np.broadcast_to(self.y0(rx, lb), eta.shape)
        weights = np.asarray(weights, dtype=float)
        if weights.shape == eta.shape:
            w = weights
        else:
            w = np.broadcast_to(weights.reshape((-1,) + (1,) * L.ndim), eta.shape)
        if ground:
            c = np.broadcast_to(np.asarray(high_cloud, float), L.shape)
            if np.any(~np.isfinite(c)) or np.any((c < 0) | (c > 1)): 
                raise ValueError("high cloud fraction outside [0,1]")
            tc = np.exp(-tau_cirrus*airmass_kasten_young(elev))
            eta = np.concatenate((eta, eta*tc), axis=0)
            y0 = np.concatenate((y0,y0), axis=0)
            w = np.concatenate((w*(1-c),w*c), axis=0)
        # domain_valid：Rytov 弱起伏适用域标记（V_R<1）。数值上的饱和/孔径平均
        # 只提供域外工程闭合；是否中止、保留外推值或写 NaN 由 dataset 的
        # domain_policy = error / extrapolate / mark 显式决定。
        # aperture_active：D_Rx>ρ_c（孔径平均生效）标记；scint_index 为最终所用 σ_I²。
        return dict(
                    eta=eta, y0=y0, weights=w,
                    domain_valid=vr < 1,
                    moment_compatible=moment_compatible,
                    rytov=vr,
                    scint_index=si if ground else np.zeros_like(L),
                    aperture_active=(ap_active if ground else np.zeros_like(L, dtype=bool)),
                    correlation_width=(rho_c if ground else np.zeros_like(L)),
                    tail=tail, collection_mean=bp, collection_mean_truncated=actual_mean,
                    collection_mean_bias=actual_mean-bp, collection_second_moment=actual_second)
