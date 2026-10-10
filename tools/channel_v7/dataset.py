"""v7.5 block computation and diagnostics for finite-key datasets."""
from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr

from tools.channel_v7.model import asymptotic_components, asymptotic_from_components
from tools.channel_v2.rate_model_v2 import sun_elevation_deg
from tools.dataset_v2.finite_key import key_length_gains


STAT_FIELDS = {
    "gain": (3,), "error_gain": (3,), "usable_fraction": (),
    "model_domain": (), "domain_visible_substeps": (), "domain_valid_substeps": (),
    "moment_compatible": (), "moment_visible_substeps": (), "moment_compatible_substeps": (),
    "rytov": (), "scint_index": (), "aperture_active": (),
    "collection_tail": (),
    "collection_mean": (), "collection_mean_truncated": (),
    "collection_mean_bias": (), "collection_second_moment": (),
    "single_photon_yield": (), "proxy_per_pulse": (),
}


def generate_station_cloud_state(
    c_lm,
    n_slots,
    *,
    seed=2023,
    slot_s=60.0,
    tau_cld_s=3600.0,
):
    """Generate the v7.5 station-scale low/mid-cloud process.

    c_lm has shape (G, H) and is the hourly combined low/mid cloud fraction.
    Hour h is applied to all scheduling slots whose UTC start time lies in that
    hour (zero-order hold on the validated hourly validity-time axis).
    Each GS owns one stationary Gaussian AR(1) hidden variable. The resulting
    C_g(t) is shared by all ground-terminated links of that station and is
    generated once per scheduling slot, independently of geometry sub-stepping.
    """
    c_lm = np.asarray(c_lm, dtype=float)
    if c_lm.ndim != 2:
        raise ValueError("c_lm must have shape (stations, hours)")
    if np.any(~np.isfinite(c_lm)) or np.any((c_lm < 0.0) | (c_lm > 1.0)):
        raise ValueError("low/mid cloud fraction must be finite in [0, 1]")
    if isinstance(n_slots, (bool, np.bool_)) or int(n_slots) != n_slots or int(n_slots) < 1:
        raise ValueError("n_slots must be a positive integer")
    n_slots = int(n_slots)
    slot_s = float(slot_s)
    tau_cld_s = float(tau_cld_s)
    if not np.isfinite(slot_s) or slot_s <= 0.0:
        raise ValueError("slot_s must be finite and positive")
    if not np.isfinite(tau_cld_s) or tau_cld_s <= 0.0:
        raise ValueError("tau_cld_s must be finite and positive")

    # Do not silently carry the final hour forever when the weather input is
    # shorter than the requested horizon. That would fabricate future weather.
    hour = (np.arange(n_slots, dtype=np.float64) * slot_s // 3600.0).astype(np.int64)
    if c_lm.shape[1] == 0 or hour[-1] >= c_lm.shape[1]:
        raise ValueError(
            "station cloud input has insufficient hourly data for the requested slots"
        )
    f = np.clip(1.0 - c_lm[:, hour], 0.0, 1.0)
    rho = float(np.exp(-slot_s / tau_cld_s))
    innovation_scale = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
    rng = np.random.default_rng(int(seed))
    z = rng.standard_normal(c_lm.shape[0])
    out = np.empty((c_lm.shape[0], n_slots), dtype=np.uint8)
    out[:, 0] = (ndtr(z) < f[:, 0]).astype(np.uint8)
    for t in range(1, n_slots):
        z = rho * z + innovation_scale * rng.standard_normal(c_lm.shape[0])
        out[:, t] = (ndtr(z) < f[:, t]).astype(np.uint8)
    return out, rho


def _capture_time_s(cfg, slot_s):
    """Resolve v7.5 capture time, with compatibility for old gamma configs."""
    if hasattr(cfg, "capture_time_s"):
        t_acq = float(cfg.capture_time_s)
    elif hasattr(cfg, "capture_gamma"):
        gamma = float(cfg.capture_gamma)
        if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("legacy capture_gamma must be finite in [0, 1]")
        t_acq = slot_s * (1.0 - gamma)
    else:
        t_acq = 0.0
    if not np.isfinite(t_acq) or not 0.0 <= t_acq <= slot_s:
        raise ValueError("capture_time_s must be finite in [0, slot_s]")
    return t_acq


def compute_block(state, b0, b1, geometry, sat_positions):
    """Return full-slot and first-activation rates plus diagnostics.

    v7.5 invariants:
    * one station cloud state C_g(t) is shared by all ground links in a slot;
      geometry sub-stepping never re-samples cloud randomness;
    * first activation samples the actual post-capture time window;
    * sufficient statistics are pulse-weighted across sub-steps before the
      finite-key or asymptotic mapping is applied.
    """
    cfg, fk = state["cfg"], state["fk"]
    if cfg.hap_mobility != "static":
        raise ValueError("v7.5 baseline requires static HAP")
    if cfg.domain_policy not in ("error", "extrapolate", "mark"):
        raise ValueError("domain_policy must be 'error', 'extrapolate', or 'mark'")
    if cfg.protocol not in ("finite_key", "asymptotic"):
        raise ValueError("protocol must be 'finite_key' or 'asymptotic'")
    if getattr(cfg, "cloud_model", "station_markov") != "station_markov":
        raise ValueError("v7.5 baseline requires cloud_model='station_markov'")

    raw_n = getattr(cfg, "intra_slot_samples", 1)
    if isinstance(raw_n, (bool, np.bool_)) or int(raw_n) != float(raw_n) or int(raw_n) < 1:
        raise ValueError("intra_slot_samples must be a positive integer")
    n_sub = int(raw_n)
    slot_s = float(fk.slot_s)
    t_acq = _capture_time_s(cfg, slot_s)
    gamma = 1.0 - t_acq / slot_s

    m = state["model"]
    G = state["G"]
    t = np.arange(b0, b1, dtype=np.float64)
    B = len(t)
    L = len(state["link_u"])
    rates = np.zeros((B, L), np.float32)
    first_rates = np.zeros_like(rates)
    geometry_visible = np.zeros((B, L), dtype=np.uint8)
    stats = {k: np.zeros((B, L) + shape, np.float64) for k, shape in STAT_FIELDS.items()}
    first_qkd = {
        "gain": np.zeros((B, L, len(fk.mu)), np.float64),
        "single_photon_yield": np.zeros((B, L), np.float64),
        "usable_fraction": np.zeros((B, L), np.float64),
        "proxy_per_pulse": np.zeros((B, L), np.float64),
    }
    visible = np.zeros(L, np.int64)
    positive = np.zeros(L, np.int64)
    blocked = np.zeros(L, np.int64)

    cloud_state = np.asarray(state.get("cloud_state"))
    if cloud_state.ndim != 2 or cloud_state.shape[0] != G or cloud_state.shape[1] < b1:
        raise ValueError("station_markov requires cloud_state with shape (G, >=period slots)")
    cloud_hi = np.asarray(state.get("c_hi"))
    if cloud_hi.ndim != 2 or cloud_hi.shape[0] != G:
        raise ValueError("v7.5 requires high-cloud array with shape (G, hours)")

    offs = (np.arange(n_sub, dtype=float) + 0.5) / n_sub

    def timeline(window_fraction, start_fraction):
        tt = (t[None, :] + start_fraction + window_fraction * offs[:, None]).reshape(-1)
        slot_local = np.tile(np.arange(B, dtype=np.int64), n_sub)
        slot_global = slot_local + b0
        hr = np.minimum(
            (tt // 60.0).astype(np.int64),
            state["a_cn2"].shape[1] - 1,
        )
        sat = sat_positions(state["sat_cfgs"], tt)
        sun = sun_elevation_deg(tt[None, :], state["gs_lat"][:, None], state["gs_lon"][:, None])
        return tt, slot_local, slot_global, hr, sat, sun

    full_tl = timeline(1.0, 0.0)
    first_tl = None if gamma <= 0.0 else timeline(gamma, 1.0 - gamma)

    def coords(ids, sat, n_time):
        result = [np.empty((len(ids), n_time)) for _ in range(3)]
        for i, node in enumerate(ids):
            if node >= 2 * G:
                for j in range(3):
                    result[j][i] = sat[j][node - 2 * G]
            else:
                idx = node % G
                result[0][i], result[1][i] = state["gs_lat"][idx], state["gs_lon"][idx]
                result[2][i] = (
                    cfg.hap_alt_km if node >= G
                    else float(np.asarray(state.get("gs_alt_km", np.zeros(G)))[idx])
                )
        return result

    def evaluate(cols, lt, window_fraction, tl, *, diagnostics=False):
        nL = len(cols)

        def empty_result():
            diag = None
            if diagnostics:
                diag = {
                    key: np.zeros((nL, B) + shape, dtype=float)
                    for key, shape in STAT_FIELDS.items()
                }
            return dict(
                rate=np.zeros((nL, B), dtype=float),
                visible=np.zeros((nL, B), dtype=bool),
                blocked=np.zeros((nL, B), dtype=bool),
                diagnostics=diag,
                gain=np.zeros((nL, B, len(fk.mu)), dtype=float),
                error_gain=np.zeros((nL, B, len(fk.mu)), dtype=float),
                n_pulses=np.zeros((nL, B), dtype=float),
                y0=np.zeros((nL, B), dtype=float),
                y1=np.zeros((nL, B), dtype=float),
                t1=np.zeros((nL, B), dtype=float),
                proxy_per_pulse=np.zeros((nL, B), dtype=float),
            )

        if tl is None or window_fraction <= 0.0:
            return empty_result()

        _, slot_local, slot_global, hr_sub, sat, sun = tl
        Bn = len(slot_local)
        u, v = state["link_u"][cols], state["link_v"][cols]
        dist, elev, los, graze = geometry(coords(u, sat, Bn), coords(v, sat, Bn))
        if lt == "SAT-SAT":
            vis = (
                los
                & (graze >= m.p.isl_min_grazing_alt_km)
                & (dist * 1000 <= m.p.isl_max_dist_m)
            )
        else:
            vis = los & (elev >= cfg.alpha_min_deg)
        vis_slot = vis.reshape(nL, n_sub, B).any(axis=1)
        li, ti = np.nonzero(vis)

        n_slot = np.zeros((nL, B), dtype=float)
        q_num = np.zeros((nL, B, len(fk.mu)), dtype=float)
        e_num = np.zeros_like(q_num)
        proxy_num = np.zeros((nL, B), dtype=float)
        y0_num = np.zeros((nL, B), dtype=float)
        y1_num = np.zeros((nL, B), dtype=float)
        t1_num = np.zeros((nL, B), dtype=float)
        diag = (
            {
                key: np.zeros((nL, B) + shape, dtype=float)
                for key, shape in STAT_FIELDS.items()
            }
            if diagnostics else None
        )
        cloud_by_link_slot = np.ones((nL, B), dtype=bool)
        # With no visible samples there is no incompatible optical state.
        # Keep this initialized for the common invalidation path below.
        slot_moment_ok = np.ones((nL, B), dtype=bool)
        slot_domain_ok = np.ones((nL, B), dtype=bool)
        if lt in ("HAP-GS", "SAT-GS"):
            cloud_by_link_slot = cloud_state[u, b0:b1].astype(bool)

        if len(li):
            kw = {}
            if lt in ("HAP-GS", "SAT-GS"):
                g = u[li]
                high_cloud = cloud_hi[g, hr_sub[ti]]
                if np.any(~np.isfinite(high_cloud)) or np.any((high_cloud < 0) | (high_cloud > 1)):
                    raise ValueError("high-cloud fraction outside [0, 1]")
                a = state["a_cn2"][g, hr_sub[ti]]
                era5_ok = state.get("era5_ok")
                if era5_ok is not None:
                    valid_era5 = np.asarray(era5_ok)[g, hr_sub[ti]]
                    a = np.where(valid_era5, a, np.where(sun[g, ti] > 0, m.p.a_day, m.p.a_night))
                aod = state["aod"][g, hr_sub[ti]]
                sun_g = sun[g, ti]

                # v7.5 atmospheric integrals depend on the real GS elevation.
                # Group samples by GS so RateModelV7 receives one scalar h_GS
                # per local AGL-HV profile while preserving vectorized work
                # inside each station group.
                parts = []
                positions = []
                gs_alt = np.asarray(state.get("gs_alt_km", np.zeros(G)), dtype=float)
                for gs_idx in np.unique(g):
                    sel = np.flatnonzero(g == gs_idx)
                    part = m.optical_states(
                        lt,
                        dist[li[sel], ti[sel]] * 1000,
                        elev[li[sel], ti[sel]],
                        A=a[sel],
                        aod=aod[sel],
                        sun=sun_g[sel],
                        high_cloud=high_cloud[sel],
                        tau_cirrus=cfg.tau_cirrus,
                        h_gs_km=float(gs_alt[int(gs_idx)]),
                    )
                    parts.append(part)
                    positions.append(sel)
                concat_pos = np.concatenate(positions)
                inv = np.argsort(concat_pos)
                optical = {}
                for key in parts[0]:
                    arrays = [np.asarray(part[key]) for part in parts]
                    if arrays[0].ndim >= 2:
                        optical[key] = np.concatenate(arrays, axis=1)[:, inv]
                    else:
                        optical[key] = np.concatenate(arrays, axis=0)[inv]
            else:
                if lt == "SAT-HAP":
                    kw = dict(sun=sun[u[li] - G, ti])
                optical = m.optical_states(
                    lt, dist[li, ti] * 1000, elev[li, ti], **kw
                )
            valid = np.asarray(optical["domain_valid"], dtype=bool)
            moment_ok = np.asarray(optical.get("moment_compatible", np.ones_like(valid)), dtype=bool)
            # Physical bounded-moment compatibility is independent of the
            # weak-fluctuation domain.  Any slot containing a visible
            # incompatible sub-step is invalid as a physical rate sample.
            moment_grid = np.zeros((nL, Bn), dtype=bool)
            moment_grid[li, ti] = moment_ok
            moment_visible = vis.reshape(nL, n_sub, B).sum(axis=1)
            moment_count = moment_grid.reshape(nL, n_sub, B).sum(axis=1)
            slot_moment_ok = (~vis_slot) | (moment_count == moment_visible)
            domain_grid = np.zeros((nL, Bn), dtype=bool)
            domain_grid[li, ti] = valid
            slot_domain_ok = (~vis_slot) | (
                domain_grid.reshape(nL, n_sub, B).sum(axis=1) == moment_visible
            )
            if np.any(~valid) and cfg.domain_policy == "error":
                raise ValueError(
                    f"v7.5 weak-fluctuation domain exceeded: {lt}, block {b0}, "
                    f"{np.count_nonzero(~valid)} samples; use domain_policy='extrapolate' "
                    "only for the documented bounded engineering closure"
                )

            q_raw, err_raw, y0_raw, y1_raw, t1_raw = asymptotic_components(
                optical["eta"], optical["y0"], optical["weights"], fk
            )
            proxy_raw = asymptotic_from_components(q_raw, err_raw, y0_raw, y1_raw, t1_raw, fk)
            # All per-slot QKD statistics here are conditional on low/mid cloud being clear.
            # build_dataset_v2.py writes two derived rate surfaces: k_max is the
            # 60 s finite-key control surface, while k_proxy is the block-compatible
            # asymptotic policy surface. The realized C_g(t) stays separate and is
            # applied only after action selection, so observations never leak it.
            pulses_per_visible_substep = math.floor(
                fk.f_rep * (slot_s * window_fraction / n_sub)
            )
            pulses = np.full(
                len(li), float(pulses_per_visible_substep), dtype=float
            )

            np.add.at(n_slot, (li, slot_local[ti]), pulses)
            for k in range(len(fk.mu)):
                np.add.at(q_num[..., k], (li, slot_local[ti]), pulses * q_raw[:, k])
                np.add.at(e_num[..., k], (li, slot_local[ti]), pulses * err_raw[:, k])
            np.add.at(proxy_num, (li, slot_local[ti]), pulses * proxy_raw)
            np.add.at(y0_num, (li, slot_local[ti]), pulses * y0_raw)
            np.add.at(y1_num, (li, slot_local[ti]), pulses * y1_raw)
            np.add.at(t1_num, (li, slot_local[ti]), pulses * t1_raw)

            if diagnostics:
                visible_count = vis.reshape(nL, n_sub, B).sum(axis=1)

                def avg_grid(values):
                    arr = np.zeros((nL, Bn), dtype=float)
                    arr[li, ti] = np.asarray(values, dtype=float)
                    summed = arr.reshape(nL, n_sub, B).sum(axis=1)
                    return np.divide(
                        summed,
                        visible_count,
                        out=np.zeros_like(summed, dtype=float),
                        where=visible_count > 0,
                    )

                q_avg = np.divide(
                    q_num, n_slot[..., None],
                    out=np.zeros_like(q_num), where=n_slot[..., None] > 0,
                )
                e_avg = np.divide(
                    e_num, n_slot[..., None],
                    out=np.zeros_like(e_num), where=n_slot[..., None] > 0,
                )
                proxy_avg = np.divide(
                    proxy_num, n_slot,
                    out=np.zeros_like(proxy_num), where=n_slot > 0,
                )
                y1_avg = np.divide(
                    y1_num, n_slot,
                    out=np.zeros_like(y1_num), where=n_slot > 0,
                )
                vgrid = np.zeros((nL, Bn), dtype=bool)
                vgrid[li, ti] = valid
                valid_count = vgrid.reshape(nL, n_sub, B).sum(axis=1)
                valid_fraction = np.divide(
                    valid_count,
                    visible_count,
                    out=np.zeros_like(valid_count, dtype=float),
                    where=visible_count > 0,
                )
                moment_fraction = np.divide(
                    moment_count,
                    visible_count,
                    out=np.zeros_like(moment_count, dtype=float),
                    where=visible_count > 0,
                )
                diag = dict(
                    gain=q_avg,
                    error_gain=e_avg,
                    usable_fraction=n_slot / fk.n_pulses,
                    model_domain=valid_fraction,
                    domain_visible_substeps=visible_count.astype(float),
                    domain_valid_substeps=valid_count.astype(float),
                    moment_compatible=moment_fraction,
                    moment_visible_substeps=visible_count.astype(float),
                    moment_compatible_substeps=moment_count.astype(float),
                    rytov=avg_grid(optical["rytov"]),
                    scint_index=avg_grid(optical["scint_index"]),
                    aperture_active=avg_grid(optical["aperture_active"].astype(float)),
                    collection_tail=avg_grid(optical["tail"]),
                    collection_mean=avg_grid(optical["collection_mean"]),
                    collection_mean_truncated=avg_grid(optical["collection_mean_truncated"]),
                    collection_mean_bias=avg_grid(optical["collection_mean_bias"]),
                    collection_second_moment=avg_grid(optical["collection_second_moment"]),
                    single_photon_yield=y1_avg,
                    proxy_per_pulse=proxy_avg,
                )
        q_avg = np.divide(
            q_num, n_slot[..., None],
            out=np.zeros_like(q_num), where=n_slot[..., None] > 0,
        )
        e_avg = np.divide(
            e_num, n_slot[..., None],
            out=np.zeros_like(e_num), where=n_slot[..., None] > 0,
        )
        proxy_avg = np.divide(
            proxy_num, n_slot,
            out=np.zeros_like(proxy_num), where=n_slot > 0,
        )
        y0_avg = np.divide(y0_num, n_slot, out=np.zeros_like(y0_num), where=n_slot > 0)
        y1_avg = np.divide(y1_num, n_slot, out=np.zeros_like(y1_num), where=n_slot > 0)
        t1_avg = np.divide(t1_num, n_slot, out=np.zeros_like(t1_num), where=n_slot > 0)
        if cfg.protocol == "finite_key":
            rate = key_length_gains(q_avg, e_avg, n_slot, fk) / slot_s
        else:
            rate = n_slot * proxy_avg / slot_s
        # Unlike weak-domain extrapolation, bounded-moment incompatibility has
        # no admissible Beta closure.  Always invalidate the affected slot
        # rather than silently substituting/clipping a distribution.  This
        # applies equally to full-slot and acquisition-window statistics.
        invalid_moment = vis_slot & ~slot_moment_ok
        # Mark both acquisition and full-slot statistics, even when diagnostic
        # output is disabled. Otherwise k_proxy/block accumulation can silently
        # turn an explicitly marked rate back into usable key material.
        if cfg.domain_policy == "mark":
            invalid_moment |= vis_slot & ~slot_domain_ok
        rate = np.where(invalid_moment, np.nan, rate)
        q_avg = np.where(invalid_moment[..., None], np.nan, q_avg)
        e_avg = np.where(invalid_moment[..., None], np.nan, e_avg)
        y0_avg = np.where(invalid_moment, np.nan, y0_avg)
        y1_avg = np.where(invalid_moment, np.nan, y1_avg)
        t1_avg = np.where(invalid_moment, np.nan, t1_avg)
        proxy_avg = np.where(invalid_moment, np.nan, proxy_avg)
        n_slot = np.where(invalid_moment, np.nan, n_slot)
        if diag is not None:
            # These are the H5 block-sufficient statistics.  Keep physical
            # diagnostics such as V_R/S_I/m_c finite for root-cause analysis,
            # but never persist placeholder QKD statistics for an incompatible
            # bounded moment target.
            diag["gain"] = q_avg
            diag["error_gain"] = e_avg
            diag["usable_fraction"] = n_slot / fk.n_pulses
            diag["single_photon_yield"] = y1_avg
            diag["proxy_per_pulse"] = proxy_avg

        blocked_slot = vis_slot & ~cloud_by_link_slot
        return dict(
            rate=rate,
            visible=vis_slot,
            blocked=blocked_slot,
            diagnostics=diag,
            gain=q_avg,
            error_gain=e_avg,
            n_pulses=n_slot,
            y0=y0_avg,
            y1=y1_avg,
            t1=t1_avg,
            proxy_per_pulse=proxy_avg,
        )

    for lt in ("HAP-GS", "SAT-GS", "SAT-HAP", "SAT-SAT"):
        allcols = np.flatnonzero(state["link_type"] == lt)
        for start in range(0, len(allcols), 32):
            cols = allcols[start:start + 32]
            full = evaluate(cols, lt, 1.0, full_tl, diagnostics=True)
            first = evaluate(cols, lt, gamma, first_tl, diagnostics=False)
            rates[:, cols] = full["rate"].T
            first_rates[:, cols] = first["rate"].T
            geometry_visible[:, cols] = full["visible"].T.astype(np.uint8)
            first_qkd["gain"][:, cols] = np.moveaxis(first["gain"], 0, 1)
            first_qkd["single_photon_yield"][:, cols] = first["y1"].T
            first_qkd["usable_fraction"][:, cols] = (first["n_pulses"] / fk.n_pulses).T
            first_qkd["proxy_per_pulse"][:, cols] = first["proxy_per_pulse"].T
            visible[cols] += full["visible"].sum(axis=1)
            positive[cols] += (np.nan_to_num(full["rate"], nan=0.0) > 0).sum(axis=1)
            blocked[cols] += full["blocked"].sum(axis=1)
            if full["diagnostics"] is not None:
                for key, value in full["diagnostics"].items():
                    stats[key][:, cols] = np.moveaxis(value, 0, 1)

    return b0, rates, first_rates, geometry_visible, visible, positive, blocked, stats, first_qkd
