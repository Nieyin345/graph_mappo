"""Compare the v2 rate model (paper §2 second draft) with the current dataset k_max.

Validation only: reads dataset/global/ (read-only) and .tmp/channel_v2/weather/,
writes to .tmp/channel_v2/out/.  Production code and data are not modified.

usage: python tools/channel_v2/compare_v2.py [--days 14,45,...] [--variants base,spad200,...]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from rate_model_v2 import (ParamsV2, RateModelV2, cn2_monin_obukhov, p_clear_lund_shanklin,  # noqa: E402
                           sun_elevation_deg)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DS = os.path.join(ROOT, "dataset", "global")
WX = os.path.join(ROOT, ".tmp", "channel_v2", "weather")
OUT = os.path.join(ROOT, ".tmp", "channel_v2", "out")
ERA5 = os.path.join(ROOT, ".tmp", "channel_v2", "era5")
R_E = 6371.0
DEFAULT_DAYS = [14, 45, 73, 104, 134, 165, 195, 226, 257, 287, 318, 348]  # 15th of each month
TYPES = ["SAT-GS", "HAP-GS", "SAT-HAP", "SAT-SAT"]


def variant_params(name: str) -> tuple[ParamsV2, float]:
    """Return (params, cn2 scale) for a named variant."""
    p = ParamsV2()
    scale = 1.0
    for tok in name.split("+"):
        if tok == "base":
            pass
        elif tok == "spad200":          # cooled free-running InGaAs (ID230 class)
            p.detector = dict(p.detector, HAP=(0.25, 200.0), SAT=(0.25, 200.0))
        elif tok == "cn2x2":
            scale = 2.0
        elif tok == "cn2x0.5":
            scale = 0.5
        elif tok == "d04":              # smaller ground aperture: less collection, better Strehl
            p.d_rx = dict(p.d_rx, GS=0.4)
        elif tok == "era5":             # near-ground Cn2 from ERA5 fluxes + Monin-Obukhov, eqs. (14)-(16)
            p.cn2_source = "era5"
        elif tok == "fb":               # opt-in: stations without an ERA5 file fall back to day/night A
            p.era5_fallback = True
        elif tok.startswith("zr"):         # reference height for eq. (15), e.g. zr2 = 2 m
            p.z_ref = float(tok[2:].replace("p", "."))
        elif tok == "rx05":             # larger SAT/HAP receivers only (W0 unchanged)
            p.d_rx = dict(p.d_rx, SAT=0.5, HAP=0.5)
        elif tok.startswith("kx"):         # lumped airmass excess loss, e.g. kx1 = 1 dB per air mass
            p.kappa_x_db = float(tok[2:].replace("p", "."))
        elif tok == "smf7a":            # old closed-form SMF coupling, draft eq. (7a) until 09-30
            p.smf_model = "core_halo"
        elif tok == "isl_d05":          # larger satellite ISL optics
            p.w0 = dict(p.w0, SAT=0.15)
            p.d_rx = dict(p.d_rx, SAT=0.5, HAP=0.5)
        else:
            raise ValueError(tok)
    return p, scale


def ecef(lat, lon, alt):
    la, lo = np.radians(lat), np.radians(lon)
    r = R_E + alt
    return r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)


def geometry(c1, c2):
    """distance [km], elevation at lower endpoint [deg], LOS not blocked, grazing altitude [km]."""
    (la1, lo1, a1), (la2, lo2, a2) = c1, c2
    x1, y1, z1 = ecef(la1, lo1, a1)
    x2, y2, z2 = ecef(la2, lo2, a2)
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    d2 = dx * dx + dy * dy + dz * dz
    d = np.sqrt(d2)
    t = -(x1 * dx + y1 * dy + z1 * dz) / np.maximum(d2, 1e-12)
    cx, cy, cz = x1 + t * dx, y1 + t * dy, z1 + t * dz
    minr = np.sqrt(cx * cx + cy * cy + cz * cz)
    inside = (t > 0) & (t < 1)
    los = ~(inside & (minr < R_E))
    graze = np.where(inside, minr, np.minimum(R_E + a1, R_E + a2)) - R_E
    low1 = a1 <= a2
    ola, olo = np.where(low1, la1, la2), np.where(low1, lo1, lo2)
    ux, uy, uz = ecef(ola, olo, np.zeros_like(ola))
    ox, oy, oz = ecef(ola, olo, np.where(low1, a1, a2))
    tx, ty, tz = (np.where(low1, x2, x1), np.where(low1, y2, y1), np.where(low1, z2, z1))
    lx, ly, lz = tx - ox, ty - oy, tz - oz
    nrm = np.sqrt(lx * lx + ly * ly + lz * lz)
    un = np.sqrt(ux * ux + uy * uy + uz * uz)
    sin_e = (lx * ux + ly * uy + lz * uz) / np.maximum(nrm * un, 1e-12)
    return d, np.degrees(np.arcsin(np.clip(sin_e, -1, 1))), los, graze


def load_weather(gs_names):
    wx = {}
    for n in gs_names:
        df = pd.read_csv(os.path.join(WX, f"{n}.csv"))
        aod = df["aod550"].astype(float)
        aod = aod.fillna(aod.median() if aod.notna().any() else 0.1)
        wx[n] = dict(cloud=df["cloud_cover"].astype(float).fillna(0).values / 100.0,
                     aod=aod.values)
    return wx


def load_era5(gs_names):
    """ERA5 hourly fluxes written by fetch_era5_cds.py; None for stations not downloaded yet."""
    out = {}
    for n in gs_names:
        f = os.path.join(ERA5, f"{n}.csv")
        out[n] = pd.read_csv(f) if os.path.exists(f) else None
    return out


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default=",".join(map(str, DEFAULT_DAYS)))
    ap.add_argument("--variants", default="base,spad200,spad200+cn2x2,spad200+cn2x0.5,spad200+d04")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output dir (default .tmp/channel_v2/out)")
    args = ap.parse_args()
    if args.out:
        OUT = args.out
    days = [int(x) for x in args.days.split(",")]
    variants = args.variants.split(",")
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()

    nodes = pd.read_csv(os.path.join(DS, "node_registry.csv"))
    links = pd.read_csv(os.path.join(DS, "link_registry.csv"))
    ntype = nodes["type"].values
    gs_idx = np.flatnonzero(ntype == "GS")
    gs_names = nodes["name"].values
    wx = load_weather(gs_names[gs_idx])
    era5 = load_era5(gs_names[gs_idx])
    n_era5 = sum(v is not None for v in era5.values())
    if any("era5" in v for v in variants):
        print(f"ERA5 fluxes available for {n_era5}/{len(gs_idx)} ground stations", flush=True)

    with h5py.File(os.path.join(DS, "hap_positions.h5"), "r") as f:
        hap_pos = f["positions"][:]
        hap_ids = f["node_ids"][:]
    fs = h5py.File(os.path.join(DS, "sat_positions.h5"), "r")
    fl = h5py.File(os.path.join(DS, "link_data.h5"), "r")
    sat_ids = fs["node_ids"][:]
    hap_col = {int(n): i for i, n in enumerate(hap_ids)}
    sat_col = {int(n): i for i, n in enumerate(sat_ids)}
    sat_alt = float(np.median(fs["positions"][0:1440:60, :, 2]))

    rng = np.random.default_rng(args.seed)
    # one uniform per GS per hour for the whole year (eq. 13)
    u_cloud = {int(g): rng.random(8760) for g in gs_idx}

    models = {}
    for v in variants:
        p, sc = variant_params(v)
        models[v] = (RateModelV2(p, sat_alt_km=sat_alt), sc)

    rec = {v: {k: [] for k in ("type", "link", "t", "elev", "day", "old", "clr", "exp", "smp", "T")}
           for v in variants}

    for di, d in enumerate(days):
        t0 = d * 1440
        tt = np.arange(t0, t0 + 1440)
        hours = tt // 60
        sat_blk = fs["positions"][t0:t0 + 1440]
        kmax_blk = fl["k_max"][t0:t0 + 1440]
        hap_blk = hap_pos[np.arange(1440) % hap_pos.shape[0]]

        def coords(nid):
            if ntype[nid] == "GS":
                r = nodes.iloc[nid]
                z = np.zeros(1440)
                return (z + r["lat"], z + r["lon"], z)
            if ntype[nid] == "HAP":
                c = hap_blk[:, hap_col[nid]]
            else:
                c = sat_blk[:, sat_col[nid]]
            return (c[:, 0].astype(float), c[:, 1].astype(float), c[:, 2].astype(float))

        for _, lk in links.iterrows():
            lid, u, v_, lt = int(lk["link_id"]), int(lk["node_u"]), int(lk["node_v"]), lk["link_type"]
            cu, cv = coords(u), coords(v_)
            dist, elev, los, graze = geometry(cu, cv)
            old = kmax_blk[:, lid].astype(float)
            if lt in ("SAT-GS", "HAP-GS"):
                vis = los & (elev >= 5.0)
                g = u if ntype[u] == "GS" else v_
                glat, glon = nodes.iloc[g]["lat"], nodes.iloc[g]["lon"]
                sun = sun_elevation_deg(tt, glat, glon)
                cloud = wx[gs_names[g]]["cloud"][hours]
                aod = wx[gs_names[g]]["aod"][hours]
                pclr = p_clear_lund_shanklin(elev, cloud)
                smp = (u_cloud[g][hours] < pclr).astype(float)
            else:
                vis = los & (dist > 0)
                if lt == "SAT-SAT":
                    vis &= graze >= 15.0   # same troposphere block as the current ISL code
                h = u if ntype[u] == "HAP" else (v_ if ntype[v_] == "HAP" else u)
                c = coords(h)
                sun = sun_elevation_deg(tt, c[0], c[1])
                pclr = np.ones(1440)
                smp = np.ones(1440)
                aod = None
            sel = vis | (old > 0)
            if not sel.any():
                continue
            idx = np.flatnonzero(sel)
            for vname, (m, sc) in models.items():
                a_cn2 = None
                if lt in ("SAT-GS", "HAP-GS"):
                    e5 = era5.get(gs_names[g])
                    use_e5 = getattr(m.p, "cn2_source", "daynight") == "era5"
                    if use_e5 and e5 is None:
                        if not getattr(m.p, "era5_fallback", False):
                            raise RuntimeError(f"no ERA5 file for {gs_names[g]}")
                        use_e5 = False
                    if use_e5:
                        cn2_h = cn2_monin_obukhov(e5["H_up_Wm2"].values, e5["ustar"].values, e5["t2m_K"].values,
                                                  e5["sp_Pa"].values, z=m.p.z_ref, cn2_min=m.p.cn2_min,
                                                  zeta_max=m.p.zeta_max)
                        a_cn2 = cn2_h[hours][idx] * sc
                    else:
                        a_cn2 = np.where(sun[idx] > 0, m.p.a_day, m.p.a_night) * sc
                r, parts = m.link_rate(lt, dist[idx] * 1e3, elev[idx], sun_rx_deg=sun[idx],
                                       aod550=None if aod is None else aod[idx], a_cn2=a_cn2,
                                       return_parts=True)
                r = np.where(vis[idx], r, 0.0)
                R = rec[vname]
                R["type"].append(np.full(len(idx), TYPES.index(lt), np.int8))
                R["link"].append(np.full(len(idx), lid, np.int32))
                R["t"].append(tt[idx].astype(np.int32))
                R["elev"].append(elev[idx].astype(np.float32))
                R["day"].append((sun[idx] > 0).astype(np.int8))
                R["old"].append(old[idx].astype(np.float32))
                R["clr"].append(r.astype(np.float32))
                R["exp"].append((r * pclr[idx]).astype(np.float32))
                R["smp"].append((r * smp[idx]).astype(np.float32))
                T = parts.get("T", 0.0)
                R["T"].append(np.broadcast_to(np.asarray(T, np.float32), (len(idx),)).copy())
        print(f"day {d} done ({di + 1}/{len(days)}), {time.time() - t_start:.0f}s", flush=True)

    fs.close()
    fl.close()
    summary = {}
    for vname in variants:
        R = {k: np.concatenate(v) for k, v in rec[vname].items()}
        np.savez_compressed(os.path.join(OUT, f"records_{vname}.npz"), **R)
        summary[vname] = summarize(R, n_steps=len(days) * 1440)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    write_report(summary, variants, days, sat_alt)
    try:
        make_figures(variants[0], np.load(os.path.join(OUT, f"records_{variants[0]}.npz")))
    except Exception as exc:  # noqa: BLE001
        print("figure skipped:", exc)
    print(f"total {time.time() - t_start:.0f}s -> {OUT}")
    return 0


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def summarize(R, n_steps):
    from scipy.stats import spearmanr
    out = {}
    for ti, lt in enumerate(TYPES):
        m = R["type"] == ti
        if not m.any():
            continue
        old, clr, exp_, smp = R["old"][m], R["clr"][m], R["exp"][m], R["smp"][m]
        elev, day, link = R["elev"][m], R["day"][m], R["link"][m]
        o = {}
        o["n_links"] = int(len(np.unique(link)))
        o["old_pos_frac_of_records"] = float((old > 0).mean())
        o["new_pos_frac_of_records"] = float((clr > 0).mean())
        o["both_pos"] = float(((old > 0) & (clr > 0)).mean())
        o["old_only"] = float(((old > 0) & (clr <= 0)).mean())
        o["new_only"] = float(((old <= 0) & (clr > 0)).mean())
        po, pn = old[old > 0], clr[clr > 0]
        o["old_bps"] = dict(p10=pct(po, 10), p50=pct(po, 50), p90=pct(po, 90), mean=float(po.mean()) if len(po) else 0)
        o["new_clear_bps"] = dict(p10=pct(pn, 10), p50=pct(pn, 50), p90=pct(pn, 90), mean=float(pn.mean()) if len(pn) else 0)
        # network-level: mean over all (link, step) incl. zeros, per link
        o["sum_old_per_step"] = float(old.sum() / n_steps)
        o["sum_new_exp_per_step"] = float(exp_.sum() / n_steps)
        o["sum_new_clr_per_step"] = float(clr.sum() / n_steps)
        o["sum_new_smp_per_step"] = float(smp.sum() / n_steps)
        o["active_links_per_step_old"] = float((old > 0).sum() / n_steps)
        o["active_links_per_step_new_smp"] = float((smp > 0).sum() / n_steps)
        df = pd.DataFrame({"link": link, "old": old, "exp": exp_})
        g = df.groupby("link").mean()
        if len(g) > 2:
            o["spearman_linkmean_old_vs_new"] = float(spearmanr(g["old"], g["exp"]).correlation)
        both = (old > 0) & (clr > 0)
        if both.sum() > 10:
            o["spearman_time_old_vs_new"] = float(spearmanr(old[both], clr[both]).correlation)
            o["median_ratio_new_over_old"] = float(np.median(clr[both] / old[both]))
        if lt in ("SAT-GS", "HAP-GS"):
            bins = [5, 10, 15, 20, 30, 45, 60, 90.1]
            rows = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                b = (elev >= lo) & (elev < hi)
                row = dict(bin=f"{lo}-{int(hi)}", n=int(b.sum()))
                for nm, arr in (("old", old), ("new", clr)):
                    for dn, dm in (("night", day == 0), ("day", day == 1)):
                        s = arr[b & dm]
                        row[f"{nm}_{dn}_p50"] = pct(s, 50)
                row["old_mean"] = float(old[b].mean()) if b.any() else float("nan")
                row["new_exp_mean"] = float(exp_[b].mean()) if b.any() else float("nan")
                row["cloud_avail"] = float((exp_[b].sum() / max(clr[b].sum(), 1e-9)))
                rows.append(row)
            o["by_elev"] = rows
            o["T_max"] = float(R["T"][m].max())
            o["T_p99"] = pct(R["T"][m], 99)
        out[lt] = o
    tot_old = sum(out[t]["sum_old_per_step"] for t in out)
    tot_new = sum(out[t]["sum_new_exp_per_step"] for t in out)
    for t in out:
        out[t]["share_old"] = out[t]["sum_old_per_step"] / tot_old if tot_old else 0
        out[t]["share_new_exp"] = out[t]["sum_new_exp_per_step"] / tot_new if tot_new else 0
    return out


def fmt(x, nd=0):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if abs(x) >= 1e6:
        return f"{x / 1e6:.2f}M"
    if abs(x) >= 1e4:
        return f"{x / 1e3:.1f}k"
    return f"{x:.{nd}f}"


def write_report(summary, variants, days, sat_alt):
    L = ["# 速率模型 v2 与现有 k_max 对比（自动生成）", "",
         f"抽样日（年内序号，0 起）：{days}，每天 1440 个时隙；卫星高度中位数 {sat_alt:.0f} km。",
         "old = dataset/global/link_data.h5 的 k_max；new = §2 第二稿模型（clear 为晴空，exp 为乘无云概率的期望，smp 为按式 (13) 抽样）。",
         "近地 Cn² 暂用昼夜两档（Liorni 2019：白天 2.75e-14，夜间 1.10e-14），ERA5 通量需 CDS 账号，尚未接入。", ""]
    for v in variants:
        S = summary[v]
        L += [f"## 变体 `{v}`", "",
              "| 类型 | 链路数 | old>0 | new>0 | 仅old>0 | 仅new>0 | old p50 | new p50 | old p90 | new p90 | new/old 中位比 | 时间Spearman | 链路均值Spearman |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in TYPES:
            if t not in S:
                continue
            o = S[t]
            L.append(f"| {t} | {o['n_links']} | {o['old_pos_frac_of_records']:.2f} | {o['new_pos_frac_of_records']:.2f} | "
                     f"{o['old_only']:.2f} | {o['new_only']:.2f} | {fmt(o['old_bps']['p50'])} | {fmt(o['new_clear_bps']['p50'])} | "
                     f"{fmt(o['old_bps']['p90'])} | {fmt(o['new_clear_bps']['p90'])} | {fmt(o.get('median_ratio_new_over_old'), 2)} | "
                     f"{fmt(o.get('spearman_time_old_vs_new'), 2)} | {fmt(o.get('spearman_linkmean_old_vs_new'), 2)} |")
        L += ["", "| 类型 | 每时隙活跃链路 old | new(抽样) | 每时隙总速率 old | new 期望 | 容量占比 old | new |", "|---|---|---|---|---|---|---|"]
        for t in TYPES:
            if t not in S:
                continue
            o = S[t]
            L.append(f"| {t} | {o['active_links_per_step_old']:.1f} | {o['active_links_per_step_new_smp']:.1f} | "
                     f"{fmt(o['sum_old_per_step'])} | {fmt(o['sum_new_exp_per_step'])} | {o['share_old']:.2f} | {o['share_new_exp']:.2f} |")
        for t in ("SAT-GS", "HAP-GS"):
            if t not in S:
                continue
            L += ["", f"**{t} 按仰角分箱**（p50 为速率>0 与 =0 一起的中位数；T_max={S[t]['T_max']:.3f}, T_p99={S[t]['T_p99']:.3f}）", "",
                  "| 仰角 | 样本 | old 夜 p50 | new 夜 p50 | old 昼 p50 | new 昼 p50 | old 均值 | new 期望均值 | 云可用率 |", "|---|---|---|---|---|---|---|---|---|"]
            for r in S[t]["by_elev"]:
                L.append(f"| {r['bin']} | {r['n']} | {fmt(r['old_night_p50'])} | {fmt(r['new_night_p50'])} | {fmt(r['old_day_p50'])} | "
                         f"{fmt(r['new_day_p50'])} | {fmt(r['old_mean'])} | {fmt(r['new_exp_mean'])} | {r['cloud_avail']:.2f} |")
        L.append("")
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def make_figures(vname, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ti, lt in enumerate(TYPES):
        ax = axes[ti]
        m = R["type"] == ti
        for key, lab in (("old", "old k_max"), ("clr", "new clear"), ("exp", "new expected")):
            x = R[key][m]
            x = np.sort(x[x > 0])
            if len(x):
                ax.plot(x, np.linspace(0, 1, len(x)), label=lab)
        ax.set_xscale("log"); ax.set_title(lt); ax.set_xlabel("bps"); ax.grid(alpha=.3)
    axes[0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"cdf_{vname}.png"), dpi=110)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, lt in zip(axes, ("SAT-GS", "HAP-GS")):
        m = R["type"] == TYPES.index(lt)
        e = R["elev"][m]
        sub = np.random.default_rng(0).choice(np.flatnonzero(m), size=min(40000, m.sum()), replace=False)
        ax.scatter(R["elev"][sub], np.maximum(R["old"][sub], 1e-1), s=1, alpha=.3, label="old k_max")
        ax.scatter(R["elev"][sub], np.maximum(R["clr"][sub], 1e-1), s=1, alpha=.3, label="new clear")
        ax.set_yscale("log"); ax.set_ylim(1, None); ax.set_xlabel("elevation (deg)"); ax.set_ylabel("bps")
        ax.set_title(lt); ax.grid(alpha=.3); ax.legend(markerscale=8)
        del e
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"elev_{vname}.png"), dpi=110)


if __name__ == "__main__":
    sys.exit(main())
