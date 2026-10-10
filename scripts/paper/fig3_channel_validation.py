# -*- coding: utf-8 -*-
"""LEGACY Fig. 3: historical kappa_x=1 model and global_v2c distributions.

2026-10-04: this script is kept ONLY for historical reproducibility. It is not a
validation of the revised main model (kappa_x=0). Outputs go to a legacy subfolder;
new main-model rate curves and dataset histograms must be regenerated separately.

(a) Micius anchor (Liao 2017, 848.6 nm, 1200 km / elev 19.8 deg):
    model beam-spread+pointing loss vs the reported 22 dB diffraction (+<3 dB pointing);
    model clear-sky atmosphere vs the reported 3-8 dB "absorption + turbulence".
    NOTE: the excess term kappa_x is CALIBRATED from this residual (supplementary A.4),
    so only the beam/pointing comparison is an independent check; the figure states this.
(b) Model S-G slot key rate vs elevation (Table I, 1550 nm, clear night) with the
    reported final-key averages of Micius (1.1 kbps) and Jinan-1 (~3 kbps over the
    360 s window of a pass, 1.07 Mbit) as order-of-magnitude references.
(c) Distribution of positive per-slot rates of the four link types in dataset/global_v2c
    (time stride 10 -> 52,560 of 525,600 slots), with positive-sample shares.

Outputs: docs/images/legacy_kappa_x1/fig3_channel_validation.{pdf,png} + fig3_data.json.
Run:     conda run -n pytorch python scripts/paper/fig3_channel_validation.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

if not hasattr(np, "trapezoid"):      # numpy < 2.0 (pytorch env)
    np.trapezoid = np.trapz

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "channel_v2"))
sys.path.insert(0, os.path.join(ROOT, "tools", "dataset_v2"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rate_model_v2 import ParamsV2, RateModelV2, airmass_kasten_young, rayleigh_tau  # noqa: E402
from finite_key import FiniteKeyParams, slot_rate  # noqa: E402

OUT_DIR = os.path.join(ROOT, "docs", "images", "legacy_kappa_x1")
os.makedirs(OUT_DIR, exist_ok=True)
DATA = {"model_status": "legacy_not_revised_main", "kappa_x_db": 1.0, "dataset": "global_v2c"}

# ---------------------------------------------------------------- (a) Micius anchors
# Liao 2017 (main text): tx 300 mm Cassegrain, far-field divergence ~10 urad (full),
# rx 1.0 m; tracking ~1.2 urad; at 1200 km: diffraction 22 dB, pointing < 3 dB,
# atmosphere+turbulence 3-8 dB. Geometry (suppl. A.4): 1200 km <-> elev 19.8 deg, m=2.93.
LAM_M = 848.6e-9
L_M = 1200e3
ELEV_M = 19.8
A_RX_M = 0.5                      # 1.0 m aperture radius
SIG_M = 1.2e-6                    # tracking accuracy [rad]
THETA_HALF = 5e-6                 # ~10 urad full-angle divergence
W0_M = LAM_M / (math.pi * THETA_HALF)   # Gaussian-beam waist giving that divergence

k_m = 2 * math.pi / LAM_M
lam0 = 2 * L_M / (k_m * W0_M ** 2)
W2 = W0_M ** 2 * (1 + lam0 ** 2)            # vacuum diffraction only (downlink Gamma ~ 0)
bp = 1 - math.exp(-2 * A_RX_M ** 2 / (W2 + 4 * (SIG_M * L_M) ** 2))
bp_db = -10 * math.log10(bp)
bp_db_nopoint = -10 * math.log10(1 - math.exp(-2 * A_RX_M ** 2 / W2))

# clear-sky atmosphere at 848.6 nm, Xinglong winter (suppl. A.4): AOD550 = 0.150, EAE 1.07
m_air = airmass_kasten_young(ELEV_M)
tau_r = rayleigh_tau(LAM_M * 1e6)
tau_aer = 0.150 * (LAM_M * 1e9 / 550.0) ** (-1.07)
atm_clear_db = 10 * math.log10(math.e) * (tau_r + tau_aer) * float(m_air)
kx_range_850 = (0.70, 1.99, 3.29)           # dB per excess air mass (calibrated)
excess_db = tuple(kx * (float(m_air) - 1.0) for kx in kx_range_850)

DATA["a"] = dict(model_beam_pointing_db=bp_db, model_beam_only_db=bp_db_nopoint,
                 liao_diffraction_db=22.0, liao_pointing_max_db=3.0,
                 model_atm_clear_db=atm_clear_db, model_excess_db=excess_db,
                 liao_atm_turb_db=(3.0, 8.0), airmass=float(m_air),
                 w0_m=W0_M, note="kappa_x calibrated from the Liao residual (not independent)")

# ---------------------------------------------------------------- (b) rate vs elevation
p = ParamsV2(kappa_x_db=1.0)  # explicit historical setting; never inherit the revised default
model = RateModelV2(p, sat_alt_km=567.0)
fk = FiniteKeyParams()
R_E = 6371e3
H_S = 567e3

def slant(elev_deg):
    e = np.radians(elev_deg)
    return np.sqrt((R_E + H_S) ** 2 - (R_E * np.cos(e)) ** 2) - R_E * np.sin(e)

elev = np.linspace(5.0, 90.0, 120)
Ls = slant(elev)
_, parts = model.link_rate("SAT-GS", Ls, elev, sun_rx_deg=np.full_like(elev, -10.0),
                           aod550=np.full_like(elev, 0.10),
                           a_cn2=np.full_like(elev, p.a_night), return_parts=True)
r_fk = slot_rate(parts["eta"], parts["y0"], fk)
DATA["b"] = dict(elev=elev.tolist(), rate_bps=np.asarray(r_fk).tolist(),
                 micius_final_bps=1.1e3, jinan_avg_bps=1.07e6 / 360.0,  # order-of-magnitude: best-pass total over the 360 s demo window
                 eta_db_at_30=float(-10 * np.log10(parts["eta"][np.argmin(abs(elev - 30))])))

# ---------------------------------------------------------------- (c) dataset histograms
import csv
import h5py

DS = os.path.join(ROOT, "dataset", "global_v2c")
with open(os.path.join(DS, "link_registry.csv"), newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
ltype = np.array([r["link_type"] for r in rows])
TYPES = ["HAP-GS", "SAT-GS", "SAT-HAP", "SAT-SAT"]
cols = {t: np.nonzero(ltype == t)[0] for t in TYPES}

STRIDE, CHUNK = 10, 4000
edges = np.arange(0.0, 6.6 + 1e-9, 0.1)     # log10(bit/s) bins
hists = {t: np.zeros(len(edges) - 1, dtype=np.int64) for t in TYPES}
n_all = {t: 0 for t in TYPES}
n_pos = {t: 0 for t in TYPES}
with h5py.File(os.path.join(DS, "link_data.h5"), "r") as h5:
    k = h5["k_max"]
    T = k.shape[0]
    for s in range(0, T, CHUNK * STRIDE):
        sl = k[s:s + CHUNK * STRIDE:STRIDE, :]
        for t in TYPES:
            block = sl[:, cols[t]]
            n_all[t] += block.size
            pos = block[block > 0]
            n_pos[t] += pos.size
            if pos.size:
                hists[t] += np.histogram(np.log10(pos), bins=edges)[0]
DATA["c"] = dict(edges=edges.tolist(),
                 hists={t: hists[t].tolist() for t in TYPES},
                 share={t: n_pos[t] / max(n_all[t], 1) for t in TYPES},
                 n_links={t: int(cols[t].size) for t in TYPES}, stride=STRIDE)

# ---------------------------------------------------------------- draw
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9.5,
                     "legend.fontsize": 7.5, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.1))

ax = axes[0]
xs = [0, 1]
model_vals = [bp_db, atm_clear_db + excess_db[1]]
ax.bar([x - 0.17 for x in xs], [bp_db, atm_clear_db], width=0.34,
       color=["#4878a8", "#4878a8"], label="model (independent part)")
ax.bar(xs[1] - 0.17, excess_db[1], width=0.34, bottom=atm_clear_db, color="#a8c4dd",
       label=r"model excess $\kappa_x(m{-}1)$ (calibrated)")
ax.errorbar(xs[1] - 0.17, atm_clear_db + excess_db[1],
            yerr=[[excess_db[1] - excess_db[0]], [excess_db[2] - excess_db[1]]],
            fmt="none", ecolor="#2b4a68", capsize=3, lw=1)
ax.errorbar(xs[0] + 0.17, 22.0 + 1.5, yerr=[[1.5], [1.5]], fmt="s", ms=4.5,
            color="#c44e52", capsize=3, lw=1, label="Liao 2017 (reported)")
ax.errorbar(xs[1] + 0.17, 5.5, yerr=[[2.5], [2.5]], fmt="s", ms=4.5,
            color="#c44e52", capsize=3, lw=1)
ax.set_xticks(xs)
ax.set_xticklabels(["beam spread\n+ pointing", "atmosphere\n+ turbulence"])
ax.set_ylabel("loss at 1200 km (dB)")
ax.set_title("(a) Micius anchor, 848.6 nm, elev. 19.8$^\\circ$")
ax.legend(frameon=False, loc="upper left")
ax.set_ylim(0, 34)

ax = axes[1]
ax.semilogy(elev, np.maximum(np.asarray(r_fk), 1.0), color="#4878a8", lw=1.6,
            label="model S–G, 1550 nm,\nclear night (Table I)")
ax.axhline(1.1e3, color="#c44e52", ls="--", lw=1.1)
ax.text(23, 1.35e3, "Micius final key, 1.1 kbps", color="#c44e52", fontsize=7.5)
ax.axhline(1.07e6 / 360.0, color="#dd8452", ls="--", lw=1.1)
ax.text(38, 3.9e3, "Jinan-1, $\\approx$3 kbps (up to 1.07 Mbit / pass)", color="#dd8452", fontsize=7.5)
ax.set_xlabel("elevation (deg)")
ax.set_ylabel("slot key rate (bit/s)")
ax.set_title("(b) model rate vs. reported final keys")
ax.set_xlim(5, 90)
ax.set_ylim(1, 3e5)
ax.legend(frameon=False, loc="lower right")

ax = axes[2]
colors = {"HAP-GS": "#4878a8", "SAT-GS": "#c44e52", "SAT-HAP": "#55a868", "SAT-SAT": "#8172b3"}
centers = (edges[:-1] + edges[1:]) / 2
for t in TYPES:
    h = hists[t].astype(float)
    if h.sum() == 0:
        continue
    ax.plot(centers, h / h.sum() / 0.1, color=colors[t], lw=1.4,
            label=f"{t} ({100 * DATA['c']['share'][t]:.1f}% > 0)")
ax.set_xlabel(r"$\log_{10}$ slot rate (bit/s), positive samples")
ax.set_ylabel("probability density")
ax.set_title("(c) dataset global_v2c, 2023, stride 10")
ax.legend(frameon=False, fontsize=7)
ax.set_xlim(0, 6.6)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_channel_validation.pdf"))
fig.savefig(os.path.join(OUT_DIR, "fig3_channel_validation.png"), dpi=200)
with open(os.path.join(OUT_DIR, "fig3_data.json"), "w", encoding="utf-8") as f:
    json.dump(DATA, f, ensure_ascii=False, indent=1)
print("model beam+pointing @1200km: %.2f dB (beam only %.2f) vs Liao 22(+<3)" % (bp_db, bp_db_nopoint))
print("clear-sky atm: %.2f dB; excess mid: %.2f dB; airmass %.2f" % (atm_clear_db, excess_db[1], m_air))
print("rate@30deg: %.0f bit/s; eta@30: %.1f dB" % (float(np.interp(30, elev, r_fk)), DATA["b"]["eta_db_at_30"]))
print("positive shares:", {t: round(DATA["c"]["share"][t], 4) for t in TYPES})
print("saved ->", OUT_DIR)
