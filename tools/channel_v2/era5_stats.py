"""Per-station ERA5 near-ground Cn2 statistics (paper Table 4). Read-only; prints markdown."""
import os, sys, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rate_model_v2 import cn2_monin_obukhov
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
reg = pd.read_csv(os.path.join(ROOT, "dataset", "global", "node_registry.csv"))
reg = reg[reg["type"] == "GS"]
E = os.path.join(ROOT, ".tmp", "channel_v2", "era5")
rows = []; allz = {z: {"d": [], "n": [], "f": []} for z in (2, 5, 10, 20)}
for _, r in reg.iterrows():
    f = os.path.join(E, r["name"] + ".csv")
    if not os.path.exists(f):
        rows.append((r["name"], None)); continue
    d = pd.read_csv(f, parse_dates=["time"])
    lt = (d.time.dt.hour + d.time.dt.minute / 60 + r["lon"] / 15) % 24
    day = ((lt >= 10) & (lt <= 15)).values; night = ((lt >= 22) | (lt <= 3)).values
    res = {"n": len(d), "miss": int(d.isna().sum().sum()),
           "H13": float(d.H_up_Wm2[(lt >= 12) & (lt < 14)].median()), "us": float(d.ustar.median())}
    for z in (2, 5, 10, 20):
        c = cn2_monin_obukhov(d.H_up_Wm2.values, d.ustar.values, d.t2m_K.values, d.sp_Pa.values, z=float(z))
        allz[z]["d"].append(c[day]); allz[z]["n"].append(c[night]); allz[z]["f"].append(c <= 1.0001e-16)
        if z == 10:
            res.update(day=np.median(c[day]), night=np.median(c[night]), p90=np.percentile(c, 90), floor=np.mean(c <= 1.0001e-16))
    rows.append((r["name"], res))
print("| 站 | 纬度 | 小时数 | 缺测 | 当地 13 时 H 中位 (W/m²) | u* 中位 (m/s) | Cn² 白天中位 | Cn² 夜间中位 | Cn² p90 | 触下限比例 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for (name, res), lat in zip(rows, reg["lat"]):
    if res is None:
        print(f"| {name} | {lat:.1f} | — | — | — | — | — | — | — | — |"); continue
    print(f"| {name} | {lat:.1f} | {res['n']} | {res['miss']} | {res['H13']:.0f} | {res['us']:.2f} | {res['day']:.1e} | {res['night']:.1e} | {res['p90']:.1e} | {res['floor']:.3f} |")
print()
print("| z_ref | 白天中位 | 夜间中位 | 触下限比例 |"); print("|---|---|---|---|")
for z in (2, 5, 10, 20):
    if not allz[z]["d"]: continue
    print(f"| {z} m | {np.median(np.concatenate(allz[z]['d'])):.1e} | {np.median(np.concatenate(allz[z]['n'])):.1e} | {np.mean(np.concatenate(allz[z]['f'])):.3f} |")
print("stations:", sum(r is not None for _, r in rows))
