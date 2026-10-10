"""Summarize and plot the controlled study; run on the server."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()
root = args.directory
with (root/"static_grid.csv").open() as f:
    grid = list(csv.DictReader(f))
with (root/"daily_summary.csv").open() as f:
    summary = list(csv.DictReader(f))
with (root/"links.csv").open() as f:
    links = list(csv.DictReader(f))
types = np.array([r["link_type"] for r in links])
checks = []


def check(name, ok):
    checks.append(dict(name=name,passed=bool(ok)))


with h5py.File(root/"clear.h5") as clear:
    for scenario in ("aerosol", "strong_turbulence", "cirrus", "broken_cloud", "overcast"):
        with h5py.File(root/f"{scenario}.h5") as other:
            ground = np.isin(types,["SAT-GS","HAP-GS"])
            for key in ("conditional_finite60_bps", "conditional_proxy_bps", "released300_bits"):
                check(f"{scenario}: non-ground {key} unchanged",
                      np.allclose(clear[key][:,~ground],other[key][:,~ground],equal_nan=True))
            if scenario in ("aerosol", "strong_turbulence", "cirrus"):
                check(f"{scenario}: ground conditional rate does not increase",
                      np.all(other["conditional_finite60_bps"][:,ground] <= clear["conditional_finite60_bps"][:,ground]+1e-6))
            else:
                check(f"{scenario}: cloud gate does not change conditional rate",
                      np.array_equal(other["conditional_finite60_bps"][:],clear["conditional_finite60_bps"][:]))
            if scenario == "overcast":
                for key in ("released60_bits","released300_bits"):
                    check(f"overcast ground {key} zero",np.all(other[key][:,ground] == 0))
    for key in ("conditional_finite60_bps", "conditional_proxy_bps", "released60_bits", "released300_bits"):
        check(f"clear {key} finite nonnegative",np.all(np.isfinite(clear[key][:])) and np.all(clear[key][:]>=0))

for row in grid:
    if row["moment_compatible"] == "True":
        check(f"stationary block monotonic {row['scenario']} {row['link_type']} {row['day']} {row['distance_km']}",
              float(row["conditional_finite_300_bps"])+1e-6 >= float(row["conditional_finite_60_bps"]))
lookup = {(r["scenario"],r["link_type"],r["day"],r["distance_km"]):r for r in grid}
for row in grid:
    if row["day"] == "True" and row["moment_compatible"] == "True":
        night = lookup[(row["scenario"],row["link_type"],"False",row["distance_km"])]
        check(f"day no better than night {row['scenario']} {row['link_type']} {row['distance_km']}",
              float(row["conditional_finite_60_bps"]) <= float(night["conditional_finite_60_bps"])+1e-6)

pass_rows = []
for scenario in sorted(set(r["scenario"] for r in summary)):
    with h5py.File(root/f"{scenario}.h5") as f:
        visible = f["geometry_visible"][:].astype(bool)
        bits = f["released300_bits"][:]
        for col in np.flatnonzero(types == "SAT-GS"):
            boundaries = np.diff(np.r_[False,visible[:,col],False].astype(int))
            for start,end in zip(np.flatnonzero(boundaries==1),np.flatnonzero(boundaries==-1)):
                # A deactivated contact releases its last block in slot end.
                key_bits = bits[start:min(end+1,len(bits)),col].sum()
                pass_rows.append(dict(scenario=scenario,link_id=int(col),start_slot=int(start),
                    end_slot_exclusive=int(end),contact_seconds=int((end-start)*60),
                    bits_300s_blocks=float(key_bits),effective_contact_bps=float(key_bits/((end-start)*60))))
with (root/"sat_ground_contacts.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(pass_rows[0])); writer.writeheader(); writer.writerows(pass_rows)

fig, axes = plt.subplots(2,2,figsize=(11,8),layout="constrained")
for ax,lt in zip(axes.flat,("HAP-GS","SAT-GS","SAT-HAP","SAT-SAT")):
    rows=[r for r in grid if r["scenario"]=="clear" and r["link_type"]==lt and r["day"]=="False"]
    rows.sort(key=lambda r:float(r["distance_km"]))
    x=np.array([float(r["distance_km"]) for r in rows])
    for key,label in (("conditional_finite_60_bps","Finite, 60 s"),("conditional_finite_300_bps","Finite, 300 s"),("conditional_proxy_bps","Asymptotic proxy")):
        y=np.array([float(r[key]) for r in rows])
        ax.plot(x,y,marker="o",label=label)
    ax.set_yscale("symlog",linthresh=10)
    ax.set(title=lt,xlabel="Slant distance (km)",ylabel="Conditional key rate (bit/s)")
    ax.grid(True,alpha=.25)
    ax.legend(fontsize=8)
fig.suptitle("Controlled clear-night links: stationary geometry, no acquisition loss")
fig.savefig(root/"rate_distance.png",dpi=180)
fig.savefig(root/"rate_distance.pdf")
plt.close(fig)

fig, ax = plt.subplots(figsize=(10,4),layout="constrained")
scenarios=["clear","aerosol","strong_turbulence","cirrus","broken_cloud","overcast"]
for i,lt in enumerate(("HAP-GS","SAT-GS","SAT-HAP","SAT-SAT")):
    rows={r["scenario"]:r for r in summary if r["link_type"]==lt}
    ref=float(rows["clear"]["total_released300_bits"])
    ax.bar(np.arange(len(scenarios))+(i-1.5)*.19,
           [float(rows[s]["total_released300_bits"])/ref for s in scenarios],width=.19,label=lt)
ax.set_xticks(np.arange(len(scenarios)),["Clear","Aerosol","Stronger\nturbulence","Cirrus","50% low/mid\ncloud input","Overcast"])
ax.set(ylabel="24 h released key / clear scenario",ylim=(0,1.1))
ax.set_title("Independent contacts, 300 s finite blocks, 5 s acquisition")
ax.legend(ncol=4); ax.grid(axis="y",alpha=.2)
fig.savefig(root/"weather_response.png",dpi=180)
fig.savefig(root/"weather_response.pdf")
plt.close(fig)
(root/"checks.json").write_text(json.dumps(checks,indent=2))
print(json.dumps(dict(checks=len(checks),passed=sum(x["passed"] for x in checks),
    failed=[x for x in checks if not x["passed"]]),indent=2))
if not all(item["passed"] for item in checks):
    raise SystemExit("Short-rate study consistency checks failed; inspect checks.json")
