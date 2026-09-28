"""Main figure: held-out success rate vs offered load (mean over 30 request seeds x 3 hotspot sets)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

loads = [1.0, 1.5, 2.0, 3.0]
data = {  # held-out means, fixed-loader runs (b14/b15; heldout_b15.py, 2026-09-28)
    "Path-Greedy (expert)":           [0.7658, 0.7403, 0.7008, 0.6290],
    "GNN + attention":                [0.7643, 0.7398, 0.7165, 0.6576],
    "Ours w/o attention":             [0.7800, 0.7702, 0.7640, 0.7288],
    "Ours: GNN + attention + forecast": [0.7996, 0.7871, 0.7703, 0.7365],
}
ub = [0.8536, 0.8440, 0.8348, 0.8155]
style = {
    "Path-Greedy (expert)":       dict(color="#555555", marker="s", ls="-"),
    "GNN + attention":            dict(color="#8da0cb", marker="^", ls="-"),
    "Ours w/o attention":         dict(color="#66c2a5", marker="D", ls="--"),
    "Ours: GNN + attention + forecast": dict(color="#d62728", marker="o", ls="-", lw=2.2),
}
fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)
ax.plot(loads, ub, color="black", ls=":", lw=1.4, label="Fluid-LP upper bound")
for name, ys in data.items():
    xs = [x for x, y in zip(loads, ys) if y is not None]
    ys2 = [y for y in ys if y is not None]
    ax.plot(xs, ys2, label=name, ms=4.5, **style[name])
ax.set_xlabel("Offered load (x baseline arrival rate)")
ax.set_ylabel("Success rate (held-out)")
ax.set_xticks(loads); ax.set_xticklabels(["1x", "1.5x", "2x", "3x"])
ax.set_ylim(0.60, 0.87); ax.grid(alpha=0.3)
ax.legend(fontsize=6.3, loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig("/home/user/paper_figs/fig_main.png"); fig.savefig("/home/user/paper_figs/fig_main.svg")
print("saved")
