"""Publication-ready matplotlib style for QKD-RL figures.

Outputs vector formats (SVG/PDF) suitable for LaTeX/Word manuscripts plus a
high-resolution PNG preview. Keeps a color-blind-safe palette, Times-like
fonts, compact layout and minimal chart junk.
"""
from __future__ import annotations

import matplotlib

# Headless backend: evaluation exports SVG/PDF/PNG files, no GUI needed.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Color-blind safe palette (Okabe-Ito).
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "sky": "#56B4E9",
    "pink": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#8C8C8C",
}
POLICY_COLORS = {
    "random": PALETTE["grey"],
    "greedy_rate": PALETTE["blue"],
    "greedy_demand": PALETTE["orange"],
    "greedy_qkp": PALETTE["green"],
    "graph_mappo": PALETTE["red"],
    "rolling_milp": PALETTE["sky"],
    "MAPPO-MLP": PALETTE["pink"],
}

# Times-like serif stack (fallback order matters on Windows).
FONT_FAMILY = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
FONT_SIZE = 8.0


def apply_paper_style() -> None:
    """Apply manuscript-ready rcParams once."""
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": FONT_FAMILY,
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE + 1,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE - 0.5,
            "ytick.labelsize": FONT_SIZE - 0.5,
            "legend.fontsize": FONT_SIZE - 0.5,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "errorbar.capsize": 2.5,
            "axes.grid": False,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "pdf.fonttype": 42,  # TrueType curves -> clean vector text
            "ps.fonttype": 42,
            "svg.fonttype": "path",  # text as paths so fonts embed anywhere
        }
    )
