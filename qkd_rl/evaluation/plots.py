"""Publication-ready figures from evaluator records (vector output)."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Suppress harmless libpng iCCP warnings from matplotlib figure saves
warnings.filterwarnings("ignore", message=".*iCCP.*")

from qkd_rl.evaluation.evaluator import EpisodeRecord
from qkd_rl.evaluation.style import POLICY_COLORS, apply_paper_style


def read_train_history(path: Path) -> list[dict]:
    """Parse a ``metrics.jsonl`` file produced by :class:`MAPPOTrainer`.

    Returns only training rows (records carrying an ``update`` key), in file
    order; per-update eval records are skipped.
    """
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "update" in record and "mean_reward" in record:
                rows.append(record)
    return rows


def _rolling_smooth(values, window: int) -> np.ndarray:
    """Centred moving average, keeping edge points raw.

    ``np.convolve(mode="same")`` with a window as large as the series applies
    a triangular weight profile to the ends, which can look like a real
    learning-curve rise and fall even on flat data. Using ``mode="valid"``
    and leaving the edge points unsmoothed avoids that artifact.
    """
    arr = np.asarray(values, dtype=float)
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    smooth = np.convolve(arr, kernel, mode="valid")
    out = arr.copy()
    pad = (len(arr) - len(smooth)) // 2
    out[pad : pad + len(smooth)] = smooth
    return out


def _series_grid(histories: list[list[dict]], field: str) -> tuple[np.ndarray, np.ndarray]:
    """Align per-run series on a shared update axis -> (updates, runs x updates)."""
    updates = sorted({int(record["update"]) for run in histories for record in run})
    matrix = np.full((len(updates), len(histories)), np.nan)
    for j, run in enumerate(histories):
        by_update = {int(record["update"]): float(record[field]) for record in run if field in record}
        for i, update in enumerate(updates):
            if update in by_update:
                matrix[i, j] = by_update[update]
    return np.asarray(updates, dtype=float), matrix


def plot_learning_curve(
    histories: list[list[dict]] | list[dict],
    path: Path,
    title: str = "",
    window: int = 20,
) -> list[Path]:
    """Training curves with rolling mean and seed confidence bands.

    ``histories`` is one training run (list of update records) or several
    runs across seeds; when several runs are given, each panel shows the
    mean curve plus a one-std band (shaded).
    """
    apply_paper_style()
    if histories and isinstance(histories[0], dict):
        histories = [histories]
    fields = [
        ("mean_reward", "Mean reward"),
        ("mean_success_rate", "Success rate"),
        ("mean_served_keys", "Served keys"),
        ("actor_loss", "Actor loss"),
        ("critic_loss", "Critic loss"),
        ("entropy", "Entropy"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.7))
    for ax, (field, label) in zip(axes.flat, fields):
        updates, matrix = _series_grid(histories, field)
        if matrix.size == 0:
            ax.set_visible(False)
            continue
        mean = np.nanmean(matrix, axis=1)
        std = np.nanstd(matrix, axis=1)
        smooth = _rolling_smooth(mean, window)
        ax.plot(updates, smooth, color=POLICY_COLORS["graph_mappo"], linewidth=1.2)
        if len(histories) > 1:
            band = _rolling_smooth(std, window)
            ax.fill_between(
                updates,
                smooth - band,
                smooth + band,
                color=POLICY_COLORS["graph_mappo"],
                alpha=0.15,
                linewidth=0.0,
            )
        ax.set_title(label, pad=3)
        ax.set_xlabel("Update")
        ax.margins(x=0.01)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout(pad=0.7)
    if title:
        fig.suptitle(title, fontsize=9, y=1.02)
    return save_figure(fig, path)


def save_figure(fig, path: Path) -> list[Path]:
    """Save a figure as SVG + PDF (vector, for manuscripts) and PNG (preview)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ("svg", "pdf", "png"):
        out = path.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", transparent=False)
        outputs.append(out)
    plt.close(fig)
    return outputs


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0) if len(arr) > 1 else 0.0)


def plot_policy_comparison(records: list[EpisodeRecord], path: Path, title: str = "") -> list[Path]:
    """Grouped bar chart with error bars: one panel per headline metric."""
    apply_paper_style()
    metrics = [
        ("success_rate", "Success rate", "ratio", 0.0),
        ("total_reward", "Total reward", "", None),
        ("served_keys", "Served keys", "", None),
        ("failed_keys", "Failed keys", "", None),
        ("conflict_count", "Conflicts", "count", 0.0),
    ]
    policies: list[str] = []
    by_policy: dict[str, list[EpisodeRecord]] = {}
    for record in records:
        by_policy.setdefault(record.policy, []).append(record)
    policies = sorted(by_policy)

    n = len(metrics)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8, 1.15 * nrows + 1.6))
    axes = list(axes.flat)
    for extra in range(n, len(axes)):
        axes[extra].set_visible(False)
    x = np.arange(len(policies))
    for ax, (attr, label, _unit, _floor) in zip(axes, metrics):
        means = [_mean_std([getattr(r, attr) for r in by_policy[p]])[0] for p in policies]
        stds = [_mean_std([getattr(r, attr) for r in by_policy[p]])[1] for p in policies]
        colors = [POLICY_COLORS.get(p, "#333333") for p in policies]
        ax.bar(x, means, yerr=stds, width=0.62, color=colors, edgecolor="black", linewidth=0.4, capsize=2.5)
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=18, ha="right", fontsize=6.5)
        ax.set_title(label, pad=3)
        ax.margins(y=0.18)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout(pad=0.8)
    if title:
        fig.suptitle(title, fontsize=9, y=1.02)
    return save_figure(fig, path)


def _rolling_mean(values, window: int = 20) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def plot_episode_timeline(
    step_rows_by_policy: dict[str, list[dict]],
    path: Path,
    episode_index: int = 0,
    title: str = "",
) -> list[Path]:
    """Per-step curves for one episode of each policy.

    Panels: reward (raw + rolling mean), served/failed keys, QKP utilization.
    """
    apply_paper_style()
    fig, axes = plt.subplots(3, 1, figsize=(4.6, 4.8), sharex=True)
    for policy, rows in step_rows_by_policy.items():
        rows = [r for r in rows if r["episode"] == episode_index]
        if not rows:
            continue
        steps = [r["step"] for r in rows]
        reward = [r["reward"] for r in rows]
        served = np.cumsum([r["served_keys"] for r in rows])
        failed = np.cumsum([r["failed_keys"] for r in rows])
        qkp = [r["qkp_utilization"] for r in rows]
        color = POLICY_COLORS.get(policy, "#333333")

        axes[0].plot(steps, reward, color=color, linewidth=0.5, alpha=0.35)
        axes[0].plot(steps, _rolling_mean(reward), color=color, label=policy, linewidth=1.2)
        axes[1].plot(steps, served, color=color, linestyle="-", label=f"{policy} (served)")
        axes[1].plot(steps, failed, color=color, linestyle="--", label=f"{policy} (failed)")
        axes[2].plot(steps, qkp, color=color, linewidth=1.1, label=policy)

    axes[0].set_ylabel("Reward / step")
    axes[0].legend(ncol=len(step_rows_by_policy), loc="upper right", fontsize=6.5)
    axes[1].set_ylabel("Cumulative keys")
    axes[1].legend(ncol=2, loc="upper left", fontsize=6.5)
    axes[2].set_ylabel("QKP utilization")
    axes[2].set_xlabel("Time slot t")
    fig.align_ylabels(axes)
    fig.tight_layout(pad=0.6)
    if title:
        fig.suptitle(title, fontsize=9, y=1.0)
    return save_figure(fig, path)

