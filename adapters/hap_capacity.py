from __future__ import annotations

"""HAP-GS capacity provider backed by the real StratoQ SKR traces.

Source of truth: ``ref_work/HAP-GS-QKD/K_MAX_checkpoint.pkl`` (72 directed HAP-GS
links x 48 slots, bps), computed by StratoQ's balloon_qnet channel model. We export it
once to ``inputs/raw/hap_kmax_traces.json`` (done) so the framework carries the data itself
and never imports NetSquid or ``ref_work`` at runtime.

Each HAP-GS link in a scenario is assigned one real trace, chosen deterministically from
the pool by hashing the link key, then resampled to the scenario's slot count. This keeps
capacity *shapes* physically faithful without needing StratoQ's exact topology semantics.

If neither the JSON export nor the source pickle is available, a smooth synthetic
diurnal profile is used so the framework still runs.
"""

import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional

from adapters.capacity_provider import resample_trace

_ADAPTER_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ADAPTER_DIR.parent
_REPO_ROOT = _PROJECT_ROOT.parent
_JSON_CACHE = _ADAPTER_DIR / "hap_kmax_traces.json"
_LEGACY_JSON_CACHE = _PROJECT_ROOT / "inputs" / "raw" / "hap_kmax_traces.json"
_PICKLE_SRC = _REPO_ROOT / "ref_work" / "HAP-GS-QKD" / "K_MAX_checkpoint.pkl"


def _load_trace_pool() -> Optional[List[List[float]]]:
    json_cache = _JSON_CACHE if _JSON_CACHE.exists() else _LEGACY_JSON_CACHE
    if json_cache.exists():
        with open(json_cache, "r") as fh:
            return json.load(fh)["traces"]
    if _PICKLE_SRC.exists():
        with open(_PICKLE_SRC, "rb") as fh:
            obj = pickle.load(fh)
        traces = [[float(v) for v in row] for row in obj if max(row) > 0]
        # opportunistically write the neutral cache for next time
        try:
            _JSON_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(_JSON_CACHE, "w") as fh:
                json.dump({"source": str(_PICKLE_SRC), "unit": "bps", "traces": traces}, fh)
        except OSError:
            pass
        return traces
    return None


def _synthetic_diurnal(link_key: str, time_slots: List[int]) -> Dict[int, float]:
    """Fallback: bell-shaped daily profile, HAP-scale peak (~1e4 bps)."""

    seed = int(hashlib.md5(link_key.encode()).hexdigest(), 16)
    peak = 8_000.0 + (seed % 6000)
    phase = (seed % 24) / 24.0
    n = max(len(time_slots), 1)
    out: Dict[int, float] = {}
    for i, t in enumerate(time_slots):
        frac = i / n
        out[int(t)] = peak * (0.4 + 0.6 * math.sin(math.pi * (frac + phase)) ** 4)
    return out


class HAPCapacityProvider:
    """Assigns real StratoQ HAP-GS SKR traces to links (synthetic fallback)."""

    def __init__(self, scale: float = 1.0) -> None:
        self._pool = _load_trace_pool()
        self._scale = float(scale)
        self.using_real_traces = self._pool is not None

    def capacity_by_time(
        self, link_key: str, time_slots: List[int], peak_bps: Optional[float] = None
    ) -> Dict[int, float]:
        if self._pool:
            idx = int(hashlib.md5(link_key.encode()).hexdigest(), 16) % len(self._pool)
            profile = resample_trace(self._pool[idx], time_slots)
            if peak_bps is not None:
                # keep the real diurnal *shape*, renormalize amplitude to a physical peak
                cur_peak = max(profile.values()) or 1.0
                factor = peak_bps / cur_peak
                return {t: v * factor for t, v in profile.items()}
            return {t: v * self._scale for t, v in profile.items()}
        prof = _synthetic_diurnal(link_key, time_slots)
        if peak_bps is not None:
            cur = max(prof.values()) or 1.0
            return {t: v * (peak_bps / cur) for t, v in prof.items()}
        return prof


def hap_skr_peak_bps(distance_km: float) -> float:
    """Distance -> peak HAP-GS SKR (bps). Monotone decreasing, StratoQ-scale.

    ~few thousand bps for the 20-70 km HAP-GS ranges in the regional topology.
    """

    return 4000.0 / (1.0 + (distance_km / 40.0) ** 2)
