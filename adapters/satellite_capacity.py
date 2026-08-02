from __future__ import annotations

"""Satellite-GS and Satellite-HAP capacity providers.

**SAT-GS default: ``TNSMSatelliteCapacityProvider``** — strictly TNSM on both steps: the channel
loss ``FSeff(theta, cc)`` from TNSM's own orbit+diffraction+atmosphere code, and the loss->rate
map from the SatQuMA **finite-key BB84 optimiser** (real nonlinear finite-key rate, not a linear
scaling). Both are baked into ``inputs/raw/sat_gs_tnsm_channel.json`` (785 nm); weather is the
cloud axis of that table. The overhead peak is rescaled to keep the link types comparable.

``SyntheticSatelliteCapacityProvider`` is the older elevation-window placeholder (peak-at-zenith
bell over short passes); it now serves as the base class supplying the shared pass-window /
elevation geometry that the TNSM provider reuses.

A single satellite has ONE pass schedule over a region; to keep SAT-GS and SAT-HAP links
physically coherent they should share the same pass centres (use ``satellite_pass_centers``
and pass ``centers=`` to both providers).

``SatelliteHAPCapacityProvider`` models the SAT->HAP downlink. Because the HAP sits at
~25 km, above most of the atmosphere:
  - it suffers far less atmospheric loss than SAT->GS  -> **higher peak SKR**;
  - the higher vantage sees the satellite above the min-elevation for longer -> a **wider
    visibility window**. At a pass edge the satellite can reach the HAP while it cannot yet
    reach the ground -> this is exactly what lets a HAP relay keys up/down as a backup.

``SatQuMACapacityProvider`` is the hook for the real pipeline (channel/orbital + atmosphere
+ finite-key BB84 from ``ref_work/Satellite-GS-QKD/``); it is intentionally a stub.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

_ADAPTER_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ADAPTER_DIR.parent
_TNSM_CHANNEL_JSON = _ADAPTER_DIR / "sat_gs_tnsm_channel.json"
_LEGACY_TNSM_CHANNEL_JSON = _PROJECT_ROOT / "inputs" / "raw" / "sat_gs_tnsm_channel.json"


def satellite_pass_centers(passes_per_horizon: int = 3, offset: float = 0.0) -> List[float]:
    """Evenly spaced pass centres (as fractions of the horizon) shared by a satellite's links."""

    return [((k + 0.5) / passes_per_horizon + offset) % 1.0 for k in range(passes_per_horizon)]


class SyntheticSatelliteCapacityProvider:
    """Elevation-window visibility model for SAT-GS links."""

    def __init__(
        self,
        passes_per_horizon: int = 3,
        pass_fraction: float = 0.12,
        peak_bps_range: tuple = (600.0, 3000.0),
        shape_exponent: float = 2.0,
        centers: Optional[List[float]] = None,
    ) -> None:
        self.passes_per_horizon = passes_per_horizon
        self.pass_fraction = pass_fraction  # width of a pass as fraction of horizon
        self.peak_lo, self.peak_hi = peak_bps_range
        self.shape_exponent = shape_exponent
        self.centers = centers  # if given, all links share this pass schedule

    def _centres(self, seed: int) -> List[float]:
        if self.centers is not None:
            return self.centers
        # per-link schedule derived from the link key (backward-compatible fallback)
        offset = (seed % 100) / 100.0 * (1.0 / self.passes_per_horizon)
        return [
            (k + 0.5) / self.passes_per_horizon + offset - (1.0 / self.passes_per_horizon) * 0.5
            for k in range(self.passes_per_horizon)
        ]

    def capacity_by_time(self, link_key: str, time_slots: List[int]) -> Dict[int, float]:
        seed = int(hashlib.md5(link_key.encode()).hexdigest(), 16)
        peak = self.peak_lo + (seed % 1000) / 1000.0 * (self.peak_hi - self.peak_lo)
        n = max(len(time_slots), 1)
        width = max(self.pass_fraction, 1.0 / n)
        centres = self._centres(seed)

        out: Dict[int, float] = {}
        for i, t in enumerate(time_slots):
            frac = i / n
            val = 0.0
            for c in centres:
                d = abs(frac - c)
                if d <= width / 2.0:
                    u = 1.0 - (d / (width / 2.0))  # 1 at pass centre (max elevation), 0 at edges
                    # bell peaked at the pass centre: f(1)=1 (max elevation), f(0)=0 (horizon)
                    val = max(val, peak * (math.sin(math.pi * 0.5 * u) ** self.shape_exponent))
            out[int(t)] = val
        return out

    def elevation_by_time(self, link_key: str, time_slots: List[int]) -> Dict[int, float]:
        """Satellite elevation angle (deg) per slot: 90 deg overhead at a pass centre, →0 at
        the window edges (0 outside a pass). Used by the weather layer (loss ~ 1/sin α)."""
        seed = int(hashlib.md5(link_key.encode()).hexdigest(), 16)
        n = max(len(time_slots), 1)
        width = max(self.pass_fraction, 1.0 / n)
        centres = self._centres(seed)
        out: Dict[int, float] = {}
        for i, t in enumerate(time_slots):
            frac = i / n
            best_u = 0.0
            for c in centres:
                d = abs(frac - c)
                if d <= width / 2.0:
                    best_u = max(best_u, 1.0 - d / (width / 2.0))
            out[int(t)] = best_u * 90.0
        return out


class TNSMSatelliteCapacityProvider(SyntheticSatelliteCapacityProvider):
    """SAT-GS capacity strictly aligned to TNSM (785 nm) — both steps of the chain:

    **(1) channel loss** — ``FSeff(theta, cc)`` from TNSM's own channel code
    (``get_losses`` = ``circular_polarorbit`` orbit + ``diffract`` + ``get_f_atm(cc)`` /
    ``Transmittance.csv``); **(2) loss -> rate** — the **SatQuMA finite-key BB84 optimiser**
    (``key_length`` + ``optimise_key``), run over a constant-efficiency block so the key rate
    at each loss is the real finite-key value (nonlinear, with a hard threshold cutoff), NOT a
    linear scaling. Both are baked once into ``inputs/raw/sat_gs_tnsm_channel.json`` as

        skr_shape(theta, cc) = rate_fk(FSeff(theta, cc)) / rate_fk(FSeff(90 deg, 0 %))  in [0, 1]
        SKR(theta, cc)       = peak_bps . skr_shape(theta, cc)

    So the **clear-sky** capacity (cc=0) and the **weather** (cloud) attenuation are the same
    finite-key curve — weather is the ``cc`` axis, not a bolted-on multiplier. The absolute
    finite-key rate at overhead-clear is ~199 kbps (fs=100 MHz, 200 s block; stored in the JSON
    ``finite_key`` block); ``peak_bps`` rescales that so the three link types stay comparable in
    the backup scenario (set ``peak_bps`` to the absolute value for true-magnitude runs).

    Pass ``cloud_by_time={slot: cloud_pct}`` (0-100) per link for the weather; default is clear.
    Share ``centers`` with the SAT-HAP provider for a coherent pass schedule.
    """

    def __init__(
        self,
        passes_per_horizon: int = 3,
        pass_fraction: float = 0.14,
        peak_bps: float = 2975.0,
        min_elev_deg: float = 5.0,
        centers: Optional[List[float]] = None,
        table_path: Optional[Path] = None,
    ) -> None:
        super().__init__(
            passes_per_horizon=passes_per_horizon,
            pass_fraction=pass_fraction,
            centers=centers,
        )
        self.peak_bps = float(peak_bps)
        self.min_elev_deg = float(min_elev_deg)
        resolved_table = table_path or (
            _TNSM_CHANNEL_JSON if _TNSM_CHANNEL_JSON.exists() else _LEGACY_TNSM_CHANNEL_JSON
        )
        with open(resolved_table, "r") as fh:
            tbl = json.load(fh)
        self._theta = [float(x) for x in tbl["theta_grid_deg"]]
        self._cloud = [float(x) for x in tbl["cloud_grid_pct"]]
        self._shape = {int(k): [float(v) for v in row] for k, row in tbl["skr_shape"].items()}
        self.wavelength_nm = float(tbl.get("wavelength_nm", 785.0))
        self.abs_rate_zenith_clear_bps = float(
            tbl.get("finite_key", {}).get("abs_rate_zenith_clear_bps", 0.0)
        )

    @staticmethod
    def _bracket(grid: List[float], x: float):
        """Return (i, j, w) so that value ~ (1-w)*grid[i] + w*grid[j] endpoints."""
        if x <= grid[0]:
            return 0, 0, 0.0
        if x >= grid[-1]:
            return len(grid) - 1, len(grid) - 1, 0.0
        for k in range(1, len(grid)):
            if x <= grid[k]:
                w = (x - grid[k - 1]) / (grid[k] - grid[k - 1])
                return k - 1, k, w
        return len(grid) - 1, len(grid) - 1, 0.0

    def _shape_at(self, theta_deg: float, cloud_pct: float) -> float:
        """Bilinear interpolation of the finite-key rate shape over (elevation, cloud)."""
        ti, tj, tw = self._bracket(self._theta, theta_deg)
        ci, cj, cw = self._bracket(self._cloud, cloud_pct)
        th_lo, th_hi = int(self._theta[ti]), int(self._theta[tj])
        e00, e01 = self._shape[th_lo][ci], self._shape[th_lo][cj]
        e10, e11 = self._shape[th_hi][ci], self._shape[th_hi][cj]
        lo = e00 * (1 - cw) + e01 * cw
        hi = e10 * (1 - cw) + e11 * cw
        return lo * (1 - tw) + hi * tw

    def skr_at(self, theta_deg: float, cloud_pct: float = 0.0) -> float:
        """SAT-GS key rate (bps) at a given elevation and cloud cover (TNSM finite-key)."""
        if theta_deg < self.min_elev_deg:
            return 0.0
        return self.peak_bps * self._shape_at(theta_deg, cloud_pct)

    def capacity_by_time(
        self, link_key: str, time_slots: List[int], cloud_by_time: Optional[Dict[int, float]] = None
    ) -> Dict[int, float]:
        elev = self.elevation_by_time(link_key, time_slots)  # 90 deg at pass centre, ->0 at edges
        out: Dict[int, float] = {}
        for t in time_slots:
            cc = 0.0 if cloud_by_time is None else float(cloud_by_time.get(int(t), 0.0))
            out[int(t)] = self.skr_at(elev[int(t)], cc)
        return out


class SatelliteHAPCapacityProvider(SyntheticSatelliteCapacityProvider):
    """SAT->HAP downlink: higher peak and wider window than SAT->GS (see module docstring)."""

    def __init__(self, centers: Optional[List[float]] = None) -> None:
        super().__init__(
            passes_per_horizon=3,
            pass_fraction=0.30,            # wider window (higher vantage)
            peak_bps_range=(2500.0, 6000.0),  # higher peak (above the atmosphere)
            shape_exponent=1.5,
            centers=centers,
        )


class PhysicalSatelliteHAPProvider:
    """SAT->HAP capacity from the FSO channel physics (``hap_satellite_channel``).

    Maps the satellite pass geometry to a zenith angle per slot (0° at the pass centre =
    overhead, up to ``zenith_max_deg`` at the window edges — the HAP's high vantage gives a
    larger max zenith, hence a wider window) and evaluates the channel SKR there. The peak
    (overhead) and the window width both follow from the geometry, not hand-set constants.
    Share ``centers`` with the SAT-GS provider for a coherent pass schedule.
    """

    def __init__(
        self,
        centers: Optional[List[float]] = None,
        pass_fraction: float = 0.30,
        zenith_max_deg: float = 85.0,
        channel=None,
    ) -> None:
        from adapters.hap_satellite_channel import HAPSatelliteChannel
        self.centers = centers
        self.pass_fraction = pass_fraction
        self.zenith_max_deg = zenith_max_deg
        self.channel = channel or HAPSatelliteChannel()

    def _centres(self) -> List[float]:
        return self.centers if self.centers is not None else satellite_pass_centers(3)

    def capacity_by_time(self, link_key: str, time_slots: List[int]) -> Dict[int, float]:
        n = max(len(time_slots), 1)
        width = max(self.pass_fraction, 1.0 / n)
        centres = self._centres()
        out: Dict[int, float] = {}
        for i, t in enumerate(time_slots):
            frac = i / n
            best = 0.0
            for c in centres:
                d = abs(frac - c)
                if d <= width / 2.0:
                    u = d / (width / 2.0)                    # 0 at centre, 1 at edge
                    zenith = u * self.zenith_max_deg          # overhead -> horizon
                    best = max(best, self.channel.skr_bps(zenith))
            out[int(t)] = best
        return out


class SatQuMACapacityProvider:
    """Superseded by ``TNSMSatelliteCapacityProvider`` (kept for reference only).

    The full TNSM chain — orbit (``circular_polarorbit``) + diffraction + atmosphere
    (``get_losses``) + finite-key BB84 optimiser (``key_length``/``optimise_key``) — now runs
    offline in ``scratchpad/gen_finitekey_table.py`` and its output feeds
    ``TNSMSatelliteCapacityProvider``. The external orbit-propagator trace turned out NOT to be
    required: TNSM's own analytic orbit substitutes for it. Original stub notes below.
    To implement directly, feed the ``ref_work/Satellite-GS-QKD/`` engine:
      - an orbit/elevation trace per (GS, RAAN) : columns ``currentTime, elevationAngle,
        range`` (produced by an external propagator; not in the repo),
      - atmosphere transmittance vs elevation x cloud-cover (``channel/atmosphere/
        Transmittance.csv``),
      - diffraction + finite-key BB84 ``key/protocols/key_efficient_BB84.key_length``,
    then integrate the per-slot rate. Output must be ``{slot: bps}`` to match the
    synthetic provider so it is a drop-in replacement.
    """

    def capacity_by_time(self, link_key: str, time_slots: List[int]) -> Dict[int, float]:
        raise NotImplementedError(
            "SatQuMA real-capacity path requires an external orbit propagator trace; "
            "use SyntheticSatelliteCapacityProvider until those inputs are supplied."
        )
