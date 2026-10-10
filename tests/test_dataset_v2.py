# -*- coding: utf-8 -*-
"""tools/dataset_v2: finite-key length, orbit propagation and an end-to-end smoke build."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "dataset_v2"))
sys.path.insert(0, str(ROOT / "tools" / "data_generation"))

import build_dataset_v2 as B  # noqa: E402
import finite_key as F  # noqa: E402

# Scalar-reference cases, with the v7.5 background/signal overlap convention
# T_mu = Y0/2 + e_d*(1-Y0)*(1-exp(-eta*mu)). The strongest case gains 3 bits
# relative to the older convention that omitted (1-Y0) on the signal term.
REFERENCE = [(-14.1, 5e-7, 17971663.0), (-32.0, 5e-7, 220133.0), (-35.8, 5e-7, 64955.0),
             (-40.0, 5e-7, 1668.0), (-34.3, 5.8e-6, 31802.0), (-34.3, 8e-7, 103545.0),
             (-40.3, 5e-7, 0.0), (-35.5, 5.8e-6, 0.0)]


def test_key_length_matches_scalar_reference():
    db = np.array([r[0] for r in REFERENCE])
    y0 = np.array([r[1] for r in REFERENCE])
    got = F.key_length(10 ** (db / 10), y0)
    np.testing.assert_array_equal(got, [r[2] for r in REFERENCE])


def test_key_length_is_monotone_in_transmittance_and_zero_far_away():
    eta = 10 ** (np.linspace(-60, -1, 400) / 10)
    ell = F.key_length(eta, 5e-7)
    assert ell[0] == 0.0 and ell[-1] > 0
    assert np.all(np.diff(ell) >= 0)
    assert np.all(F.key_length(eta, 1e-3) <= ell)       # more background never helps
    assert F.slot_rate(10 ** -3.58, 5e-7) == pytest.approx(1083.0, abs=2)   # paper III-B: ~1.1 kbps


def test_propagator_matches_dataset_builder():
    bgt = pytest.importorskip("build_global_tensor")
    from legacy_data import GlobalSatellites
    t = np.array([0.0, 61.0, 3600.0, 86399.0, 525600.0 * 60.0])
    for name in ("legacy", "walker"):
        for cfg in GlobalSatellites.get_satellites(name):
            a = np.array(B.propagate(cfg, t))
            b = np.array(bgt._vectorized_sat_positions(cfg, t))
            assert np.abs(a - b).max() < 1e-6


def _write_inputs(tmp: Path) -> Path:
    st = tmp / "stations.csv"
    st.write_text("name,lat,lon,continent,base,alt_km,h_ERA5\nAlpha,48.85,2.35,Europe,1,0.1,100\nBeta,40.42,-3.70,Europe,0,0.6,600\n",
                  encoding="utf-8")
    for sub in ("wx", "era5"):
        (tmp / sub).mkdir()
    hours = [(datetime(2023, 1, 1) + timedelta(hours=i)).isoformat() for i in range(B.YEAR_HOURS)]
    for n in ("Alpha", "Beta"):
        with open(tmp / "wx" / f"{n}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time", "cloud_cover", "visibility", "weather_source_model"])
            w.writerows([[h, 0.0, 23000.0, "era5"] for h in hours])
        with open(tmp / "era5" / f"{n}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time", "H_up_Wm2", "ustar", "t2m_K", "sp_Pa"])
            w.writerows([[h, 50.0, 0.3, 288.0, 101325.0] for h in hours])
    return st


def test_smoke_build(tmp_path):
    """End-to-end build with the DEFAULT channel model (v7 since 2026-10-07)."""
    import h5py
    st = _write_inputs(tmp_path)
    layers = _write_layers(tmp_path, low={"Alpha": 0.0, "Beta": 0.0},
                           mid={"Alpha": 0.0, "Beta": 0.0}, high={"Alpha": 0.0, "Beta": 0.0})
    out = tmp_path / "ds"
    common = ["--stations", str(st), "--weather-dir", str(tmp_path / "wx"),
              "--era5-dir", str(tmp_path / "era5"), "--cloud-layers-dir", str(layers)]
    assert B.main(common + ["--check-inputs"]) == 0
    with pytest.raises(SystemExit):                      # never writes into dataset/global
        B.main(common + ["--out", str(ROOT / "dataset" / "global"), "--days", "1", "--workers", "1"])
    # NO --channel-model flag: this exercises the default path (v7).
    assert B.main(common + ["--out", str(out), "--days", "1", "--workers", "1"]) == 0
    for f in ("node_registry.csv", "link_registry.csv", "link_registry_skipped.csv", "link_data.h5",
              "sat_positions.h5", "hap_positions.h5", "build_metadata.json", "rate_stats.json", "link_diag.csv"):
        assert (out / f).exists(), f
    links = list(csv.DictReader(open(out / "link_registry.csv", encoding="utf-8")))
    nodes = list(csv.DictReader(open(out / "node_registry.csv", encoding="utf-8")))
    assert [n["type"] for n in nodes].count("GS") == 2 and len(nodes) == 2 + 2 + 30
    with h5py.File(out / "link_data.h5", "r") as f:
        k = f["k_max"][:]
        proxy = f["k_proxy"][:]
        assert k.shape == proxy.shape == (1440, len(links))
        assert len(f["link_registry"]) == len(links)
        assert "k_max_first" in f and "k_proxy_first" in f
        assert f.attrs["rate_model"] == "v7_finite_key"
        assert f.attrs["policy_rate_recommended"] == "k_proxy"
    assert np.isfinite(k).all() and (k >= 0).all()
    # v7.5 retains candidates by the training-observation proxy, not by a
    # 60 s finite-key threshold. A retained edge may therefore have k_max=0
    # for every isolated slot but still become useful after multi-slot block
    # aggregation; every retained edge must instead be positive in k_proxy.
    assert np.isfinite(proxy).all() and (proxy >= 0).all() and (proxy.max(axis=0) > 0).all()
    own = [i for i, r in enumerate(links) if r["link_type"] == "HAP-GS"
           and int(r["node_v"]) - int(r["node_u"]) == 2]      # HAP right above its own station
    assert len(own) == 2 and (k[:, own] > 0).mean() > 0.99   # cloud-free input -> always usable
    meta = json.loads((out / "build_metadata.json").read_text(encoding="utf-8"))
    assert meta["num_links"] == len(links)
    assert meta["channel_model"] == "v7"                     # the DEFAULT is v7
    with pytest.raises(SystemExit):                      # existing output needs --overwrite
        B.main(common + ["--out", str(out), "--days", "1", "--workers", "1"])


def test_key_length_block_size_override():
    fk = F.FiniteKeyParams()
    eta = 10 ** (np.array([-14.1, -30.0, -35.8, -40.0]) / 10)
    full = F.key_length(eta, 5e-7)
    np.testing.assert_array_equal(F.key_length(eta, 5e-7, fk, n_pulses=fk.n_pulses), full)   # bit-identical
    half = F.key_length(eta, 5e-7, fk, n_pulses=0.5 * fk.n_pulses)
    assert np.all(half <= 0.5 * full + 1)             # finite-size penalty: smaller block, less than linear
    assert half[0] > 0.45 * full[0]                   # strong link: nearly linear in the block size
    tiny = F.key_length(eta, 5e-7, fk, n_pulses=np.array([1e3, 1e3, 1e3, 1e3]))
    assert np.all(tiny == 0)


def _write_layers(tmp: Path, low: dict, mid: dict, high: dict, name: str = "layers") -> Path:
    d = tmp / name
    d.mkdir(exist_ok=True)
    for n in ("Alpha", "Beta"):
        with open(d / f"{n}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "source_model"])
            w.writerows([[(datetime(2023, 1, 1) + timedelta(hours=i)).isoformat(),
                          0.0, low[n], mid[n], high[n], "era5"] for i in range(B.YEAR_HOURS)])
    return d


def test_v7_station_markov_cloud_is_reproducible_and_persisted(tmp_path):
    """v7.5 uses one station-scale persistent cloud state per GS and slot."""
    import h5py
    st = _write_inputs(tmp_path)
    common = [
        "--channel-model", "v7", "--stations", str(st),
        "--weather-dir", str(tmp_path / "wx"), "--era5-dir", str(tmp_path / "era5"),
        "--days", "1", "--workers", "1", "--no-stats", "--keep-zero-links",
        "--cloud-seed", "77", "--cloud-tau-s", "1800",
    ]
    layers = _write_layers(
        tmp_path, name="layers_markov",
        low={"Alpha": 50.0, "Beta": 0.0},
        mid={"Alpha": 0.0, "Beta": 0.0},
        high={"Alpha": 0.0, "Beta": 0.0},
    )
    assert B.main(common + ["--cloud-layers-dir", str(tmp_path / "nope"), "--check-inputs"]) == 1
    assert B.main(common + ["--cloud-layers-dir", str(layers), "--check-inputs"]) == 0
    out1, out2 = tmp_path / "cloud1", tmp_path / "cloud2"
    assert B.main(common + ["--cloud-layers-dir", str(layers), "--out", str(out1)]) == 0
    assert B.main(common + ["--cloud-layers-dir", str(layers), "--out", str(out2)]) == 0
    with h5py.File(out1 / "link_data.h5", "r") as f1, h5py.File(out2 / "link_data.h5", "r") as f2:
        c1, c2 = f1["station_cloud_state"][:], f2["station_cloud_state"][:]
        np.testing.assert_array_equal(c1, c2)
        assert c1.shape == (1440, 2)
        assert f1.attrs["cloud_model"] == "station_markov"
        assert f1.attrs["cloud_tau_s"] == pytest.approx(1800.0)
def test_v7_smoke_build_writes_capture_adjusted_finite_key_rates(tmp_path):
    import h5py

    st = _write_inputs(tmp_path)
    layers = _write_layers(
        tmp_path,
        low={"Alpha": 0.0, "Beta": 0.0},
        mid={"Alpha": 0.0, "Beta": 0.0},
        high={"Alpha": 0.0, "Beta": 0.0},
    )
    out = tmp_path / "ds_v7"
    args = [
        "--channel-model", "v7",
        "--stations", str(st),
        "--weather-dir", str(tmp_path / "wx"),
        "--era5-dir", str(tmp_path / "era5"),
        "--cloud-layers-dir", str(layers),
        "--out", str(out), "--days", "1", "--workers", "1", "--no-stats",
    ]
    assert B.main(args) == 0
    with h5py.File(out / "link_data.h5", "r") as f:
        full = f["k_max"][:]
        first = f["k_max_first"][:]
        proxy = f["k_proxy"][:]
        proxy_first = f["k_proxy_first"][:]
        assert full.shape == first.shape == proxy.shape == proxy_first.shape == (1440, len(f["link_registry"]))
        assert f.attrs["rate_model"] == "v7_finite_key"
        assert f.attrs["domain_policy"] == "extrapolate"
        assert f.attrs["capture_time_s"] == pytest.approx(5.0)
        assert f.attrs["capture_gamma"] == pytest.approx(55.0 / 60.0)
        assert "station_cloud_state" in f
        for key in (
            "qkd_gain", "qkd_y1", "qkd_usable_fraction",
            "qkd_gain_first", "qkd_y1_first", "qkd_usable_fraction_first",
        ):
            assert key in f
        assert f["qkd_gain"].shape == (1440, len(f["link_registry"]), 3)
        assert f.attrs["qkd_stats_semantics"] == "conditional_clear_block_sufficient_stats_v7_5"
        assert f.attrs["policy_rate_recommended"] == "k_proxy"
    assert np.isfinite(full).all() and np.isfinite(first).all()
    assert np.isfinite(proxy).all() and np.isfinite(proxy_first).all()
    assert np.all(first >= 0) and np.all(full >= 0)
    assert np.all(proxy >= 0) and np.all(proxy_first >= 0)
    assert np.any(first > 0.0)
    assert np.any(full > 0)
    assert np.any(proxy > 0.0)
    assert np.any(np.abs(first - full) > 1e-6)  # capture window is actually re-sampled
    meta = json.loads((out / "build_metadata.json").read_text(encoding="utf-8"))
    assert meta["channel_model"] == "v7"
    assert meta["capture_time_s"] == pytest.approx(5.0)
    assert meta["capture_gamma"] == pytest.approx(55.0 / 60.0)
    assert meta["cloud_model"] == "station_markov"
    assert "performance estimator" in meta["finite_key_scope"]
    fill = meta["weather_fill"]["Alpha"]
    assert fill["most_regularization"]["n_hours"] == 8760
    assert fill["most_regularization"]["near_neutral_flux_fraction"] >= 0.0
    assert fill["hv_match_floor"]["n_hours"] == 8760
    assert 0.0 <= fill["hv_match_floor"]["background_exceeds_reference_fraction"] <= 1.0


def test_v7_intra_slot_substepping_builds_and_matches_slot_shape(tmp_path):
    """--intra-slot-samples sub-samples each 60 s slot; output stays (T, L), finite, nonneg."""
    import h5py

    st = _write_inputs(tmp_path)
    layers = _write_layers(tmp_path, low={"Alpha": 0.0, "Beta": 0.0},
                           mid={"Alpha": 0.0, "Beta": 0.0}, high={"Alpha": 0.0, "Beta": 0.0})
    common = ["--channel-model", "v7", "--stations", str(st),
              "--weather-dir", str(tmp_path / "wx"), "--era5-dir", str(tmp_path / "era5"),
              "--cloud-layers-dir", str(layers), "--days", "1", "--workers", "1", "--no-stats"]
    out1, out8 = tmp_path / "n1", tmp_path / "n8"
    assert B.main(common + ["--out", str(out1), "--intra-slot-samples", "1"]) == 0
    assert B.main(common + ["--out", str(out8), "--intra-slot-samples", "8"]) == 0
    with h5py.File(out1 / "link_data.h5", "r") as f1, h5py.File(out8 / "link_data.h5", "r") as f8:
        a, b = f1["k_max"][:], f8["k_max"][:]
        c1, c8 = f1["station_cloud_state"][:], f8["station_cloud_state"][:]
        assert a.shape == b.shape == (1440, len(f1["link_registry"]))
        np.testing.assert_array_equal(c1, c8)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert np.all(a >= 0) and np.all(b >= 0)
    meta = json.loads((out8 / "build_metadata.json").read_text(encoding="utf-8"))
    assert meta["intra_slot_samples"] == 8
    assert meta["aperture_averaging"] is True
    assert meta["saturate_scint"] is True
