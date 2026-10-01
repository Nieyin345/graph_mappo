# -*- coding: utf-8 -*-
"""tools/dataset_v2: finite-key length, orbit propagation and an end-to-end smoke build."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "dataset_v2"))
sys.path.insert(0, str(ROOT / "tools" / "data_generation"))

import build_dataset_v2 as B  # noqa: E402
import finite_key as F  # noqa: E402

# (total transmittance dB, Y0, key bits per 60 s block) from the scalar reference q/fk/fk_fixed2.py L()
REFERENCE = [(-14.1, 5e-7, 17971660.0), (-32.0, 5e-7, 220133.0), (-35.8, 5e-7, 64955.0),
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
    st.write_text("name,lat,lon,continent,base\nAlpha,48.85,2.35,Europe,1\nBeta,40.42,-3.70,Europe,0\n",
                  encoding="utf-8")
    for sub in ("wx", "era5"):
        (tmp / sub).mkdir()
    hours = [f"2023-01-01T00:00" for _ in range(B.YEAR_HOURS)]
    for n in ("Alpha", "Beta"):
        with open(tmp / "wx" / f"{n}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time", "cloud_cover", "aod550"])
            w.writerows([[h, 0.0, 0.1] for h in hours])
        with open(tmp / "era5" / f"{n}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time", "H_up_Wm2", "ustar", "t2m_K", "sp_Pa"])
            w.writerows([[h, 50.0, 0.3, 288.0, 101325.0] for h in hours])
    return st


def test_smoke_build(tmp_path):
    import h5py
    st = _write_inputs(tmp_path)
    out = tmp_path / "ds"
    common = ["--stations", str(st), "--weather-dir", str(tmp_path / "wx"), "--era5-dir", str(tmp_path / "era5")]
    assert B.main(common + ["--check-inputs"]) == 0
    with pytest.raises(SystemExit):                      # never writes into dataset/global
        B.main(common + ["--out", str(ROOT / "dataset" / "global"), "--days", "1", "--workers", "1"])
    assert B.main(common + ["--out", str(out), "--days", "1", "--workers", "1"]) == 0
    for f in ("node_registry.csv", "link_registry.csv", "link_registry_skipped.csv", "link_data.h5",
              "sat_positions.h5", "hap_positions.h5", "build_metadata.json", "rate_stats.json", "link_diag.csv"):
        assert (out / f).exists(), f
    links = list(csv.DictReader(open(out / "link_registry.csv", encoding="utf-8")))
    nodes = list(csv.DictReader(open(out / "node_registry.csv", encoding="utf-8")))
    assert [n["type"] for n in nodes].count("GS") == 2 and len(nodes) == 2 + 2 + 30
    with h5py.File(out / "link_data.h5", "r") as f:
        k = f["k_max"][:]
        assert k.shape == (1440, len(links)) and len(f["link_registry"]) == len(links)
    assert np.isfinite(k).all() and (k >= 0).all() and (k.max(axis=0) > 0).all()
    own = [i for i, r in enumerate(links) if r["link_type"] == "HAP-GS"
           and int(r["node_v"]) - int(r["node_u"]) == 2]      # HAP right above its own station
    assert len(own) == 2 and (k[:, own] > 0).mean() > 0.99   # cloud-free input -> always usable
    meta = json.loads((out / "build_metadata.json").read_text(encoding="utf-8"))
    assert meta["num_links"] == len(links) and meta["build_config"]["sat_constellation"] == "walker"
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


def _write_layers(tmp: Path, low: dict, mid: dict, high: dict) -> Path:
    d = tmp / "layers"
    d.mkdir(exist_ok=True)
    for n in ("Alpha", "Beta"):
        with open(d / f"{n}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"])
            w.writerows([["2023-01-01T00:00", 0.0, low[n], mid[n], high[n]]] * B.YEAR_HOURS)
    return d


def test_coverage_cloud_model(tmp_path):
    import h5py
    st = _write_inputs(tmp_path)
    common = ["--stations", str(st), "--weather-dir", str(tmp_path / "wx"), "--era5-dir", str(tmp_path / "era5"),
              "--days", "1", "--workers", "1", "--no-stats", "--keep-zero-links"]
    cov = common + ["--cloud-model", "coverage_layers"]
    assert B.main(cov[:6] + ["--cloud-model", "coverage_layers", "--check-inputs"]) == 1   # layer files missing
    ldir = _write_layers(tmp_path, low={"Alpha": 50.0, "Beta": 0.0}, mid={"Alpha": 0.0, "Beta": 0.0},
                         high={"Alpha": 0.0, "Beta": 100.0})
    assert B.main(cov + ["--cloud-layers-dir", str(ldir), "--check-inputs"]) == 0
    assert B.main(common + ["--out", str(tmp_path / "clear")]) == 0               # binary, cloud_cover = 0
    assert B.main(cov + ["--cloud-layers-dir", str(ldir), "--out", str(tmp_path / "cov")]) == 0
    reg = list(csv.DictReader(open(tmp_path / "cov" / "link_registry.csv", encoding="utf-8")))
    assert reg == list(csv.DictReader(open(tmp_path / "clear" / "link_registry.csv", encoding="utf-8")))
    with h5py.File(tmp_path / "clear" / "link_data.h5", "r") as f:
        k0 = f["k_max"][:]
    with h5py.File(tmp_path / "cov" / "link_data.h5", "r") as f:
        k1 = f["k_max"][:]
    col = {(int(r["node_u"]), int(r["node_v"])): i for i, r in enumerate(reg)}
    a, b = col[(0, 2)], col[(1, 3)]                    # own HAP-GS links of Alpha and Beta
    # Alpha: half the slot cloud-free -> about half the key, never zero (no binary outage)
    assert (k1[:, a] > 0).all() and np.all(k1[:, a] < 0.5 * k0[:, a] + 1) and np.all(k1[:, a] > 0.45 * k0[:, a])
    # Beta: cirrus overhead = attenuation exp(-0.3) at zenith, link stays usable
    assert (k1[:, b] > 0).all() and np.all(k1[:, b] < k0[:, b]) and np.all(k1[:, b] > 0.5 * k0[:, b])
    sats = [i for i, r in enumerate(reg) if r["link_type"] in ("SAT-HAP", "SAT-SAT")]
    np.testing.assert_array_equal(k1[:, sats], k0[:, sats])    # only ground-terminated links change
    meta = json.loads((tmp_path / "cov" / "build_metadata.json").read_text(encoding="utf-8"))
    assert meta["build_config"]["cloud_model"] == "coverage_layers"
    ldir2 = tmp_path / "l2"
    ldir2.mkdir()
    for n in ("Alpha", "Beta"):                          # overcast low cloud -> ground links off
        (ldir2 / f"{n}.csv").write_text("time,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high\n"
                                        + "2023-01-01T00:00,100,100,0,0\n" * B.YEAR_HOURS, encoding="utf-8")
    assert B.main(cov + ["--cloud-layers-dir", str(ldir2), "--out", str(tmp_path / "ovc")]) == 0
    with h5py.File(tmp_path / "ovc" / "link_data.h5", "r") as f:
        k2 = f["k_max"][:]
    gl = [i for i, r in enumerate(reg) if r["link_type"] in ("HAP-GS", "SAT-GS")]
    assert (k2[:, gl] == 0).all()
