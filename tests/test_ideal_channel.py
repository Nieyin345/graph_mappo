# -*- coding: utf-8 -*-
"""理想信道对照（build_global_tensor.py --ideal-channel）：只替换天气输入，其余不变。

论文“环境感知的价值”实验要求：理想版与真实版**只差天气**。这里钉住：
  1. 普通模式读取天气 CSV（云量按 /100 换算、太阳辐射照抄）；
  2. ideal_channel=True 时即使 CSV 存在也不读，所有量取默认值（无云/雨/雪、能见度 24 km、太阳辐射 0），
     且 coverage 记为 source="ideal_channel"；
  3. 默认值与 SAT-SAT/HAP-SAT 链路用的 _DEFAULT_WEATHER 一致（两版不经大气的链路逐位相同的前提）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "data_generation"))

# The builder imports xarray/cdsapi/pyproj (data-generation deps, not in the
# training venv); skip rather than break collection of the training suite.
pytest.importorskip("xarray")
pytest.importorskip("cdsapi")

import build_global_tensor as bgt

T = 48          # 两天，按小时步长
THETA = 3600    # repeat_factor = 1


@pytest.fixture()
def fake_root(tmp_path, monkeypatch):
    station = bgt.GlobalGroundStations.get_stations()[0]["name"]
    wdir = tmp_path / "weather" / "2023"
    wdir.mkdir(parents=True)
    rows = ["cloud_cover_low,cloud_cover_mid,cloud_cover_high,rain_mm,snow_cm,temperature_2m,"
            "relative_humidity_2m,wind_speed_10m,direct_radiation_w,visibility_m"]
    rows += ["80,40,20,1.5,0,5,90,7,300,3000"] * T
    (wdir / f"{station}_weather.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(bgt, "_PROJECT_ROOT", str(tmp_path))
    return [SimpleNamespace(tag=station)]


def test_real_mode_reads_csv(fake_root):
    wd, cov = bgt.load_weather_data(fake_root, T, THETA)
    w = wd[0]
    assert cov[0]["source"] == "csv" and cov[0]["covered_steps"] == T
    assert np.allclose(w["c_low"], 0.8) and np.allclose(w["c_mid"], 0.4)
    assert np.allclose(w["sf"], 300.0) and np.allclose(w["vis"], 3000.0) and np.allclose(w["rain"], 1.5)


def test_ideal_mode_ignores_csv(fake_root):
    wd, cov = bgt.load_weather_data(fake_root, T, THETA, ideal_channel=True)
    w = wd[0]
    assert cov[0]["source"] == "ideal_channel"
    assert cov[0]["covered_steps"] == 0 and cov[0]["fallback_steps"] == T
    for key in ("c_low", "c_mid", "c_high", "rain", "snow", "sf"):
        assert not w[key].any(), key
    assert np.allclose(w["vis"], 24000.0)


def test_ideal_defaults_match_non_atmospheric_default():
    """理想版地面站天气 == 不经大气链路所用的 _DEFAULT_WEATHER（同一套“晴空无日照”值）。"""
    wd, _ = bgt.load_weather_data([SimpleNamespace(tag="__no_such_station__")], T, THETA, ideal_channel=True)
    default = bgt._build_default_weather(T)
    for key, arr in default.items():
        np.testing.assert_array_equal(wd[0][key], arr, err_msg=key)
