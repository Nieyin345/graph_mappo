"""park_v3 gravity request model: opt-in, calibrated, reproducible."""
from pathlib import Path

import numpy as np
import pytest

from qkd_rl.env.request import RequestGenerator

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "configs" / "park" / "gs_demand_mass_45.csv"
GS = ["London", "Paris", "Tokyo", "Sydney", "NewYork", "Dakar"]


def _cfg(**kw):
    cfg = {
        "arrival_rate": 0.16,
        "amount_mean": 5.0e5,
        "amount_max": 2.5e6,
        "deadline_steps": 30,
        "steps_per_hour": 60,
        "distribution_version": "park_v3",
        "park_mass_csv": str(CSV),
    }
    cfg.update(kw)
    return cfg


def test_daily_mean_rate_equals_arrival_rate():
    gen = RequestGenerator(GS, _cfg(), seed=0)
    mean = np.mean([gen._park_pair_rates(t).sum() for t in range(0, 1440, 5)])
    assert mean == pytest.approx(0.16, rel=1e-3)


def test_gravity_and_diurnal_shape():
    gen = RequestGenerator(GS, _cfg(), seed=0)
    pairs = gen._pairs
    r = gen._park_pair_rates(14 * 60)  # 14:00 UTC = 14:00 London (winter)
    lp = r[pairs.index(("London", "Paris"))]
    dk = r[pairs.index(("Dakar", "Tokyo"))]
    assert lp > 50 * dk
    # London local 14:00 is its peak, 02:00 its trough
    i = pairs.index(("London", "Paris"))
    assert gen._park_pair_rates(14 * 60)[i] > gen._park_pair_rates(2 * 60)[i]


def test_reproducible_and_counts():
    a = RequestGenerator(GS, _cfg(), seed=7)
    b = RequestGenerator(GS, _cfg(), seed=7)
    ra = [(q.src_gs, q.dst_gs, q.amount) for t in range(600) for q in a.generate(t)]
    rb = [(q.src_gs, q.dst_gs, q.amount) for t in range(600) for q in b.generate(t)]
    assert ra == rb and len(ra) > 0
    assert all(x[2] < 2.5e6 for x in ra)


def test_default_version_unchanged_and_missing_city_rejected():
    base = {"arrival_rate": 0.8, "amount_mean": 1e5, "deadline_steps": 30}
    gen = RequestGenerator(GS, base, seed=3)
    assert gen._park_base is None
    with pytest.raises(ValueError):
        RequestGenerator(GS + ["Atlantis"], _cfg(), seed=0)
