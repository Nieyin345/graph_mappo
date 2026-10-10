"""Reject malformed v7.5 channel inputs instead of silently generating plausible rates."""
import numpy as np
import pytest
from tools.channel_v7.dataset import generate_station_cloud_state
from tools.channel_v7.model import RateModelV7

def test_cloud_state_cannot_extend_weather_series_silently():
    hourly = np.full((1, 1), 0.3)
    out, _ = generate_station_cloud_state(hourly, 60, slot_s=60, seed=5)
    assert out.shape == (1, 60)
    with pytest.raises(ValueError, match="hour"):
        generate_station_cloud_state(hourly, 61, slot_s=60, seed=5)

def test_cloud_state_cannot_use_empty_hourly_data():
    with pytest.raises(ValueError, match="hour"):
        generate_station_cloud_state(np.empty((1, 0)), 1)

@pytest.mark.parametrize("ltype", ["SAT-GSS", "HAP-SAT", "BAD"])
def test_unknown_link_type_rejected(ltype):
    model=RateModelV7()
    with pytest.raises(ValueError, match="link type"):
        model.optical_states(ltype, np.array([550_000.0]), np.array([60.0]))

@pytest.mark.parametrize("distance", [0.0, -100.0, np.nan, np.inf])
def test_nonphysical_slant_distance_rejected(distance):
    model=RateModelV7()
    with pytest.raises(ValueError, match="slant"):
        model.optical_states("SAT-SAT", np.array([distance]), np.array([60.0]))
