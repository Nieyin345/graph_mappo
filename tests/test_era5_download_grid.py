"""Prevent CDS area bounds from shifting the requested ERA5 grid."""
import pandas as pd

from tools.channel_v2.fetch_era5_cds import _request
from tools.channel_v2.fetch_era5_boxes import box


def test_station_request_matches_nearest_standard_grid_cell():
    # Observed old download returned latitude 37.9 for San Francisco,
    # while the point time-series API returned 37.75.
    request = _request(37.7749, -122.4194, [1])
    assert request["area"] == [37.75, -122.5, 37.75, -122.5]


def test_cluster_bounds_preserve_global_grid_origin():
    stations = pd.DataFrame({
        "lat": [37.7749, 49.2827],
        "lon": [-122.4194, -123.1207],
    })
    assert box(stations) == [49.25, -123.0, 37.75, -122.5]
