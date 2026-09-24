from __future__ import annotations

import math

import pytest

from qkd_rl.baselines.receding_horizon_milp import _require_finite_capacities


def test_milp_capacity_contract_accepts_complete_finite_map() -> None:
    capacities = {"e1": 100.0, "e2": 50}
    assert _require_finite_capacities(capacities, ["e1", "e2"]) == {
        "e1": 100.0, "e2": 50.0
    }


@pytest.mark.parametrize(
    "capacities, match",
    [
        ({"e1": 100.0}, "missing"),
        ({"e1": 100.0, "e2": math.inf}, "non_finite_or_invalid"),
        ({"e1": 100.0, "e2": "bad"}, "non_finite_or_invalid"),
    ],
)
def test_milp_capacity_contract_rejects_missing_or_invalid(capacities, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _require_finite_capacities(capacities, ["e1", "e2"])
