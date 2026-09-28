"""Layout-aware widening of edge-input weights when a PHYSICAL edge feature is added.

Edge feature rows are ``[physical block | demand block]`` (physical edges carry
zeros in the demand block and vice versa). Adding a physical feature such as
``include_path_recency`` inserts a column at the END OF THE PHYSICAL BLOCK, i.e.
in the MIDDLE of the row. The old widening appended the zero column at the end
of the row, which shifted every demand-block weight by one column: the new
feature was multiplied by the (never-trained) first demand weight and each
demand feature by its neighbour's weight -- a scrambled warm start, contrary to
the "forward bit-equivalent" log line (found 2026-09-28: update-0 of the
recency-only arm differed from its control).

Pinned here:
  1. with an edge layout, the zero column goes at the old physical width, and
     the widened layer reproduces the old layer EXACTLY on physical and demand
     rows for ANY value of the new feature;
  2. the Adam moments are widened the same way;
  3. without a layout, or when the demand block changed, the old append rule is
     kept (no silent reinterpretation).
"""

from __future__ import annotations

import torch

from qkd_rl.rl.algos.checkpoint_compat import (
    _upgrade_optimizer_state_for_model,
    _upgrade_state_dict_for_model,
)

OLD_P, NEW_P, DEM = 5, 6, 4
LAYOUT = (OLD_P, DEM, NEW_P, DEM)


class _Enc(torch.nn.Module):
    def __init__(self, phys, dem):
        super().__init__()
        self.edge_proj_phys = torch.nn.Sequential(torch.nn.Linear(phys + dem, 8))
        self.edge_proj_demand = torch.nn.Sequential(torch.nn.Linear(phys + dem, 8))


class _Model(torch.nn.Module):
    def __init__(self, phys, dem):
        super().__init__()
        self.encoder = _Enc(phys, dem)


def _rows(n, phys, dem, seed, new_value=None):
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(n, OLD_P, generator=g)
    d = torch.randn(n, dem, generator=g)
    zp, zd = torch.zeros(n, OLD_P), torch.zeros(n, dem)
    if new_value is None:
        return torch.cat([p, zd], 1), torch.cat([zp, d], 1)
    col = torch.full((n, 1), float(new_value))
    return torch.cat([p, col, zd], 1), torch.cat([zp, torch.zeros(n, 1), d], 1)


def _upgrade(layout):
    torch.manual_seed(0)
    old = _Model(OLD_P, DEM)
    new = _Model(NEW_P, DEM)
    state, widened, dropped, missing, unexpected = _upgrade_state_dict_for_model(
        new, old.state_dict(), edge_layout=layout
    )
    assert len(widened) == 2 and not dropped and not missing and not unexpected
    new.load_state_dict(state)
    return old, new


def test_layout_insert_keeps_forward_exact_for_any_new_value():
    old, new = _upgrade(LAYOUT)
    phys_old, dem_old = _rows(7, OLD_P, DEM, seed=1)
    for v in (0.0, 3.7, -12.0):
        phys_new, dem_new = _rows(7, NEW_P, DEM, seed=1, new_value=v)
        for proj in ("edge_proj_phys", "edge_proj_demand"):
            f_old = getattr(old.encoder, proj)
            f_new = getattr(new.encoder, proj)
            # Not bit-exact: an extra (zero-weighted) column changes the BLAS
            # summation order (observed |diff| ~6e-8). The weight layout itself
            # is pinned exactly below; a misplaced column gives O(1) errors here.
            torch.testing.assert_close(f_new(phys_new), f_old(phys_old), atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(f_new(dem_new), f_old(dem_old), atol=1e-6, rtol=1e-5)
    w_old = old.encoder.edge_proj_demand[0].weight
    w_new = new.encoder.edge_proj_demand[0].weight
    assert torch.equal(w_new[:, OLD_P], torch.zeros(8))
    assert torch.equal(w_new[:, OLD_P + 1:], w_old[:, OLD_P:])


def test_adam_moments_widened_with_the_same_layout():
    torch.manual_seed(0)
    old = _Model(OLD_P, DEM)
    opt_old = torch.optim.Adam(old.parameters(), lr=1e-3)
    loss = sum(m[0](torch.randn(3, OLD_P + DEM)).pow(2).sum() for m in (old.encoder.edge_proj_phys, old.encoder.edge_proj_demand))
    loss.backward()
    opt_old.step()
    new = _Model(NEW_P, DEM)
    opt_new = torch.optim.Adam(new.parameters(), lr=1e-3)
    saved = opt_old.state_dict()
    direct, _msgs = _upgrade_optimizer_state_for_model(new, opt_new, saved, edge_layout=LAYOUT)
    assert direct is not None
    names = [n for n, _ in new.named_parameters()]
    i = names.index("encoder.edge_proj_demand.0.weight")
    ea_old = saved["state"][i]["exp_avg"]
    ea_new = direct[dict(new.named_parameters())["encoder.edge_proj_demand.0.weight"]]["exp_avg"]
    assert ea_new.shape == (8, NEW_P + DEM)
    assert torch.equal(ea_new[:, :OLD_P], ea_old[:, :OLD_P])
    assert torch.equal(ea_new[:, OLD_P], torch.zeros(8))
    assert torch.equal(ea_new[:, OLD_P + 1:], ea_old[:, OLD_P:])


def test_without_layout_or_changed_demand_block_keeps_append_rule():
    for layout in (None, (OLD_P, DEM, NEW_P - 1, DEM + 1)):
        old, new = _upgrade(layout)
        w_old = old.encoder.edge_proj_demand[0].weight
        w_new = new.encoder.edge_proj_demand[0].weight
        assert torch.equal(w_new[:, : OLD_P + DEM], w_old)
        assert torch.equal(w_new[:, -1], torch.zeros(8))
