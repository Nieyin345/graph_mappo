"""Regression guards for the 2026-10-04 no-excess main channel model.
Run on the scientific server: python -m unittest discover -s tests -p test_channel_no_excess.py -v
No weather files, datasets, Monte Carlo, or training jobs are needed.
"""
from pathlib import Path
import contextlib
import importlib.util
import io
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("tools/channel_v2", "tools/dataset_v2"):
    sys.path.insert(0, str(ROOT / sub))
from rate_model_v2 import ParamsV2, RateModelV2, airmass_kasten_young
from finite_key import FiniteKeyParams, slot_rate


class NoExcessRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = RateModelV2(ParamsV2(), sat_alt_km=550.0)
        cls.legacy = RateModelV2(ParamsV2(kappa_x_db=1.0), sat_alt_km=550.0)
        cls.elev = np.array([5., 10., 20., 30., 60., 90.])

    def ground(self, model, kind="SAT-GS", elev=None):
        e = self.elev if elev is None else np.asarray(elev)
        h = 550e3 if kind == "SAT-GS" else 20e3
        r = 6371e3
        distance = np.sqrt((r+h)**2 - (r*np.cos(np.radians(e)))**2) - r*np.sin(np.radians(e))
        return model.link_rate(kind, distance, e, sun_rx_deg=-10., aod550=0.1,
                               a_cn2=model.p.a_night, return_parts=True)[1]

    def test_default_is_off(self):
        self.assertEqual(ParamsV2().kappa_x_db, 0.)
        self.assertEqual(self.main.p.kappa_x_db, 0.)

    def test_invalid_coefficients_rejected(self):
        for x in (-1., float("nan"), float("inf"), -float("inf")):
            with self.subTest(x=x), self.assertRaises(ValueError):
                ParamsV2(kappa_x_db=x)

    def test_ground_main_is_physical_product(self):
        for kind in ("SAT-GS", "HAP-GS"):
            with self.subTest(kind=kind):
                p = self.ground(self.main, kind)
                np.testing.assert_array_equal(p["xs"], np.ones_like(self.elev))
                np.testing.assert_allclose(p["eta"], p["bp"]*p["phi"]*p["mol"]*p["aer"]*p["sys"]*p.get("upper", 1.0), rtol=1e-12)
                self.assertTrue(np.all(np.isfinite(p["eta"])))
                self.assertTrue(np.all((p["eta"] >= 0.) & (p["eta"] <= 1.)))

    def test_only_opt_in_excess_changes_ground_transmittance(self):
        expected = 10**(-np.maximum(airmass_kasten_young(self.elev)-1, 0)/10)
        for kind in ("SAT-GS", "HAP-GS"):
            with self.subTest(kind=kind):
                a, b = self.ground(self.main, kind), self.ground(self.legacy, kind)
                for key in ("bp", "T", "phi", "r0", "mol", "aer", "sys", "y0", "upper"):
                    np.testing.assert_array_equal(a[key], b[key])
                np.testing.assert_allclose(b["eta"], a["eta"]*expected, rtol=2e-15, atol=0)
                np.testing.assert_allclose(b["xs"], expected, rtol=2e-15)
                self.assertTrue(np.all(b["eta"] <= a["eta"]))

    def test_scalar_and_zenith(self):
        for kind in ("SAT-GS", "HAP-GS"):
            for elev in (5., 30., 90.):
                with self.subTest(kind=kind, elev=elev):
                    a, b = self.ground(self.main, kind, elev), self.ground(self.legacy, kind, elev)
                    self.assertEqual(float(a["xs"]), 1.)
                    self.assertTrue(np.isfinite(a["eta"]))
                    if elev == 90.:
                        self.assertEqual(float(a["eta"]), float(b["eta"]))

    def test_high_links_unchanged(self):
        for kind in ("SAT-HAP", "SAT-SAT"):
            with self.subTest(kind=kind):
                kw = dict(sun_rx_deg=-10., return_parts=True)
                a = self.main.link_rate(kind, np.array([500e3, 1000e3, 2000e3]), 30., **kw)
                b = self.legacy.link_rate(kind, np.array([500e3, 1000e3, 2000e3]), 30., **kw)
                np.testing.assert_array_equal(a[0], b[0])
                for key in a[1]:
                    np.testing.assert_array_equal(a[1][key], b[1][key])

    def test_finite_key_accepts_revised_eta(self):
        fk = FiniteKeyParams()
        for kind in ("SAT-GS", "HAP-GS"):
            with self.subTest(kind=kind):
                a, b = self.ground(self.main, kind), self.ground(self.legacy, kind)
                ra, rb = slot_rate(a["eta"], a["y0"], fk), slot_rate(b["eta"], b["y0"], fk)
                self.assertEqual(ra.shape, self.elev.shape)
                self.assertTrue(np.all(np.isfinite(ra)))
                self.assertTrue(np.all(ra >= 0.))
                self.assertTrue(np.all(ra >= rb))  # this fixed regression grid, not a universal theorem
                self.assertTrue(np.any(ra > rb))

    def test_builder_rejects_invalid_cli_before_reading_inputs(self):
        from build_dataset_v2 import main
        # v7 CLI validations (the legacy --kappa-x-db flag was removed with the v2 default)
        for flag, values in (("--capture-gamma", ("-1", "nan", "inf")),
                             ("--intra-slot-samples", ("-1", "0"))):
            for value in values:
                with self.subTest(flag=flag, value=value), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as cm:
                        main([flag, value])
                    self.assertEqual(cm.exception.code, 2)

    def test_old_model_reproduced_when_reference_supplied(self):
        # Release validation adds an untouched source snapshot alongside this test.
        path = ROOT / "tools/channel_v2/rate_model_v2_before_change.py"
        if not path.exists():
            self.skipTest("optional pre-change source snapshot not present")
        spec = importlib.util.spec_from_file_location("rate_model_v2_before_change", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        original = mod.RateModelV2(mod.ParamsV2(), sat_alt_km=550.0)
        self.assertEqual(original.p.kappa_x_db, 1.0)
        for kind in ("SAT-GS", "HAP-GS"):
            a, b = self.ground(original, kind), self.ground(self.legacy, kind)
            for key in a:
                np.testing.assert_allclose(a[key], b[key], rtol=2e-15, atol=0)
            fk = FiniteKeyParams()
            np.testing.assert_array_equal(slot_rate(a["eta"], a["y0"], fk),
                                          slot_rate(b["eta"], b["y0"], fk))


if __name__ == "__main__":
    unittest.main()
