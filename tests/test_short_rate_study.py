import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path.cwd()))
from tools.short_channel_rate_study import released_keys
from tools.dataset_v2.finite_key import FiniteKeyParams, QKDBlockAccumulator


class ReleaseAudit(unittest.TestCase):
    def data(self, slot=60):
        fk = FiniteKeyParams(slot_s=slot)
        rng = np.random.default_rng(927)
        q = np.empty((23, 4, 3))
        q[..., 0] = rng.uniform(.001, .02, (23, 4))
        q[..., 1] = .4*q[..., 0]
        q[..., 2] = 5e-7
        stats = dict(gain=q, usable_fraction=rng.choice([.25,.5,1.],(23,4)))
        first = dict(gain=q*.85, usable_fraction=stats['usable_fraction']*.8)
        visible = rng.random((23,4)) > .2
        visible[:,0] = True
        visible[:,1] = False
        visible[-1,2] = True
        gates = rng.random((23,4)) > .4
        return fk, stats, first, visible, gates

    def reference(self, fk, stats, first, visible, gates, duration):
        result = np.zeros(visible.shape)
        for col in range(visible.shape[1]):
            acc = QKDBlockAccumulator(fk)
            previous = False
            for step in range(len(visible)):
                active = bool(visible[step,col])
                if previous and not active:
                    result[step,col] += acc.close_finite_key()
                if active:
                    selected = first if not previous else stats
                    q = selected['gain'][step,col]
                    e = fk.e_d*q + (.5-fk.e_d)*q[-1]
                    n = selected['usable_fraction'][step,col]*fk.n_pulses*float(gates[step,col])
                    acc.add_slot(q,e,n)
                    if acc.slot_count*fk.slot_s >= duration:
                        result[step,col] += acc.close_finite_key()
                previous = active
            result[-1,col] += acc.close_finite_key()
        return result

    def test_matches_production_accumulator(self):
        for duration in (60,300):
            for slot in (30,60):
                with self.subTest(slot=slot,duration=duration):
                    fk,s,f,v,g=self.data(slot)
                    actual=released_keys(s,f,v,g,fk,duration)
                    expected=self.reference(fk,s,f,v,g,duration)
                    np.testing.assert_allclose(actual,expected,rtol=0,atol=1)

    def test_invalid_retained_slot_marks_block(self):
        fk,s,f,v,g=self.data()
        s['gain'][1,0]=np.nan
        s['usable_fraction'][1,0]=np.nan
        g[:,0]=True
        out=released_keys(s,f,v,g,fk,300)
        self.assertTrue(np.isnan(out[4,0]))
        self.assertTrue(np.isfinite(out[9,0]))

    def test_invalid_blocked_slot_does_not_poison_retained_counts(self):
        fk,s,f,v,g=self.data()
        s['gain'][1,0]=np.nan
        s['usable_fraction'][1,0]=np.nan
        g[1,0]=False
        out=released_keys(s,f,v,g,fk,300)
        self.assertTrue(np.isfinite(out).all())

    def test_fully_blocked_is_zero(self):
        fk,s,f,v,g=self.data()
        self.assertTrue(np.all(released_keys(s,f,v,np.zeros_like(g),fk,300)==0))

    def test_invalid_block_duration_rejected(self):
        fk,s,f,v,g=self.data()
        for duration in (0,-1,61,np.nan):
            with self.assertRaises(ValueError):
                released_keys(s,f,v,g,fk,duration)


if __name__=='__main__':
    unittest.main(verbosity=2)
