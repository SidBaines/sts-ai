from __future__ import annotations
from types import SimpleNamespace
import unittest
import numpy as np
from sts_policy.strategic_learning import actor_loss_weights


class ActorActWeightTests(unittest.TestCase):
    def test_equal_act_mass_excludes_forced_rows_and_preserves_total_scale(self):
        acts=[1]*9+[2]*3+[3]+[1]
        rows=[SimpleNamespace(observation={'state':{'act':a}}) for a in acts]
        indices=list(range(13))
        weights=actor_loss_weights(rows,indices,True)
        masses=[sum(float(weights[i]) for i in indices if acts[i]==act) for act in (1,2,3)]
        np.testing.assert_allclose(masses,[13/3]*3,rtol=1e-6)
        self.assertEqual(float(weights[-1]),0.)
        uniform=actor_loss_weights(rows,indices,False)
        np.testing.assert_array_equal(uniform[:13],np.ones(13))
        self.assertEqual(float(uniform[-1]),0.)


if __name__=='__main__':unittest.main()
