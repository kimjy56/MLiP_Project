import unittest
from unittest.mock import patch
import torch
import numpy as np
import train


class FakeIndex:
    name_mode = 'baseline'
    ids = ['V'+str(i) for i in range(70)]
    lookup = {v:i for i,v in enumerate(ids)}

    def candidates(self, row, use_aliases=False):
        return list(range(70))

    def features(self, row, candidates, include_name_features=False):
        x=np.zeros((len(candidates),len(train.FEATURES)),dtype=np.float32)
        for j,i in enumerate(candidates):
            x[j,train.FEATURES.index('ratio')]=i/100
            x[j,train.FEATURES.index('phonetic_ratio')]=i/110
            x[j,train.FEATURES.index('best_gap')]=(i%13)/10
            x[j,train.FEATURES.index('length_ratio')]=(70-i)/70
            x[j,train.FEATURES.index('international_exact')]=i%2
        return x


class CandidateFeatureTests(unittest.TestCase):
    def test_mining_is_invariant_to_feature_column_order(self):
        row={'vessel_name':'ALPHA','target_vessel_id':'V0','sample_id':'S'}
        index=FakeIndex()
        first=train.training_pair_for_row(row,index,.1)
        with patch.object(train,'FEATURES',list(reversed(train.FEATURES))):
            second=train.training_pair_for_row(row,index,.1)
        self.assertEqual(first[5],second[5])
        self.assertIn('V0',first[5])
        self.assertEqual(sum(first[1]),1)


if __name__=='__main__':
    unittest.main()
