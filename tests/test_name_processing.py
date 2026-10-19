"""Contract checks for identity-sensitive names and backwards compatibility."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
from name_processing import name_view, NameFeatureIndex, weighted_similarity, phone_spelling
from matcher import CatalogIndex, FEATURES, feature_names, make_assets


class NameContracts(unittest.TestCase):
    def test_normalization_and_number_boundaries(self):
        self.assertEqual(name_view('ＯＣＥＡＮ－０７').roman, 'OCEAN07')
        self.assertEqual(name_view('오션 스타').tokens, ('OSYEON', 'SEUTA'))
        self.assertEqual(name_view('OCEAN 007').numbers, ('7',))
        self.assertNotEqual(name_view('1 23').numbers, name_view('12 3').numbers)
        self.assertEqual(name_view(None).tokens, ())

    def test_phonetic_costs(self):
        score = lambda a,b: weighted_similarity(phone_spelling(a), phone_spelling(b))
        self.assertGreater(score('STEVEN', 'STEPHEN'), score('STEVEN', 'STEQEN'))
        self.assertGreater(score('STAR', 'STAL'), score('STAR', 'STAX'))
        self.assertLess(score('OCEAN1', 'OCEAN2'), 1)
        for a,b in [('OCEAN', 'OSYEON'), ('A', 'ABC'), ('PHIL', 'FIL')]:
            self.assertAlmostEqual(score(a,b), score(b,a))
            self.assertTrue(0 <= score(a,b) <= 1)

    def test_discriminative_tokens_and_empty_names(self):
        idx = NameFeatureIndex(['MORNING COMPOSER', 'MORNING CELLO', 'OCEAN 7', 'OCEAN 8'])
        x = idx.features('MORNING COMPOSER', [0,1], 'all')
        self.assertGreater(x[0, 12], x[1, 12])  # residual phonetic score
        n = idx.features('OCEAN 7', [2,3], 'all')
        self.assertEqual(n[:,1].tolist(), [0,1])
        self.assertTrue(np.isfinite(idx.features(None, [0], 'all')).all())
        self.assertEqual(idx.features(None, [], 'all').shape, (0,14))

    def test_baseline_features_and_candidates_unchanged(self):
        catalog = pd.DataFrame({'vessel_id':['A','B'], 'vessel_name':['OCEAN 7','OCEAN 8'],
            'gross_tonnage':[100,101], 'vessel_type':['cargo']*2, 'flag_name':['KR']*2})
        train = pd.DataFrame([{'vessel_name':'오션 7', 'call_sign':'C',
            'gross_tonnage':100, 'international_gross_tonnage':100,
            'vessel_type':'cargo', 'flag_name':'KR', 'target_vessel_id':'A'}])
        row = train.iloc[0].to_dict()
        base = CatalogIndex(catalog, make_assets(train, catalog))
        improved = CatalogIndex(catalog, make_assets(train, catalog, 'all'))
        candidates = base.candidates(row, use_aliases=False)
        self.assertEqual(candidates, improved.candidates(row, use_aliases=False))
        np.testing.assert_array_equal(base.features(row,candidates),
            improved.features(row,candidates)[:,:len(FEATURES)])
        self.assertEqual(improved.features(row,candidates).shape[1], len(feature_names('all')))


if __name__ == '__main__':
    unittest.main()
