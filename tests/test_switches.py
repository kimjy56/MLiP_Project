import sys
from pathlib import Path
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
from matcher import CatalogIndex, make_assets, prepare_row, FEATURES


class TestSwitches(unittest.TestCase):
    def setUp(self):
        self.catalog = pd.DataFrame({
            'vessel_id': ['V001', 'V002'],
            'vessel_name': ['HAN GUL SHIP', 'LATIN SHIP'],
            'gross_tonnage': [1000.0, 2000.0],
            'vessel_type': ['Cargo', 'Tanker'],
            'flag_name': ['KR', 'PA']
        })
        self.train = pd.DataFrame({
            'sample_id': ['S1', 'S2'],
            'vessel_name': ['한글 선박', 'LATIN SHIP'],
            'call_sign': ['CALL1', 'CALL2'],
            'gross_tonnage': [1000.0, 2000.0],
            'international_gross_tonnage': [1000.0, 2000.0],
            'flag_name': ['KR', 'PA'],
            'vessel_type': ['Cargo', 'Tanker'],
            'target_vessel_id': ['V001', 'V002']
        })

    def test_hangul_ignore_type_flag_switch(self):
        # Default: hangul_ignore_type_flag=False
        assets_default = make_assets(self.train, self.catalog, hangul_ignore_type_flag=False)
        index_default = CatalogIndex(self.catalog, assets_default)
        
        # Switched on: hangul_ignore_type_flag=True
        assets_switched = make_assets(self.train, self.catalog, hangul_ignore_type_flag=True)
        index_switched = CatalogIndex(self.catalog, assets_switched)

        ko_row = self.train.iloc[0].to_dict()
        feats_def = index_default.features(ko_row, [0, 1])
        feats_sw = index_switched.features(ko_row, [0, 1])

        type_idx = FEATURES.index('type_probability')
        flag_idx = FEATURES.index('flag_probability')

        # In switched index for Korean row, type and flag probs must be 0
        self.assertTrue(np.all(feats_sw[:, type_idx] == 0.0))
        self.assertTrue(np.all(feats_sw[:, flag_idx] == 0.0))

        # For Latin row, they must remain identical
        latin_row = self.train.iloc[1].to_dict()
        latin_def = index_default.features(latin_row, [0, 1])
        latin_sw = index_switched.features(latin_row, [0, 1])
        np.testing.assert_array_equal(latin_def[:, type_idx], latin_sw[:, type_idx])
        np.testing.assert_array_equal(latin_def[:, flag_idx], latin_sw[:, flag_idx])

    def test_use_train_lookup_switch(self):
        assets_on = make_assets(self.train, self.catalog, use_train_lookup=True)
        index_on = CatalogIndex(self.catalog, assets_on)
        self.assertEqual(index_on.direct({'call_sign': 'CALL1'}), 'V001')

        assets_off = make_assets(self.train, self.catalog, use_train_lookup=False)
        index_off = CatalogIndex(self.catalog, assets_off)
        self.assertIsNone(index_off.direct({'call_sign': 'CALL1'}))

    def test_korean_expansion_switch(self):
        row = {'vessel_name': '케이엠티씨 서울', 'gross_tonnage': 1000.0}
        prep_off = prepare_row(row, expand_korean=False)
        prep_on = prepare_row(row, expand_korean=True)
        
        # Off: unidecode is 'keiemtissiseoul'
        self.assertEqual(prep_off['_name'][1], 'KEIEMTISSISEOUL')
        # On: expand_korean converts 케이엠티씨 -> KMTC
        self.assertEqual(prep_on['_name'][1], 'KMTCSEOUL')


if __name__ == '__main__':
    unittest.main()
