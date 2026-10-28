"""Scoring/row-independence checks on synthetic public-format fixtures."""
import importlib.util
from pathlib import Path
import sys
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_engine(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_engine(ROOT / 'artifacts/AS-06/script.py', '_as06_test_base')
NEW = load_engine(ROOT / 'AS-07/script.py', '_as07_test_new')


class TestAS07JamoScoring(unittest.TestCase):
    def setUp(self):
        self.catalog = pd.DataFrame({
            'vessel_id': ['V1', 'V2', 'V3'],
            'vessel_name': ['BLUE STAR', 'GOLD MOON', 'RED SKY'],
            'gross_tonnage': [1000, 1000, 1000],
            'vessel_type': ['Cargo'] * 3, 'flag_name': ['Panama'] * 3,
        })
        self.fit = pd.DataFrame([{
            'sample_id': 'fit', 'target_vessel_id': 'V1', 'vessel_name': 'BLUE STAR',
            'call_sign': 'CALL1', 'gross_tonnage': 1000, 'international_gross_tonnage': 1000,
            'vessel_type': 'Cargo', 'flag_name': 'Panama', 'arrival_year': 2023,
        }])
        self.row = self.fit.drop(columns='target_vessel_id').copy()
        self.row['sample_id'] = 'probe'
        self.row['call_sign'] = 'UNSEEN'
        self.row['vessel_name'] = '\uac00\ub098\ub2e4'
        self.ko_map = {'BLUE STAR': ['\uac00\ub098\ub2e4'], 'GOLD MOON': ['\uac00\ub098\ub77c']}
        self.model = NEW.fit_matching_model(self.fit, self.catalog, use_train_lookup=False)

    def predict(self, module, data=None, mapping=None, **kwargs):
        idx = module.CatalogIndex(self.catalog, full_ko_map=self.ko_map if mapping is None else mapping)
        return module.predict_rows(self.row if data is None else data, self.model, idx,
            use_train_lookup=False, top_candidates=10, **kwargs)

    def test_legacy_preserves_original_predictions_ranks_and_scores(self):
        baseline = self.predict(BASE, gold_ids=['V1'])
        self.assertEqual(baseline, self.predict(NEW, gold_ids=['V1']))
        self.assertEqual(baseline, self.predict(NEW, gold_ids=['V1'], jamo_score_mode='legacy'))

    def test_separate_channel_adds_exactly_one_bounded_bonus(self):
        without = dict((vid, score) for score, vid in self.predict(NEW, mapping={}, jamo_score_mode='separate')[2][0])
        separate = dict((vid, score) for score, vid in self.predict(NEW, jamo_score_mode='separate')[2][0])
        legacy = dict((vid, score) for score, vid in self.predict(NEW, jamo_score_mode='legacy')[2][0])
        self.assertAlmostEqual(separate['V1'] - without['V1'], 1.8, places=5)
        for vid in separate:
            self.assertGreaterEqual(separate[vid] - without[vid], -0.000002)
            self.assertLessEqual(separate[vid] - without[vid], 1.800002)
        self.assertGreater(legacy['V1'] - without['V1'], 1.8)

    def test_candidates_and_non_hangul_scores_are_unchanged(self):
        mixed = pd.concat([self.row, self.fit.drop(columns='target_vessel_id')], ignore_index=True)
        a, b = [], []
        old = self.predict(NEW, data=mixed, jamo_score_mode='legacy', candidate_diagnostics=a)
        new = self.predict(NEW, data=mixed, jamo_score_mode='separate', candidate_diagnostics=b)
        self.assertEqual(a, b)
        self.assertEqual(old[2][1], new[2][1])
        self.assertEqual(a[0]['count'], 3)

    def test_row_independence_and_diagnostic_gold_does_not_affect_output(self):
        mixed = pd.concat([self.row, self.fit.drop(columns='target_vessel_id')], ignore_index=True)
        for mode in ['legacy', 'separate']:
            full = self.predict(NEW, data=mixed, jamo_score_mode=mode)
            reverse = self.predict(NEW, data=mixed.iloc[::-1], jamo_score_mode=mode)
            self.assertEqual(full[0], reverse[0][::-1])
            self.assertEqual(full[2], reverse[2][::-1])
            for i in range(len(mixed)):
                single = self.predict(NEW, data=mixed.iloc[[i]], jamo_score_mode=mode, gold_ids=['V3'])
                self.assertEqual(full[0][i], single[0][0])
                self.assertEqual(full[2][i], single[2][0])

    def test_invalid_mode_rejected_and_model_setting_respected(self):
        with self.assertRaises(ValueError):
            self.predict(NEW, jamo_score_mode='invalid')
        self.model['jamo_score_mode'] = 'separate'
        self.assertEqual(self.predict(NEW), self.predict(NEW, jamo_score_mode='separate'))


if __name__ == '__main__':
    unittest.main()