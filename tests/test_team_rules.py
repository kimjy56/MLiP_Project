"""Synthetic contract tests, not competition performance evidence."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model'))
import script
import train as training
import train_oof
from matcher import CatalogIndex, Matcher, QUERY_COLUMNS, make_assets


def fixture():
    catalog = pd.DataFrame({'vessel_id': [f'{i:03}' for i in range(5)],
        'vessel_name': ['ALPHA', 'BRAVO', 'CHARLIE', 'DELTA', 'ECHO'],
        'gross_tonnage': [100, 200, 300, 400, 500],
        'vessel_type': ['cargo'] * 5, 'flag_name': ['KR'] * 5})
    rows = []
    for k, vessel in catalog.iterrows():
        for j in range(2):
            rows.append({'sample_id': f'{k}{j:02}', 'target_vessel_id': vessel.vessel_id,
                'vessel_name': vessel.vessel_name if j == 0 else f'선박 {k}',
                'call_sign': f'C{k}', 'gross_tonnage': vessel.gross_tonnage,
                'international_gross_tonnage': vessel.gross_tonnage,
                'vessel_type': 'cargo', 'flag_name': 'KR'})
    train = pd.DataFrame(rows)
    folds = train[['sample_id', 'target_vessel_id']].copy()
    folds['fold'] = np.repeat(np.arange(5), 2)
    return train, catalog, folds


class TeamRules(unittest.TestCase):
    def test_lookup_switch_covers_calls_and_aliases_and_old_assets(self):
        train, catalog, _ = fixture()
        assets = make_assets(train, catalog, retrieval_channels=[])
        row = train.iloc[-1].to_dict()
        active = CatalogIndex(catalog, assets)
        disabled = CatalogIndex(catalog, assets, use_train_lookup=False)
        self.assertEqual(active.direct(row), '004')
        self.assertEqual(active.candidates(row), [4])
        self.assertIsNone(disabled.direct(row))
        self.assertEqual(disabled.candidates(row), [0])
        self.assertTrue(assets['use_train_lookup'])  # Override did not mutate assets.
        assets.pop('use_train_lookup')
        self.assertEqual(CatalogIndex(catalog, assets).direct(row), '004')
        assets['use_train_lookup'] = False
        self.assertEqual(CatalogIndex(catalog, assets).candidates(row), [0])

    def test_fold_alignment_and_invalid_inputs(self):
        train, catalog, folds = fixture()
        aligned = train_oof.validate_folds(train, catalog, folds.iloc[::-1], expected_rows=10)
        self.assertEqual(aligned.sample_id.tolist(), train.sample_id.tolist())
        self.assertEqual(aligned.fold.tolist(), folds.fold.tolist())
        cases = []
        wrong = folds.copy(); wrong.loc[0, 'sample_id'] = 'unknown'; cases.append((train, catalog, wrong))
        wrong = folds.copy(); wrong.loc[0, 'sample_id'] = wrong.loc[1, 'sample_id']; cases.append((train, catalog, wrong))
        wrong = folds.copy(); wrong.loc[0, 'target_vessel_id'] = '001'; cases.append((train, catalog, wrong))
        wrong = folds.copy(); wrong.loc[0, 'fold'] = 1; cases.append((train, catalog, wrong))
        for value in [None, 0.5, 5, True, 'invalid']:
            wrong = folds.copy(); wrong['fold'] = wrong.fold.astype(object)
            wrong.loc[0, 'fold'] = value; cases.append((train, catalog, wrong))
        wrong = folds.copy(); wrong.loc[wrong.fold == 4, 'fold'] = 3; cases.append((train, catalog, wrong))
        wrong = train.copy(); wrong.loc[0, 'sample_id'] = ''; cases.append((wrong, catalog, folds))
        wrong = catalog.copy(); wrong.loc[0, 'vessel_id'] = '001'; cases.append((train, wrong, folds))
        cases.append((train, catalog.iloc[1:], folds))
        cases.append((train, catalog, folds.iloc[:-1]))
        for i, (tr, cat, fd) in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError):
                train_oof.validate_folds(tr, cat, fd, expected_rows=10)
        with self.assertRaises(ValueError):
            train_oof.validate_folds(train, catalog, folds)  # Production count fixed at 12,222.

    def test_csv_identifiers_preserve_zeros_and_na(self):
        train, catalog, _ = fixture()
        train.loc[0, 'sample_id'] = 'NA'
        train.loc[0, 'call_sign'] = '001'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train.to_csv(root / 'train.csv', index=False)
            catalog.to_csv(root / 'vessel_catalog.csv', index=False)
            loaded, loaded_catalog = training.read_data(root)
        self.assertEqual(loaded.sample_id.iloc[0], 'NA')
        self.assertEqual(loaded.call_sign.iloc[0], '001')
        self.assertEqual(loaded_catalog.vessel_id.iloc[0], '000')

    def test_prediction_labels_never_reach_retrieval(self):
        train, catalog, _ = fixture()
        matcher = Matcher.__new__(Matcher)
        matcher.index = CatalogIndex(catalog, make_assets(train, catalog), use_train_lookup=False)
        class ConstantModel:
            def predict(self, features, **kwargs):
                self_test.assertTrue(all(pool['num_threads'] <= 1 for pool in threadpool_info()))
                return np.ones(len(features))
        self_test = self
        matcher.model = ConstantModel()
        original = matcher.index.candidates
        def checked(row, **kwargs):
            self.assertEqual(set(row), set(QUERY_COLUMNS))
            return original(row, **kwargs)
        with patch.object(matcher.index, 'candidates', side_effect=checked):
            expected = matcher.predict(train)
            corrupted = train.assign(target_vessel_id='not-a-vessel', fold=999)
            self.assertEqual(expected, matcher.predict(corrupted))
            pred, _, _ = matcher.predict(corrupted, diagnostics=True)
            self.assertEqual(expected, pred)
        self.assertEqual(matcher.predict(train.iloc[:0]), [])

    def test_validation_passes_no_label_or_fold(self):
        train, _, _ = fixture()
        class SpyMatcher:
            def predict_details(inner, queries):
                self.assertEqual(list(queries.columns), QUERY_COLUMNS)
                return ['000'] * len(queries), [['000']] * len(queries), [False] * len(queries)
        metrics, predictions = training.evaluate(SpyMatcher(), train.assign(fold=0), {'000'})
        self.assertEqual(len(predictions), len(train))
        self.assertAlmostEqual(metrics['weighted_accuracy'], .2)
        # An absent language group is unmeasured, not zero accuracy or an exception.
        metrics, _ = training.evaluate(SpyMatcher(), train.iloc[::2], set())
        self.assertIsNone(metrics['weighted_accuracy'])

    def test_fixed_five_fold_training_is_isolated_and_loadable(self):
        train, catalog, folds = fixture()
        before = train.copy(deep=True)
        params = {**training.MODEL_PARAMETERS, 'n_estimators': 4, 'min_child_samples': 1}
        real_fit = train_oof.fit
        fitted_targets = []
        def checked_fit(frame, cat, destination, *args):
            k = int(destination.parent.name.split('_')[1])
            self.assertNotIn('fold', frame.columns)
            self.assertNotIn(f'{k:03}', set(frame.target_vessel_id))
            fitted_targets.append(set(frame.target_vessel_id))
            return real_fit(frame, cat, destination, *args)
        with tempfile.TemporaryDirectory() as directory, patch.dict(training.MODEL_PARAMETERS, params), \
                patch.object(train_oof, 'fit', side_effect=checked_fit):
            root = Path(directory) / 'run'
            oof, summaries = train_oof.generate_oof(train, catalog, folds.iloc[::-1], root,
                name_mode='all', phonetic_search=2, retrieval_channels=list('SNPT'),
                use_train_lookup=False, expected_rows=10)
            self.assertEqual(len(fitted_targets), 5)
            self.assertEqual(oof.sample_id.tolist(), train.sample_id.tolist())
            self.assertEqual(oof.fold.tolist(), folds.fold.tolist())
            self.assertEqual(set(oof.columns), {'sample_id', 'pred_vessel_id', 'fold'})
            self.assertTrue(oof.pred_vessel_id.isin(catalog.vessel_id).all())
            for k, summary in enumerate(summaries):
                assets = json.loads((root / f'fold_{k}/model/assets.json').read_text(encoding='utf-8'))
                self.assertFalse(assets['use_train_lookup'])
                self.assertNotIn(f'C{k}', assets['calls'])
                self.assertFalse(any(f'{k:03}' in ids for ids in assets['aliases'].values()))
                self.assertTrue(summary['row_independence']['passed'])
            with self.assertRaises(FileExistsError):
                train_oof.generate_oof(train, catalog, folds, root, expected_rows=10)
            self.assertFalse(list(root.rglob('*.zip')))
        pd.testing.assert_frame_equal(train, before)

    def test_summary_weights_rows_within_each_language(self):
        summaries = []
        for ko_rows, ko_accuracy, en_rows, en_accuracy in [(1, 1., 9, 0.), (9, 0., 1, 1.)]:
            validation = {'seconds': 2.}
            for name, count, accuracy in [('hangul', ko_rows, ko_accuracy),
                    ('non_hangul', en_rows, en_accuracy), ('seen', 0, None), ('unseen', 10, .1)]:
                validation[name + '_rows'] = count
                validation[name + '_accuracy'] = accuracy
                validation[name + '_candidate_recall'] = accuracy
            summaries.append({'validation': validation, 'fit': {'fit_seconds': 3.}})
        summary = train_oof.summarize_folds(summaries)
        self.assertAlmostEqual(summary['weighted_accuracy'], .1)
        self.assertIsNone(summary['seen_accuracy'])
        self.assertEqual(summary['training_seconds'], 6.)

    def test_entrypoint_paths_and_lookup_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir(); (root / 'open').mkdir()
            (root / 'open/test.csv').touch()
            with patch.object(script, 'ROOT', root):
                self.assertEqual(script.input_path('test.csv'), root / 'open/test.csv')
                (root / 'data/test.csv').touch()
                self.assertEqual(script.input_path('test.csv'), root / 'data/test.csv')
                explicit = root / 'missing.csv'
                self.assertEqual(script.input_path('test.csv', explicit), explicit)
                with self.assertRaises(FileNotFoundError):
                    script.input_path('vessel_catalog.csv')
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(script.lookup_override())
            for value, expected in [('0', False), ('false', False), ('1', True), ('TRUE', True)]:
                os.environ['USE_TRAIN_LOOKUP'] = value
                self.assertEqual(script.lookup_override(), expected)
            os.environ['USE_TRAIN_LOOKUP'] = 'typo'
            with self.assertRaises(ValueError):
                script.lookup_override()

    def test_real_entrypoint_with_saved_model_and_empty_queries(self):
        train, catalog, _ = fixture()
        params = {**training.MODEL_PARAMETERS, 'n_estimators': 4, 'min_child_samples': 1}
        with tempfile.TemporaryDirectory() as directory, patch.dict(training.MODEL_PARAMETERS, params), \
                patch.dict(os.environ, {'USE_TRAIN_LOOKUP': '0'}), \
                patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')), \
                patch('socket.create_connection', side_effect=AssertionError('Network forbidden')):
            root = Path(directory) / '모델 검증'
            root.mkdir()
            training.fit(train, catalog, root / 'model', use_train_lookup=True)
            (root / 'open').mkdir()
            catalog.to_csv(root / 'open/vessel_catalog.csv', index=False)
            preserved = {p: train_oof.sha256(p) for p in [root / 'open/vessel_catalog.csv',
                         root / 'model/assets.json', root / 'model/ranker.txt']}
            with patch.object(script, 'ROOT', root):
                expected = None
                for queries in [train, train.iloc[::-1], train.iloc[::2], train.iloc[:1], train.iloc[:0]]:
                    queries.loc[:, QUERY_COLUMNS].to_csv(root / 'open/test.csv', index=False)
                    before = train_oof.sha256(root / 'open/test.csv')
                    path = script.run()
                    result = pd.read_csv(path, dtype=str, keep_default_na=False)
                    self.assertEqual(list(result.columns), ['sample_id', 'target_vessel_id'])
                    self.assertEqual(result.sample_id.tolist(), queries.sample_id.tolist())
                    observed = dict(zip(result.sample_id, result.target_vessel_id))
                    if expected is None:
                        expected = observed
                    self.assertEqual(observed, {s: expected[s] for s in queries.sample_id})
                    self.assertEqual(before, train_oof.sha256(root / 'open/test.csv'))
                    self.assertTrue(all(train_oof.sha256(p) == digest for p, digest in preserved.items()))
            self.assertFalse(list(root.rglob('*.zip')))


if __name__ == '__main__':
    unittest.main()
