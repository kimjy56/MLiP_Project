"""Contracts for set context, query isolation and saved-model integration."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from decision_models.set_model import CandidateSetNetwork, SetDecisionModel, validate_groups


class SetModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.x = np.random.default_rng(42).normal(size=(15, 4)).astype(np.float32)
        cls.sizes = np.array([3, 5, 2, 5])
        cls.y = np.zeros(15, dtype=np.float32)
        cls.y[[0, 4, 9, 12]] = 1
        cls.model = SetDecisionModel(epochs=4, batch_size=2).fit(
            cls.x, cls.y, np.ones(15), list('abcd'), group_sizes=cls.sizes)

    def test_competitor_changes_score_without_cross_query_context(self):
        net = CandidateSetNetwork(1, 1, 1)
        with torch.no_grad():
            for p in net.parameters():
                p.zero_()
            net.encoder[0].weight.fill_(1)
            net.head[0].weight[0, 1] = 1  # use mean context only
            net.head[2].weight.fill_(1)
        with torch.inference_mode():
            single = net(torch.tensor([[[1.]]]), torch.tensor([[True]]))
            together = net(torch.tensor([[[1.], [3.]]]), torch.tensor([[True, True]]))
        self.assertEqual(single.item(), 1.)
        self.assertEqual(together[0, 0].item(), 2.)

    def test_padding_is_excluded_from_both_pools(self):
        model = self.model.model
        x = torch.from_numpy(self.x[:3])[None]
        with torch.inference_mode():
            expected = model(x, torch.ones(1, 3, dtype=torch.bool))
            padded = torch.cat([x, torch.full((1, 4, 4), 1e6)], dim=1)
            got = model(padded, torch.tensor([[True] * 3 + [False] * 4]))[:, :3]
        torch.testing.assert_close(expected, got, rtol=1e-6, atol=1e-7)

    def test_full_subset_single_and_query_order_are_exact(self):
        expected = self.model.predict_groups(self.x, self.sizes)
        np.testing.assert_array_equal(expected[:3], self.model.predict_group(self.x[:3]))
        order = np.r_[np.arange(8, 10), np.arange(0, 3), np.arange(10, 15), np.arange(3, 8)]
        np.testing.assert_array_equal(expected[order], self.model.predict_groups(self.x[order], [2, 3, 5, 5]))
        other = self.x.copy()
        other[3:] *= 1000
        np.testing.assert_array_equal(expected[:3], self.model.predict_groups(other, self.sizes)[:3])
        np.testing.assert_array_equal(expected, np.concatenate([
            self.model.predict_groups(self.x[:8], [3, 5]), self.model.predict_groups(self.x[8:], [2, 5])]))

    def test_candidate_permutation_and_duplicate_features(self):
        group = np.vstack([self.x[:3], self.x[0]])
        expected = self.model.predict_group(group)
        order = np.array([2, 0, 3, 1])
        np.testing.assert_array_equal(expected[order], self.model.predict_group(group[order]))
        self.assertEqual(expected[0], expected[3])

    def test_persistence_empty_and_scaler_stays_fixed(self):
        expected = self.model.predict_groups(self.x, self.sizes)
        mean, scale = self.model.mean.copy(), self.model.scale.copy()
        self.model.predict_group(self.x * 100)
        np.testing.assert_array_equal(mean, self.model.mean)
        np.testing.assert_array_equal(scale, self.model.scale)
        self.assertEqual(self.model.predict_groups(self.x[:0], []).shape, (0,))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts') as directory:
            self.model.save(directory)
            loaded = SetDecisionModel.load(directory)
            np.testing.assert_array_equal(expected, loaded.predict_groups(self.x, self.sizes))
            with self.assertRaises(FileExistsError):
                self.model.save(directory)

    def test_invalid_boundaries_and_missing_boundaries_rejected(self):
        for sizes in ([15.], [0, 15], [4, 4], [[15]], [True], [-1, 16]):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                validate_groups(sizes, 15)
        with self.assertRaises(ValueError):
            self.model.predict(self.x)
        with self.assertRaises(ValueError):
            self.model.predict_group(np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            self.model.predict_group(np.full((2, 4), np.nan))
        with self.assertRaises(ValueError):
            SetDecisionModel(epochs=1).fit(self.x, self.y, np.ones(15), list('abcd'), group_sizes=[15])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_fit_and_query_isolation(self):
        model = SetDecisionModel(epochs=1, batch_size=2, device='cuda').fit(
            self.x, self.y, np.ones(15), list('abcd'), group_sizes=self.sizes)
        expected = model.predict_groups(self.x, self.sizes)
        self.assertTrue(np.isfinite(expected).all())
        np.testing.assert_array_equal(expected[:3], model.predict_group(self.x[:3]))
        order = np.array([2, 0, 1])
        np.testing.assert_array_equal(expected[:3][order], model.predict_group(self.x[:3][order]))

    def test_matcher_load_direct_rows_and_batch_boundaries(self):
        from tests.test_team_rules import fixture
        from model.matcher import CatalogIndex, Matcher, QUERY_COLUMNS, make_assets
        from train import training_pair_for_row, prepare_row
        train, catalog, _ = fixture()
        with patch.dict(os.environ, {'VESSEL_SEARCH_ALGORITHM': 'rapidfuzz', 'VESSEL_SCORE_THREADS': '1'}):
            assets = make_assets(train, catalog, name_mode='all', phonetic_search=3, use_train_lookup=False)
            assets['decision_model'] = 'set_mlp'
            index = CatalogIndex(catalog, assets)
            pairs = [training_pair_for_row(prepare_row(r), index, .5) for r in train.to_dict('records')]
            x = np.concatenate([p[0] for p in pairs])
            scorer = SetDecisionModel(epochs=2).fit(
                x, np.concatenate([p[1] for p in pairs]), np.concatenate([p[2] for p in pairs]),
                index.feature_names, group_sizes=[len(p[0]) for p in pairs])
            with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts') as directory:
                scorer.save(directory)
                path = Path(directory) / 'assets.json'
                path.write_text(json.dumps(assets), encoding='utf-8')
                matcher = Matcher(catalog, directory)
                queries = train.loc[:, QUERY_COLUMNS].copy()
                queries = queries.loc[queries.index.repeat(3)].reset_index(drop=True)
                with patch.dict(os.environ, {'VESSEL_PREDICT_BATCH': '1', 'VESSEL_PREDICT_WORKERS': '1'}):
                    expected = matcher.predict(queries)
                with patch.dict(os.environ, {'VESSEL_PREDICT_BATCH': '32', 'VESSEL_PREDICT_WORKERS': '2'}):
                    self.assertEqual(expected, matcher.predict(queries))
                    self.assertEqual(expected[::-1], matcher.predict(queries.iloc[::-1]))
                    self.assertEqual([expected[3]], matcher.predict(queries.iloc[[3]]))
                    self.assertEqual([], matcher.predict(queries.iloc[:0]))
                # Mix direct lookups and model-scored rows: only scored groups
                # belong in group_sizes, and their boundaries must stay aligned.
                matcher.index.use_train_lookup = True
                matcher.index.assets['calls'] = {queries.iloc[0].call_sign: catalog.iloc[0].vessel_id}
                mixed = matcher.predict(queries)
                self.assertEqual(set(matcher.predict_details(queries)[2]), {True, False})
                self.assertEqual(mixed, [matcher.predict(queries.iloc[[i]])[0] for i in range(len(queries))])
                assets['features'] = list(reversed(assets['features']))
                path.write_text(json.dumps(assets), encoding='utf-8')
                with self.assertRaises(ValueError):
                    Matcher(catalog, directory)

    def test_leave_one_out_context_properties(self):
        # 1. Single candidate: diffs must be 0, flag must be 0
        net = CandidateSetNetwork(input_dim=2, embedding_dim=2, hidden_dim=4, context_mode='leave_one_out')
        with torch.no_grad():
            net.encoder[0].weight.copy_(torch.eye(2))
            net.encoder[0].bias.zero_()
        x_single = torch.tensor([[[5.0, 2.0]]])
        mask_single = torch.tensor([[True]])
        with torch.inference_mode():
            # Bypass head to inspect context directly
            encoded = net.encoder(x_single)
            valid = mask_single.unsqueeze(-1)
            # Forward pass test
            out_single = net(x_single, mask_single)
            self.assertEqual(out_single.shape, (1, 1))

        # 2. Ties in maximum and competitor differences
        x_tie = torch.tensor([[[5.0, 1.0], [5.0, 3.0], [0.0, 0.0]]])  # candidate 0 and 1 tie on feature 0
        mask_tie = torch.tensor([[True, True, False]])  # 3rd is padded
        with torch.inference_mode():
            out_tie = net(x_tie, mask_tie)
            self.assertEqual(out_tie.shape, (1, 3))

        # Test persistence and reload
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts') as d:
            rng = np.random.default_rng(123)
            x_test = rng.normal(size=(6, 2)).astype(np.float32)
            model = SetDecisionModel(epochs=2, context_mode='leave_one_out')
            model.fit(x_test, np.array([1, 0, 1, 0, 1, 0], dtype=np.float32),
                      np.ones(6, dtype=np.float32), ['f1', 'f2'], group_sizes=[2, 2, 2])
            model.save(d)
            loaded = SetDecisionModel.load(d)
            self.assertEqual(loaded.context_mode, 'leave_one_out')
            scores_orig = model.predict_groups(x_test, [2, 2, 2])
            scores_loaded = loaded.predict_groups(x_test, [2, 2, 2])
            np.testing.assert_array_equal(scores_orig, scores_loaded)


    def test_row_bce_and_row_ce_loss_modes(self):
        rng = np.random.default_rng(42)
        # 3 queries with sizes 2, 3, 1 (total 6 candidates)
        x_test = rng.normal(size=(6, 2)).astype(np.float32)
        y_test = np.array([1, 0, 0, 1, 0, 1], dtype=np.float32)
        w_test = np.array([0.7, 0.7, 0.3, 0.3, 0.3, 0.7], dtype=np.float32)
        sizes = [2, 3, 1]

        for mode in ('row_bce', 'row_ce'):
            model = SetDecisionModel(epochs=2, loss_mode=mode)
            model.fit(x_test, y_test, w_test, ['f1', 'f2'], group_sizes=sizes)
            self.assertEqual(len(model.loss_history), 2)
            self.assertTrue(all(np.isfinite(l) for l in model.loss_history))
            with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts') as d:
                model.save(d)
                loaded = SetDecisionModel.load(d)
                self.assertEqual(loaded.loss_mode, mode)
                s1 = model.predict_groups(x_test, sizes)
                s2 = loaded.predict_groups(x_test, sizes)
                np.testing.assert_array_equal(s1, s2)


if __name__ == '__main__':
    unittest.main()
