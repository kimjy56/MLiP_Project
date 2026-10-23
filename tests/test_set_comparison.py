import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from decision_models.mlp_model import MLPDecisionModel
from decision_models.set_model import SetDecisionModel
from experiments.compare_set_model import IndependentNetwork, MatchedBatchMLP, score_cache, select_and_rank


class SetComparisonTests(unittest.TestCase):
    def test_matched_training_export_and_scaling(self):
        torch.set_num_threads(1)
        x = np.random.default_rng(1).normal(size=(15, 4)).astype(np.float32)
        y = np.zeros(15, dtype=np.float32)
        y[[1, 6, 11]] = 1
        w = np.ones(15, dtype=np.float32)
        sizes = np.array([4, 6, 5])
        baseline = MatchedBatchMLP(epochs=2, batch_size=2).fit(x, y, w, list('abcd'), group_sizes=sizes)
        contextual = SetDecisionModel(epochs=2, batch_size=2).fit(x, y, w, list('abcd'), group_sizes=sizes)
        np.testing.assert_array_equal(baseline.mean, contextual.mean)
        np.testing.assert_array_equal(baseline.scale, contextual.scale)
        scorer = baseline.export_scorer()
        np.testing.assert_allclose(baseline.predict_groups(x, sizes), scorer.predict(x), rtol=1e-6, atol=1e-7)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts') as directory:
            scorer.save(directory)
            loaded = MLPDecisionModel.load(directory, device='cpu')
            np.testing.assert_array_equal(score_cache(scorer, x, sizes), score_cache(loaded, x, sizes))

    def test_baseline_does_not_use_competitor_context(self):
        net = IndependentNetwork(4).eval()
        x = torch.ones(1, 3, 4)
        with torch.inference_mode():
            expected = net(x, torch.ones(1, 3, dtype=torch.bool))[0, 0].clone()
            x[0, 1:] *= 1000
            got = net(x, torch.ones(1, 3, dtype=torch.bool))[0, 0]
        torch.testing.assert_close(expected, got, rtol=0, atol=0)

    def test_gold_rank_ties_and_missing_gold(self):
        predictions, ranks = select_and_rank(np.array([.5, .5, .2, .7, .1]),
                                             [['A', 'B'], ['C', 'D'], ['E']], ['B', 'D', 'Z'])
        self.assertEqual(predictions, ['A', 'D', 'E'])
        self.assertEqual(ranks, [2, 1, 0])


if __name__ == '__main__':
    unittest.main()
