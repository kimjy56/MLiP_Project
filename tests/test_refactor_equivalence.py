"""Compare working-tree snapshots with refactored code, including real OOF caches.

Snapshots are local validation artifacts, never submission dependencies.
"""
import importlib.util
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from decision_models.mlp_model import MLPDecisionModel
from decision_models.set_model import SetDecisionModel
from decision_models.common import file_sha256, validate_training


SNAPSHOTS = Path('artifacts/REFACTOR-02/before')


class CommonProcessingTests(unittest.TestCase):
    def test_streaming_digest_across_chunk_boundary(self):
        with tempfile.TemporaryDirectory(dir='artifacts') as directory:
            path = Path(directory) / 'cache.bin'
            for payload in (b'', b'a' * (1024 * 1024 + 7)):
                path.write_bytes(payload)
                self.assertEqual(file_sha256(path), hashlib.sha256(payload).hexdigest())

    def test_binary_validation_acceptance_matches_previous_rule(self):
        for labels in ([0, 1], [1, 0, 1], [0], [1], [], [0, 2], [0, .5, 1], [0, np.nan]):
            y = np.asarray(labels, dtype=np.float32)
            x = np.zeros((len(y), 2), dtype=np.float32)
            if set(np.unique(y)) == {0, 1}:
                actual = validate_training(x, y, np.ones(len(y)))
                np.testing.assert_array_equal(actual[1], y)
            else:
                with self.assertRaises(ValueError):
                    validate_training(x, y, np.ones(len(y)))


def previous_class(filename, classname):
    spec = importlib.util.spec_from_file_location(
        'decision_models._before_' + filename, SNAPSHOTS / (filename + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, classname)


@unittest.skipUnless(SNAPSHOTS.exists(), 'Local before-refactor snapshots required')
class RefactorEquivalenceTests(unittest.TestCase):
    def test_training_parameters_and_scores_exact(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(42)
        x = rng.normal(size=(24, 4)).astype(np.float32)
        x[:, -1] = 7  # Constant columns must retain the same scale fallback.
        y = np.tile([1, 0, 0], 8).astype(np.float32)
        weights = np.linspace(.5, 2, len(x), dtype=np.float32)
        for filename, cls in [('mlp_model', MLPDecisionModel), ('set_model', SetDecisionModel)]:
            with self.subTest(model=filename):
                previous = previous_class(filename, cls.__name__)
                kwargs = {'group_sizes': [3] * 8} if cls is SetDecisionModel else {}
                before = previous(epochs=2, batch_size=4, device='cpu').fit(x, y, weights, list('abcd'), **kwargs)
                after = cls(epochs=2, batch_size=4, device='cpu').fit(x, y, weights, list('abcd'), **kwargs)
                np.testing.assert_array_equal(before.mean, after.mean)
                np.testing.assert_array_equal(before.scale, after.scale)
                for key, tensor in before.model.state_dict().items():
                    self.assertTrue(torch.equal(tensor, after.model.state_dict()[key]), key)
                np.testing.assert_array_equal(before.predict(x, **kwargs), after.predict(x, **kwargs))

    def test_all_saved_g5_scores_exact(self):
        root = Path('artifacts/AS-SET-02')
        if not (root / 'summary.json').exists():
            self.skipTest('Completed G5 caches required')
        for fold in range(5):
            directory = root / f'fold_{fold}'
            with np.load(directory / 'validation_pairs.npz', allow_pickle=False) as data:
                x, sizes = data['x'], data['group_sizes']
            offsets = np.r_[0, np.cumsum(sizes)]
            for key, filename, cls in [('mlp', 'mlp_model', MLPDecisionModel), ('set', 'set_model', SetDecisionModel)]:
                with self.subTest(fold=fold, model=key):
                    before = previous_class(filename, cls.__name__).load(directory / key, device='cpu')
                    after = cls.load(directory / key, device='cpu')
                    for start in range(0, len(sizes), 64):
                        stop = min(start + 64, len(sizes))
                        batch = x[offsets[start]:offsets[stop]]
                        kwargs = {'group_sizes': sizes[start:stop]} if key == 'set' else {}
                        np.testing.assert_array_equal(before.predict(batch, num_threads=1, **kwargs),
                                                      after.predict(batch, num_threads=1, **kwargs))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_gpu_chunk_boundary_exact(self):
        before = previous_class('mlp_model', 'MLPDecisionModel').load('artifacts/AS-SET-02/fold_0/mlp', device='cuda')
        after = MLPDecisionModel.load('artifacts/AS-SET-02/fold_0/mlp', device='cuda')
        rng = np.random.default_rng(9)
        x = rng.normal(size=(16385, len(after.feature_names))).astype(np.float32)
        for count in (0, 1, 16384, 16385):
            np.testing.assert_array_equal(before.predict(x[:count]), after.predict(x[:count]))


if __name__ == '__main__':
    unittest.main()
