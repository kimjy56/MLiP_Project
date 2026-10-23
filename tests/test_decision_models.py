import tempfile
import unittest
from pathlib import Path
import numpy as np
from decision_models.mlp_model import MLPDecisionModel
from decision_models.catboost_model import CatBoostDecisionModel


class DecisionModelTests(unittest.TestCase):
    def test_mlp_persistence_and_row_independence(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(40, 4)).astype(np.float32)
        y = (x[:, 0] > 0).astype(np.float32)
        model = MLPDecisionModel(epochs=2, batch_size=8).fit(x, y, np.ones(40), list('abcd'))
        expected = model.predict(x)
        np.testing.assert_array_equal(expected, model.predict(x[::-1])[::-1])
        np.testing.assert_allclose(expected, np.concatenate([model.predict(row[None]) for row in x]), rtol=1e-5, atol=1e-6)
        self.assertEqual(model.predict(x[:0]).shape, (0,))
        with tempfile.TemporaryDirectory(dir='artifacts') as directory:
            model.save(Path(directory))
            restored = MLPDecisionModel.load(directory)
            np.testing.assert_array_equal(expected, restored.predict(x))

    def test_catboost_optional_import(self):
        self.assertEqual(CatBoostDecisionModel().feature_names, [])


if __name__ == '__main__':
    unittest.main()
