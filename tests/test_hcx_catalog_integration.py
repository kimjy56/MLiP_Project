"""Exercise actual HCX evaluator wiring with deterministic generated text."""
import os
import unittest
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '4'
import pandas as pd
import torch
from experiments.train_hyperclovax_sft import evaluate_pilot

torch.set_num_threads(4)

class Inputs(dict):
    def to(self, device):
        return self

class Tokenizer:
    eos_token_id = 0
    def __call__(self, prompt, return_tensors):
        return Inputs(input_ids=torch.tensor([[1]]))
    def decode(self, ids, skip_special_tokens):
        return ' polar brait\nextra text'

class Model:
    def eval(self):
        return self
    def generate(self, **kwargs):
        return torch.tensor([[1, 2]])

class HCXScorerIntegrationTests(unittest.TestCase):
    def test_evaluator_default_and_legacy_scorer(self):
        pilot = pd.DataFrame([
            dict(sample_id='a', source='input a', target='POLAR BRIGHT', hangul=True),
            dict(sample_id='b', source='input b', target='POLAR BRIGHT', hangul=False)])
        catalog = ['POLA', 'POLAR BRIGHT']
        current, metrics, _ = evaluate_pilot(Model(), Tokenizer(), pilot, catalog)
        legacy, _, _ = evaluate_pilot(Model(), Tokenizer(), pilot, catalog, catalog_scorer='WRatio')
        self.assertEqual(current.generated_text.tolist(), ['POLAR BRAIT'] * 2)
        self.assertEqual(current.predicted_catalog_name.tolist(), ['POLAR BRIGHT'] * 2)
        self.assertEqual(legacy.predicted_catalog_name.tolist(), ['POLA'] * 2)
        self.assertEqual(metrics['weighted_name_exact_accuracy'], 1.0)
        self.assertFalse(current.direct_exact_match.any())

if __name__ == '__main__':
    unittest.main()
