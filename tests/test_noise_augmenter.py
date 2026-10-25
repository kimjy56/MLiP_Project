"""Unit tests for noise augmentation engine."""
import random
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from name_decoder.noise_augmenter import (
    augment_english_name,
    augment_hangul_name,
    generate_augmented_pairs,
)


class TestNoiseAugmenter(unittest.TestCase):
    def test_english_augmentation_deterministic(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        name = "HEMISPHERE"
        out1 = augment_english_name(name, rng1)
        out2 = augment_english_name(name, rng2)
        self.assertEqual(out1, out2)
        self.assertTrue(len(out1) > 0)

    def test_hangul_augmentation_deterministic(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        name = "모닝 레나"
        out1 = augment_hangul_name(name, rng1)
        out2 = augment_hangul_name(name, rng2)
        self.assertEqual(out1, out2)
        self.assertTrue(len(out1) > 0)

    def test_generate_augmented_pairs(self):
        df = pd.DataFrame([
            {"catalog_name": "HEMISPHERE", "observed_name": "HEMISPHERE", "hangul": False},
            {"catalog_name": "MORNING LENA", "observed_name": "모닝 레나", "hangul": True},
            {"catalog_name": "ALWINE", "observed_name": "ALWINE", "hangul": False},
        ])
        aug1 = generate_augmented_pairs(df, seed=20261022, augment_factor=1.0)
        aug2 = generate_augmented_pairs(df, seed=20261022, augment_factor=1.0)
        self.assertEqual(aug1, aug2)
        self.assertTrue(len(aug1) > 0)
        for cat, noisy in aug1:
            self.assertIsInstance(cat, str)
            self.assertIsInstance(noisy, str)
            self.assertTrue(len(cat) > 0)
            self.assertTrue(len(noisy) > 0)


if __name__ == "__main__":
    unittest.main()
