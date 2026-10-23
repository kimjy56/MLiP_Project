import unittest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
from torch_indexer import TorchNgramIndexer


class TestTorchNgramIndexer(unittest.TestCase):
    def setUp(self):
        self.catalog = [
            "ALPHA STAR",
            "BETA MOON",
            "GAMMA SUN",
            "DELTA WIND",
            "OCEAN GLORY",
            "KOREA EXPRESS",
        ]
        self.indexer = TorchNgramIndexer(self.catalog)

    def test_exact_query(self):
        res = self.indexer.search("ALPHA STAR", count=2)
        self.assertGreater(len(res), 0)
        self.assertEqual(res[0], 0)

    def test_fuzzy_query(self):
        res = self.indexer.search("ALFA STAR", count=2)
        self.assertGreater(len(res), 0)
        self.assertEqual(res[0], 0)

    def test_empty_and_none_queries(self):
        self.assertEqual(self.indexer.search(None, count=2), [])
        self.assertEqual(self.indexer.search("", count=2), [])
        self.assertEqual(self.indexer.search("   ", count=2), [])

    def test_batch_matches_single_independent_queries(self):
        queries = ["ALPHA STAR", "BETA MOON", None, "OCEAN GLORY", ""]
        batch_res = self.indexer.batch_search(queries, count=3)
        for q, b_items in zip(queries, batch_res):
            single_items = self.indexer.search(q, count=3)
            self.assertEqual(b_items, single_items)


if __name__ == "__main__":
    unittest.main()
