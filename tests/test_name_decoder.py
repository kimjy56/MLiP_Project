"""Software-correctness fixtures, not external vessel validation data."""
import unittest
import tempfile
from pathlib import Path

import pandas as pd
import torch
from transformers import ByT5Tokenizer, T5Config, T5ForConditionalGeneration

from name_decoder.catalog import ByteNameTrie, CatalogNames, canonical_name
from name_decoder.data import prepare_frames, prepare_final_frames, read_frames, select_pilot
from name_decoder.model import ByT5NameDecoder


def fixtures():
    train = pd.DataFrame([
        {"sample_id": "s0", "target_vessel_id": "v0", "vessel_name": "에이스원"},
        {"sample_id": "s1", "target_vessel_id": "v1", "vessel_name": "비원"},
        {"sample_id": "s2", "target_vessel_id": "v1", "vessel_name": "비원"},
        {"sample_id": "s3", "target_vessel_id": "v2", "vessel_name": "비원"},
    ])
    catalog = pd.DataFrame([
        {"vessel_id": "v0", "vessel_name": "ACE ONE"},
        {"vessel_id": "v1", "vessel_name": "B ONE"},
        {"vessel_id": "v2", "vessel_name": "B 1"},
    ])
    folds = train[["sample_id", "target_vessel_id"]].copy()
    folds["fold"] = [0, 1, 1, 2]
    return train, catalog, folds


class CatalogTests(unittest.TestCase):
    def test_name_keys_preserve_numbers_and_punctuation(self):
        self.assertEqual(canonical_name("  Ace   01-B "), "ACE 01-B")
        self.assertNotEqual(canonical_name("ACE ONE"), canonical_name("ACE 1"))

    def test_same_name_keeps_every_id(self):
        catalog = CatalogNames([{"vessel_id": "z", "vessel_name": "ace"},
                                {"vessel_id": "a", "vessel_name": "ACE"}])
        self.assertEqual(catalog.ids_by_name["ACE"], ("a", "z"))

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            CatalogNames([{"vessel_id": "a", "vessel_name": "A"},
                          {"vessel_id": "a", "vessel_name": "B"}])

    def test_terminal_is_not_confused_with_a_longer_name(self):
        trie = ByteNameTrie(["ACE", "ACE ONE"])
        prefix = [0] + [b + 3 for b in b"ACE"]
        self.assertEqual(trie.allowed(prefix), [1, 35])
        self.assertEqual(trie.allowed(prefix + [1]), [0])
        with self.assertRaises(ValueError):
            trie.allowed([0, 255])

    def test_unicode_bytes_match_tokenizer(self):
        name = "가 01"
        tokenizer = ByT5Tokenizer()
        tokens = tokenizer.encode(name)
        trie = ByteNameTrie([name])
        prefix = [0]
        for token in tokens:
            self.assertIn(token, trie.allowed(prefix))
            prefix.append(token)


class DataTests(unittest.TestCase):
    def test_final_training_keeps_all_official_pairs_without_a_split(self):
        train, catalog, _ = fixtures()
        pairs, rows, index = prepare_final_frames(train, catalog)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(set(pairs.target), set(index.names))
        self.assertNotIn("fold", rows.columns)

    def test_final_training_does_not_require_folds_but_evaluation_does(self):
        train, catalog, _ = fixtures()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            train.to_csv(path / "train.csv", index=False)
            catalog.to_csv(path / "vessel_catalog.csv", index=False)
            missing = path / "missing.csv"
            pairs, val, fit, _ = read_frames(path, missing, 0, final_train=True)
            self.assertTrue(val.empty)
            self.assertEqual(len(fit), 4)
            self.assertEqual(len(pairs), 3)
            with self.assertRaisesRegex(FileNotFoundError, "Common fold file missing"):
                read_frames(path, missing, 0)

    def test_no_heldout_training_and_ambiguous_targets_remain(self):
        pairs, val, fit, catalog = prepare_frames(*fixtures(), heldout_fold=0)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(set(pairs.target), {"B ONE", "B 1"})
        self.assertEqual(set(val.target_vessel_id), {"v0"})
        self.assertTrue(set(fit.target_vessel_id).isdisjoint(val.target_vessel_id))
        self.assertIn("ACE ONE", catalog.names)  # Official catalog, not validation labels.

    def test_cross_fold_vessel_is_rejected(self):
        train, catalog, folds = fixtures()
        folds.loc[2, "fold"] = 0
        with self.assertRaisesRegex(ValueError, "crosses folds"):
            prepare_frames(train, catalog, folds, 0)

    def test_mismatched_fold_labels_are_rejected(self):
        train, catalog, folds = fixtures()
        folds.loc[0, "target_vessel_id"] = "v1"
        with self.assertRaisesRegex(ValueError, "disagree"):
            prepare_frames(train, catalog, folds, 0)

    def test_pilot_selection_does_not_depend_on_input_order(self):
        _, val, _, _ = prepare_frames(*fixtures(), heldout_fold=0)
        self.assertEqual(select_pilot(val, 1, 42).sample_id.tolist(),
                         select_pilot(val.iloc[::-1], 1, 42).sample_id.tolist())


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        config = T5Config(vocab_size=384, d_model=16, d_ff=32, d_kv=8,
                          num_layers=1, num_decoder_layers=1, num_heads=2,
                          dropout_rate=0., pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
        catalog = CatalogNames([{"vessel_id": str(i), "vessel_name": n}
                                for i, n in enumerate(["A", "A 1", "B", "가", "ACE"])])
        cls.decoder = ByT5NameDecoder(T5ForConditionalGeneration(config), ByT5Tokenizer(), catalog)

    def test_constrained_decoding_and_row_independence(self):
        first = self.decoder.decode_many(["에이", "비"], top_k=5)
        second = self.decoder.decode_many(["비", "에이"], top_k=5)
        self.assertEqual(first, second[::-1])
        self.assertEqual(first[0], self.decoder.decode("에이", top_k=5))
        for candidates in first:
            self.assertEqual(len(candidates), 5)
            self.assertEqual(len({c.name for c in candidates}), 5)
            for candidate in candidates:
                self.assertIn(candidate.name, self.decoder.catalog.names)

    def test_top_one_uses_same_search_and_preserves_all_matching_ids(self):
        from unittest.mock import patch
        from name_decoder.catalog import NameCandidate
        ranked = [NameCandidate("ACE", -1.2, ("v1", "v2")),
                  NameCandidate("B", -2.0, ("v3",))]
        with patch.object(self.decoder, "decode", return_value=ranked) as decode:
            result = self.decoder.predict("에이스")
        decode.assert_called_once_with("에이스", top_k=10)
        self.assertEqual(result.predicted_name, "ACE")
        self.assertEqual(result.vessel_ids, ("v1", "v2"))
        self.assertEqual(result.candidates, tuple(ranked))

    def test_invalid_inputs_fail_instead_of_truncating(self):
        for source in ("", "A" * 257):
            with self.assertRaises(ValueError):
                self.decoder.decode(source)
        with self.assertRaises(ValueError):
            self.decoder.decode("A", top_k=0)

    def test_untied_checkpoint_preserves_config_weights_and_candidate_names(self):
        config = T5Config(vocab_size=384, d_model=16, d_ff=32, d_kv=8,
                          num_layers=1, num_decoder_layers=1, num_heads=2,
                          dropout_rate=0., pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
        config.tie_word_embeddings = False
        config.scale_decoder_outputs = False
        config._attn_implementation = "eager"
        decoder = ByT5NameDecoder(T5ForConditionalGeneration(config), ByT5Tokenizer(), self.decoder.catalog)
        cache = Path(__file__).resolve().parents[1] / ".cache/name_decoder_tests"
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as folder:
            decoder.model.save_pretrained(folder)
            decoder.tokenizer.save_pretrained(folder)
            loaded = ByT5NameDecoder.load(folder, decoder.catalog)
            self.assertFalse(loaded.model.config.tie_word_embeddings)
            self.assertFalse(loaded.model.config.scale_decoder_outputs)
            for key, value in decoder.model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.model.state_dict()[key]), key)
            self.assertEqual([c.name for c in decoder.decode("에이스원", 5)],
                             [c.name for c in loaded.decode("에이스원", 5)])


if __name__ == "__main__":
    unittest.main()
