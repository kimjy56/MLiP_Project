"""Tests for EXAONE reverse transliteration module."""
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from name_decoder.exaone_reverse import build_reverse_prompt, encode_reverse_example


class DummyTokenizer:
    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = 2
        self.all_special_ids = [0, 1, 2]

    def encode(self, text, add_special_tokens=False):
        # Deterministic ascii/character tokenization for unit tests
        return [ord(c) % 100 + 10 for c in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr((t - 10) % 100 + 32) for t in token_ids)

    def __len__(self):
        return 120


class TestExaoneReverse(unittest.TestCase):
    def test_build_reverse_prompt(self):
        prompt = build_reverse_prompt("바우 세실")
        self.assertEqual(prompt, "[KO] 바우 세실 => [EN] ")
        with self.assertRaises(ValueError):
            build_reverse_prompt("")

    def test_encode_reverse_example(self):
        tok = DummyTokenizer()
        encoded = encode_reverse_example(tok, "바우 세실", "BOW CECIL")
        self.assertIn("input_ids", encoded)
        self.assertIn("labels", encoded)
        prompt_len = len(tok.encode(build_reverse_prompt("바우 세실"), add_special_tokens=False))
        self.assertTrue(all(v == -100 for v in encoded["labels"][:prompt_len]))
        self.assertEqual(encoded["labels"][-1], tok.eos_token_id)
        self.assertNotEqual(encoded["labels"][prompt_len], -100)


if __name__ == "__main__":
    unittest.main()
