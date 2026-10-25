"""Unit tests for V2 Task-Prefix Noisy Channel prompt encoding."""
import json
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from name_decoder.noisy_channel import (
    PROMPT_VERSION_V1,
    PROMPT_VERSION_V2,
    encode_channel_example,
    collate_channel_examples,
)


class TestNoisyChannelV2Prompt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load local tokenizer
        tok_path = ROOT / "artifacts/AS-HCX-NC-01/model"
        cls.tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)

    def test_v2_hangul_task_prefix(self):
        cat_name = "MARSGRACHT"
        obs_name = "마스그라치"
        example = encode_channel_example(
            self.tokenizer, cat_name, obs_name, prompt_version=PROMPT_VERSION_V2
        )
        prompt_text = self.tokenizer.decode(example.input_ids[:example.prompt_length])
        self.assertIn("KO_TRANSLITERATION", prompt_text)
        self.assertIn('"task": "KO_TRANSLITERATION"', prompt_text)
        self.assertIn('"catalog_name": "MARSGRACHT"', prompt_text)

        # Target verification
        target_tokens = example.input_ids[example.prompt_length:-1]
        decoded_target = self.tokenizer.decode(target_tokens)
        self.assertEqual(decoded_target, obs_name)
        self.assertEqual(example.input_ids[-1], self.tokenizer.eos_token_id)

    def test_v2_english_task_prefix(self):
        cat_name = "HEMISPHERE"
        obs_name = "M/V HEMISPEHRE"
        example = encode_channel_example(
            self.tokenizer, cat_name, obs_name, prompt_version=PROMPT_VERSION_V2
        )
        prompt_text = self.tokenizer.decode(example.input_ids[:example.prompt_length])
        self.assertIn("EN_TYPO_RECOVERY", prompt_text)
        self.assertIn('"task": "EN_TYPO_RECOVERY"', prompt_text)
        self.assertIn('"catalog_name": "HEMISPHERE"', prompt_text)

        # Target verification
        target_tokens = example.input_ids[example.prompt_length:-1]
        decoded_target = self.tokenizer.decode(target_tokens)
        self.assertEqual(decoded_target, obs_name)

    def test_v1_backward_compatibility(self):
        cat_name = "MARSGRACHT"
        obs_name = "마스그라치"
        example_v1 = encode_channel_example(
            self.tokenizer, cat_name, obs_name, prompt_version=PROMPT_VERSION_V1
        )
        prompt_text_v1 = self.tokenizer.decode(example_v1.input_ids[:example_v1.prompt_length])
        self.assertNotIn("KO_TRANSLITERATION", prompt_text_v1)
        self.assertIn('"catalog_name": "MARSGRACHT"', prompt_text_v1)


if __name__ == "__main__":
    unittest.main()
