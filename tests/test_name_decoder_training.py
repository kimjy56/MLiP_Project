"""Tiny software fixtures only: never vessel-model performance evidence."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch
from transformers import ByT5Tokenizer, T5Config, T5ForConditionalGeneration

from experiments import train_name_decoder as runner


class TrainingMathTests(unittest.TestCase):
    def test_budget_preserves_old_steps_and_counts_weighted_draws(self):
        self.assertEqual(runner.training_steps(4685, 8, 2), 200)
        self.assertEqual(runner.training_steps(4685, 8, 2, epochs=10), 2929)
        self.assertEqual(runner.training_steps(4685, 8, 2, steps=9), 9)
        for kwargs in ({"epochs": 0}, {"epochs": float("nan")},
                       {"steps": 0}, {"steps": 2, "epochs": 3}):
            with self.assertRaises(ValueError):
                runner.training_steps(10, 2, 1, **kwargs)

    def test_row_loss_ignores_padding_and_equalizes_name_lengths(self):
        logits = torch.tensor([[[3., 0.], [0., 2.], [8., -2.]],
                               [[0., 1.], [2., 0.], [0., 2.]]], requires_grad=True)
        labels = torch.tensor([[0, -100, -100], [1, 0, 1]])
        first = torch.nn.functional.cross_entropy(logits[0, :1], labels[0, :1])
        second = torch.nn.functional.cross_entropy(logits[1], labels[1])
        row = runner.sequence_loss(logits, labels, "row")
        token = runner.sequence_loss(logits, labels, "token")
        torch.testing.assert_close(row, (first + second) / 2)
        torch.testing.assert_close(token, (first + second * 3) / 4)
        row.backward()
        self.assertEqual(logits.grad[0, 1:].abs().sum().item(), 0.)
        with self.assertRaises(ValueError):
            runner.sequence_loss(logits, torch.full_like(labels, -100), "row")

    def test_name_metrics_do_not_claim_id_accuracy(self):
        rows = pd.DataFrame({"hangul": [True, True, False], "target": ["A", "B", "C"]})
        result = runner.name_metrics(rows, [["A"], ["X", "B"], ["C"]])
        self.assertEqual(result["hangul"]["name_exact_accuracy"], .5)
        self.assertEqual(result["hangul"]["gold_name_hits_top_5"], 2)
        self.assertAlmostEqual(result["weighted_name_exact_accuracy"], .65)
        self.assertNotIn("weighted_accuracy", result)
        one_group = runner.name_metrics(rows.iloc[:2], [["A"], ["B"]])
        self.assertIsNone(one_group["weighted_name_exact_accuracy"])
        with self.assertRaises(ValueError):
            runner.name_metrics(rows, [["A"]])


class WorkflowTests(unittest.TestCase):
    def test_train_save_reload_and_evaluate_or_final_train(self):
        template = runner.ROOT / "project_log/templates/experiment.json"
        for final_train in (False, True):
            with self.subTest(final_train=final_train), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "project_log/templates").mkdir(parents=True)
                (root / "project_log/experiments").mkdir()
                (root / "project_log/templates/experiment.json").write_bytes(template.read_bytes())
                data = root / "data"
                data.mkdir()
                train = pd.DataFrame({"sample_id": ["s0", "s1", "s2"],
                                      "vessel_name": ["에이", "B", "시"],
                                      "target_vessel_id": ["v0", "v1", "v2"]})
                train.to_csv(data / "train.csv", index=False)
                pd.DataFrame({"vessel_id": ["v0", "v1", "v2"],
                              "vessel_name": ["A", "B", "C"]}).to_csv(data / "vessel_catalog.csv", index=False)
                folds = root / "folds.csv"
                if not final_train:
                    train[["sample_id", "target_vessel_id"]].assign(fold=[0, 1, 2]).to_csv(folds, index=False)
                base = root / "base"
                torch.manual_seed(42)
                config = T5Config(vocab_size=384, d_model=8, d_ff=16, d_kv=4,
                                  num_layers=1, num_decoder_layers=1, num_heads=2,
                                  dropout_rate=0., pad_token_id=0, eos_token_id=1, decoder_start_token_id=0,
                                  tie_word_embeddings=False)
                config.tie_word_embeddings = False
                config.scale_decoder_outputs = False
                T5ForConditionalGeneration(config).save_pretrained(base)
                ByT5Tokenizer().save_pretrained(base)
                (base / "download_manifest.json").write_text('{"software_fixture": true}')
                argv = ["train_name_decoder.py", "--experiment-id", "FIXTURE-01",
                        "--data-dir", str(data), "--folds", str(folds), "--base-model", str(base),
                        "--steps", "1", "--batch-size", "2", "--accumulation", "1",
                        "--loss-reduction", "row", "--device", "cpu"]
                if final_train:
                    argv.append("--final-train")
                with patch.object(runner, "ROOT", root), patch("sys.argv", argv), \
                     patch.object(runner.subprocess, "check_output", return_value="fixture"), \
                     contextlib.redirect_stdout(io.StringIO()):
                    runner.main()
                    record_path = root / "project_log/experiments/FIXTURE-01.json"
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                    self.assertEqual(record["status"], "completed")
                    self.assertTrue(record["serialization"]["top_10_names_equal"])
                    self.assertIsNone(record["metrics"]["weighted_accuracy"])
                    predictions = root / "artifacts/FIXTURE-01/pilot_predictions.csv"
                    self.assertEqual(predictions.exists(), not final_train)
                    if final_train:
                        self.assertIsNone(record["pilot_results"])
                        self.assertIsNone(record["data"]["split_file_sha256"])
                    else:
                        self.assertIn("predicted_name", pd.read_csv(predictions).columns)
                    self.assertTrue((root / "artifacts/FIXTURE-01/model/name_decoder_manifest.json").is_file())
                    original = record_path.read_bytes()
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        runner.main()
                    self.assertEqual(record_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
