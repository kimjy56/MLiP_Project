"""Unit tests for HyperCLOVANameDecoder and Beam Search aggregation wiring."""
import unittest
import torch
from name_decoder import CatalogNames, HyperCLOVANameDecoder, NameCandidate, NamePrediction


class MockInputs(dict):
    def to(self, device):
        return self


class MockTokenizer:
    eos_token_id = 2

    def __call__(self, prompt, return_tensors):
        return MockInputs(input_ids=torch.tensor([[10, 11]]))

    def decode(self, ids, skip_special_tokens=True):
        # Deterministically return based on the sequence
        seq = tuple(ids.tolist())
        if seq == (20,):
            return "POLAR BRAHT\nextra"
        elif seq == (21,):
            return "POLAR BRIGHT\nextra"
        return "UNKNOWN VESSEL"


class MockModel:
    device = torch.device("cpu")

    def eval(self):
        return self

    def generate(self, **kwargs):
        num_return = kwargs.get("num_return_sequences", 1)
        if num_return == 1:
            return torch.tensor([[10, 11, 20]])
        # Return 2 beam sequences
        return torch.tensor([[10, 11, 20], [10, 11, 21]])


class HyperCLOVADecoderTests(unittest.TestCase):
    def setUp(self):
        records = [
            {"vessel_id": "V1", "vessel_name": "POLAR BRIGHT"},
            {"vessel_id": "V2", "vessel_name": "POLAR BEAR"},
            {"vessel_id": "V3", "vessel_name": "SAKURA TIGER"},
        ]
        self.catalog = CatalogNames(records)

    def test_single_sequence_decode(self):
        decoder = HyperCLOVANameDecoder(
            model=MockModel(),
            tokenizer=MockTokenizer(),
            catalog=self.catalog,
            num_beams=1,
            num_return_sequences=1,
            scorer="ratio",
        )
        candidates = decoder.decode("폴라 브라이트", top_k=2)
        self.assertEqual(len(candidates), 2)
        self.assertIsInstance(candidates[0], NameCandidate)
        # POLAR BRAHT matches POLAR BRIGHT closely
        self.assertEqual(candidates[0].name, "POLAR BRIGHT")
        self.assertEqual(candidates[0].vessel_ids, ("V1",))

        pred = decoder.predict("폴라 브라이트", top_k=2)
        self.assertIsInstance(pred, NamePrediction)
        self.assertEqual(pred.predicted_name, "POLAR BRIGHT")

    def test_multi_beam_aggregation(self):
        decoder = HyperCLOVANameDecoder(
            model=MockModel(),
            tokenizer=MockTokenizer(),
            catalog=self.catalog,
            num_beams=2,
            num_return_sequences=2,
            scorer="ratio",
        )
        candidates = decoder.decode("폴라 브라이트", top_k=2)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].name, "POLAR BRIGHT")
        self.assertEqual(candidates[0].vessel_ids, ("V1",))


if __name__ == "__main__":
    unittest.main()
