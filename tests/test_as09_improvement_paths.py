"""Synthetic end-to-end checks of experimental candidate-union paths."""
import unittest

import numpy as np
import pandas as pd

from experimental_matchers.as09_candidate_extension import as09, predict_row
from experimental_matchers.en_global_retrieval import CatalogNgramIndex, compare_row as compare_en
from experimental_matchers.exaone_candidate_union import ReverseDecoderCandidates, compare_row as compare_ko
from experimental_matchers.lookup_conflict import predict_lookup_conflict


def fixture():
    catalog = pd.DataFrame({
        "vessel_id": ["V009", "V003", "V004", "V099"],
        "vessel_name": ["SILVER WIND 9", "SILVER WIND 3", "SILVER WIND 3", "GOLDEN SKY"],
        "gross_tonnage": [900., 300., 310., 400.],
        "vessel_type": ["cargo"] * 4,
        "flag_name": ["KR"] * 4,
    })
    index = as09.CatalogIndex(catalog, full_ko_map={})
    model = {"kr_mode": "AR06", "en_mode": "EN_V3", "dense_mode": "none",
             "callsign_lookup": {"C1": "V009"}, "name_lookup": {},
             "use_train_lookup_default": True,
             "type_trans_prob": {"cargo": {"cargo": 1.}},
             "flag_trans_prob": {"KR|MID": {"KR": 1.}}}
    return model, index


def query(name, call_sign=""):
    return pd.DataFrame([{"sample_id": "query", "vessel_name": name, "call_sign": call_sign,
                          "gross_tonnage": np.nan, "international_gross_tonnage": np.nan,
                          "flag_name": "KR", "vessel_type": "cargo", "arrival_year": 2021,
                          "target_vessel_id": "must_not_reach_prediction", "fold": 4}])


class FakeDecoder:
    def decode(self, names, **kwargs):
        return [[{"text": "SILVER WIND 3", "score": 0., "score_type": "beam_rank"}]]


class As09ImprovementPaths(unittest.TestCase):
    def test_candidate_union_preserves_fallback_and_legacy_scores(self):
        model, index = fixture()
        q = query("ZZZZZ")
        original = as09.predict_rows(q.drop(columns=["target_vessel_id", "fold"]), model, index,
                                     use_train_lookup=True, top_candidates=index.n)
        base_pred, baseline = predict_row(q, model, index, diagnostic=True)
        self.assertEqual(base_pred, original[0][0])
        self.assertEqual(baseline, tuple(original[2][0]))
        expanded_pred, expanded = predict_row(q, model, index, candidates=["V003"], diagnostic=True)
        base_scores, expanded_scores = dict((vid, score) for score, vid in baseline), dict((vid, score) for score, vid in expanded)
        self.assertTrue(set(base_scores) <= set(expanded_scores))
        self.assertEqual(base_scores, {vid: expanded_scores[vid] for vid in base_scores})
        self.assertIn("V003", expanded_scores)
        self.assertIn(expanded_pred, expanded_scores)

    def test_lookup_conflict_keeps_original_candidate(self):
        model, index = fixture()
        baseline = predict_row(query("SILVER WIND 3", "C1"), model, index)
        result = predict_lookup_conflict(query("SILVER WIND 3", "C1"), model, index)
        self.assertEqual(baseline, "V009")
        self.assertEqual(result.route, "reconsidered")
        self.assertIn("number_conflict", result.reasons)
        self.assertIn("V009", {vid for _, vid in result.candidates})
        self.assertIn(result.prediction, {"V003", "V004"})
        unchanged = predict_lookup_conflict(query("SILVER WIND 9", "C1"), model, index)
        self.assertEqual((unchanged.route, unchanged.prediction), ("unchanged", "V009"))
        index.word_Alpha[index.vid_to_idx["V009"]] = "한글이름"
        cross_script = predict_lookup_conflict(query("SILVER WIND", "C1"), model, index)
        self.assertEqual((cross_script.route, cross_script.prediction), ("unchanged", "V009"))
        invalid_tonnage = query("SILVER WIND 3", "C1")
        invalid_tonnage["gross_tonnage"] = np.inf
        invalid_result = predict_lookup_conflict(invalid_tonnage, model, index)
        self.assertIsNone(invalid_result.lookup_id)
        self.assertEqual(invalid_result.prediction, predict_row(invalid_tonnage, model, index))
        model["use_train_lookup_default"] = False
        disabled = predict_lookup_conflict(query("SILVER WIND 3", "C1"), model, index)
        self.assertIsNone(disabled.lookup_id)
        self.assertEqual(disabled.prediction, predict_row(query("SILVER WIND 3", "C1"), model, index))

    def test_union_does_not_change_the_200_candidate_fallback(self):
        names = ([f"ABC {i}" for i in range(180)] + [f"ABZ {i}" for i in range(40)] +
                 [f"XMN {i}" for i in range(20)])
        catalog = pd.DataFrame({"vessel_id": [f"T{i:03d}" for i in range(240)],
            "vessel_name": names, "gross_tonnage": [100.] * 240,
            "vessel_type": ["cargo"] * 240, "flag_name": ["KR"] * 240})
        index = as09.CatalogIndex(catalog, full_ko_map={})
        model = {"kr_mode": "AR06", "en_mode": "EN_V3", "dense_mode": "none"}
        _, baseline = predict_row(query("ABCXYZ"), model, index, diagnostic=True)
        _, expanded = predict_row(query("ABCXYZ"), model, index,
                                  candidates=[f"T{i:03d}" for i in range(220, 240)], diagnostic=True)
        self.assertGreaterEqual(len(baseline), 220)
        self.assertTrue({vid for _, vid in baseline} <= {vid for _, vid in expanded})

    def test_korean_extension_matches_original_without_extra_candidates(self):
        model, index = fixture()
        q = query("실버 윈드 3")
        raw = q.drop(columns=["target_vessel_id", "fold"])
        original = as09.predict_rows(raw, model, index, top_candidates=index.n, dense_mode="none")
        predicted, candidates = predict_row(q, model, index, diagnostic=True, dense_mode="none")
        self.assertEqual(predicted, original[0][0])
        self.assertEqual(candidates, tuple(original[2][0]))
        missing_transition = dict(model, flag_trans_prob={})
        with self.assertRaisesRegex(RuntimeError, "transition tables are required"):
            predict_row(q, missing_transition, index, dense_mode="none")

    def test_non_korean_ngram_index_and_union(self):
        model, index = fixture()
        ngram = CatalogNgramIndex(index)
        # Identical catalog names are retained as distinct IDs and tie by ID.
        self.assertEqual(ngram.search("SILVER WIND 3", count=2), ("V003", "V004"))
        comparison = compare_en(query("AILVER WIND 3"), model, index, ngram, count=3)
        self.assertTrue({vid for _, vid in comparison.baseline_candidates} <=
                        {vid for _, vid in comparison.expanded_candidates})
        self.assertTrue({"V003", "V004"} <= set(comparison.ngram_candidates))
        self.assertEqual(comparison, compare_en(query("AILVER WIND 3"), model, index, ngram, count=3))
        lookup = compare_en(query("SILVER WIND 3", "C1"), model, index, ngram)
        self.assertEqual((lookup.route, lookup.expanded_prediction, lookup.ngram_candidates),
                         ("lookup_bypass", "V009", ()))

    def test_exaone_candidate_union_and_invalid_input(self):
        model, index = fixture()
        retriever = ReverseDecoderCandidates(index, FakeDecoder())
        self.assertEqual(retriever.search("실버/윈드"), ())
        comparison = compare_ko(query("실버 윈드 3"), model, index, retriever, dense_mode="none")
        self.assertTrue({"V003", "V004"} <= {c.vessel_id for c in comparison.decoder_candidates})
        self.assertTrue({vid for _, vid in comparison.baseline_candidates} <=
                        {vid for _, vid in comparison.expanded_candidates})
        self.assertEqual(comparison, compare_ko(query("실버 윈드 3"), model, index, retriever, dense_mode="none"))
        lookup = compare_ko(query("실버 윈드 3", "C1"), model, index, retriever, dense_mode="none")
        self.assertEqual((lookup.route, lookup.expanded_prediction, lookup.decoder_candidates),
                         ("lookup_bypass", "V009", ()))


if __name__ == "__main__":
    unittest.main()
