import re
import unittest
from name_decoder.pronunciation_quality import clean_sample, select_variants, schedule_names


def sample(text, score=-0.2, finished=True):
    return {'text': text, 'mean_logprob': score, 'finished': finished}


class TestPronunciationQuality(unittest.TestCase):
    def choose(self, originals, samples, name='VESSEL 7', **kw):
        return select_variants(name, originals, samples,
            to_jamo=lambda text: re.sub(r'[\s0-9]', '', text),
            number_signature=lambda text: tuple(re.findall(r'\d+', text)), **kw)

    def test_schedule_uses_asset_coverage_only(self):
        self.assertEqual(schedule_names(['C','A','B','B'], {'A':['a','b'],'B':['b']}, 'deficient'), ['B','C'])
        self.assertEqual(schedule_names(['B','A'], {}, 'all'), ['A','B'])
        with self.assertRaises(ValueError):
            schedule_names([], {}, 'invalid')

    def test_rejects_explanations_latin_nonhangul_and_repetition(self):
        for text in ['BLUE', '123', '\uac00\n\ub098', '\uac00' * 6, '\uac00: \ub098']:
            self.assertIsNone(clean_sample(text))
        self.assertEqual(clean_sample(' "\uac00  \ub098 7" '), '\uac00 \ub098 7')

    def test_preserves_baseline_exactly_and_filters_unfinished_numeric_errors(self):
        originals = ['\uac00\ub098\ub2e4 7']
        variants, reasons = self.choose(originals, [sample('\uac00\ub098\ub77c 8'), sample('\uac00\ub098\ub77c 7', finished=False)])
        self.assertEqual(variants, originals)
        self.assertEqual(reasons['no_eos'], 1)
        self.assertEqual(reasons['number_signature_mismatch'], 1)
        self.assertEqual(originals, ['\uac00\ub098\ub2e4 7'])

    def test_raw_jamo_dedup_and_same_prefix_late_variant(self):
        variants, _ = self.choose(['\uac00\ub098\ub2e4 7'], [sample('\uac00 \ub098\ub2e4 7'), sample('\uac00\ub098\ub77c 7')])
        self.assertEqual(variants, ['\uac00\ub098\ub2e4 7', '\uac00\ub098\ub77c 7'])

    def test_deterministic_order_and_cap(self):
        samples = [sample('\uac00\ub098\ub77c 7'), sample('\uac00\ub098\ub9c8 7'), sample('\uac00\ub098\ubc14 7')]
        a, _ = self.choose(['\uac00\ub098\ub2e4 7'], samples, max_variants=3)
        b, _ = self.choose(['\uac00\ub098\ub2e4 7'], samples[::-1], max_variants=3)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 3)

    def test_logprob_and_anchor_quality(self):
        variants, reasons = self.choose(['\uac00\ub098\ub2e4 7'], [sample('\uac00\ub098\ub77c 7'), sample('\uac00\ub098\ub9c8 7', -4), sample('\ud0c0\ud30c\ud558 7')])
        self.assertEqual(variants, ['\uac00\ub098\ub2e4 7', '\uac00\ub098\ub77c 7'])
        self.assertEqual(reasons['relative_logprob'], 1)
        self.assertEqual(reasons['anchor_dissimilar'], 1)

    def test_missing_base_can_be_filled_but_existing_base_cannot_be_truncated(self):
        variants, _ = self.choose([], [sample('\uac00\ub098 7')])
        self.assertEqual(variants, ['\uac00\ub098 7'])
        with self.assertRaises(ValueError):
            self.choose(['a','b'], [], max_variants=1)


if __name__ == '__main__':
    unittest.main()