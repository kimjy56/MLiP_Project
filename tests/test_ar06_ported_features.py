import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'model') not in sys.path:
    sys.path.insert(0, str(ROOT / 'model'))

from model.name_processing import expand_korean_name, name_view, parse_korean_acronym_token
from model.matcher import CatalogIndex, make_assets, prepare_row


class TestAR06PortedFeatures(unittest.TestCase):
    def test_pinyin_contractions_and_lexicon(self):
        # 1. Pinyin contractions & ports
        s1 = expand_korean_name("닝하이투오")
        self.assertIn("NINGHAI", s1)
        self.assertIn("TUO", s1)

        s2 = expand_korean_name("바오루이")
        self.assertIn("BAORUI", s2)

        s3 = expand_korean_name("샤오룽")
        self.assertIn("XIAO", s3)

        # 2. Ferry and maritime loanwords
        self.assertEqual(expand_korean_name("에코누리"), "ECONURI")
        self.assertEqual(expand_korean_name("한우리"), "HANWOORI")
        self.assertEqual(expand_korean_name("코모도어"), "COMMODORE")
        self.assertEqual(expand_korean_name("쓰시마"), "TSUSHIMA")
        self.assertEqual(expand_korean_name("누산타라"), "NUSANTARA")
        self.assertEqual(expand_korean_name("썬크루저"), "SUNCRUISER")

        # 3. Colloquial acronym syllables
        self.assertEqual(parse_korean_acronym_token("재이에스"), "JS")
        self.assertEqual(parse_korean_acronym_token("앰엠씨"), "MMC")
        self.assertEqual(parse_korean_acronym_token("애프에이"), "FA")

    def test_l_ord_sorted_token_retrieval(self):
        # Catalog has "MORNING PACIFIC", arrival row is "PACIFIC MORNING"
        catalog = pd.DataFrame({
            'vessel_id': ['V_CAT_01', 'V_CAT_02'],
            'vessel_name': ['MORNING PACIFIC', 'PACIFIC STAR'],
            'gross_tonnage': [50000.0, 50000.0],
            'vessel_type': ['Cargo', 'Cargo'],
            'flag_name': ['Panama', 'Panama']
        })
        train = pd.DataFrame([{
            'sample_id': 'S_TRAIN_01',
            'vessel_name': 'MORNING PACIFIC',
            'call_sign': 'CALL1',
            'gross_tonnage': 50000.0,
            'international_gross_tonnage': 50000.0,
            'vessel_type': 'Cargo',
            'flag_name': 'Panama',
            'target_vessel_id': 'V_CAT_01'
        }])
        assets = make_assets(train, catalog, retrieval_channels=['S'])
        index = CatalogIndex(catalog, assets)

        # Query with inverted word order: "PACIFIC MORNING"
        query_row = {
            'sample_id': 'Q1',
            'vessel_name': 'PACIFIC MORNING',
            'call_sign': 'CALL_X',
            'gross_tonnage': 50000.0,
            'international_gross_tonnage': 50000.0,
            'vessel_type': 'Cargo',
            'flag_name': 'Panama'
        }
        cands = index.channel_candidates(query_row, 'S')
        # Candidate 0 ("MORNING PACIFIC") must be retrieved despite reversed word order
        self.assertIn(0, cands)

    def test_high_similarity_tonnage_penalty_gating(self):
        # Catalog has candidate 0 with sim=1.0 and 15% tonnage difference,
        # and 35 random noise candidates with 0% tonnage difference but low similarity (sim=0.20).
        cat_names = ['PACIFIC CARRIER'] + [f'UNRELATED SHIP {i}' for i in range(35)]
        cat_gts = [10000.0 * 1.15] + [10000.0 for _ in range(35)]
        vids = [f'V_{i:03}' for i in range(36)]
        catalog = pd.DataFrame({
            'vessel_id': vids,
            'vessel_name': cat_names,
            'gross_tonnage': cat_gts,
            'vessel_type': ['Cargo'] * 36,
            'flag_name': ['Panama'] * 36
        })
        train = pd.DataFrame([{
            'sample_id': 'S0',
            'vessel_name': 'PACIFIC CARRIER',
            'call_sign': 'C0',
            'gross_tonnage': 10000.0,
            'international_gross_tonnage': 10000.0,
            'vessel_type': 'Cargo',
            'flag_name': 'Panama',
            'target_vessel_id': 'V_000'
        }])
        assets = make_assets(train, catalog, retrieval_channels=['S'])
        index = CatalogIndex(catalog, assets)

        query_row = {
            'sample_id': 'Q1',
            'vessel_name': 'PACIFIC CARRIER',
            'call_sign': 'CX',
            'gross_tonnage': 10000.0,
            'international_gross_tonnage': 10000.0,
            'vessel_type': 'Cargo',
            'flag_name': 'Panama'
        }
        all_cands = list(range(36))
        # Pruning with max_candidates=10
        pruned = index.prune_candidates_by_tonnage(query_row, all_cands, max_candidates=10)
        # Candidate 0 must be in the top 10 pruned candidates
        self.assertIn(0, pruned)


if __name__ == '__main__':
    unittest.main()
