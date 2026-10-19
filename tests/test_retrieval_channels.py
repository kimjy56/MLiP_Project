import itertools
import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model'))
from matcher import CatalogIndex, make_assets, validate_channels
from ngram_retrieval import CharacterNgramIndex


class RetrievalContracts(unittest.TestCase):
    def test_ngram_ties_missing_and_query_independence(self):
        index=CharacterNgramIndex(['ALPHA STAR','ALPHA STAR','BETA MOON'])
        expected=index.search('alpha star',2)
        self.assertEqual(expected,[0,1])
        before=index.vectorizer.idf_.copy()
        for name in [None,'','베타 문','xxxxxxx','ALPHA']:
            index.search(name,3)
        self.assertEqual(index.search('alpha star',2),expected)
        self.assertTrue((before==index.vectorizer.idf_).all())
        self.assertEqual(index.search(None,3),[])
        self.assertEqual(index.search('xxxxxxx',3),[])
        self.assertEqual(CharacterNgramIndex(['A','']).search('A',3),[])

    def test_all_sixteen_unions_and_empty_fallback(self):
        catalog=pd.DataFrame({'vessel_id':['A','B','C'],
            'vessel_name':['ALPHA STAR','ALPHA MOON','BETA STAR'],
            'gross_tonnage':[100,200,300],'vessel_type':['cargo']*3,'flag_name':['KR']*3})
        train=pd.DataFrame([{'target_vessel_id':'A','call_sign':'C1','vessel_name':'알파 스타',
            'gross_tonnage':100,'international_gross_tonnage':100,'vessel_type':'cargo','flag_name':'KR'}])
        assets=make_assets(train,catalog,'all',24,list('SNPT'))
        index=CatalogIndex(catalog,assets)
        row=train.iloc[0].to_dict()
        groups={c:set(index.channel_candidates(row,c)) for c in 'SNPT'}
        for bits in itertools.product([False,True],repeat=4):
            channels=[c for c,on in zip('SNPT',bits) if on]
            index.retrieval_channels=channels
            expected=sorted(set().union(*(groups[c] for c in channels))) or [0]
            self.assertEqual(index.candidates(row,use_aliases=False),expected)
        with self.assertRaises(ValueError):validate_channels(['N','N'])
        with self.assertRaises(ValueError):validate_channels(['unknown'])


if __name__=='__main__':unittest.main()
