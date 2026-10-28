"""Requested named examples: post-generation diagnostics only, never training fixtures."""
import json
import os
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
MAP=Path(os.environ.get("EXAONE_ACCEPTANCE_MAP",ROOT/"artifacts/LEX-EXAONE-GENERATE-01/catalog_exaone_ko_map.json"))


class GeneratedAcceptance(unittest.TestCase):
    def check_case(self,name,expected):
        if not MAP.is_file():
            self.skipTest("Real EXAONE map not generated; model weights unavailable")
        mapping=json.loads(MAP.read_text(encoding="utf-8"))
        self.assertTrue(set(mapping.get(name,[]))&set(expected),"Requested pronunciation absent; diagnostic only, do not inject labels")

    def test_bow_cecil(self):
        self.check_case("BOW CECIL",["바우 세실"])

    def test_cnc_dream(self):
        self.check_case("CNC DREAM",["씨엔씨 드림"])

    def test_keoyoung_star3(self):
        self.check_case("KEOYOUNG STAR 3",["거영 스타 3","거영스타3"])


if __name__=="__main__":
    unittest.main()
