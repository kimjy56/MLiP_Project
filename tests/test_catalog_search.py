import unittest
from name_decoder.catalog_search import CatalogNameSearch

class CatalogSearchTests(unittest.TestCase):
    def test_short_partial_match(self):
        for query, short, gold in [("POLAR BRAIT", "POLA", "POLAR BRIGHT"),
                                   ("KITI BOM", "BO", "KITI BHUM")]:
            with self.subTest(query=query):
                self.assertEqual(CatalogNameSearch([gold, short], "WRatio").top_names(query)[0], short)
                self.assertEqual(CatalogNameSearch([gold, short]).top_names(query)[0], gold)

    def test_order_independence_and_catalog_membership(self):
        names = ["CAT", "BAT", "CAT", "SHIP 1", "SHIP 10"]
        search = CatalogNameSearch(names)
        expected = {q: search.top_names(q) for q in ["HAT", "SHIP 1", ""]}
        reverse = CatalogNameSearch(reversed(names))
        for q in reversed(list(expected)):
            self.assertEqual(reverse.top_names(q), expected[q])
        self.assertEqual(expected["HAT"][:2], ["BAT", "CAT"])
        self.assertEqual(expected["SHIP 1"][0], "SHIP 1")
        self.assertTrue(all(set(v) <= set(names) for v in expected.values()))

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            CatalogNameSearch([])
        with self.assertRaises(ValueError):
            CatalogNameSearch(["A"], "partial_ratio")
        with self.assertRaises(ValueError):
            CatalogNameSearch(["A"]).top_names("A", limit=11)

if __name__ == "__main__":
    unittest.main()
