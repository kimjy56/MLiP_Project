import re
import unittest
from name_decoder.pronunciation_ownership import PronunciationOwnershipGuard


class TestPronunciationOwnership(unittest.TestCase):
    def guard(self, baseline):
        return PronunciationOwnershipGuard(baseline,
            to_jamo=lambda text:re.sub(r'[\d\s]','',text),
            number_signature=lambda text:tuple(re.findall(r'\d+',text)))

    def test_accepts_unique_owner(self):
        guard=self.guard({'OWN':['abcdef'],'OTHER':['uvwxyz']})
        self.assertTrue(guard.check('OWN','abcdeg')[0])

    def test_rejects_competing_owner(self):
        guard=self.guard({'OWN':['abcdef'],'OTHER':['abcxef']})
        self.assertEqual(guard.check('OWN','abcxeg'),(False,'competing_owner_beats_or_ties'))

    def test_rejects_ties_and_missing_anchors(self):
        guard=self.guard({'OWN':['abcxde'],'OTHER':['abcwde']})
        self.assertFalse(guard.check('OWN','abcyde')[0])
        self.assertEqual(guard.check('MISSING','abcdef'),(False,'no_original_anchor'))

    def test_own_multiple_hits_cannot_hide_competitor(self):
        guard=self.guard({'OWN':['abcxde','abcwde'],'OTHER':['abczde']})
        self.assertEqual(guard.hit_limit,3)
        self.assertFalse(guard.check('OWN','abcyde')[0])

    def test_numbers_separate_competitors_and_duplicate_anchors_do_not_inflate_limit(self):
        guard=self.guard({'OWN':['abcdef 7','abcdef 7'],'OTHER':['abcdeg 8']})
        self.assertEqual(guard.hit_limit,2)
        self.assertTrue(guard.check('OWN','abcdeg 7')[0])
        self.assertFalse(guard.check('OWN','abcdeg 8')[0])


if __name__=='__main__':unittest.main()