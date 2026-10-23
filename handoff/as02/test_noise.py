import unittest
from independent_noise import noise_copy


class NoiseTests(unittest.TestCase):
    def test_preserve_boundaries_digits_and_identity(self):
        cat = {'vessel_id':'V1','gross_tonnage':1000}
        for i in range(300):
            row = {'sample_id':str(i),'target_vessel_id':'V1','vessel_name':'ABCDEFGH 12',
                   'gross_tonnage':1000,'international_gross_tonnage':1000,'flag_name':'K'}
            original = dict(row)
            copy, flags = noise_copy(row,cat)
            self.assertEqual(row, original)
            self.assertEqual((copy,flags), noise_copy(row,cat))
            word,number = copy['vessel_name'].split()
            self.assertEqual(number,'12')
            self.assertTrue(word.startswith('AB') and word.endswith('GH'))
            self.assertEqual(copy['target_vessel_id'],'V1')
            if flags['ton_missing']:
                self.assertEqual(copy['gross_tonnage'],'')
                self.assertEqual(copy['international_gross_tonnage'],'')
            elif flags['ton_shift_applied']:
                self.assertTrue(.03 <= abs(copy['gross_tonnage']/1000-1) <= .05)

    def test_korean_unchanged(self):
        row = {'sample_id':'X','target_vessel_id':'V1','vessel_name':'한국호'}
        copy,flags = noise_copy(row,{'vessel_id':'V1'})
        self.assertEqual(copy,row)
        self.assertFalse(flags['eligible'])


if __name__ == '__main__':
    unittest.main()
