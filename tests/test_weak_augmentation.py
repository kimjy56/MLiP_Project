import unittest
import pandas as pd
from experiments.weak_augmentation import augment


class WeakAugmentationTests(unittest.TestCase):
    def test_deterministic_digits_ids_and_source_preserved(self):
        training = pd.DataFrame([dict(sample_id=str(i),target_vessel_id='V'+str(i),
            vessel_name='OCEAN STAR 12' if i%2 else '대한민국선박 12',call_sign='CALL',
            gross_tonnage='100',international_gross_tonnage='101',flag_name='K',vessel_type='C')
            for i in range(100)])
        catalog=training[['target_vessel_id','vessel_name']].rename(columns={'target_vessel_id':'vessel_id'})
        before=training.copy(deep=True)
        a,stats=augment(training,catalog,fraction=1)
        b,stats2=augment(training,catalog,fraction=1)
        pd.testing.assert_frame_equal(a,b)
        pd.testing.assert_frame_equal(training,before)
        self.assertEqual(stats,stats2)
        self.assertEqual(set(a.target_vessel_id),set(training.target_vessel_id))
        self.assertTrue(a.vessel_name.str.contains('12').all())
        self.assertTrue(a.call_sign.eq('').all())
        self.assertGreater(stats['operations']['delete_one_letter'],0)

    def test_no_change_with_zero_fraction(self):
        frame=pd.DataFrame([dict(sample_id='1',target_vessel_id='A',vessel_name='ALPHA',
            call_sign='',gross_tonnage='',international_gross_tonnage='',flag_name='',vessel_type='')])
        catalog=pd.DataFrame({'vessel_id':['A'],'vessel_name':['ALPHA']})
        result,stats=augment(frame,catalog,fraction=0)
        self.assertTrue(result.empty)
        self.assertEqual(stats['rows'],0)

    def test_known_wrong_id_collision_is_rejected(self):
        frame=pd.DataFrame([dict(sample_id=str(i),target_vessel_id='A',vessel_name='ALPHA',
            call_sign='',gross_tonnage=str(i),international_gross_tonnage='',flag_name='',vessel_type='') for i in range(100)])
        catalog=pd.DataFrame({'vessel_id':['A']+['B']*5,
            'vessel_name':['ALPHA','LPHA','APHA','ALHA','ALPA','ALPH']})
        result,stats=augment(frame,catalog,fraction=1)
        self.assertGreater(stats['operations']['rejected_known_collision'],0)
        self.assertTrue(result.vessel_name.eq('ALPHA').all())


if __name__=='__main__':
    unittest.main()
