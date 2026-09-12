import unittest
from xpyd.tiered_policy import select_medium,route_pair


class TieredPolicyTests(unittest.TestCase):
    def test_near_tie_prefers_lower_clock_with_headroom(self):
        rows=[]
        for axis,key in [('P','prefill_frequency_mhz'),('D','decode_frequency_mhz')]:
            for freq,energy,ttft in [(900,100.2,420),(1110,100.1,390),(1200,100,370)]:
                rows.append(dict(workload_id='small_light',axis=axis,slo_met=True,
                    measured_energy_j=energy,ttft_ms=ttft,tpot_ms=60,**{key:freq}))
        state=select_medium(rows,{'small_light':{'value':{'slo_met':True}}},(2520,1500))
        self.assertEqual(state['frequencies'],(900,900))
        self.assertEqual(route_pair('small_light',state)[0],('P1','D1'))
        state['eligible']=['small_light']
        self.assertEqual(route_pair('small_light',state)[0],('P2','D2'))
        self.assertEqual(route_pair('unknown',state)[0],('P1','D1'))
        self.assertEqual(route_pair('small_light',state,True)[0],('P1','D1'))

    def test_joint_tier_covers_class_representatives_and_excludes_fallback(self):
        rows=[]
        table={w:{'value':{'slo_met':True}} for w in ['small_light','prefill_medium']}
        table['both_heavy']={'value':{'slo_met':False}}
        for w,p,d in [('small_light',900,900),('prefill_medium',1110,975)]:
            for axis,key,f in [('P','prefill_frequency_mhz',p),('D','decode_frequency_mhz',d)]:
                rows.append(dict(workload_id=w,axis=axis,slo_met=True,measured_energy_j=100,
                    ttft_ms=300,tpot_ms=60,**{key:f}))
        state=select_medium(rows,table,(2520,1500))
        self.assertEqual(state['frequencies'],(1110,975))
        self.assertNotIn('both_heavy',state['representatives'])

    def test_no_candidates_does_not_invent_medium(self):
        state=select_medium([],{},(2520,1500))
        self.assertIsNone(state['frequencies'])
        self.assertEqual(route_pair('small_light',state)[0],('P1','D1'))
