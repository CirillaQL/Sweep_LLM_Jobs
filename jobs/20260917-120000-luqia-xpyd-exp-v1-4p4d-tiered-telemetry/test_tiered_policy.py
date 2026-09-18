import json
from pathlib import Path
import tempfile
import unittest
from xpyd.tiered_policy import controlled_tier_route,fixed_tier_route,select_medium,route_pair


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

    def test_exp_v1_fixed_tiers_use_distinct_endpoint_pairs(self):
        tiers={
            'high': {'pair':['P1','D1'],'prefill_mhz':2520,'decode_mhz':1500},
            'middle': {'pair':['P2','D2'],'prefill_mhz':1710,'decode_mhz':975},
            'low': {'pair':['P3','D3'],'prefill_mhz':900,'decode_mhz':450},
        }
        self.assertEqual(fixed_tier_route('high',tiers),('high',('P1','D1'),(2520,1500)))
        self.assertEqual(fixed_tier_route('middle',tiers),('middle',('P2','D2'),(1710,975)))
        self.assertEqual(fixed_tier_route('low',tiers),('low',('P3','D3'),(900,450)))
        self.assertEqual(fixed_tier_route('low',tiers,force_high=True)[0],'high')

    def test_route_control_can_cross_connect_and_invalid_control_falls_back(self):
        flow={
            'tiers': {
                'high': {'pair':['P1','D1']},
                'middle': {'pair':['P2','D2']},
                'low': {'pair':['P3','D3']},
            },
            'endpoint_frequency_mhz': {'P1':2520,'P2':1710,'P3':900,'D1':1500,'D2':975,'D3':450},
        }
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'routes.json'
            flow['route_control_path']=str(path)
            self.assertEqual(controlled_tier_route('middle',flow)[1:4],(('P2','D2'),(1710,975),'default_same_index'))
            path.write_text(json.dumps({'schema_version':1,'routes':{'middle':{'prefill_endpoint_id':'P1','decode_endpoint_id':'D2'}}}))
            tier,pair,targets,source,error=controlled_tier_route('middle',flow)
            self.assertEqual((tier,pair,targets,source,error),('middle',('P1','D2'),(2520,975),'route_control_file',None))
            path.write_text(json.dumps({'schema_version':1,'routes':{'high':{'prefill_endpoint_id':'P3','decode_endpoint_id':'D2'}}}))
            self.assertEqual(controlled_tier_route('middle',flow,force_high=True)[:4],
                             ('high',('P1','D1'),(2520,1500),'default_same_index'))
            path.write_text('{bad json')
            self.assertEqual(controlled_tier_route('middle',flow)[1],('P2','D2'))
            self.assertIsNotNone(controlled_tier_route('middle',flow)[4])
            path.write_text(json.dumps({'schema_version':2,'routes':{}}))
            self.assertIsNotNone(controlled_tier_route('middle',flow)[4])
