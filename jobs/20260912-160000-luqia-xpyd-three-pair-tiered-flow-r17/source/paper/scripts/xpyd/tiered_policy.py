"""Noise-aware fixed-tier selection from this run only, not historical priors."""
import math


def select_medium(candidates, table, high, energy_band=.01, headroom=.10, ceiling=.80):
    representatives = {}
    for workload_id, entry in table.items():
        value=entry['value']
        if value is None or not value['slo_met']:
            continue
        frequencies=[]
        for axis,key in [('P','prefill_frequency_mhz'),('D','decode_frequency_mhz')]:
            rr=[r for r in candidates if r.get('workload_id')==workload_id
                and r.get('axis')==axis and r.get('slo_met')
                and r['ttft_ms'] < 500*(1-headroom) and r['tpot_ms'] <= 200*(1-headroom)
                and math.isfinite(r['measured_energy_j']) and r['measured_energy_j']>0]
            if not rr:break
            best=min(r['measured_energy_j'] for r in rr)
            near=[r for r in rr if r['measured_energy_j']<=best*(1+energy_band)]
            frequencies.append(min(r[key] for r in near))
        if len(frequencies)==2 and all(f<=h*ceiling for f,h in zip(frequencies,high)):
            representatives[workload_id]=tuple(frequencies)
    if not representatives:
        return dict(frequencies=None,representatives={},eligible=[],status='no_medium_candidate')
    # One fixed shared tier must cover each admitted class's representative.
    common=tuple(max(f[i] for f in representatives.values()) for i in (0,1))
    return dict(frequencies=common,representatives=representatives,eligible=[],
                status='awaiting_joint_confirmation')


def route_pair(workload_id, state, force_high=False):
    if not force_high and workload_id in state.get('eligible',[]):
        return ('P2','D2'),'medium_tier_confirmed'
    return ('P1','D1'),'high_tier_unknown_failed_or_ineligible'
