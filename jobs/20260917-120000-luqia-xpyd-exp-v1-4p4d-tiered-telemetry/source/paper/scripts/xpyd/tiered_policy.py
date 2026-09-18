"""Noise-aware selection plus exp-v1's validated live P/D route control."""
import json
import math
import os
from pathlib import Path


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


def fixed_tier_route(tier_name, tiers, force_high=False):
    """Return one immutable production pair and frequency target for exp-v1."""
    selected = "high" if force_high else str(tier_name)
    if selected not in ("high", "middle", "low"):
        raise ValueError("tier must be high, middle, or low")
    value = dict(tiers[selected])
    pair = tuple(value["pair"])
    if len(pair) != 2 or not pair[0].startswith("P") or not pair[1].startswith("D"):
        raise ValueError("tier pair must contain one P and one D endpoint")
    return selected, pair, (int(value["prefill_mhz"]), int(value["decode_mhz"]))


def controlled_tier_route(tier_name, flow, force_high=False):
    """Resolve one high/middle/low slot to any supported production P/D pair.

    A missing control file retains the configured same-index default. An invalid
    control value is ignored and returned as audit evidence instead of changing
    the active route. Forced-high evaluation never accepts a live override.
    """
    selected = "high" if force_high else str(tier_name)
    if selected not in ("high", "middle", "low"):
        raise ValueError("tier must be high, middle, or low")
    default = tuple(flow["tiers"][selected]["pair"])
    endpoints = dict(flow["endpoint_frequency_mhz"])
    source = "default_same_index"
    control_error = None
    control_path = os.path.expandvars(str(flow.get("route_control_path", "")))
    if control_path and not force_high:
        path = Path(control_path)
        if path.exists():
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(document, dict) or document.get("schema_version") != 1:
                    raise ValueError("route control must be a schema_version=1 object")
                routes = document.get("routes", {})
                if not isinstance(routes, dict):
                    raise ValueError("route control routes must be an object")
                override = routes.get(selected)
                if override is not None:
                    if isinstance(override, dict):
                        pair = (override["prefill_endpoint_id"], override["decode_endpoint_id"])
                    else:
                        pair = tuple(override)
                    if len(pair) != 2 or pair[0] not in ("P1", "P2", "P3") or pair[1] not in ("D1", "D2", "D3"):
                        raise ValueError("route must be one P1-P3 and one D1-D3 endpoint")
                    default = pair
                    source = "route_control_file"
            except (OSError, ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
                control_error = "%s: %s" % (type(exc).__name__, exc)
    if default[0] not in endpoints or default[1] not in endpoints:
        raise ValueError("route references an endpoint without a frequency target")
    return selected, default, (int(endpoints[default[0]]), int(endpoints[default[1]])), source, control_error
