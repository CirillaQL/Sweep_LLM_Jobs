# XpYd Phase 3C real multi-endpoint substrate validation

Verdict: **PASS**

This is infrastructure validation with deterministic baseline routing; it makes no routing-novelty or optimization claim.

| Pair | Requests |
|---|---:|
| P0->D0 | 1 |
| P0->D1 | 1 |
| P1->D0 | 1 |
| P1->D1 | 1 |

| Endpoint | Role | Node/GPU | Requests | Gross energy (J) | Clock valid |
|---|---|---|---:|---:|---|
| D0 | decode | ganymede/0 | 2 | 1347.813 | True |
| D1 | decode | ganymede/1 | 2 | 1330.098 | True |
| P0 | prefill | uranus/0 | 2 | 2333.652 | True |
| P1 | prefill | uranus/1 | 2 | 2375.508 | True |

Hard gates: {"endpoint_assignment": true, "endpoint_coverage": true, "explicit_compatibility": true, "fixed_clocks": true, "logical_request_id": true, "no_invalidating_thermal_or_hw_slowdown": true, "no_resource_overlap": true, "nvml_energy_windows": true, "pair_coverage": true, "phase3a_token_semantics": true, "real_sse_and_latency": true, "requested_output_tokens": true}
